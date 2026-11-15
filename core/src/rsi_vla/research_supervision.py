"""Complete per-request Research budgets and separate task-control accounting."""

from __future__ import annotations

import hashlib
import ast
import json
import os
import inspect
import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .storage import atomic_json, identifier, ResearchStore
from .process_accounting import StorageMeter


def read(path):
    return json.loads(Path(path).read_text())


def request_for_operation(host, operation):
    request_id = operation.get("payload", {}).get("request_id")
    return next(
        (
            row
            for row in host.core_config.get("research_requests", [])
            if row["request_id"] == request_id
        ),
        None,
    )


def business_roots(host, request):
    """Include every unique request archive, permit, SDK/scratch and business receipt."""
    rid = identifier(request["request_id"])
    store = Path(host.core_config["research_store_ref"])
    roots = [
        request["output_ref"],
        request["scratch_ref"],
        request["permit_ref"],
        request["preflight_ref"],
        request["domain_policy_ref"],
        str(store / "*" / (rid + ".json")),
        str(store / "*" / (rid + "-*.json")),
        str(store / "jobs" / (identifier(request["job_id"]) + ".json")),
        str(store / "bundles" / (identifier(request["reserved_bundle_id"]) + ".json")),
    ]
    for operation in host.core_config["operations"]:
        if operation.get("payload", {}).get("request_id") != rid:
            continue
        oid = identifier(operation["operation_id"])
        roots += [
            str(host.root / "jobs" / oid),
            str(Path(host.config["continuation_root"]) / "receipts" / (oid + ".json")),
            str(
                Path(host.core_config["isolation_task_root"])
                / "operator/role_requests"
                / (oid + "-*.json")
            ),
            str(store / "role_receipts" / (oid + "-*.json")),
        ]
        feedback = operation["payload"].get("feedback_id")
        if feedback:
            roots.append(str(store / "*" / (identifier(feedback) + "*.json")))
    meter_root = host.root / "research-meters" / rid
    roots += [
        str(meter_root / "lifecycle.json"),
        str(meter_root / "business-state.json"),
    ]
    return list(dict.fromkeys(roots))


def lifecycle(host, operation):
    if host.core_config.get("research_request_envelope") != "complete_v2":
        return None
    request = request_for_operation(host, operation)
    if request is None:
        return None
    directory = host.root / "research-meters" / identifier(request["request_id"])
    path = directory / "lifecycle.json"
    if path.exists():
        return read(path)
    now = datetime.now(timezone.utc)
    cap = read(host.core_config["protocol_ref"])["task_policy"]["research_per_request"]
    value = {
        "schema": "research-request-lifecycle-v2",
        "scope_id": host.config["scope_id"],
        "request_id": request["request_id"],
        "owner_uid": os.getuid(),
        "started_at": now.isoformat(),
        "deadline": (now + timedelta(seconds=cap["wall_seconds"])).isoformat(),
        "storage_bytes": cap["storage_bytes"],
        "business_roots": business_roots(host, request),
        "business_state_ref": str(directory / "business-state.json"),
        "scope_state_ref": str(host.root / "scope-storage-state.json"),
    }
    from .process_accounting import logical_snapshot

    shared = [
        str(
            Path(host.core_config["research_store_ref"])
            / "jobs"
            / (identifier(request["job_id"]) + ".json")
        ),
        str(
            Path(host.core_config["research_store_ref"])
            / "bundles"
            / (identifier(request["reserved_bundle_id"]) + ".json")
        ),
    ]
    business = StorageMeter(
        value["business_roots"],
        baseline={
            "schema": "logical-storage-baseline-v2",
            "files": logical_snapshot(shared),
        },
    )
    atomic_json(value["business_state_ref"], business.state())
    if not Path(value["scope_state_ref"]).exists():
        atomic_json(value["scope_state_ref"], StorageMeter([host.ledger.root]).state())
    atomic_json(path, value)
    return value


def complete_context(host, operation, context):
    value = lifecycle(host, operation)
    if value is None:
        return context
    remaining = (
        datetime.fromisoformat(value["deadline"]) - datetime.now(timezone.utc)
    ).total_seconds()
    if remaining <= 0:
        raise PermissionError(
            "complete Research request deadline expired; no new model dispatch"
        )
    state = host.ledger.read()
    held = sum(
        host.ledger._held(row, "storage_bytes")
        for key, row in state["reservations"].items()
        if key != operation["operation_id"]
    )
    reserve = sum(
        row["upper_bound_bytes"]
        for row in state.get("historical_control_uncertainties", [])
    )
    return {
        **context,
        "limits": {
            **context["limits"],
            "wall_seconds": remaining,
            "storage_bytes": value["storage_bytes"],
        },
        "storage_write_roots": value["business_roots"],
        "storage_meter_state_ref": value["business_state_ref"],
        "scope_accounting_roots": [str(host.ledger.root)],
        "scope_meter_state_ref": value["scope_state_ref"],
        "scope_storage_remaining_bytes": state["limits"]["storage_bytes"]
        - state["used"]["storage_bytes"]
        - held
        - reserve,
        "scope_current_disk_limit_bytes": state["limits"]["current_disk_bytes"],
        "request_lifecycle_ref": str(
            Path(value["business_state_ref"]).parent / "lifecycle.json"
        ),
    }


def compact_input_binding(path):
    """Bind a readonly permit to exact archive content without duplicating its body."""
    path = Path(path).resolve(strict=True)
    return {
        "input_ref": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def bill_control(host, operation_id, report):
    amount = report.get("control_storage_delta_lower_bound")
    if amount is None:
        return
    with host.ledger.lock():
        state = host.ledger._read()
        rows = state.setdefault("control_storage_charges", [])
        entry = {
            "operation_id": operation_id,
            "storage_bytes": amount,
            "meter_version": "logical-path-v2",
            "scope_report_ref": str(
                host.root / "jobs" / operation_id / "supervisor.json"
            ),
        }
        if any(row["operation_id"] == operation_id for row in rows):
            if entry not in rows:
                raise ValueError("conflicting control-storage charge")
        else:
            rows.append(entry)
            state["used"]["storage_bytes"] += amount
            atomic_json(host.ledger.path, state)
    phase = read(host.core_config["phase_ref"])
    entries = phase.setdefault("host_control_storage_charges", [])
    if entry not in entries:
        entries.append(entry)
        phase["used"]["storage_bytes"] += amount
        phase["remaining"]["storage_bytes"] = max(
            0, phase["limits"]["storage_bytes"] - phase["used"]["storage_bytes"]
        )
        atomic_json(host.core_config["phase_ref"], phase)


def bill_research_overhead(host, operation, entry):
    """Keep prepare/feedback and command overhead in the same original job."""
    if host.core_config.get("research_request_envelope") != "complete_v2":
        return
    request = request_for_operation(host, operation)
    if request is None:
        return
    from .research import ResearchJob

    store = ResearchStore(host.core_config["research_store_ref"])
    job = ResearchJob(
        store,
        request["job_id"],
        request.get("base_bundle_id", host.core_config["base_bundle_id"]),
        request.get("job_limits"),
    )
    costs = {
        "whole_job_wall_seconds": entry["wall_seconds"],
        "artifact_bytes": entry["storage_bytes"],
    }
    costs = {key: value for key, value in costs.items() if key in job.state["limits"]}
    reason = "complete Research host overhead " + operation["operation_id"]
    previous = [row for row in job.state["charges"] if row["reason"] == reason]
    expected = {"reason": reason, "costs": costs}
    if previous:
        if previous != [expected]:
            raise ValueError("complete Research overhead charge changed")
    else:
        job.consume(costs, reason)


def research_actor_evidence(host):
    """Verify the actual new stdio SDK process inside the role's fresh PID namespace."""
    import openai_codex
    from openai_codex.client import CodexClient

    manifest = Path(host.core_config["source_manifest_ref"])
    base = manifest.parent / "source"
    bounded = base / "core/src/rsi_vla/bounded_runner.py"
    isolation = base / "scripts/run_isolated_role.py"
    sdk_api = Path(inspect.getsourcefile(openai_codex.Codex))
    sdk_client = Path(inspect.getsourcefile(CodexClient))
    verified = (
        "with openai_codex.Codex(config=config) as client:" in bounded.read_text()
        and '"--unshare-pid"' in isolation.read_text()
        and '"--die-with-parent"' in isolation.read_text()
        and "self._client.start()" in inspect.getsource(openai_codex.Codex.__init__)
        and "subprocess.Popen(" in inspect.getsource(CodexClient.start)
        and "stdio://" in inspect.getsource(CodexClient.start)
    )
    return {
        "established": verified,
        "basis": "new Codex client starts a new stdio app-server in the isolated role namespace",
        "source_refs": [str(bounded), str(isolation), str(sdk_api), str(sdk_client)],
    }


def historical_control_reserve(host, operation_id, report):
    """Bound the actual logical-growth uncertainty, not repeated identical writes."""
    stop_ref = host.root / "observer-storage-stop.json"
    if not stop_ref.exists():
        return None
    stop = read(stop_ref)
    job = host.root / "jobs" / operation_id
    start = datetime.fromisoformat(read(job / "submission.json")["at"]) + timedelta(
        seconds=report["wall_seconds"]
    )
    end = datetime.fromisoformat(stop["at"])
    if end <= start:
        return None
    core = host.root / "observer/core-state.json"
    progress = Path(host.config["continuation_root"]) / "coordinator/progress.json"
    signals = progress.parent / "signals.json"
    stable_core = (
        core.exists()
        and core.read_bytes() == progress.read_bytes()
        and progress.stat().st_mtime <= start.timestamp()
        and (not signals.exists() or signals.stat().st_mtime <= start.timestamp())
    )
    hub = host.root / "observer/hub-state.json"
    database_ref = Path(host.config["autoloop_home"]) / "hub.db"
    with sqlite3.connect(
        "file:" + str(database_ref) + "?mode=ro", uri=True
    ) as database:
        count, max_fact = database.execute(
            "SELECT count(*),coalesce(max(length(fact)),0) FROM facts WHERE json_extract(fact,'$.at')>=? AND json_extract(fact,'$.at')<=?",
            (start.timestamp() * 1000, end.timestamp() * 1000),
        ).fetchone()
    hub_bound = (hub.stat().st_size if hub.exists() else 0) + max_fact
    core_bound = (
        0
        if stable_core
        else (core.stat().st_size if core.exists() else progress.stat().st_size)
    )
    why = sum(
        path.stat().st_size
        for path in (host.root / "observer").glob("run-*-why.json")
        if path.stat().st_mtime >= start.timestamp()
    )
    database_files_bound = sum(
        path.stat().st_size
        for path in (
            database_ref,
            Path(str(database_ref) + "-wal"),
            Path(str(database_ref) + "-shm"),
        )
        if path.exists()
    )
    upper = (count + 1) * (hub_bound + core_bound + database_files_bound) + why
    entry = {
        "operation_id": operation_id,
        "from": start.isoformat(),
        "until": end.isoformat(),
        "actual_additional_logical_growth": "unknown",
        "upper_bound_bytes": upper,
        "upper_bound_is_reserve_not_actual": True,
        "semantic_fact_changes": count,
        "same_core_snapshot_rewrites_excluded": stable_core,
        "retained_sqlite_control_files_bound_bytes": database_files_bound,
        "basis": "unchanged logical snapshot rewrites cost no new bytes; retained snapshot sizes plus actual hub change count bound unknown positive growth",
        "observer_stop_ref": str(stop_ref),
        "legacy_v1_reports_retained_no_refund": True,
    }
    with host.ledger.lock():
        state = host.ledger._read()
        entries = state.setdefault("historical_control_uncertainties", [])
        previous = next(
            (row for row in entries if row["operation_id"] == operation_id), None
        )
        if previous is None:
            entries.append(entry)
            state.setdefault("storage_measurement_versions", []).append(
                {
                    "from": "inode-v1",
                    "to": "logical-path-v2",
                    "legacy_overcount": "known possible; exact history unknown; no refund",
                }
            )
            atomic_json(host.ledger.path, state)
        elif previous != entry:
            raise ValueError("historical control uncertainty evidence changed")
    return entry


def research_policy_rejection_evidence(host, operation_id, report):
    """Recompute the original source gate without applying the saved response again."""
    from .task_completion import TaskCompletionBackend
    from .research_tools import opportunity_source_eligibility

    operation = next(
        row
        for row in host.core_config["operations"]
        if row["operation_id"] == operation_id
    )
    request = request_for_operation(host, operation)
    if operation["kind"] != "research" or request is None:
        return None
    output, job = Path(request["output_ref"]), host.root / "jobs" / operation_id
    store = ResearchStore(host.core_config["research_store_ref"])
    required = [
        output / "sdk-result.json",
        output / "request_gate.json",
        output / "supervisor.json",
        output / "model_requests.jsonl",
        store.path("llm_executions", request["request_id"]),
    ]
    if not all(path.exists() for path in required):
        return None
    context, outer = (
        read(output / "runtime_context.json"),
        read(job / "runtime_context.json"),
    )
    identity = {
        "run_id": request["request_id"],
        "episode_id": "research-only",
        "bundle_id": request.get("base_bundle_id", host.core_config["base_bundle_id"]),
        "protocol_id": read(host.core_config["protocol_ref"])["protocol_id"],
    }
    sdk, gate, native = (
        read(output / "sdk-result.json"),
        read(output / "request_gate.json"),
        read(output / "supervisor.json"),
    )
    events = [
        json.loads(line)
        for line in (output / "model_requests.jsonl").read_text().splitlines()
        if line
    ]
    admitted = [row for row in events if row.get("kind") == "request_admitted"]
    terminal = [row for row in events if row.get("kind") == "response_terminal"]
    if (
        report.get("status") != "completed"
        or report.get("exit_code") != 1
        or report.get("stop_reason") is not None
        or report.get("cleanup_confirmed") is not True
        or report.get("remaining_process_ids") != []
        or report["storage_bytes"] > outer["limits"]["storage_bytes"]
        or report["wall_seconds"] > outer["limits"]["wall_seconds"]
        or any(
            context.get(key) != value or native.get("identity", {}).get(key) != value
            for key, value in identity.items()
        )
        or Path(context["authorization_ref"]).resolve()
        != Path(request["permit_ref"]).resolve()
        or native.get("status") != "completed"
        or native.get("exit_code") != 0
        or native.get("cleanup_confirmed") is not True
        or native.get("remaining_process_ids") != []
        or gate.get("identity") != identity
        or gate.get("requests_forwarded") != 1
        or gate.get("cleanup_complete") is not True
        or gate.get("active_requests_at_close") != 0
        or gate.get("unfinished_request_numbers") != []
        or str(sdk.get("status")).rsplit(".", 1)[-1] != "completed"
        or len(admitted) != 1
        or len(terminal) != 1
        or admitted[0].get("identity") != identity
        or terminal[0].get("identity") != identity
        or admitted[0].get("model") != "gpt-6.1-sol"
        or admitted[0].get("reasoning_effort") != "high"
        or terminal[0].get("backend_model") != "gpt-6.1-sol"
    ):
        return None
    response = json.loads(sdk["final_response"])
    if response.get("request_id") != request["request_id"] or response.get(
        "source"
    ) not in {"normal", "failure", "technical_diagnostic"}:
        return None
    if not response.get("result_ids") or not set(response["result_ids"]) <= set(
        request["result_ids"]
    ):
        return None
    execution = store.get("llm_executions", request["request_id"])
    usage = terminal[0].get("usage") or {}
    tokens = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": (usage.get("input_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }
    if (
        execution.get("request_id") != request["request_id"]
        or execution.get("job_id") != request["job_id"]
        or execution.get("base_bundle_id") != identity["bundle_id"]
        or execution.get("costs", {}).get("model_requests") != 1
        or execution.get("transport_facts", {}).get("effective_model") != "gpt-6.1-sol"
        or execution.get("transport_facts", {}).get("effective_reasoning_effort")
        != "high"
        or any(
            type(value) is not int or value < 0 or execution["costs"].get(key) != value
            for key, value in tokens.items()
        )
    ):
        return None
    tools, _ = TaskCompletionBackend(host.core_config).tools(request["request_id"])
    results = tools.query_development(response["result_ids"])
    eligibility = opportunity_source_eligibility(response["source"], results)
    if eligibility["eligible"]:
        return None
    expected_error = "ValueError: " + eligibility["error"]
    error_refs = []
    for path in (store.root / "role_receipts").glob(
        operation_id + "-execute-revision-*.json"
    ):
        value = read(path)
        if (
            value.get("role") == "research"
            and value.get("request", {}).get("configuration") == host.core_config
            and value["request"].get("task", {}).get("operation") == operation
            and value["request"]["task"].get("action") == "execute"
            and value.get("error") == expected_error
            and "value" not in value
        ):
            error_refs.append(str(path))
    policy_source = (
        Path(host.core_config["source_manifest_ref"]).parent
        / "source/core/src/rsi_vla/research_tools.py"
    )
    if not error_refs or eligibility["error"] not in policy_source.read_text():
        return None
    if (
        (output / "validated-response.json").exists()
        or store.path("opportunities", request["request_id"] + "-opportunity").exists()
        or store.path("research_decisions", request["request_id"] + "-defer").exists()
        or store.path("bounded_proposals", request["request_id"]).exists()
    ):
        return None
    supplement_ref = store.path("research_evidence_supplements", request["request_id"])
    supplement = (
        store.get("research_evidence_supplements", request["request_id"])
        if supplement_ref.exists()
        else None
    )
    if supplement and (
        supplement.get("job_id") != request["job_id"]
        or supplement.get("request_id") != request["request_id"]
        or supplement.get("base_bundle_id") != identity["bundle_id"]
    ):
        return None
    return {
        "policy_rejection_kind": "source_evidence_ineligible",
        "source_original": response["source"],
        "decision_original": response["decision"],
        "eligibility": eligibility,
        "role_error_refs": error_refs,
        "policy_source_ref": str(policy_source),
        "response_ref": str(output / "sdk-result.json"),
        "response_transport_status": "completed",
        "response_application": "policy_rejected",
        "llm_execution_ref": str(store.path("llm_executions", request["request_id"])),
        "response_tokens": tokens,
        "evidence_supplement_ref": str(supplement_ref) if supplement else None,
        "saved_evidence_read_groups": len(supplement["reads"]) if supplement else 0,
        "additional_evidence_forwarded_after_read": False,
        "opportunity_created": False,
        "defer_created": False,
        "schema_validation_replayed": False,
        "response_application_replayed": False,
    }


def close_policy_rejected_research(host, operation_id, report):
    """Publish a typed business rejection backed by the original native records."""
    from .midterm_host import (
        immutable_json,
        inspect_owned_runtime_processes,
        capture_closure_sources,
    )

    operation = next(
        row
        for row in host.core_config["operations"]
        if row["operation_id"] == operation_id
    )
    job = host.root / "jobs" / identifier(operation_id)
    target = job / "host-ended-receipt.json"
    if target.exists():
        return read(target)
    evidence = research_policy_rejection_evidence(host, operation_id, report)
    if evidence is None:
        raise PermissionError(
            "original Research facts do not establish a typed source-policy rejection"
        )
    request = request_for_operation(host, operation)
    record = host.ledger.read()["reservations"][operation_id]
    scope = host.ledger.authority(active=False)
    submission, process = read(job / "submission.json"), read(job / "process.json")
    if (
        operation["dynamic"] is not True
        or submission.get("operation") != operation
        or submission.get("owner_uid") != scope["owner_uid"]
        or process.get("owner_uid") != scope["owner_uid"]
        or not all(process.get(key) for key in ("pid", "start_ticks", "boot_id"))
        or record.get("status") != "settled"
        or record.get("cleanup_confirmed") is not True
        or not record.get("submitted_at")
        or record.get("scope_id") != scope["scope_id"]
        or record.get("owner_uid") != scope["owner_uid"]
        or record.get("operation_id") != operation_id
        or record.get("budget") != host.config["operations"][operation_id]["budget"]
        or record.get("actual", {}).get("model_requests") != 1
        or record["actual"].get("research_requests") != 1
        or record.get("phase_ref") != host.core_config["phase_ref"]
        or record.get("gpu_devices") != []
        or report.get("identity", {}).get("run_id") != operation_id
    ):
        raise PermissionError(
            "policy rejection owner/commit/settled native costs are unresolved"
        )
    actors = research_actor_evidence(host)
    if not actors["established"]:
        raise PermissionError("fresh Research SDK ownership source proof is absent")
    processes = inspect_owned_runtime_processes(
        host.root / "host-process.json",
        operation_id,
        Path(request["output_ref"]),
        operation_anchor_ref=job / "process.json",
        actor_evidence=actors,
    )
    if processes["live_processes"] or processes["unreadable_owned_processes"]:
        raise PermissionError(
            "policy-rejected Research owners are not fully reconciled"
        )
    source = capture_closure_sources(
        host, host.root / "control-policy-rejection-source"
    )
    closure = {
        "schema": "rsi-host-ended-operation-receipt-v1",
        "scope_id": scope["scope_id"],
        "owner_uid": scope["owner_uid"],
        "operation": operation,
        "status": "completed",
        "value": {
            "operation_id": operation_id,
            "request_id": request["request_id"],
            "execution_ended": True,
            "execution_status": "policy_rejected",
            "research_policy_rejected": True,
            "research_outcome": "policy_rejected",
            "model_requests": 1,
            "research_requests": 1,
            "supervisor_ref": str(job / "supervisor.json"),
            "actual": record["actual"],
            "process_evidence": processes,
            "source_manifest_ref": str(source),
            "original_costs_already_settled": True,
            "model_retried": False,
            "source_label_changed": False,
            "gpu_technical_recovery": False,
            **evidence,
        },
    }
    immutable_json(target, closure)
    return closure


def close_failed_research_preparation(host, operation_id, report):
    """Close the proved CPU selector failure before any native Research submission."""
    from .midterm_host import (
        immutable_json,
        inspect_owned_runtime_processes,
        capture_closure_sources,
    )

    operation = next(
        row
        for row in host.core_config["operations"]
        if row["operation_id"] == operation_id
    )
    template = host.config["operations"][operation_id]
    request = request_for_operation(host, operation)
    if (
        operation["kind"] != "research_prepare"
        or operation["dynamic"] is not False
        or request is None
        or template.get("gpu_devices")
        or set(template["budget"]) - {"storage_bytes", "current_disk_bytes"}
    ):
        raise PermissionError(
            "CPU preparation closure cannot admit Runtime/model/native work"
        )
    job = host.root / "jobs" / identifier(operation_id)
    target = job / "host-ended-receipt.json"
    if target.exists():
        return read(target)
    original, process = read(job / "submission.json"), read(job / "process.json")
    record = host.ledger.read()["reservations"][operation_id]
    scope = host.ledger.authority(active=False)
    if (
        original.get("operation") != operation
        or original.get("owner_uid") != scope["owner_uid"]
        or process.get("owner_uid") != scope["owner_uid"]
        or not all(process.get(key) for key in ("pid", "start_ticks", "boot_id"))
        or record.get("scope_id") != scope["scope_id"]
        or record.get("owner_uid") != scope["owner_uid"]
        or record.get("operation_id") != operation_id
        or record.get("budget") != template["budget"]
        or record.get("phase_ref") != host.core_config["phase_ref"]
        or record.get("gpu_devices") != []
        or record.get("status") != "settled"
        or record.get("cleanup_confirmed") is not True
        or not record.get("submitted_at")
        or report.get("identity", {}).get("run_id") != operation_id
        or report.get("status") != "completed"
        or report.get("exit_code") != 1
        or report.get("cleanup_confirmed") is not True
        or report.get("remaining_process_ids") != []
    ):
        raise PermissionError(
            "original CPU preparation owner/commit/end/costs are unresolved"
        )
    output = Path(request["output_ref"])
    store = ResearchStore(host.core_config["research_store_ref"])
    if (
        output.exists()
        or store.path("bounded_requests", request["request_id"]).exists()
        or store.path("bounded_transports", request["request_id"]).exists()
    ):
        raise PermissionError(
            "native output or completed preparation exists; CPU absent-submission closure refused"
        )
    archive = store.get("llm_requests", request["request_id"])
    if (
        archive.get("request_id") != request["request_id"]
        or archive.get("job_id") != request["job_id"]
        or archive.get("bundle_id")
        != request.get("base_bundle_id", host.core_config["base_bundle_id"])
    ):
        raise PermissionError("partial request archive belongs to another registration")
    policy = read(request["domain_policy_ref"])
    projection = policy.get("evidence_input", {}).get("sdk_projection")
    if (
        projection != "terminal_history_catalog_v2"
        or policy.get("evidence_selection", {}).get("sdk_projection") != projection
    ):
        raise PermissionError("CPU failure is not the registered catalog selector case")
    snapshot = Path(host.core_config["source_manifest_ref"]).parent / "source"
    transport_source = snapshot / "core/src/rsi_vla/research_transport.py"
    tree = ast.parse(transport_source.read_text())
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_input"
    )
    expected_error, allowed = None, None
    for node in ast.walk(selector):
        test = node.test if isinstance(node, ast.If) else None
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "projection"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotIn)
        ):
            values = ast.literal_eval(test.comparators[0])
            for branch in node.body:
                if (
                    isinstance(branch, ast.Raise)
                    and isinstance(branch.exc, ast.Call)
                    and isinstance(branch.exc.func, ast.Name)
                    and branch.exc.func.id == "ValueError"
                ):
                    expected_error = "ValueError: " + ast.literal_eval(
                        branch.exc.args[0]
                    )
                    allowed = list(values)
    if (
        expected_error != "ValueError: unsupported explicit SDK field projection"
        or projection in (allowed or ())
    ):
        raise PermissionError("frozen source does not prove the failed selector branch")
    receipts = []
    for path in (store.root / "role_receipts").glob(
        operation_id + "-execute-revision-*.json"
    ):
        value = read(path)
        if (
            value.get("request", {}).get("configuration") == host.core_config
            and value["request"].get("task", {}).get("operation") == operation
        ):
            if (
                path.stat().st_uid != scope["owner_uid"]
                or value.get("role") != "research"
                or value["request"]["task"].get("action") != "execute"
                or value.get("error") != expected_error
                or "value" in value
            ):
                raise PermissionError(
                    "original role return contradicts the CPU selector failure"
                )
            receipts.append(str(path))
    if not receipts:
        raise PermissionError("original bound CPU preparation error receipt is absent")
    isolation = snapshot / "scripts/run_isolated_role.py"
    prepare_source = (
        (snapshot / "core/src/rsi_vla/task_completion.py")
        .read_text()
        .split('            if kind == "research_prepare":', 1)[1]
        .split("            from rsi_vla.bounded_runner import execute", 1)[0]
    )
    actor = {
        "established": "adapter.prepare_request(" in prepare_source
        and "execute(" not in prepare_source
        and '"--unshare-pid"' in isolation.read_text()
        and '"--die-with-parent"' in isolation.read_text(),
        "basis": "registered CPU prepare branch failed in select_input before native execute; isolated role has fresh PID namespace",
        "source_refs": [
            str(transport_source),
            str(snapshot / "core/src/rsi_vla/task_completion.py"),
            str(isolation),
        ],
    }
    if not actor["established"]:
        raise PermissionError(
            "CPU-only prepare and isolated ownership source proof is absent"
        )
    processes = inspect_owned_runtime_processes(
        host.root / "host-process.json",
        operation_id,
        output,
        operation_anchor_ref=job / "process.json",
        actor_evidence=actor,
    )
    if processes["live_processes"] or processes["unreadable_owned_processes"]:
        raise PermissionError("CPU preparation owners are not fully reconciled")
    source = capture_closure_sources(host, job / "closure-control/source")
    closure = {
        "schema": "rsi-host-ended-operation-receipt-v1",
        "scope_id": scope["scope_id"],
        "owner_uid": scope["owner_uid"],
        "operation": operation,
        "status": "completed",
        "value": {
            "operation_id": operation_id,
            "request_id": request["request_id"],
            "execution_ended": True,
            "execution_status": "failed_cpu_preparation",
            "native_outcome": "not_started",
            "research_technical_stop": True,
            "research_preparation_technical_stop": True,
            "technical_terminal": "research_preparation_technical_stop",
            "model_requests": 0,
            "research_requests": 0,
            "supervisor_ref": str(job / "supervisor.json"),
            "role_error_refs": receipts,
            "partial_archive_ref": str(
                store.path("llm_requests", request["request_id"])
            ),
            "selector_source_ref": str(transport_source),
            "requested_projection": projection,
            "failed_allowed_projections": allowed,
            "process_evidence": processes,
            "source_manifest_ref": str(source),
            "actual": record["actual"],
            "original_costs_already_settled": True,
            "original_lifecycle_reset": False,
            "native_result_synthesized": False,
            "old_request_resubmitted": False,
            "gpu_technical_recovery": False,
        },
    }
    immutable_json(target, closure)
    return closure


def withheld_response_evidence(host, request, job, report, identity, events):
    """Read a completed native response that the complete request budget forbids applying."""
    output = Path(request["output_ref"])
    sdk_ref = output / "sdk-result.json"
    terminal = [row for row in events if row.get("kind") == "response_terminal"]
    if not sdk_ref.exists() and not terminal:
        return None
    sdk, native, gate = (
        read(sdk_ref),
        read(output / "supervisor.json"),
        read(output / "request_gate.json"),
    )
    context = read(job / "runtime_context.json")
    if (
        report.get("stop_reason") != "storage_bytes"
        or report["storage_bytes"] <= context["limits"]["storage_bytes"]
        or str(sdk.get("status")).rsplit(".", 1)[-1] != "completed"
        or any(
            native.get("identity", {}).get(key) != value
            for key, value in identity.items()
        )
        or native.get("status") != "completed"
        or native.get("exit_code") != 0
        or native.get("cleanup_confirmed") is not True
        or native.get("remaining_process_ids") != []
        or gate.get("identity") != identity
        or gate.get("requests_forwarded") != 1
        or gate.get("cleanup_complete") is not True
        or gate.get("active_requests_at_close") != 0
        or gate.get("unfinished_request_numbers") != []
        or len(terminal) != 1
        or terminal[0].get("identity") != identity
        or terminal[0].get("request_number") != 1
        or terminal[0].get("backend_model") != "gpt-6.1-sol"
    ):
        raise PermissionError(
            "completed response lacks bound native closure and complete-request budget stop"
        )
    store = ResearchStore(host.core_config["research_store_ref"])
    if (output / "validated-response.json").exists() or store.path(
        "llm_executions", request["request_id"]
    ).exists():
        raise PermissionError(
            "response has entered application; budget-withheld closure requires separate adjudication"
        )
    response = json.loads(sdk["final_response"])
    if response.get("request_id") != request["request_id"]:
        raise PermissionError("completed SDK response belongs to another request")
    usage = terminal[0].get("usage") or {}
    tokens = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": (usage.get("input_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }
    if (
        any(type(value) is not int or value < 0 for value in tokens.values())
        or tokens["cached_input_tokens"] > tokens["input_tokens"]
    ):
        raise PermissionError("completed response token usage is not closed")
    return {
        "response_transport_status": "completed",
        "response_ref": str(sdk_ref),
        "response_application": "withheld_budget_stop",
        "response_schema_validation": "not_run_due_complete_request_budget_stop",
        "response_tokens": tokens,
        "response_usage_ref": str(output / "model_requests.jsonl"),
        "native_supervisor_ref": str(output / "supervisor.json"),
        "native_gate_ref": str(output / "request_gate.json"),
        "native_execution_costs_ref": str(output / "execution-costs.json"),
        "whole_request_storage_bytes": report["storage_bytes"],
        "whole_request_limit_bytes": context["limits"]["storage_bytes"],
    }


def close_ended_research(host, operation_id, report):
    """Close an ended interrupted request with known admission count and unknown response."""
    from .midterm_host import (
        inspect_owned_runtime_processes,
        capture_closure_sources,
        immutable_json,
    )

    operation = next(
        row
        for row in host.core_config["operations"]
        if row["operation_id"] == operation_id
    )
    request = request_for_operation(host, operation)
    if operation["kind"] != "research" or request is None:
        raise ValueError("ended Research requires its registered request")
    job = host.root / "jobs" / identifier(operation_id)
    target = job / "host-ended-receipt.json"
    if target.exists():
        return read(target)
    original = read(job / "submission.json")
    if (
        original["operation"] != operation
        or original["owner_uid"] != os.getuid()
        or report.get("identity", {}).get("run_id") != operation_id
        or report.get("status") != "interrupted"
        or report.get("cleanup_confirmed") is not True
        or report.get("remaining_process_ids") != []
    ):
        raise PermissionError(
            "Research supervisor owner/identity/cleanup is unresolved"
        )
    output = Path(request["output_ref"])
    context = read(output / "runtime_context.json")
    identity = {
        "run_id": request["request_id"],
        "episode_id": "research-only",
        "bundle_id": request.get("base_bundle_id", host.core_config["base_bundle_id"]),
        "protocol_id": read(host.core_config["protocol_ref"])["protocol_id"],
    }
    if (
        any(context.get(key) != value for key, value in identity.items())
        or context.get("job_id") != request["job_id"]
        or Path(context["authorization_ref"]).resolve()
        != Path(request["permit_ref"]).resolve()
        or read(output / "child-started.json").get("request_id")
        != request["request_id"]
    ):
        raise PermissionError(
            "Research native request identity differs from its original registration"
        )
    actors = research_actor_evidence(host)
    if not actors["established"]:
        raise PermissionError("fresh Research SDK/namespace source proof unavailable")
    processes = inspect_owned_runtime_processes(
        host.root / "host-process.json",
        operation_id,
        output,
        operation_anchor_ref=job / "process.json",
        actor_evidence=actors,
    )
    if processes["live_processes"] or processes["unreadable_owned_processes"]:
        raise PermissionError("Research owners/forwarders not fully reconciled")
    gate = read(output / "request_gate_live.json")
    gate_source = (
        Path(host.core_config["source_manifest_ref"]).parent
        / "source/core/src/rsi_vla/adapters/request_gate.py"
    ).read_text()
    admit = gate_source.split("    def admit(self, payload):", 1)[1].split(
        "    def observe_event(", 1
    )[0]
    post = gate_source.split("            def do_POST(self):", 1)[1]
    events_ref = output / "model_requests.jsonl"
    events = (
        [json.loads(line) for line in events_ref.read_text().splitlines() if line]
        if events_ref.exists()
        else []
    )
    forwarded = gate.get("requests_forwarded")
    admitted = [row for row in events if row.get("kind") == "request_admitted"]
    if (
        any(
            gate.get("identity", {}).get(key) != value
            for key, value in identity.items()
        )
        or type(forwarded) is not int
        or forwarded not in {0, 1}
        or len(admitted) != forwarded
        or any(
            row.get("identity") != identity
            or row.get("request_number") != 1
            or row.get("model") != "gpt-6.1-sol"
            or row.get("reasoning_effort") != "high"
            for row in admitted
        )
        or admit.index("self.sent += 1") > admit.index("self.publish_live()")
        or post.index("gate.admit(payload)") > post.index("asyncio.run(self.forward(")
    ):
        raise PermissionError(
            "exact bounded Research admission count cannot be established"
        )
    response = (
        withheld_response_evidence(host, request, job, report, identity, events)
        if forwarded
        else None
    )
    before = host.ledger.read()["reservations"][operation_id]
    if not (job / "closure-control/previous-reservation.json").exists():
        immutable_json(job / "closure-control/previous-reservation.json", before)
    actual = {
        **before.get("actual", {}),
        "model_requests": forwarded,
        "research_requests": forwarded,
    }
    actual.setdefault(
        "storage_bytes",
        report.get("business_storage_delta_bytes", report["storage_bytes"]),
    )
    current = report.get("storage_accounting", {}).get("current_new_bytes")
    if current is not None:
        actual.setdefault("current_disk_bytes", current)
    if "business_storage_delta_bytes" in report:
        host.charge_phase(operation_id, report)
    host.ledger.settle(operation_id, actual=actual, cleanup_confirmed=True)
    bill_control(host, operation_id, report)
    uncertainty = historical_control_reserve(host, operation_id, report)
    # Preserve the existing charged legacy envelope in the nested job too;
    # this is not a claim that the over-counted v1 business artifact size is exact.
    from .research import ResearchJob

    store = ResearchStore(host.core_config["research_store_ref"])
    research_job = ResearchJob(
        store,
        request["job_id"],
        request.get("base_bundle_id", host.core_config["base_bundle_id"]),
        request.get("job_limits"),
    )
    costs = {
        "model_requests": forwarded,
        "whole_job_wall_seconds": report["wall_seconds"],
        "artifact_bytes": actual["storage_bytes"],
    }
    if response:
        costs.update(response["response_tokens"])
    reason = "failed LLM startup " + request["request_id"]
    prior = [row for row in research_job.state["charges"] if row["reason"] == reason]
    if not prior and "business_storage_delta_bytes" not in report:
        research_job.consume(
            {
                key: value
                for key, value in costs.items()
                if key in research_job.state["limits"]
            },
            reason,
        )
    if forwarded:
        reason = "interrupted admitted Research " + request["request_id"]
        count_cost = {"model_requests": forwarded}
        prior = [
            row for row in research_job.state["charges"] if row["reason"] == reason
        ]
        if prior and prior != [{"reason": reason, "costs": count_cost}]:
            raise ValueError("interrupted Research admitted count changed")
        native_charges = [
            row
            for row in research_job.state["charges"]
            if row["reason"] == "llm " + request["request_id"]
        ]
        if native_charges and (
            len(native_charges) != 1
            or native_charges[0]["costs"].get("model_requests") != forwarded
        ):
            raise ValueError(
                "native Research admission charge contradicts gate evidence"
            )
        if (
            not prior
            and not native_charges
            and "business_storage_delta_bytes" in report
        ):
            research_job.consume(count_cost, reason)
        phase = read(host.core_config["phase_ref"])
        entry = {
            "operation_id": operation_id,
            "request_id": request["request_id"],
            "model_requests": forwarded,
            "evidence_ref": str(events_ref),
        }
        rows = phase.setdefault("host_ended_research_requests", [])
        if not any(row["operation_id"] == operation_id for row in rows):
            if not any(
                row["request_id"] == request["request_id"]
                for row in phase.get("research_charges", [])
            ):
                phase["used"]["model_requests"] += forwarded
                phase["remaining"]["model_requests"] = max(
                    0,
                    phase["limits"]["model_requests"] - phase["used"]["model_requests"],
                )
            rows.append(entry)
            atomic_json(host.core_config["phase_ref"], phase)
        elif entry not in rows:
            raise ValueError("interrupted Research phase count changed")
    store.put(
        "research_interrupted_costs" if forwarded else "research_startup_costs",
        request["request_id"] + ("-interrupted" if forwarded else "-startup"),
        {
            "costs": costs,
            "storage_basis": "logical-path-v2 command delta; original overshoot retained"
            if "business_storage_delta_bytes" in report
            else "legacy v1 scope measurement retained without refund; exact business bytes unknown",
            "source_ref": str(job / "supervisor.json"),
            "scope_id": host.config["scope_id"],
            "operation_id": operation_id,
        },
    )
    source = capture_closure_sources(host, job / "closure-control/source")
    value = {
        "operation_id": operation_id,
        "request_id": request["request_id"],
        "execution_ended": True,
        "execution_status": "interrupted",
        "research_technical_stop": True,
        "research_outcome": "unknown",
        "model_requests": forwarded,
        "research_requests": forwarded,
        "technical_terminal": "research_budget_stop_response_withheld"
        if response
        else (
            "research_forwarded_response_unknown"
            if forwarded
            else "research_pre_forward_technical_stop"
        ),
        "response_tokens": response["response_tokens"] if response else "unknown",
        "response_received": response is not None,
        **(response or {}),
        "supervisor_ref": str(job / "supervisor.json"),
        "process_evidence": processes,
        "gate_live_ref": str(output / "request_gate_live.json"),
        "source_manifest_ref": str(source),
        "historical_control_uncertainty": uncertainty,
        "actual": actual,
        "original_over_limit_reports_retained": True,
        "sdk_response_synthesized": False,
        "resubmitted": False,
        "gpu_technical_recovery": False,
    }
    closure = {
        "schema": "rsi-host-ended-operation-receipt-v1",
        "scope_id": host.config["scope_id"],
        "owner_uid": os.getuid(),
        "operation": operation,
        "status": "completed",
        "value": value,
    }
    immutable_json(target, closure)
    return closure


def prepare_research_storage_recovery(
    previous_configuration_ref,
    stage_id,
    suffix,
    *,
    research_sdk_projection=None,
    research_request_limit=2,
):
    """Register linked new requests after a proved old interruption legally ends."""
    from .midterm_host import MidtermHost, build_configuration, immutable_json

    identifier(stage_id)
    identifier(suffix)
    if type(research_request_limit) is not int or research_request_limit not in {1, 2}:
        raise ValueError("Research request limit must be one or two")
    if research_sdk_projection not in {
        None,
        "terminal_history_v1",
        "terminal_history_catalog_v2",
    }:
        raise ValueError("unregistered Research SDK projection")
    previous = MidtermHost(previous_configuration_ref)
    scope = previous.ledger.authority()
    progress = previous.coordinator().status()
    if progress.get("status") != "completed" or progress.get("terminal") not in {
        "research_pre_forward_technical_stop",
        "research_forwarded_response_unknown",
        "research_budget_stop_response_withheld",
        "research_preparation_technical_stop",
    }:
        raise PermissionError(
            "old Research must first complete its supported ended-request technical stop"
        )
    original = previous.core_config
    first = next(
        op for op in original["operations"] if op["kind"] == "research_prepare"
    )
    start = next(
        key
        for key, node in original["continuation"]["nodes"].items()
        if node.get("operation_id") == first["operation_id"]
    )
    mapping = {
        row["request_id"]: identifier(row["request_id"] + "-" + suffix)
        for row in original["research_requests"]
    }

    def replace(value):
        if isinstance(value, str):
            for old, new in mapping.items():
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    core = replace(deepcopy(original))
    nodes = core["continuation"]["nodes"]
    if research_request_limit == 1:
        first_request_id = mapping[first["payload"]["request_id"]]
        feedback = next(
            row
            for row in core["operations"]
            if row["kind"] == "research_feedback"
            and row["payload"]["request_id"] == first_request_id
        )
        feedback_node = next(
            node
            for node in nodes.values()
            if node.get("operation_id") == feedback["operation_id"]
        )
        terminal = (
            "supplement-limit"
            if nodes.get("supplement-limit") == {"terminal": "supplement-limit"}
            else "registered-single-request-end"
        )
        nodes.setdefault(terminal, {"terminal": "registered-single-request-end"})
        redirected = 0
        for edge in feedback_node["transitions"]:
            if edge.get("when") == {
                "path": ["next_action"],
                "equals": "prepare_evidence_followup",
            }:
                edge["next"] = terminal
                redirected += 1
        if redirected != 1:
            raise ValueError(
                "single-request graph requires the original first-feedback supplement route"
            )
    reachable = set()

    def visit(key):
        if key in reachable:
            return
        reachable.add(key)
        for edge in nodes[key].get("transitions", []):
            visit(edge["next"])

    visit(start)
    core["continuation"]["start"] = start
    core["continuation"]["nodes"] = {
        key: value for key, value in nodes.items() if key in reachable
    }
    used = {
        node["operation_id"]
        for node in core["continuation"]["nodes"].values()
        if "operation_id" in node
    }
    core["operations"] = [
        row for row in core["operations"] if row["operation_id"] in used
    ]
    if research_request_limit == 1:
        kept = {
            row["payload"]["request_id"]
            for row in core["operations"]
            if row.get("payload", {}).get("request_id")
        }
        core["research_requests"] = [
            row for row in core["research_requests"] if row["request_id"] in kept
        ]
        mapping = {old: new for old, new in mapping.items() if new in kept}
    core["research_receipt_mode"] = "reference_v1"
    core["research_request_envelope"] = "complete_v2"
    core["llm_archive_format"] = "gzip_json_v1"
    core["research_archive_record_kinds"] = ["llm_requests", "bounded_requests"]
    destination = previous.ledger.root / "operator" / stage_id
    for request in core["research_requests"]:
        if (
            Path(core["research_store_ref"])
            / "llm_requests"
            / (request["request_id"] + ".json")
        ).exists():
            raise ValueError("new Research request identity already exists")
        original_request = next(
            row
            for row in original["research_requests"]
            if mapping.get(row["request_id"]) == request["request_id"]
        )
        policy = read(original_request["domain_policy_ref"])
        if research_sdk_projection is not None:
            policy.setdefault("evidence_selection", {})["sdk_projection"] = (
                research_sdk_projection
            )
        domain = destination / "research-domains" / (request["request_id"] + ".json")
        immutable_json(domain, policy)
        request["domain_policy_ref"] = str(domain)
        request["permit_ref"] = str(
            destination / "permits" / (request["request_id"] + ".json")
        )
        request["preflight_ref"] = str(
            destination / "preflights" / (request["request_id"] + ".json")
        )
    source_ref = destination / "source/manifest.json"
    original_manifest_ref = Path(original["source_manifest_ref"])
    manifest = read(original_manifest_ref)
    files = list(
        dict.fromkeys(
            [
                *manifest["files"],
                "core/src/rsi_vla/research_supervision.py",
                "scripts/prepare_research_storage_recovery.py",
            ]
        )
    )
    changed = []
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("reviewed source path escapes repository")
        payload = (Path(core["repository_root"]) / relative).read_bytes()
        before = original_manifest_ref.parent / "source" / relative
        if not before.exists() or before.read_bytes() != payload:
            changed.append(name)
        target = source_ref.parent / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("new Research source revision changed after registration")
        target.write_bytes(payload)
    immutable_json(
        source_ref,
        {
            **manifest,
            "files": files,
            "parent_manifest_ref": str(original_manifest_ref),
            "source_revision": stage_id,
            "changed_files": changed,
            "storage_meter_version": "logical-path-v2",
        },
    )
    core["source_manifest_ref"] = str(source_ref)
    core["research_storage_recovery"] = {
        "previous_configuration_ref": str(previous.configuration_ref),
        "request_mapping": mapping,
        "old_requests_not_resubmitted": True,
        "research_sdk_projection": research_sdk_projection,
        "research_request_limit": research_request_limit,
        "evidence_followup_dispatch_authorized": any(
            row["kind"] == "evidence_followup_prepare" for row in core["operations"]
        ),
        "limits_raised": False,
        "scope_clock_reset": False,
    }
    core_ref = destination / "continuation-configuration.json"
    immutable_json(core_ref, core)
    inputs = read(previous.config["inputs_ref"])
    inputs["authorization"]["binding"]["registration_ref"] = str(
        destination / "continuation/registration.json"
    )
    inputs["authorization"]["operation_ids"] = [
        row["operation_id"] for row in core["operations"] if row["dynamic"]
    ]
    immutable_json(destination / "inputs.json", inputs)
    host_ref = destination / "host.json"
    host = build_configuration(
        core_ref,
        host_ref,
        previous.config["hub_configuration_ref"],
        python=previous.config["python"],
        script_ref=previous.config["script_ref"],
        poll_seconds=previous.config["poll_seconds"],
    )
    host.update(
        host_root=str(destination / "host-state"),
        transport_root=str(destination / "transport"),
        node=previous.config["node"],
        autoloop_root=previous.config["autoloop_root"],
        autoloop_home=previous.config["autoloop_home"],
        previous_configuration_ref=str(previous.configuration_ref),
        phase_id=scope["scope_id"] + "-" + stage_id,
        preparation_repair=core["research_storage_recovery"],
    )
    immutable_json(host_ref, host)
    current = MidtermHost(host_ref)
    current.coordinator()
    result = {
        "scope_id": scope["scope_id"],
        "configuration_ref": str(host_ref),
        "request_mapping": mapping,
        "complete_request_wall_seconds": 180,
        "complete_request_storage_bytes": 16 * 1024**2,
        "actual_dispatch": False,
        "research_request_limit": research_request_limit,
        "registered_research_requests": [
            row["request_id"] for row in core["research_requests"]
        ],
        "original_scope_job_phase_limits_and_costs_preserved": True,
    }
    immutable_json(destination / "registration-receipt.json", result)
    return result
