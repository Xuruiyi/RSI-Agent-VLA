"""Evaluation-owned revisions of completed development attempts; no new execution."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from .storage import atomic_json, identifier

SCHEMA = "rsi-development-evaluation-revision-v1"
SOURCES = (
    "RPent/robots/libero/result.py",
    "core/src/rsi_vla/adapters/rpent.py",
    "core/src/rsi_vla/adapters/runtime_closure.py",
    "core/src/rsi_vla/adapters/runtime_acceptance.py",
    "core/src/rsi_vla/development.py",
    "core/src/rsi_vla/development_reparse.py",
    "core/src/rsi_vla/task_completion.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(
                "conflicting development evaluation revision: " + str(path)
            )
        return
    atomic_json(path, value)


def verify_sources(configuration):
    manifest_ref = Path(configuration["source_manifest_ref"]).resolve(strict=True)
    manifest = read(manifest_ref)
    if not set(SOURCES) <= set(manifest["files"]):
        raise PermissionError(
            "development reparse requires the complete reviewed consumers"
        )
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("reviewed reparse source escapes repository")
        if (Path(configuration["repository_root"]) / relative).read_bytes() != (
            manifest_ref.parent / "source" / relative
        ).read_bytes():
            raise PermissionError(
                "reviewed development reparse source changed: " + name
            )
    return manifest_ref


def replay_native_finalizer(run_dir, revision_root):
    """Replay the existing robot finalizer from exactly the original runner facts."""
    from robots.libero.result import finalize_run
    from rpent.evaluation import RunFinalizationContext

    root, destination = Path(run_dir).resolve(), Path(revision_root).resolve()
    if (
        root == destination
        or destination.is_relative_to(root)
        or root.is_relative_to(destination)
    ):
        raise PermissionError(
            "evaluation revision must be outside original Runtime output"
        )
    raw = read(root / "result.json")
    transcripts = list(root.glob("transcript*.json"))
    if len(transcripts) != 1:
        raise ValueError("one original runner transcript required for reparse")
    transcript = read(transcripts[0])
    metrics = {
        key: raw.get(key)
        for key in (
            "terminated",
            "truncated",
            "mode",
            "attempt",
            "tool_successes",
            "runtime",
            "execution_arm",
            "episode_execution",
        )
    }
    planner = raw["planner"]
    context = RunFinalizationContext(
        output_dir=destination,
        robot_name=raw["robot"],
        task_desc=raw["task"],
        environment_success=transcript.get("environment_success"),
        environment_metrics=metrics,
        agent_error=raw.get("error"),
        elapsed_s=raw["elapsed_s"],
        planner=planner["backend"],
        model=planner["model"],
        reasoning_effort=planner["reasoning_effort"],
        max_turns=planner["max_turns"],
        planner_timeout_s=planner["timeout_s"],
        finish_result=raw.get("planner_finish"),
        stats=transcript.get("stats", {}),
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected = Path(finalize_run(context)).resolve()
    derived = read(selected)
    mutable = {
        "benchmark_success",
        "benchmark_status",
        "attempt_closed",
        "termination_reason",
        "verified_budget_exhaustion",
        "attempt_outcome_source",
        "missing",
        "interrupted",
    }
    if {key: value for key, value in derived.items() if key not in mutable} != {
        key: value for key, value in raw.items() if key not in mutable
    }:
        raise ValueError(
            "robot reparse changed original facts beyond terminal interpretation: "
            + ", ".join(
                sorted(
                    key
                    for key in set(raw) | set(derived)
                    if key not in mutable and raw.get(key) != derived.get(key)
                )
            )
        )
    return selected


def resolve_development_evaluation(
    evaluation_ref, run_dir, *, evaluation_store_ref, protocol=None
):
    """Resolve an explicit evaluation-owned version with original facts still bound."""
    root = Path(run_dir).resolve()
    store_root = Path(evaluation_store_ref).resolve()
    path = Path(evaluation_ref).resolve(strict=True)
    if not path.is_relative_to(store_root) or path.is_relative_to(root):
        raise PermissionError(
            "evaluation revision is outside the trusted evaluation store"
        )
    receipt = read(path)
    context = read(root / "runtime_context.json")
    identity = {
        key: context[key]
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("identity") != identity
        or receipt.get("run_dir") != str(root)
        or receipt.get("evaluation_store_ref") != str(store_root)
    ):
        raise PermissionError("evaluation revision belongs to another Runtime or store")
    for key, name in (
        ("context", "runtime_context.json"),
        ("raw_result", "result.json"),
        ("previous_independent_result", "independent_result.json"),
    ):
        if receipt["inputs"][key] != read(root / name):
            raise ValueError("original Runtime facts changed after evaluation revision")
    if protocol is not None and receipt["inputs"]["protocol"] != protocol:
        raise ValueError(
            "evaluation revision changes the original protocol or thresholds"
        )
    for reference in receipt["outputs"].values():
        if not Path(reference).resolve(strict=True).is_relative_to(path.parent):
            raise PermissionError("evaluation output escapes its registered revision")
    if read(receipt["outputs"]["audit_ref"])["identity"] != identity:
        raise ValueError("revised independent audit identity mismatch")
    verify_sources(receipt["source_configuration"])
    return receipt


def reparse_development_attempt(configuration, payload):
    """Create a new robot result, independent closure and audit in the evaluation role."""
    if os.environ.get("RSI_ISOLATED_ROLE") != "evaluation":
        raise PermissionError(
            "development reparse requires the independent evaluation role"
        )
    root = Path(payload["output_ref"]).resolve(strict=True)
    store_root = Path(configuration["evaluation_store_ref"]).resolve()
    revision_id = identifier(payload["revision_id"])
    context = read(root / "runtime_context.json")
    identity = {
        key: context[key]
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    slot = next(
        item
        for item in configuration["runtime_slots"]
        if item["run_id"] == identity["run_id"]
    )
    if (
        Path(slot["output_ref"]).resolve() != root
        or identity["protocol_id"] != read(configuration["protocol_ref"])["protocol_id"]
    ):
        raise PermissionError("reparse output differs from its registered Runtime slot")
    destination = (
        store_root
        / "development_revisions"
        / identifier(identity["run_id"])
        / revision_id
    )
    receipt_ref = destination / "revision.json"
    requested_ref = Path(payload["revision_ref"]).resolve()
    if requested_ref != receipt_ref:
        raise PermissionError(
            "reparse revision reference differs from its evaluation identity"
        )
    if receipt_ref.exists():
        receipt = resolve_development_evaluation(
            receipt_ref,
            root,
            evaluation_store_ref=store_root,
            protocol=read(configuration["protocol_ref"]),
        )
        audit = read(receipt["outputs"]["audit_ref"])
        return {
            "passed": audit["passed"],
            "outcome": audit["outcome"],
            "revision_ref": str(receipt_ref),
            "audit_ref": receipt["outputs"]["audit_ref"],
        }
    source_ref = verify_sources(configuration)
    original = {
        "context": context,
        "raw_result": read(root / "result.json"),
        "previous_independent_result": read(root / "independent_result.json"),
        "protocol": read(configuration["protocol_ref"]),
        "previous_audit": read(payload["previous_audit_ref"]),
    }
    if original["previous_audit"]["identity"] != identity:
        raise ValueError("previous audit belongs to another Runtime")
    if read(configuration["frozen_protocol_ref"]) != original["protocol"]:
        raise PermissionError("reparse changes frozen protocol thresholds")
    immutable(
        destination / "revision-input.json",
        {
            "revision_id": revision_id,
            "inputs": original,
            "source_manifest_ref": str(source_ref),
        },
    )
    robot_result_ref = replay_native_finalizer(root, destination)
    from .adapters.runtime_closure import finalize_runtime
    from .adapters.runtime_acceptance import inspect_attempt

    finalize_runtime(
        context, raw_result_path=robot_result_ref, derived_root=destination
    )
    independent_ref = destination / "independent_result.json"
    audit = inspect_attempt(root, require_vla=True, result_path=independent_ref)
    if audit["costs"] != original["previous_audit"]["costs"]:
        raise ValueError("offline evaluation revision changed historical Runtime costs")
    atomic_json(destination / "independent-audit.json", audit)
    receipt = {
        "schema": SCHEMA,
        "revision_id": revision_id,
        "identity": identity,
        "run_dir": str(root),
        "evaluation_store_ref": str(store_root),
        "inputs": original,
        "source_configuration": {
            key: configuration[key]
            for key in ("source_manifest_ref", "repository_root")
        },
        "outputs": {
            "robot_result_ref": str(robot_result_ref),
            "independent_result_ref": str(independent_ref),
            "cost_breakdown_ref": str(destination / "cost_breakdown.json"),
            "audit_ref": str(destination / "independent-audit.json"),
        },
        "original_refs": {
            "raw_result_ref": str(root / "result.json"),
            "independent_result_ref": str(root / "independent_result.json"),
            "cost_breakdown_ref": str(root / "cost_breakdown.json"),
            "audit_ref": str(Path(payload["previous_audit_ref"]).resolve()),
        },
        "original_source_snapshot_ref": context.get("code_snapshot_ref"),
        "actual_execution": False,
        "historical_costs_recharged": False,
        "thresholds_changed": False,
    }
    immutable(receipt_ref, receipt)
    return {
        "passed": audit["passed"],
        "outcome": audit["outcome"],
        "revision_ref": str(receipt_ref),
        "audit_ref": receipt["outputs"]["audit_ref"],
    }


def remaining_reparse_plan(configuration, progress, run_id, revision_id):
    """Build a linked remaining graph without reopening any completed native slot."""
    if progress.get("status") != "completed" or progress.get("paused"):
        raise PermissionError(
            "previous graph must have legally completed before evaluation repair"
        )
    core = deepcopy(configuration)
    slot = next(item for item in core["runtime_slots"] if item["run_id"] == run_id)
    runtime = next(
        item
        for item in core["operations"]
        if item["kind"] == "runtime" and item["payload"]["run_id"] == run_id
    )
    collect = next(
        item
        for item in core["operations"]
        if item["kind"] == "development_base_collect"
        and item["payload"]["output_ref"] == slot["output_ref"]
    )
    imported = next(
        item
        for item in core["operations"]
        if item["kind"] == "development_base_import"
        and item["payload"]["output_ref"] == slot["output_ref"]
    )
    if (
        runtime["operation_id"] not in progress["receipts"]
        or progress["receipts"]
        .get(collect["operation_id"], {})
        .get("value", {})
        .get("passed")
        is not False
        or imported["operation_id"] in progress["receipts"]
    ):
        raise PermissionError(
            "repair requires the completed Runtime and original rejected audit, before import"
        )
    identifier(revision_id)
    revision_ref = (
        Path(core["evaluation_store_ref"]).resolve()
        / "development_revisions"
        / identifier(run_id)
        / revision_id
        / "revision.json"
    )
    old_collect_id = collect["operation_id"]
    collect["operation_id"] = identifier(old_collect_id + "-" + revision_id)
    collect["kind"] = "development_reparse"
    collect["payload"] = {
        "output_ref": slot["output_ref"],
        "revision_id": revision_id,
        "revision_ref": str(revision_ref),
        "previous_audit_ref": collect["payload"]["audit_ref"],
    }
    imported["payload"]["evaluation_revision_ref"] = str(revision_ref)
    imported["payload"]["audit_ref"] = str(
        revision_ref.parent / "independent-audit.json"
    )
    nodes = core["continuation"]["nodes"]
    start = next(
        key
        for key, value in nodes.items()
        if value.get("operation_id") == old_collect_id
    )
    import_node = next(
        key
        for key, value in nodes.items()
        if value.get("operation_id") == imported["operation_id"]
    )
    nodes[start]["operation_id"] = collect["operation_id"]
    # The original raw v1 intentionally remains unchanged. Keep this run's
    # derivatives instead of feeding its v1 into the ordinary retention gate.
    nodes[start]["transitions"] = [
        {"when": {"path": ["passed"], "equals": True}, "next": import_node},
        nodes[start]["transitions"][-1],
    ]
    core["continuation"]["start"] = start
    reached = set()

    def visit(key):
        if key in reached:
            return
        reached.add(key)
        for edge in nodes[key].get("transitions", []):
            visit(edge["next"])

    visit(start)
    core["continuation"]["nodes"] = {
        key: value for key, value in nodes.items() if key in reached
    }
    used = {
        value["operation_id"]
        for value in core["continuation"]["nodes"].values()
        if "operation_id" in value
    }
    if used.intersection(progress["receipts"]):
        raise PermissionError("remaining graph would repeat a completed operation")
    core["operations"] = [
        item for item in core["operations"] if item["operation_id"] in used
    ]
    core["evaluation_reparse"] = {
        "run_id": run_id,
        "revision_id": revision_id,
        "revision_ref": str(revision_ref),
        "original_collect_operation_id": old_collect_id,
        "reparse_operation_id": collect["operation_id"],
        "completed_operations_not_replayed": list(progress["receipts"]),
        "runtime_restarted": False,
        "gpu_technical_recovery": False,
        "all_original_runtime_slots_preserved": True,
        "reparsed_run_retention_skipped": True,
    }
    return core


def configure_reparse_research(
    core,
    destination,
    *,
    result_ids=None,
    development_task=None,
    media_refs=None,
    sdk_projection=None,
):
    """Narrow registered Research inputs and copy their policies before registration."""
    selected = list(dict.fromkeys(result_ids)) if result_ids is not None else None
    if selected is not None and not selected:
        raise ValueError(
            "Research input selection requires at least one registered result"
        )
    if media_refs not in (None, "selected_events") or sdk_projection not in (
        None,
        "terminal_history_v1",
    ):
        raise ValueError("unsupported registered Research input projection")
    requests = core.get("research_requests", [])
    if selected is not None and any(
        not set(selected) <= set(request["result_ids"]) for request in requests
    ):
        raise PermissionError(
            "Research results must be a subset of each original registered request"
        )
    if development_task is not None:
        development = read(core["protocol_ref"])["task_sets"]["development"]
        if (
            development_task not in development["task_keys"]
            or 2 not in development["seeds"]
            or not any(
                slot["task_key"] == development_task and slot["seed"] == 2
                for slot in core["runtime_slots"]
            )
        ):
            raise PermissionError(
                "Research development task/seed 2 is outside registered development slots"
            )
    domains, copied = {}, {}
    for request in requests:
        original_ref = str(Path(request["domain_policy_ref"]).resolve())
        if original_ref not in copied:
            policy = read(original_ref)
            selection = (
                policy.setdefault("evidence_selection", {})
                if media_refs is not None or sdk_projection is not None
                else None
            )
            if media_refs is not None:
                selection["media_refs"] = media_refs
            if sdk_projection is not None:
                selection["sdk_projection"] = sdk_projection
            if selected is not None or development_task is not None:
                task = development_task or "matched400"
                hint = (
                    f" for {development_task}/seed 2"
                    if development_task is not None
                    else ""
                )
                policy["purpose"] = (
                    f"Diagnose the selected {task} development attempts and their recorded costs. "
                    f"Propose a bounded later-development hint{hint}, request exact legal development events, or defer. "
                    "This call is text-only."
                )
            copied[original_ref] = str(
                Path(destination)
                / "research-domains"
                / (identifier(request["request_id"]) + ".json")
            )
            domains[copied[original_ref]] = policy
        request["domain_policy_ref"] = copied[original_ref]
        if selected is not None:
            request["result_ids"] = list(selected)
        if development_task is not None:
            request["development_slots"] = [{"task_key": development_task, "seed": 2}]
    return domains


def prepare_reparse_stage(
    previous_configuration_ref,
    *,
    run_id,
    revision_id,
    stage_id,
    research_result_ids=None,
    research_development_task=None,
    research_media_refs=None,
    research_sdk_projection=None,
):
    """Prepare CPU evaluation repair and remaining operations for the existing stage activator."""
    from .midterm_host import MidtermHost, build_configuration

    identifier(stage_id)
    previous = MidtermHost(previous_configuration_ref)
    scope = previous.ledger.authority()
    progress = previous.coordinator().status()
    core = remaining_reparse_plan(previous.core_config, progress, run_id, revision_id)
    destination = previous.ledger.root / "operator" / stage_id
    domains = configure_reparse_research(
        core,
        destination,
        result_ids=research_result_ids,
        development_task=research_development_task,
        media_refs=research_media_refs,
        sdk_projection=research_sdk_projection,
    )
    for path, policy in domains.items():
        immutable(path, policy)
    source_ref = destination / "source/manifest.json"
    original_source = Path(core["source_manifest_ref"])
    manifest = read(original_source)
    files = list(
        dict.fromkeys(
            [*manifest["files"], *SOURCES, "scripts/prepare_development_reparse.py"]
        )
    )
    changed = []
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("source path escapes repository")
        payload = (Path(core["repository_root"]) / relative).read_bytes()
        before = original_source.parent / "source" / relative
        if not before.exists() or before.read_bytes() != payload:
            changed.append(name)
        target = source_ref.parent / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("reviewed reparse source changed after registration")
        target.write_bytes(payload)
    immutable(
        source_ref,
        {
            **manifest,
            "files": files,
            "source_revision": revision_id,
            "changed_files": changed,
            "parent_manifest_ref": str(original_source),
            "reason": "versioned native budget-owner evaluation correction; no physical task changes",
        },
    )
    core["source_manifest_ref"] = str(source_ref)
    core["evaluation_reparse"]["previous_configuration_ref"] = str(
        previous.configuration_ref
    )
    core_ref = destination / "continuation-configuration.json"
    immutable(core_ref, core)
    inputs = read(previous.config["inputs_ref"])
    inputs["authorization"]["binding"]["registration_ref"] = str(
        destination / "continuation/registration.json"
    )
    inputs["authorization"]["operation_ids"] = [
        item["operation_id"] for item in core["operations"] if item["dynamic"]
    ]
    immutable(destination / "inputs.json", inputs)
    host_ref = destination / "host.json"
    host = build_configuration(
        core_ref,
        host_ref,
        previous.config["hub_configuration_ref"],
        python=previous.config["python"],
        script_ref=previous.config["script_ref"],
        poll_seconds=previous.config["poll_seconds"],
    )
    for operation in core["operations"]:
        identity = operation["operation_id"]
        if identity in previous.config["operations"]:
            host["operations"][identity] = deepcopy(
                previous.config["operations"][identity]
            )
    host.update(
        host_root=str(destination / "host-state"),
        transport_root=str(destination / "transport"),
        node=previous.config["node"],
        autoloop_root=previous.config["autoloop_root"],
        autoloop_home=previous.config["autoloop_home"],
        previous_configuration_ref=str(previous.configuration_ref),
        phase_id=scope["scope_id"] + "-" + stage_id,
        preparation_repair=core["evaluation_reparse"],
    )
    immutable(host_ref, host)
    registered = MidtermHost(host_ref)
    registered.coordinator()
    result = {
        "scope_id": scope["scope_id"],
        "configuration_ref": str(host_ref),
        "source_manifest_ref": str(source_ref),
        "revision_ref": core["evaluation_reparse"]["revision_ref"],
        "remaining_runtime_count": sum(
            item["kind"] == "runtime" for item in core["operations"]
        ),
        "original_runtime_slot_count": len(core["runtime_slots"]),
        "actual_dispatch": False,
        "previous_progress_copied_or_reopened": False,
        "gpu_technical_recovery": False,
    }
    immutable(destination / "registration-receipt.json", result)
    return result
