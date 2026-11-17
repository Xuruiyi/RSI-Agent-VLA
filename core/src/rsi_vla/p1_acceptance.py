# Copyright 2026 RSI-Agent-VLA Authors.
"""Independent P1 acceptance from current-scope native and control evidence."""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_json


def read(ref):
    return json.loads(Path(ref).read_text())


def capture_boundaries(configuration_ref, output_ref):
    """Probe the registered Research/Runtime mounts without model or device access."""
    from .protocol10_roles import _sandbox_command, role_isolation

    config = read(configuration_ref)
    root = Path(config["isolation_task_root"]).resolve()
    hidden = Path(config["p1_acceptance"]["hidden_probe_ref"]).resolve()
    if not hidden.is_relative_to(root / "state/evaluation") or not hidden.is_file() or not hidden.read_bytes():
        raise ValueError("P1 private input must be evaluator-owned and pre-registered")
    rows = []
    for role in ("research", "runtime"):
        program = (
            "import json,os; from pathlib import Path; "
            f"hidden=Path({str(hidden)!r}); forbidden=Path({str(root / 'release/p1-forbidden.json')!r})\n"
            "try: denied_read=(hidden.read_text()=='')\nexcept OSError: denied_read=True\n"
            "try: forbidden.write_text('forbidden')\nexcept OSError: denied_write=True\nelse: denied_write=False\n"
            "assert denied_read and denied_write\n"
            "print(json.dumps(dict(uid=os.getuid(),denied_read=denied_read,denied_write=denied_write)))"
        )
        returned = subprocess.run(
            _sandbox_command(root, role, [sys.executable, "-c", program],
                             **role_isolation(config, role)),
            capture_output=True, text=True, timeout=15, check=False,
        )
        rows.append({"role": role, "returncode": returned.returncode,
                     "stdout": returned.stdout, "stderr": returned.stderr})
    value = {"schema": "rsi-p1-role-boundaries-v1", "scope_id": read(config["scope_authorization_ref"])["scope_id"],
             "configuration_ref": str(Path(configuration_ref).resolve()), "roles": rows}
    atomic_json(output_ref, value)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("P1 role-boundary probe failed; preserve native returns")
    return value


def exercise_controls(host, evidence, *, phase="inflight"):
    """Probe pause before submission; observe a live job through its Host owner."""
    probe = host.config["engineering_probe_id"]
    job = host.root / "jobs" / probe
    if phase == "pause":
        if (job / "submission.json").exists():
            raise RuntimeError("pause probe requires the held pre-submission engineering lease")
        coordinator = host.coordinator()
        scope = host.ledger.authority()
        own_pause = scope["scope_id"] + "-p1-control-pause"
        paused = coordinator.event({"type": "pause", "event_id": own_pause,
                                    "operation_id": None, "reference": str(host.configuration_ref)})
        paused_return = host.advance()
        signals = read(coordinator.root / "signals.json")
        foreign_stop = any(row["event"]["type"] in {"pause", "cancel", "terminate"}
                           and key != own_pause for key, row in signals.items())
        if foreign_stop or host.ledger.authority(active=False)["status"] != "active":
            raise RuntimeError("external pause/cancel remains authoritative; no P1 auto-resume")
        resumed = coordinator.event({"type": "resume", "event_id": scope["scope_id"] + "-p1-control-resume",
                                     "operation_id": None, "reference": str(host.configuration_ref)})
        snapshot = {"scope_id": scope["scope_id"], "paused": paused,
                    "paused_reconcile_return": paused_return, "resumed": resumed,
                    "foreign_stop_present": foreign_stop, "manual_node_dispatches": 0}
        atomic_json(evidence / "pause-observations.json", snapshot)
        if paused_return["status"] != "paused" or resumed.get("paused") or (job / "submission.json").exists():
            raise RuntimeError("P1 pause did not block the unsubmitted operation")
        return snapshot
    submission = (job / "submission.json").read_bytes()
    process = (job / "process.json").read_bytes()
    if (job / "completion.json").exists():
        raise RuntimeError("engineering inflight control window already closed")
    # Native execution can own the Coordinator lock for its whole lifecycle.
    # Host.advance is the supported nonblocking receipt/process reconciliation.
    before, after = host.advance(), host.advance()
    unchanged = (submission == (job / "submission.json").read_bytes()
                 and process == (job / "process.json").read_bytes())
    snapshot = {**read(evidence / "pause-observations.json"),
                "schema": "rsi-p1-control-observations-v1", "probe_operation_id": probe,
                "inflight_before": before, "inflight_after": after,
                "submission_unchanged": unchanged, "process_identity": json.loads(process)}
    atomic_json(evidence / "control-observations.json", snapshot)
    if not unchanged or before["status"] != "running" or after["status"] != "running":
        raise RuntimeError("P1 unresolved native return was not safely reconciled")
    return snapshot


def evaluate_p1(configuration_ref, output_ref):
    """Compute a training-consumable receipt; missing raw evidence never passes."""
    from .adapters.runtime_acceptance import inspect_attempt
    from .adapters.runtime_closure import read_model_events, request_usage
    from .midterm_host import MidtermLedger

    config = read(configuration_ref)
    core = read(config["continuation_configuration_ref"])
    scope = read(config["scope_authorization_ref"])
    root = Path(scope["write_root"]).resolve()
    destination = Path(output_ref).resolve()
    if (destination != Path(scope["p1_acceptance_ref"]).resolve()
            or not destination.is_relative_to(root / "state/evaluation")):
        raise PermissionError("P1 acceptance must use the registered evaluator-owned path")
    settings = core["p1_acceptance"]
    evidence = root / "validation/P1-autoloop"
    host_root = Path(config.get("host_root", root / "operator/midterm-host"))
    checks, details, missing = {}, {}, []
    checks["current_scope_and_authority"] = (
        scope["scope_id"] == config["scope_id"]
        and Path(core["scope_authorization_ref"]).resolve() == Path(config["scope_authorization_ref"]).resolve()
        and scope.get("status") == "active" and scope.get("dynamic_authorized") is True
        and scope.get("owner_uid") == os.getuid()
        and datetime.now(timezone.utc) < datetime.fromisoformat(scope["expires_at"].replace("Z", "+00:00"))
        and core.get("offline_fixture") is not True
    )
    try:
        starts = read(evidence / "total-start-deliveries.json")
        deliveries = starts["deliveries"]
        checks["duplicate_delivery_is_single_start"] = (
            starts["event"]["id"] == scope["scope_id"] + "-total-start"
            and starts["event"]["type"] == "midterm.start" and len(deliveries) == 3
            and all(item.get("duplicate") is True for item in deliveries[1:])
        )
        with sqlite3.connect(f"file:{Path(config['autoloop_home']) / 'hub.db'}?mode=ro", uri=True) as db:
            one = db.execute("SELECT COUNT(*) FROM events WHERE key=?", (starts["event"]["id"],)).fetchone()[0]
            timer = db.execute("SELECT COUNT(*) FROM events WHERE type='timer.due'").fetchone()[0]
            failed = db.execute("SELECT COUNT(*) FROM runs WHERE attempts>0 OR error IS NOT NULL").fetchone()[0]
        checks["native_autoloop_events_and_timers"] = one == 1 and timer > 0 and failed == 0
        details["autoloop"] = {"unique_start": one, "timer_events": timer, "failed_or_retried_runs": failed}
        restarted = read(evidence / "controlled-restart.json")
        probe_id = config["engineering_probe_id"]
        original = read(host_root / "jobs" / probe_id / "process.json")
        checks["host_restart_preserves_original_supervisor"] = (
            restarted["supervisor_survived"] is True and original == restarted["original_supervisor"]
            and restarted["old_host"]["pid"] != restarted["new_host"]["pid"]
        )
        controls = read(evidence / "control-observations.json")
        checks["pause_and_unresolved_submission_do_not_redispatch"] = (
            controls["scope_id"] == scope["scope_id"] and controls["probe_operation_id"] == probe_id
            and controls["submission_unchanged"] is True and controls["process_identity"] == original
            and controls["inflight_before"]["status"] == controls["inflight_after"]["status"] == "running"
            and controls["paused_reconcile_return"]["status"] == "paused"
            and controls["resumed"]["paused"] is False and controls["foreign_stop_present"] is False
            and controls["manual_node_dispatches"] == 0
        )
        boundary = read(settings["boundary_evidence_ref"])
        checks["actual_registered_role_boundaries"] = (
            boundary["scope_id"] == scope["scope_id"]
            and boundary["configuration_ref"] == str(Path(config["continuation_configuration_ref"]).resolve())
            and {row["role"] for row in boundary["roles"]} == {"research", "runtime"}
            and all(row["returncode"] == 0 and json.loads(row["stdout"])["denied_read"] is True
                    and json.loads(row["stdout"])["denied_write"] is True
                    and json.loads(row["stdout"])["uid"] == scope["owner_uid"] for row in boundary["roles"])
        )
        ledger = read(config["accounting_ref"])
        qualifying_operations = [
            operation["operation_id"] for operation in core.get("operations", [])
            if operation["kind"] == "runtime"
            and operation["payload"]["run_id"] in settings["runtime_run_ids"]
        ]
        recovery_model_operation = qualifying_operations[0] if len(qualifying_operations) == 1 else "p1-ambiguous-runtime"
        required = list(settings["required_operation_ids"])
        if probe_id not in required:
            required.insert(0, probe_id)
        closed = []
        prefix_inputs = {}
        for operation_id in required:
            job = host_root / "jobs" / operation_id
            completion, supervision = read(job / "completion.json"), read(job / "supervisor.json")
            record = ledger["reservations"][operation_id]
            prefix_inputs[operation_id] = {
                "completion": completion, "supervision": supervision,
                "reservation_actual": record.get("actual"),
                "reservation_status": record["status"], "reservation_missing": record.get("missing", []),
            }
            pending_debt = [item for item in ledger["unresolved_costs"] if item.get("operation_id") == operation_id]
            one_off_prior_debt = (
                record["status"] == "unknown" and bool(pending_debt)
                and not MidtermLedger.blocking_unresolved_costs(
                    scope, {**ledger, "unresolved_costs": pending_debt},
                    for_model_operation=recovery_model_operation)
            )
            closed.append(
                completion["scope_id"] == record["scope_id"] == scope["scope_id"]
                and completion["operation_id"] == operation_id
                and (record["status"] == "settled" or one_off_prior_debt)
                and completion["status"] == supervision["status"] == "completed"
                and supervision["exit_code"] == 0 and supervision["cleanup_confirmed"] is True
                and not supervision.get("remaining_process_ids")
                and completion["cleanup_confirmed"] is True and record["cleanup_confirmed"] is True
                and (not record.get("missing") or one_off_prior_debt)
                and completion["actual"] == record["actual"]
                and (job / "submission.json").is_file()
            )
        details["prefix_operations"] = prefix_inputs
        blocking_debt = MidtermLedger.blocking_unresolved_costs(
            scope, ledger, for_model_operation=recovery_model_operation)
        details["retained_token_unknowns"] = [
            {"operation_id": item["operation_id"], "missing": item["missing"],
             "actual": ledger["reservations"][item["operation_id"]]["actual"],
             "record_status": ledger["reservations"][item["operation_id"]]["status"],
             "authorization_ref": scope.get("one_off_token_unknown_recovery", {}).get("authorization_ref"),
             "token_amounts_filled_zero": False}
            for item in ledger["unresolved_costs"] if item not in blocking_debt
        ]
        checks["prefix_native_costs_and_cleanup_closed"] = bool(closed) and all(closed) and not blocking_debt
        runs, requests, current_operation_ids = [], 0, []
        for run_id in settings["runtime_run_ids"]:
            slot = next(row for row in core["runtime_slots"] if row["run_id"] == run_id)
            runtime = Path(slot["output_ref"])
            context, gate = read(runtime / "runtime_context.json"), read(runtime / "request_gate.json")
            identity = {key: context[key] for key in ("run_id", "episode_id", "bundle_id", "protocol_id")}
            events = read_model_events(runtime, gate, identity)
            usage, gaps = request_usage(events, expected_requests=gate["requests_forwarded"], require_reasoning=True)
            audit = inspect_attempt(runtime)
            matching_operations = [operation for operation in core.get("operations", [])
                                   if operation["kind"] == "runtime" and operation["payload"]["run_id"] == run_id]
            operation_id = context.get("request_scope", {}).get("operation_id")
            current_operation_ids.append(operation_id)
            admitted = [row for row in events if row["kind"] == "request_admitted"]
            bound = (len(matching_operations) == 1
                     and matching_operations[0]["operation_id"] == operation_id
                     and context["run_id"] == run_id and context["bundle_id"] == slot["bundle_id"]
                     and context["request_scope"]["scope_authorization_ref"] == config["scope_authorization_ref"]
                     and context["request_scope"]["accounting_ref"] == config["accounting_ref"]
                     and gate.get("upstream") == "https://chatgpt.com/backend-api/codex"
                     and bool(admitted) and not gaps and gate["cleanup_complete"] is True
                     and gate["active_requests_at_close"] == 0
                     and all(row["model"] == "gpt-6.1-sol" and row["reasoning_effort"] == "medium" for row in admitted)
                     and audit["passed"] is True)
            requests += len(admitted)
            runs.append({"run_id": run_id, "bound_real_chain": bound, "audit": audit, "usage": usage})
        checks["real_model_runtime_vla_roundtrip"] = bool(runs) and all(row["bound_real_chain"] for row in runs)
        checks["native_request_cost_attribution"] = (
            requests > 0 and len(current_operation_ids) == len(set(current_operation_ids))
            and all(name in required for name in current_operation_ids)
            and requests == sum(ledger["reservations"][name]["actual"].get("model_requests", 0)
                                for name in current_operation_ids)
            and ledger["used"]["model_requests"] == sum(
                record.get("actual", {}).get("model_requests", 0)
                for record in ledger["reservations"].values())
        )
        details["runtimes"] = runs
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, StopIteration) as exc:
        missing.append(f"{type(exc).__name__}: {exc}")
    passed = bool(checks) and not missing and all(value is True for value in checks.values())
    receipt = {"schema": "rsi-midterm-p1-acceptance-v1", "scope_id": scope["scope_id"],
               "status": "passed" if passed else "inconclusive", "checked_at": datetime.now(timezone.utc).isoformat(),
               "configuration_ref": str(Path(configuration_ref).resolve()), "scope_ref": config["scope_authorization_ref"],
               "checks": checks, "missing_evidence": missing, "native_details": details,
               "acceptance_scope": "current-scope unattended real component function and resource evidence; not task capability or release qualification",
               "manual_node_dispatches": 0}
    if destination.exists():
        previous = read(destination)
        compared = {key: value for key, value in previous.items() if key != "checked_at"}
        if compared != {key: value for key, value in receipt.items() if key != "checked_at"}:
            raise FileExistsError("P1 receipt already exists; preserve it and create an explicit review revision")
        receipt = previous
    else:
        atomic_json(destination, receipt)
    return {"status": receipt["status"], "p1_acceptance_ref": str(destination), "artifact_ref": str(destination)}
