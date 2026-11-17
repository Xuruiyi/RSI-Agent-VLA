# Copyright 2026 RSI-Agent-VLA Authors.
"""P1 receipt admission from raw CPU fixtures; no real component claims."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rsi_vla.p1_acceptance import capture_boundaries, evaluate_p1
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def update(path, change):
    value = read(path)
    change(value)
    atomic_json(path, value)


def create_fixture(root):
    """Use distinct synthetic scope files, real SQLite and raw JSON consumers."""
    root = root.resolve()
    for name in (
        "operator",
        "state/research",
        "state/evaluation",
        "release",
        "runtime",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    scope_id = "p1-consumer-synthetic"
    scope_ref = root / "operator/scope.json"
    accounting_ref = root / "operator/accounting.json"
    core_ref = root / "operator/core.json"
    config_ref = root / "operator/host.json"
    output_ref = root / "state/evaluation/p1.json"
    boundary_ref = root / "state/evaluation/boundaries.json"
    hidden_ref = root / "state/evaluation/hidden-probe.json"
    evidence = root / "validation/P1-autoloop"
    evidence.mkdir(parents=True)
    hub_home = root / "autoloop/home"
    hub_home.mkdir(parents=True)
    host_root = root / "operator/midterm-host"
    probe_id = "engineering-probe"
    runtime_id = "runtime-function"
    identity = {
        "run_id": "runtime-synthetic",
        "episode_id": "episode-synthetic",
        "bundle_id": "bundle-synthetic",
        "protocol_id": "protocol-synthetic",
    }
    runtime = root / "runtime" / identity["run_id"]
    runtime.mkdir()
    scope = {
        "scope_id": scope_id,
        "write_root": str(root),
        "owner_uid": os.getuid(),
        "status": "active",
        "authorized": True,
        "dynamic_authorized": True,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        "p1_acceptance_ref": str(output_ref),
        "evidence_kind": "synthetic_cpu_fixture",
        "outer_limits": {"model_requests": 4, "runtime_starts": 2, "gpu_seconds": 7200},
        "token_policy": {"mode": "observed_cumulative_stop_threshold"},
    }
    atomic_json(scope_ref, scope)
    settings = {
        "required_operation_ids": [runtime_id],
        "runtime_run_ids": [identity["run_id"]],
        "boundary_evidence_ref": str(boundary_ref),
        "hidden_probe_ref": str(hidden_ref),
    }
    core = {
        "scope_authorization_ref": str(scope_ref),
        "isolation_task_root": str(root),
        "p1_acceptance": settings,
        "role_isolation": {"hidden_paths": [str(hidden_ref)]},
        "operations": [
            {
                "operation_id": runtime_id,
                "kind": "runtime",
                "payload": {"run_id": identity["run_id"]},
            }
        ],
        "runtime_slots": [
            {
                "run_id": identity["run_id"],
                "bundle_id": identity["bundle_id"],
                "output_ref": str(runtime),
            }
        ],
        "evidence_kind": "synthetic_cpu_fixture",
    }
    atomic_json(core_ref, core)
    config = {
        "scope_id": scope_id,
        "continuation_configuration_ref": str(core_ref),
        "scope_authorization_ref": str(scope_ref),
        "accounting_ref": str(accounting_ref),
        "continuation_root": str(root / "operator/continuation"),
        "host_root": str(host_root),
        "autoloop_home": str(hub_home),
        "engineering_probe_id": probe_id,
    }
    atomic_json(config_ref, config)
    atomic_json(hidden_ref, {"private_label": "P1-SYNTHETIC-HIDDEN"})
    atomic_json(
        evidence / "total-start-deliveries.json",
        {
            "event": {"id": scope_id + "-total-start", "type": "midterm.start"},
            "deliveries": [
                {"duplicate": False},
                {"duplicate": True},
                {"duplicate": True},
            ],
        },
    )
    with sqlite3.connect(hub_home / "hub.db") as db:
        db.executescript(
            "CREATE TABLE events (key TEXT, type TEXT); "
            "CREATE TABLE runs (attempts INTEGER, error TEXT);"
        )
        db.executemany(
            "INSERT INTO events VALUES (?, ?)",
            [(scope_id + "-total-start", "midterm.start"), ("timer-1", "timer.due")],
        )
        db.execute("INSERT INTO runs VALUES (0, NULL)")
    original = {
        "pid": 12345,
        "start_time": "synthetic-start",
        "process_group_id": 12345,
    }
    atomic_json(
        evidence / "controlled-restart.json",
        {
            "supervisor_survived": True,
            "original_supervisor": original,
            "old_host": {"pid": 20000},
            "new_host": {"pid": 20001},
        },
    )
    atomic_json(
        evidence / "control-observations.json",
        {
            "scope_id": scope_id,
            "probe_operation_id": probe_id,
            "submission_unchanged": True,
            "process_identity": original,
            "inflight_before": {"status": "running"},
            "inflight_after": {"status": "running"},
            "paused_reconcile_return": {"status": "paused"},
            "resumed": {"paused": False},
            "foreign_stop_present": False,
            "manual_node_dispatches": 0,
        },
    )
    roles = [
        {
            "role": role,
            "returncode": 0,
            "stdout": json.dumps(
                {"uid": os.getuid(), "denied_read": True, "denied_write": True}
            ),
            "stderr": "",
        }
        for role in ("research", "runtime")
    ]
    atomic_json(
        boundary_ref,
        {
            "schema": "rsi-p1-role-boundaries-v1",
            "scope_id": scope_id,
            "configuration_ref": str(core_ref),
            "roles": roles,
        },
    )
    reservations = {}
    for name, requests in ((probe_id, 0), (runtime_id, 1)):
        job = host_root / "jobs" / name
        actual = {
            "model_requests": requests,
            "wall_seconds": 1.0,
            "gpu_seconds": 0.0,
            "input_tokens": 10 * requests,
            "output_tokens": 5 * requests,
            "reasoning_tokens": 2 * requests,
            "runtime_starts": requests,
            "storage_bytes": 100,
            "current_disk_bytes": 80,
        }
        atomic_json(
            job / "submission.json", {"scope_id": scope_id, "operation_id": name}
        )
        atomic_json(
            job / "completion.json",
            {
                "scope_id": scope_id,
                "operation_id": name,
                "status": "completed",
                "cleanup_confirmed": True,
                "actual": actual,
            },
        )
        atomic_json(
            job / "supervisor.json",
            {
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
            },
        )
        reservations[name] = {
            "scope_id": scope_id,
            "status": "settled",
            "cleanup_confirmed": True,
            "missing": [],
            "actual": actual,
        }
    atomic_json(host_root / "jobs" / probe_id / "process.json", original)
    used = {
        key: sum(record["actual"].get(key, 0) for record in reservations.values())
        for key in reservations[runtime_id]["actual"]
    }
    atomic_json(
        accounting_ref,
        {
            "scope_id": scope_id,
            "reservations": reservations,
            "unresolved_costs": [],
            "used": used,
        },
    )
    atomic_json(
        runtime / "runtime_context.json",
        dict(
            identity,
            request_scope={
                "scope_authorization_ref": str(scope_ref),
                "accounting_ref": str(accounting_ref),
                "operation_id": runtime_id,
            },
        ),
    )
    atomic_json(
        runtime / "request_gate.json",
        dict(
            identity,
            requests_forwarded=1,
            upstream="https://chatgpt.com/backend-api/codex",
            cleanup_complete=True,
            active_requests_at_close=0,
        ),
    )
    events = [
        {
            "identity": identity,
            "kind": "request_admitted",
            "request_number": 1,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
        },
        {
            "identity": identity,
            "kind": "response_terminal",
            "request_number": 1,
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 15,
            },
        },
    ]
    (runtime / "model_requests.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    return {
        "root": root,
        "scope": scope_ref,
        "config": config_ref,
        "core": core_ref,
        "output": output_ref,
        "boundary": boundary_ref,
        "accounting": accounting_ref,
        "evidence": evidence,
        "hub_home": hub_home,
        "host_root": host_root,
        "runtime": runtime,
        "probe_id": probe_id,
        "runtime_id": runtime_id,
    }


@pytest.fixture
def p1(tmp_path, monkeypatch):
    fixture = create_fixture(tmp_path)
    audits = []

    def audit_boundary(path):
        audits.append(Path(path))
        return {
            "passed": True,
            "native_task_outcome": "valid_failure",
            "fixture": "outer heavy Runtime audit boundary only",
        }

    monkeypatch.setattr(
        "rsi_vla.adapters.runtime_acceptance.inspect_attempt", audit_boundary
    )
    fixture["audits"] = audits
    return fixture


def consume(fixture):
    result = evaluate_p1(fixture["config"], fixture["output"])
    receipt = read(fixture["output"])
    assert result["p1_acceptance_ref"] == str(fixture["output"])
    return receipt


def test_complete_raw_prefix_accepts_technical_runtime_with_native_task_failure(p1):
    receipt = consume(p1)
    assert receipt["status"] == "passed"
    assert all(value is True for value in receipt["checks"].values())
    assert receipt["missing_evidence"] == []
    assert p1["audits"] == [p1["runtime"]]
    assert (
        receipt["native_details"]["runtimes"][0]["audit"]["native_task_outcome"]
        == "valid_failure"
    )
    assert receipt["native_details"]["runtimes"][0]["usage"]["reasoning_tokens"] == 2
    assert receipt["manual_node_dispatches"] == 0


def test_same_evidence_replays_original_receipt_without_rewriting_bytes(p1):
    assert consume(p1)["status"] == "passed"
    original = p1["output"].read_bytes()
    result = evaluate_p1(p1["config"], p1["output"])
    assert result["status"] == "passed"
    assert result["p1_acceptance_ref"] == str(p1["output"])
    assert p1["output"].read_bytes() == original


def test_changed_evidence_cannot_overwrite_an_existing_passed_receipt(p1):
    assert consume(p1)["status"] == "passed"
    original = p1["output"].read_bytes()
    update(
        p1["accounting"],
        lambda v: v.update(
            unresolved_costs=[
                {
                    "operation_id": p1["runtime_id"],
                    "missing": ["input_tokens", "output_tokens"],
                    "cleanup_confirmed": True,
                }
            ]
        ),
    )
    with pytest.raises(FileExistsError, match="preserve"):
        evaluate_p1(p1["config"], p1["output"])
    assert p1["output"].read_bytes() == original


@pytest.mark.parametrize(
    "fault", ["wrong_scope", "inactive", "not_authorized", "expired", "offline"]
)
def test_current_scope_authority_is_required(p1, fault):
    if fault == "wrong_scope":
        update(p1["config"], lambda v: v.update(scope_id="different-synthetic-scope"))
    elif fault == "offline":
        update(p1["core"], lambda v: v.update(offline_fixture=True))
    else:
        change = {
            "inactive": {"status": "paused"},
            "not_authorized": {"dynamic_authorized": False},
            "expired": {"expires_at": "2000-01-01T00:00:00Z"},
        }[fault]
        update(p1["scope"], lambda v: v.update(change))
    receipt = consume(p1)
    assert receipt["status"] == "inconclusive"
    assert receipt["checks"]["current_scope_and_authority"] is False


@pytest.mark.parametrize(
    "fault", ["missing_runtime", "empty_runtime_list", "old_receipt_only"]
)
def test_historical_p1_cannot_replace_current_runtime(p1, fault):
    atomic_json(
        p1["root"] / "historical-p1.json",
        {
            "schema": "rsi-midterm-p1-acceptance-v1",
            "scope_id": "historical-synthetic",
            "status": "passed",
        },
    )
    if fault == "empty_runtime_list":
        update(p1["core"], lambda v: v["p1_acceptance"].update(runtime_run_ids=[]))
    else:
        (p1["runtime"] / "runtime_context.json").unlink()
        if fault == "old_receipt_only":
            update(
                p1["core"],
                lambda v: v["p1_acceptance"].update(
                    p1_acceptance_ref=str(p1["root"] / "historical-p1.json")
                ),
            )
    assert consume(p1)["status"] == "inconclusive"


@pytest.mark.parametrize(
    "fault", ["restart", "duplicate", "timer", "autoloop_retry", "pause", "redispatch"]
)
def test_actual_control_and_autoloop_records_are_required(p1, fault):
    if fault == "restart":
        (p1["evidence"] / "controlled-restart.json").unlink()
    elif fault == "duplicate":
        update(
            p1["evidence"] / "total-start-deliveries.json",
            lambda v: v.update(deliveries=[{"duplicate": False}]),
        )
    elif fault in {"timer", "autoloop_retry"}:
        with sqlite3.connect(p1["hub_home"] / "hub.db") as db:
            db.execute(
                "DELETE FROM events WHERE type='timer.due'"
                if fault == "timer"
                else "UPDATE runs SET attempts=1"
            )
    else:
        update(
            p1["evidence"] / "control-observations.json",
            lambda v: (
                v.update(paused_reconcile_return={"status": "running"})
                if fault == "pause"
                else v.update(submission_unchanged=False)
            ),
        )
    assert consume(p1)["status"] == "inconclusive"


@pytest.mark.parametrize(
    "fault",
    ["unresolved", "missing_cost", "unclean", "remaining_process", "cost_mismatch"],
)
def test_unknown_costs_and_unclosed_cleanup_block_admission(p1, fault):
    operation = p1["runtime_id"]
    if fault == "unresolved":
        update(
            p1["accounting"],
            lambda v: v.update(
                unresolved_costs=[
                    {
                        "operation_id": operation,
                        "missing": ["input_tokens", "output_tokens"],
                        "cleanup_confirmed": True,
                    }
                ]
            ),
        )
    elif fault == "missing_cost":
        update(
            p1["accounting"],
            lambda v: v["reservations"][operation].update(missing=["wall_seconds"]),
        )
    elif fault == "unclean":
        update(
            p1["host_root"] / "jobs" / operation / "completion.json",
            lambda v: v.update(cleanup_confirmed=False),
        )
    elif fault == "remaining_process":
        update(
            p1["host_root"] / "jobs" / operation / "supervisor.json",
            lambda v: v.update(remaining_process_ids=[54321]),
        )
    else:
        update(
            p1["accounting"],
            lambda v: v["reservations"][operation]["actual"].update(wall_seconds=2.0),
        )
    receipt = consume(p1)
    assert receipt["status"] == "inconclusive"
    assert receipt["checks"]["prefix_native_costs_and_cleanup_closed"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "missing_reasoning",
        "wrong_model",
        "wrong_scope_binding",
        "active_request",
        "wrong_cost_count",
    ],
)
def test_real_runtime_binding_request_usage_and_reasoning_are_required(p1, fault):
    if fault in {"missing_reasoning", "wrong_model"}:
        path = p1["runtime"] / "model_requests.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "missing_reasoning":
            events[1]["usage"].pop("output_tokens_details")
        else:
            events[0]["model"] = "other-model"
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    elif fault == "wrong_scope_binding":
        update(
            p1["runtime"] / "runtime_context.json",
            lambda v: v["request_scope"].update(
                scope_authorization_ref="/synthetic/old-scope"
            ),
        )
    elif fault == "active_request":
        update(
            p1["runtime"] / "request_gate.json",
            lambda v: v.update(active_requests_at_close=1),
        )
    else:
        update(
            p1["accounting"],
            lambda v: v["reservations"][p1["runtime_id"]]["actual"].update(
                model_requests=2
            ),
        )
    assert consume(p1)["status"] == "inconclusive"


@pytest.mark.parametrize("role", ["research", "runtime"])
def test_role_boundary_failure_blocks_acceptance(p1, role):
    def fail_boundary(value):
        row = next(row for row in value["roles"] if row["role"] == role)
        row["stdout"] = json.dumps(
            {"uid": os.getuid(), "denied_read": False, "denied_write": True}
        )

    update(p1["boundary"], fail_boundary)
    receipt = consume(p1)
    assert receipt["status"] == "inconclusive"
    assert receipt["checks"]["actual_registered_role_boundaries"] is False


def test_existing_p1_receipt_is_never_overwritten(p1):
    original = b'{"status": "inconclusive", "scope_id": "prior-synthetic"}\n'
    p1["output"].write_bytes(original)
    with pytest.raises(FileExistsError, match="preserve"):
        evaluate_p1(p1["config"], p1["output"])
    assert p1["output"].read_bytes() == original


def test_output_must_be_current_registered_evaluator_path(p1):
    forbidden = p1["root"] / "state/research/p1.json"
    with pytest.raises(PermissionError, match="registered evaluator-owned"):
        evaluate_p1(p1["config"], forbidden)
    assert not forbidden.exists()
    assert p1["audits"] == []


def test_capture_boundaries_uses_real_namespace_and_preserves_private_data():
    with tempfile.TemporaryDirectory(
        prefix="p1-consumer-boundary-", dir=ROOT
    ) as directory:
        fixture = create_fixture(Path(directory))
        destination = fixture["root"] / "operator/captured-boundaries.json"
        original = (fixture["root"] / "state/evaluation/hidden-probe.json").read_bytes()
        try:
            captured = capture_boundaries(fixture["core"], destination)
        except RuntimeError as error:
            pytest.fail(f"{error}: {destination.read_text()}")
        assert {row["role"] for row in captured["roles"]} == {"research", "runtime"}
        for row in captured["roles"]:
            assert row["returncode"] == 0, row["stdout"] + row["stderr"]
            stdout = json.loads(row["stdout"])
            assert stdout == {
                "uid": os.getuid(),
                "denied_read": True,
                "denied_write": True,
            }
        assert captured == read(destination)
        assert (
            fixture["root"] / "state/evaluation/hidden-probe.json"
        ).read_bytes() == original
        assert not (fixture["root"] / "release/p1-forbidden.json").exists()


def test_capture_rejects_readonly_private_material_that_was_not_hidden():
    with tempfile.TemporaryDirectory(
        prefix="p1-consumer-boundary-", dir=ROOT
    ) as directory:
        fixture = create_fixture(Path(directory))
        update(fixture["core"], lambda value: value.update(role_isolation={}))
        destination = fixture["root"] / "operator/rejected-boundaries.json"
        with pytest.raises(RuntimeError, match="role-boundary probe failed"):
            capture_boundaries(fixture["core"], destination)
        returned = read(destination)
        assert len(returned["roles"]) == 2
        assert all(row["returncode"] != 0 for row in returned["roles"])
        assert all("AssertionError" in row["stderr"] for row in returned["roles"])
        assert not (fixture["root"] / "release/p1-forbidden.json").exists()


@pytest.mark.timeout(60)
def test_real_autoloop_monitor_controls_preserve_one_cpu_probe_and_resume_successor(
    monkeypatch,
):
    """Real monitor/Hub/Host control; CPU business replacements are not real P1."""
    import shutil
    import signal
    import socket
    import subprocess
    import time
    from copy import deepcopy

    from rsi_vla import midterm_host
    from test_host_autoloop_receipts import HostAutoloopReceiptTests

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    node = os.environ.get("RSI_TEST_NODE") or shutil.which("node")
    assert node, "Node24 is required for this explicit integration check"
    version = subprocess.check_output([node, "--version"], text=True).strip()
    assert int(version.lstrip("v").split(".")[0]) >= 24
    fixture = HostAutoloopReceiptTests(
        "test_real_hub_timer_restart_finishes_two_cpu_receipts_without_manual_advance"
    )
    fixture.setUp()
    fixture.hub_home = fixture.root / "autoloop/home"
    child_offset = len(midterm_host._children)
    summary = {"evidence_kind": "synthetic_cpu_engineering_control", "status": "failed"}
    try:
        originals = deepcopy(fixture.config["operations"])
        probe_id = fixture.scope["scope_id"] + "-engineering-probe"
        fixture.scope["cpu_operation_ids"].append(probe_id)
        fixture.scope["outer_limits"].update(
            wall_seconds=600,
            storage_bytes=256 * 1024**2,
            current_disk_bytes=256 * 1024**2,
            resource_wait_seconds=30,
        )
        atomic_json(fixture.scope_ref, fixture.scope)
        phase = {
            key: value
            for key, value in fixture.scope["outer_limits"].items()
            if key != "concurrent_gpu_devices"
        }
        atomic_json(
            fixture.phase,
            {
                "limits": phase,
                "used": dict.fromkeys(phase, 0),
                "remaining": phase.copy(),
                "unresolved_costs": [],
            },
        )
        fixture.core.update(
            repository_root=str(ROOT), p1_acceptance={"control_fixture": True}
        )
        atomic_json(fixture.core_ref, fixture.core)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        midterm_host.prepare_deployment(
            fixture.core_ref,
            fixture.host_ref,
            node=node,
            autoloop_root=ROOT / "agent-autoloop",
            port=port,
        )
        configured = read(fixture.host_ref)
        configured["operations"].update(originals)
        configured.update(inputs_ref=fixture.config["inputs_ref"], poll_seconds=0.5)
        assert configured["p1_control_enabled"] is True
        assert configured["operations"][probe_id]["command"][-1] == "20"
        atomic_json(fixture.host_ref, configured)
        atomic_json(
            fixture.hub_home / "rules.json",
            midterm_host.build_autoloop_rules(configured),
        )
        deployed = midterm_host.launch_deployment(fixture.host_ref)
        monitor = next(
            p
            for p in midterm_host._children[child_offset:]
            if p.pid == deployed["monitor_pid"]
        )
        host_root = fixture.root / "operator/midterm-host"
        evidence = fixture.root / "validation/P1-autoloop"
        deadline = time.monotonic() + 45
        while monitor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert monitor.poll() == 0, (host_root / "monitor.log").read_text()
        terminal = read(evidence / "observed-terminal.json")
        assert terminal["core"]["status"] == "completed"
        assert terminal["controlled_restart"] is True
        assert terminal["completed_timer_runs"]
        assert terminal["manual_node_dispatches"] == 0
        deliveries = read(evidence / "total-start-deliveries.json")
        assert [r["duplicate"] for r in deliveries["deliveries"]] == [False, True, True]
        assert (
            read(evidence / "engineering-lease-released.json")["wait"]["waited_seconds"]
            > 0
        )
        restarted = read(evidence / "controlled-restart.json")
        assert restarted["supervisor_survived"] is True
        assert restarted["old_host"]["pid"] != restarted["new_host"]["pid"]
        controls = read(evidence / "control-observations.json")
        assert controls["submission_unchanged"] is True
        assert controls["process_identity"] == restarted["original_supervisor"]
        assert (
            controls["inflight_before"]["status"]
            == controls["inflight_after"]["status"]
            == "running"
        )
        assert controls["paused_reconcile_return"]["status"] == "paused"
        assert controls["resumed"]["paused"] is False
        assert controls["foreign_stop_present"] is False
        assert controls["manual_node_dispatches"] == 0
        assert (evidence / "resume-control-delivery.json").is_file()
        assert fixture.counter.read_text() == "produce\nconsume\n"
        for name in (probe_id, "produce", "consume"):
            job = host_root / "jobs" / name
            assert read(job / "completion.json")["status"] == "completed"
            assert read(job / "supervisor.json")["cleanup_confirmed"] is True
        ledger = read(fixture.root / "accounting.json")
        assert set(ledger["reservations"]) == {probe_id, "produce", "consume"}
        assert all(
            row["status"] == "settled" for row in ledger["reservations"].values()
        )
        assert ledger["unresolved_costs"] == []
        assert ledger["used"]["model_requests"] == ledger["used"]["gpu_seconds"] == 0
        with sqlite3.connect(
            f"file:{fixture.hub_home / 'hub.db'}?mode=ro", uri=True
        ) as db:
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM events WHERE key=?",
                    (deliveries["event"]["id"],),
                ).fetchone()[0]
                == 1
            )
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM events WHERE type='timer.due'"
                ).fetchone()[0]
                > 0
            )
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM runs WHERE attempts>0 OR error IS NOT NULL"
                ).fetchone()[0]
                == 0
            )
        summary.update(
            status="passed",
            scope_id=fixture.scope["scope_id"],
            node_version=version,
            costs=ledger["used"],
            manual_node_dispatches=0,
        )
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for process in reversed(midterm_host._children[child_offset:]):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for pid in fixture._owned_processes():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        until = time.monotonic() + 3
        while fixture._owned_processes() and time.monotonic() < until:
            time.sleep(0.05)
        for pid in fixture._owned_processes():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        summary["remaining_owned_process_ids"] = fixture._owned_processes()
        destination = os.environ.get("RSI_P1_TEST_EVIDENCE_ROOT")
        if destination:
            Path(destination).mkdir(parents=True, exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix="p1-control-", dir=destination))
            for relative in ("validation/P1-autoloop", "operator/midterm-host"):
                source = fixture.root / relative
                if source.exists():
                    shutil.copytree(source, saved / relative)
            atomic_json(saved / "summary.json", summary)
            print(f"Synthetic P1 control evidence: {saved}", flush=True)
        fixture.doCleanups()
        assert summary["remaining_owned_process_ids"] == [], (
            "owned test processes survived cleanup"
        )


def test_changed_consistent_native_costs_cannot_reuse_existing_passed_receipt(p1):
    assert consume(p1)["status"] == "passed"
    original = p1["output"].read_bytes()
    operation = p1["runtime_id"]
    update(
        p1["accounting"],
        lambda value: value["reservations"][operation]["actual"].update(
            wall_seconds=2.0
        ),
    )
    update(
        p1["host_root"] / "jobs" / operation / "completion.json",
        lambda value: value["actual"].update(wall_seconds=2.0),
    )
    with pytest.raises(FileExistsError, match="preserve"):
        evaluate_p1(p1["config"], p1["output"])
    assert p1["output"].read_bytes() == original


def test_host_boundary_entry_follows_real_cpu_receipts_without_nested_operator_namespace():
    from copy import deepcopy

    from test_host_receipt_inputs import HostReceiptInputsTests

    fixture = HostReceiptInputsTests(
        "test_real_host_restart_native_costs_late_artifact_and_successor"
    )
    fixture.setUp()
    try:
        hidden = fixture.root / "state/evaluation/p1-private-marker.txt"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        hidden.write_text("P1-PRIVATE-TEST-MARKER")
        public_p02 = hidden.with_name("p02-admitted.json")
        atomic_json(
            public_p02, {"scope_id": fixture.scope["scope_id"], "status": "admitted"}
        )
        boundary_id = "boundary"
        fixture.scope["cpu_operation_ids"].append(boundary_id)
        atomic_json(fixture.scope_ref, fixture.scope)
        fixture.core["operations"].append(
            {
                "operation_id": boundary_id,
                "stage": "engineering",
                "kind": "p1_boundary_check",
                "dynamic": False,
                "payload": {},
            }
        )
        fixture.core["continuation"]["nodes"]["consume"]["transitions"] = [
            {"next": "boundary"}
        ]
        fixture.core["continuation"]["nodes"]["boundary"] = {
            "operation_id": boundary_id,
            "transitions": [{"next": "done"}],
        }
        fixture.core["p1_acceptance"] = {"hidden_probe_ref": str(hidden)}
        fixture.core["role_isolation"] = {"hidden_paths": [str(hidden)]}
        atomic_json(fixture.core_ref, fixture.core)
        output = fixture.operator / "boundary"
        output.mkdir()
        template = deepcopy(fixture.config["operations"]["produce"])
        for key in ("command", "exports", "late_refs"):
            template.pop(key, None)
        template.update(
            result_ref=str(output / "result.json"),
            permit_ref=str(output / "permit.json"),
            version="p1-boundary-test-v1",
            storage_write_roots=[str(output)],
        )
        fixture.config["operations"][boundary_id] = template
        atomic_json(fixture.host_ref, fixture.config)
        for operation_id in ("produce", "consume", boundary_id):
            assert fixture.host().advance()["status"] == "running"
            fixture.end_job(operation_id)
        assert fixture.host().advance()["status"] == "completed"
        assert fixture.counter.read_text() == "produce\nconsume\n"
        exported = read(output / "result.json")
        assert exported["schema"] == "rsi-cpu-operation-result-v1"
        assert exported["scope_id"] == fixture.scope["scope_id"]
        assert exported["operation_id"] == boundary_id
        assert exported["status"] == "completed"
        captured = read(exported["artifact_ref"])
        assert {row["role"] for row in captured["roles"]} == {"research", "runtime"}
        for row in captured["roles"]:
            assert row["returncode"] == 0, row
            assert json.loads(row["stdout"]) == {
                "uid": os.getuid(),
                "denied_read": True,
                "denied_write": True,
            }
        assert hidden.read_text() == "P1-PRIVATE-TEST-MARKER"
        assert read(public_p02)["status"] == "admitted"
        ledger = fixture.host().ledger.read()
        assert set(ledger["reservations"]) == {"produce", "consume", boundary_id}
        assert all(
            row["status"] == "settled" for row in ledger["reservations"].values()
        )
        assert ledger["unresolved_costs"] == []
        assert ledger["used"]["model_requests"] == ledger["used"]["gpu_seconds"] == 0
    finally:
        fixture.doCleanups()


def add_original_token_unknown(fixture, *, authorize=True):
    """Retain one original unknown prefix; only the current Runtime is audited."""
    old = "original-runtime"
    scope = read(fixture["scope"])
    actual = {
        "model_requests": 1,
        "runtime_starts": 1,
        "wall_seconds": 2.0,
        "gpu_seconds": 3.0,
        "storage_bytes": 200,
        "current_disk_bytes": 100,
    }
    job = fixture["host_root"] / "jobs" / old
    atomic_json(
        job / "submission.json", {"scope_id": scope["scope_id"], "operation_id": old}
    )
    atomic_json(
        job / "completion.json",
        {
            "scope_id": scope["scope_id"],
            "operation_id": old,
            "status": "completed",
            "cleanup_confirmed": True,
            "actual": actual,
            "unknown_dimensions": ["input_tokens", "output_tokens"],
        },
    )
    atomic_json(
        job / "supervisor.json",
        {
            "status": "completed",
            "exit_code": 0,
            "cleanup_confirmed": True,
            "remaining_process_ids": [],
        },
    )
    ledger = read(fixture["accounting"])
    ledger["reservations"][old] = {
        "scope_id": scope["scope_id"],
        "status": "unknown",
        "cleanup_confirmed": True,
        "actual": actual,
        "missing": ["input_tokens", "output_tokens"],
        "budget": {
            "model_requests": 3,
            "runtime_starts": 1,
            "wall_seconds": 1200,
            "gpu_seconds": 1200,
            "storage_bytes": 1000,
            "current_disk_bytes": 1000,
        },
    }
    ledger["unresolved_costs"] = [
        {
            "operation_id": old,
            "missing": ["input_tokens", "output_tokens"],
            "cleanup_confirmed": True,
        }
    ]
    for key, value in actual.items():
        ledger["used"][key] = ledger["used"].get(key, 0) + value
    cpu_prefix = []
    for name in ("recovery-audit", "recovery-boundary"):
        cpu_prefix.append(name)
        record = json.loads(json.dumps(ledger["reservations"][fixture["probe_id"]]))
        ledger["reservations"][name] = record
        cpu_job = fixture["host_root"] / "jobs" / name
        atomic_json(
            cpu_job / "submission.json",
            {"scope_id": scope["scope_id"], "operation_id": name},
        )
        atomic_json(
            cpu_job / "completion.json",
            {
                "scope_id": scope["scope_id"],
                "operation_id": name,
                "status": "completed",
                "cleanup_confirmed": True,
                "actual": record["actual"],
            },
        )
        atomic_json(
            cpu_job / "supervisor.json",
            {
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
            },
        )
        for key, value in record["actual"].items():
            ledger["used"][key] = ledger["used"].get(key, 0) + value
    atomic_json(fixture["accounting"], ledger)
    update(
        fixture["core"],
        lambda value: value["p1_acceptance"].update(
            required_operation_ids=[old, fixture["runtime_id"], *cpu_prefix]
        ),
    )
    authorization = fixture["root"] / "operator/one-off-authorization-v3.json"
    grant = {
        "authorized": authorize,
        "original_operation_id": old,
        "recovery_operation_id": fixture["runtime_id"],
        "authorization_ref": str(authorization),
        "old_token_values_stay_unknown": True,
        "failed_request_count_not_refunded": True,
    }
    atomic_json(
        authorization,
        {
            "scope_id": scope["scope_id"],
            "authorized": authorize,
            "one_off_token_unknown_recovery": grant,
            "hard_limits_unchanged": True,
            "expires_at": scope["expires_at"],
            "evidence_kind": "synthetic_cpu_fixture",
        },
    )
    update(
        fixture["scope"],
        lambda value: value.update(one_off_token_unknown_recovery=grant),
    )
    fixture["old_id"] = old
    return job


def test_explicit_one_off_recovery_keeps_original_unknown_and_accepts_strict_current_runtime(
    p1,
):
    old_job = add_original_token_unknown(p1)
    before = {
        name: (old_job / name).read_bytes()
        for name in ("submission.json", "completion.json", "supervisor.json")
    }
    ledger_before = p1["accounting"].read_bytes()
    receipt = consume(p1)
    assert receipt["status"] == "passed"
    assert p1["audits"] == [p1["runtime"]]
    original = receipt["native_details"]["prefix_operations"][p1["old_id"]]
    assert original["reservation_status"] == "unknown"
    assert original["reservation_missing"] == ["input_tokens", "output_tokens"]
    assert "input_tokens" not in original["reservation_actual"]
    assert "output_tokens" not in original["reservation_actual"]
    assert original["reservation_actual"]["model_requests"] == 1
    assert original["completion"]["status"] == "completed"
    retained = receipt["native_details"]["retained_token_unknowns"]
    assert len(retained) == 1
    assert retained[0]["operation_id"] == p1["old_id"]
    assert retained[0]["record_status"] == "unknown"
    assert retained[0]["missing"] == ["input_tokens", "output_tokens"]
    assert retained[0]["actual"] == original["reservation_actual"]
    assert retained[0]["token_amounts_filled_zero"] is False
    assert (
        retained[0]["authorization_ref"]
        == read(p1["scope"])["one_off_token_unknown_recovery"]["authorization_ref"]
    )
    assert p1["accounting"].read_bytes() == ledger_before
    assert all((old_job / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize(
    "fault", ["missing", "not_authorized", "different_original", "different_recovery"]
)
def test_token_unknown_is_not_exempt_without_the_exact_one_off_authority(p1, fault):
    add_original_token_unknown(p1, authorize=fault != "not_authorized")

    def change(scope):
        if fault == "missing":
            scope.pop("one_off_token_unknown_recovery")
        elif fault == "different_original":
            scope["one_off_token_unknown_recovery"]["original_operation_id"] = (
                "other-original"
            )
        elif fault == "different_recovery":
            scope["one_off_token_unknown_recovery"]["recovery_operation_id"] = (
                "other-recovery"
            )

    update(p1["scope"], change)
    assert consume(p1)["status"] == "inconclusive"


@pytest.mark.parametrize(
    "fault",
    [
        "gpu_missing",
        "reservation_unclean",
        "completion_unclean",
        "supervisor_unclean",
        "other_scope",
    ],
)
def test_one_off_recovery_requires_original_known_hard_costs_and_cleanup(p1, fault):
    job = add_original_token_unknown(p1)
    old = p1["old_id"]
    if fault == "gpu_missing":
        update(
            p1["accounting"],
            lambda value: value["reservations"][old]["actual"].pop("gpu_seconds"),
        )
        update(
            job / "completion.json", lambda value: value["actual"].pop("gpu_seconds")
        )
    elif fault == "reservation_unclean":
        update(
            p1["accounting"],
            lambda value: value["reservations"][old].update(cleanup_confirmed=False),
        )
    elif fault in {"completion_unclean", "supervisor_unclean"}:
        target = job / (
            "completion.json" if fault == "completion_unclean" else "supervisor.json"
        )
        update(target, lambda value: value.update(cleanup_confirmed=False))
    else:
        update(
            p1["accounting"],
            lambda value: value["reservations"][old].update(
                scope_id="foreign-synthetic"
            ),
        )
    assert consume(p1)["status"] == "inconclusive"


@pytest.mark.parametrize(
    "missing", ["input_tokens", "output_tokens", "reasoning_tokens", "terminal"]
)
def test_old_usage_exception_never_waives_current_runtime_usage(p1, missing):
    add_original_token_unknown(p1)
    path = p1["runtime"] / "model_requests.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if missing == "terminal":
        events = events[:1]
    elif missing == "reasoning_tokens":
        events[1]["usage"]["output_tokens_details"].pop("reasoning_tokens")
    else:
        events[1]["usage"].pop(missing)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    receipt = consume(p1)
    assert receipt["status"] == "inconclusive"
    assert receipt["checks"]["real_model_runtime_vla_roundtrip"] is False


@pytest.mark.parametrize("fault", ["current_operation_unbound", "scope_total_mismatch"])
def test_recovery_requests_are_bound_to_current_operation_and_whole_scope_totals(
    p1, fault
):
    add_original_token_unknown(p1)
    if fault == "current_operation_unbound":
        update(
            p1["runtime"] / "runtime_context.json",
            lambda value: value["request_scope"].update(operation_id=p1["old_id"]),
        )
    else:
        update(p1["accounting"], lambda value: value["used"].update(model_requests=3))
    assert consume(p1)["status"] == "inconclusive"


def test_recovery_same_evidence_is_idempotent_without_settling_original_unknown(p1):
    add_original_token_unknown(p1)
    assert consume(p1)["status"] == "passed"
    original_receipt = p1["output"].read_bytes()
    original_ledger = p1["accounting"].read_bytes()
    assert evaluate_p1(p1["config"], p1["output"])["status"] == "passed"
    assert p1["output"].read_bytes() == original_receipt
    assert p1["accounting"].read_bytes() == original_ledger


def test_changed_original_hard_costs_cannot_reuse_recovery_passed_receipt(p1):
    job = add_original_token_unknown(p1)
    assert consume(p1)["status"] == "passed"
    original = p1["output"].read_bytes()
    update(
        p1["accounting"],
        lambda value: value["reservations"][p1["old_id"]]["actual"].update(
            gpu_seconds=4.0
        ),
    )
    update(p1["accounting"], lambda value: value["used"].update(gpu_seconds=4.0))
    update(
        job / "completion.json", lambda value: value["actual"].update(gpu_seconds=4.0)
    )
    with pytest.raises(FileExistsError, match="preserve"):
        evaluate_p1(p1["config"], p1["output"])
    assert p1["output"].read_bytes() == original
