# Copyright 2026 RSI-Agent-VLA Authors.
"""Real Host submission, namespace CPU jobs, native receipts and late successors."""

import os
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rsi_vla.host_inputs import materialize_inputs
from rsi_vla.midterm_host import MidtermHost, read
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class HostReceiptInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f03-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.operator = self.root / "operator"
        self.operator.mkdir()
        self.scope_ref = self.root / "scope.json"
        self.phase = self.root / "phase.json"
        self.operations = [
            {
                "operation_id": name,
                "stage": "engineering",
                "kind": "host_command",
                "dynamic": False,
                "payload": {},
            }
            for name in ("produce", "consume")
        ]
        limits = {
            "wall_seconds": 60,
            "storage_bytes": 20000000,
            "current_disk_bytes": 20000000,
            "model_requests": 0,
            "gpu_seconds": 0,
            "concurrent_gpu_devices": 0,
        }
        self.scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": "f03-offline",
            "status": "active",
            "authorized": True,
            "dynamic_authorized": False,
            "cpu_authorized": True,
            "cpu_operation_ids": [op["operation_id"] for op in self.operations],
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "outer_limits": limits,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        atomic_json(self.scope_ref, self.scope)
        phase_limits = {
            k: v for k, v in limits.items() if k != "concurrent_gpu_devices"
        }
        atomic_json(
            self.phase,
            {
                "limits": phase_limits,
                "used": dict.fromkeys(phase_limits, 0),
                "remaining": phase_limits,
                "unresolved_costs": [],
            },
        )
        protocol = self.operator / "protocol.json"
        atomic_json(protocol, {"protocol_id": "f03-offline"})
        self.core = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": str(self.scope_ref),
            "protocol_ref": str(protocol),
            "base_bundle_id": "synthetic-v1",
            "phase_ref": str(self.phase),
            "isolation_task_root": str(self.root),
            "operations": self.operations,
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "produce",
                "nodes": {
                    "produce": {
                        "operation_id": "produce",
                        "transitions": [{"next": "consume"}],
                    },
                    "consume": {
                        "operation_id": "consume",
                        "transitions": [{"next": "done"}],
                    },
                    "done": {"terminal": "offline_complete"},
                },
            },
        }
        self.core_ref = self.operator / "core.json"
        atomic_json(self.core_ref, self.core)
        inputs = self.operator / "admission.json"
        atomic_json(inputs, {})
        self.config = {
            "schema": "rsi-midterm-host-v1",
            "scope_id": self.scope["scope_id"],
            "scope_authorization_ref": str(self.scope_ref),
            "continuation_configuration_ref": str(self.core_ref),
            "continuation_root": str(self.operator / "continuation"),
            "inputs_ref": str(inputs),
            "repository_root": str(ROOT),
            "python": sys.executable,
            "script_ref": str(ROOT / "scripts/run_midterm_host.py"),
            "operations": {},
        }
        self.counter = self.operator / "executions.txt"
        for op in self.operations:
            name = op["operation_id"]
            output = self.operator / name
            output.mkdir()
            result = output / "result.json"
            artifact = output / "artifact.json"
            code = (
                "import json,os,time; from pathlib import Path; "
                "p=json.loads(Path(os.environ['RSI_CPU_PERMIT_REF']).read_text()); "
                "i=json.loads(Path(os.environ['RSI_OPERATION_INPUT_REF']).read_text()); "
                f"Path({str(self.counter)!r}).open('a').write({name!r}+'\\n'); time.sleep(.15); "
            )
            if name == "produce":
                code += f"Path({str(artifact)!r}).write_text(json.dumps({{'theta':'synthetic-test-only'}})); "
            else:
                code += "assert json.loads(Path(i['references']['theta']).read_text())['theta']=='synthetic-test-only'; "
            code += (
                "Path(p['result_ref']).write_text(json.dumps(dict(schema='rsi-cpu-operation-result-v1', "
                "scope_id=p['scope_id'],operation_id=p['operation_id'],version=p['version'],status='completed', "
                f"theta_ref={str(artifact)!r})))"
            )
            self.config["operations"][name] = {
                "budget": {
                    "wall_seconds": 8,
                    "storage_bytes": 2000000,
                    "current_disk_bytes": 2000000,
                    "model_requests": 0,
                    "gpu_seconds": 0,
                },
                "supervisor_limits": {"wall_seconds": 8, "storage_bytes": 2000000},
                "command": [sys.executable, "-c", code],
                "result_ref": str(result),
                "permit_ref": str(output / "permit.json"),
                "permit_template": {"kind": "generic-cpu-model0"},
                "version": "synthetic-v1",
                "cpu_purpose": "analysis",
                "storage_write_roots": [str(output)],
            }
        self.config["operations"]["produce"]["exports"] = {
            "theta": {"field": "theta_ref", "version": "synthetic-v1", "type": "path"}
        }
        self.config["operations"]["consume"]["late_refs"] = {
            "theta": {
                "operation_id": "produce",
                "export": "theta",
                "version": "synthetic-v1",
            }
        }
        self.host_ref = self.operator / "host.json"
        atomic_json(self.host_ref, self.config)

    def host(self):
        return MidtermHost(self.host_ref)

    def end_job(self, name, expected="completed"):
        host = self.host()
        completion = host.root / "jobs" / name / "completion.json"
        deadline = time.monotonic() + 12
        while not completion.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        from rsi_vla.midterm_host import process_matches

        while (
            process_matches(completion.parent / "process.json")
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        log = completion.parent / "backend.log"
        self.assertTrue(
            completion.exists(), log.read_text() if log.exists() else str(completion)
        )
        self.assertEqual(read(completion)["status"], expected, log.read_text())
        return host, completion

    def produce(self):
        self.assertEqual(self.host().advance()["status"], "running")
        self.assertEqual(self.host().advance()["status"], "running")
        return self.end_job("produce")

    def test_real_host_restart_native_costs_late_artifact_and_successor(self):
        host, _completion = self.produce()
        state = host.ledger.read()
        self.assertEqual(state["reservations"]["produce"]["status"], "settled")
        self.assertEqual(state["used"]["model_requests"], 0)
        self.assertGreater(state["used"]["wall_seconds"], 0)
        self.assertEqual(self.host().advance()["status"], "running")
        host, completion = self.end_job("consume")
        self.assertEqual(host.advance()["status"], "completed")
        self.assertEqual(self.counter.read_text(), "produce\nconsume\n")
        used = host.ledger.read()["used"].copy()
        host.finish("consume", read(completion.parent / "supervisor.json"))
        self.assertEqual(host.ledger.read()["used"], used)
        self.assertEqual(host.advance()["status"], "completed")

    def test_missing_receipt_never_resubmits_and_manual_pause_survives_timer(self):
        host, _completion = self.produce()
        native = Path(self.config["continuation_root"]) / "receipts/produce.json"
        saved = read(native)
        atomic_json(native, {"operation": self.operations[0], "status": "started"})
        # Reconcile the real result after a crash between native return and saved receipt.
        state_path = host.coordinator().path
        state = read(state_path)
        state.update(pending=self.operations[0], receipts={}, status="waiting")
        atomic_json(state_path, state)
        self.assertEqual(
            host.coordinator().advance(host.inputs(), reconcile_only=True)["receipts"][
                "produce"
            ]["value"],
            saved["value"],
        )
        self.assertEqual(self.counter.read_text(), "produce\n")
        # Missing native evidence is unknown and cannot trigger execute.
        Path(self.config["operations"]["produce"]["result_ref"]).unlink()
        atomic_json(native, {"operation": self.operations[0], "status": "started"})
        atomic_json(state_path, state)
        self.assertEqual(
            host.coordinator().advance(host.inputs(), reconcile_only=True)["wait"][
                "reason"
            ],
            "submission_unknown",
        )
        self.assertEqual(self.counter.read_text(), "produce\n")
        coordinator = host.coordinator()
        coordinator.event({"event_id": "pause", "type": "pause", "operation_id": None})
        coordinator.event({"event_id": "timer", "type": "timer", "operation_id": None})
        self.assertEqual(self.host().advance()["status"], "paused")
        self.assertEqual(self.counter.read_text(), "produce\n")

    def test_late_refs_require_terminal_scope_version_cost_closure_and_no_overwrite(
        self,
    ):
        host = self.host()
        with self.assertRaises(FileNotFoundError):
            materialize_inputs(host, self.operations[1])
        host, completion = self.produce()
        materialize_inputs(host, self.operations[1])
        original = read(completion)
        atomic_json(completion, {**original, "scope_id": "other"})
        with self.assertRaises(PermissionError):
            materialize_inputs(host, self.operations[1])
        atomic_json(completion, original)
        native = Path(self.config["continuation_root"]) / "receipts/produce.json"
        saved = read(native)
        original_native = deepcopy(saved)
        saved["value"]["theta_ref"] = str(self.scope_ref)
        atomic_json(native, saved)
        with self.assertRaisesRegex(PermissionError, "artifact escapes"):
            materialize_inputs(host, self.operations[1])
        atomic_json(native, original_native)
        host.config["operations"]["consume"]["late_refs"]["theta"]["version"] = (
            "wrong-version"
        )
        with self.assertRaisesRegex(PermissionError, "versioned export"):
            materialize_inputs(host, self.operations[1])
        host.config["operations"]["consume"]["late_refs"]["theta"]["version"] = (
            "synthetic-v1"
        )
        ledger = host.ledger.read()
        original_ledger = deepcopy(ledger)
        ledger["reservations"]["produce"]["status"] = "unknown"
        atomic_json(host.ledger.path, ledger)
        with self.assertRaisesRegex(PermissionError, "cost closure"):
            materialize_inputs(host, self.operations[1])
        atomic_json(host.ledger.path, original_ledger)
        report_ref = host.root / "jobs/produce/supervisor.json"
        report = read(report_ref)
        atomic_json(
            report_ref,
            {**report, "cleanup_confirmed": False, "remaining_process_ids": [1]},
        )
        with self.assertRaisesRegex(PermissionError, "cost closure"):
            materialize_inputs(host, self.operations[1])
        atomic_json(report_ref, report)
        saved = deepcopy(original_native)
        saved["value"]["theta_ref"] = self.config["operations"]["produce"]["result_ref"]
        atomic_json(native, saved)
        with self.assertRaisesRegex(ValueError, "input or predecessor receipt changed"):
            materialize_inputs(host, self.operations[1])

    def test_dynamic_false_does_not_grant_cpu_or_model_permissions(self):
        self.scope["cpu_operation_ids"] = []
        atomic_json(self.scope_ref, self.scope)
        with self.assertRaisesRegex(PermissionError, "CPU operation authorization"):
            self.host().advance()
        self.assertFalse(self.counter.exists())

    def test_real_cpu_timeout_and_storage_stop_preserve_original_attempt(self):
        template = self.config["operations"]["produce"]
        template["supervisor_limits"]["wall_seconds"] = 0.8
        template["command"] = [sys.executable, "-c", "import time; time.sleep(10)"]
        atomic_json(self.host_ref, self.config)
        self.assertEqual(self.host().advance()["status"], "running")
        host, completion = self.end_job("produce", expected="unknown")
        report = read(completion.parent / "supervisor.json")
        self.assertEqual(report["stop_reason"], "wall_seconds")
        self.assertTrue(report["cleanup_confirmed"])
        self.assertEqual(host.advance()["status"], "unknown")
        self.assertFalse((host.root / "jobs/consume/submission.json").exists())

    def test_real_cpu_storage_limit_stops_before_successor(self):
        template = self.config["operations"]["produce"]
        target = Path(template["result_ref"]).with_name("oversized.bin")
        template["command"] = [
            sys.executable,
            "-c",
            f"from pathlib import Path; import time; Path({str(target)!r}).write_bytes(b'x'*3000000); time.sleep(5)",
        ]
        atomic_json(self.host_ref, self.config)
        self.assertEqual(self.host().advance()["status"], "running")
        host, completion = self.end_job("produce", expected="unknown")
        report = read(completion.parent / "supervisor.json")
        self.assertEqual(report["stop_reason"], "storage_bytes")
        self.assertTrue(report["cleanup_confirmed"])
        self.assertGreater(
            host.ledger.read()["used"]["storage_bytes"],
            template["budget"]["storage_bytes"],
        )
        self.assertEqual(host.advance()["status"], "unknown")

    def test_existing_baseline_consumer_runs_from_same_host_and_exports_actual_freeze(
        self,
    ):
        import test_midterm_bindings

        fixture = test_midterm_bindings.MidtermBindingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = self.operator / "arm-bindings.json"
        atomic_json(manifest, fixture.bindings)
        self.operations[0]["payload"] = {"manifest_ref": str(manifest)}
        atomic_json(self.core_ref, self.core)
        producer = self.config["operations"]["produce"]
        producer.update(
            cpu_purpose="baseline_freeze",
            command=[sys.executable, str(ROOT / "scripts/run_cpu_operation.py")],
        )
        producer["exports"]["theta"]["field"] = "artifact_ref"
        from rsi_vla.midterm_host import build_configuration

        self.core["repository_root"] = str(ROOT)
        self.operations[0].update(
            cpu_purpose="baseline_freeze",
            version="synthetic-v1",
            exports=producer["exports"],
        )
        atomic_json(self.core_ref, self.core)
        generated = build_configuration(
            self.core_ref, self.host_ref, self.operator / "hub.json"
        )["operations"]["produce"]
        # These are the fixture's explicitly reviewed task resource allocations.
        generated.update(
            budget=producer["budget"], supervisor_limits=producer["supervisor_limits"]
        )
        self.config["operations"]["produce"] = producer = generated
        consumer = self.config["operations"]["consume"]
        consumer["command"] = [
            sys.executable,
            "-c",
            "import json,os; from pathlib import Path; "
            "p=json.loads(Path(os.environ['RSI_CPU_PERMIT_REF']).read_text()); "
            "i=json.loads(Path(os.environ['RSI_OPERATION_INPUT_REF']).read_text()); "
            "b=json.loads(Path(i['references']['theta']).read_text()); "
            "assert b['schema']=='rsi-midterm-B0-v1' and b['B1_status']=='not_developed'; "
            "Path(p['result_ref']).write_text(json.dumps(dict(schema='rsi-cpu-operation-result-v1',"
            "scope_id=p['scope_id'],operation_id=p['operation_id'],version=p['version'],status='completed')))",
        ]
        atomic_json(self.host_ref, self.config)
        self.produce()
        self.assertEqual(self.host().advance()["status"], "running")
        host, _completion = self.end_job("consume")
        self.assertEqual(host.advance()["status"], "completed")
        frozen = read(Path(producer["result_ref"]).with_name("artifact.json"))
        self.assertEqual(set(frozen["arms"]), {"D", "S", "R", "A"})


if __name__ == "__main__":
    unittest.main()
