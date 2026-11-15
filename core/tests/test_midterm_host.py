"""Prevent duplicate side effects, unreserved jobs, and timer-driven pause bypass."""

import json
import os
import tempfile
import time
import shutil
import subprocess
from unittest.mock import patch
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rsi_vla.midterm_host import (
    MidtermHost,
    MidtermLedger,
    build_autoloop_rules,
    worker_receipt,
    runtime_permit_binding,
    prove_unsubmitted_preparation,
    prepare_preparation_repair,
    process_identity,
    process_matches,
    inspect_owned_runtime_processes,
    ended_request_count,
)
from rsi_vla.storage import atomic_json


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope_ref = self.root / "scope.json"
        self.phase = self.root / "phase.json"
        self.scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": "test-midterm",
            "status": "active",
            "authorized": True,
            "dynamic_authorized": True,
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "outer_limits": {
                "training_starts": 2,
                "gpu_seconds": 100,
                "storage_bytes": 1000000,
                "current_disk_bytes": 1000000,
                "concurrent_gpu_devices": 1,
                "resource_wait_seconds": 10,
            },
        }
        atomic_json(self.scope_ref, self.scope)
        atomic_json(
            self.phase,
            {
                "limits": {"gpu_seconds": 100},
                "remaining": {"gpu_seconds": 100},
                "unresolved_costs": [],
            },
        )
        self.ledger = MidtermLedger(self.scope_ref)
        self.budget = {"training_starts": 1, "gpu_seconds": 60}

    def reserve(self, name="one"):
        return self.ledger.reserve(
            name,
            budget=self.budget,
            phase_ref=self.phase,
            gpu_devices=[0],
            receipt_ref=self.root / (name + ".json"),
        )

    def test_reservation_serializes_gpu_and_resource_envelopes(self):
        self.reserve()
        with self.assertRaises(PermissionError):
            self.reserve("two")
        record = self.ledger.validate_reservation(
            "one", budget=self.budget, phase_ref=self.phase, gpu_devices=[0]
        )
        self.assertEqual(record["status"], "active")
        with self.assertRaises(ValueError):
            self.ledger.reserve(
                "one",
                budget={"gpu_seconds": 1},
                phase_ref=self.phase,
                gpu_devices=[0],
                receipt_ref=self.root / "one.json",
            )

    def test_settlement_is_idempotent_and_never_refunds_launch(self):
        self.reserve()
        self.ledger.mark_submitted("one")
        self.ledger.mark_submitted("one")
        actual = {"gpu_seconds": 5, "training_starts": 1}
        self.ledger.settle("one", actual=actual, cleanup_confirmed=True)
        self.ledger.settle("one", actual=actual, cleanup_confirmed=True)
        view = self.ledger.read()
        self.assertEqual(view["used"]["training_starts"], 1)
        self.assertEqual(view["used"]["gpu_seconds"], 5)
        self.reserve("two")

    def test_missing_cost_holds_reservation_and_blocks_dispatch(self):
        self.reserve()
        self.ledger.mark_submitted("one")
        self.ledger.settle("one", actual={}, cleanup_confirmed=True)
        self.assertEqual(self.ledger.read()["reservations"]["one"]["status"], "unknown")
        with self.assertRaises(PermissionError):
            self.reserve("two")

    def test_consumer_rechecks_live_scope_and_phase(self):
        self.reserve()
        atomic_json(
            self.phase,
            {
                "limits": {"gpu_seconds": 100},
                "remaining": {"gpu_seconds": 20},
                "unresolved_costs": [],
            },
        )
        with self.assertRaises(PermissionError):
            self.ledger.validate_reservation(
                "one", budget=self.budget, phase_ref=self.phase
            )
        self.scope["status"] = "paused"
        atomic_json(self.scope_ref, self.scope)
        with self.assertRaises(PermissionError):
            self.ledger.validate_reservation("one")

    def test_pause_and_terminal_cancel_existing_timer(self):
        for status in ("paused", "completed", "unknown", "expired"):
            receipt = worker_receipt(
                {"status": status}, scope_id="test", poll_seconds=1
            )
            self.assertEqual(receipt["later"], [{"key": "test-wake", "cancel": True}])
        receipt = worker_receipt({"status": "running"}, scope_id="test", poll_seconds=1)
        self.assertIn("at", receipt["later"][0])

    def test_rules_bind_fixed_command_and_disable_channels(self):
        cfg = {
            "scope_id": "test",
            "python": "/python",
            "script_ref": "/script",
            "configuration_ref": "/host.json",
            "repository_root": "/repo",
        }
        rules = build_autoloop_rules(cfg)
        self.assertEqual(rules["channels"], {})
        command = rules["agents"]["test-driver"]["command"]
        self.assertEqual(
            command, ["/python", "/script", "worker", "--configuration", "/host.json"]
        )
        self.assertNotIn("{prompt}", json.dumps(rules))


class DetachedHostTests(unittest.TestCase):
    setUp = LedgerTests.setUp

    def test_host_restart_reconciles_independent_supervisor_without_reexecution(self):
        operation = {
            "operation_id": "cpu-job",
            "stage": "engineering",
            "kind": "host_command",
            "dynamic": False,
            "payload": {},
        }
        core_ref = self.root / "core.json"
        protocol_ref = self.root / "protocol.json"
        inputs_ref = self.root / "inputs.json"
        result_ref = self.root / "native-result.json"
        count_ref = self.root / "executions.txt"
        atomic_json(protocol_ref, {"protocol_id": "cpu-engineering"})
        phase = {
            "limits": {"wall_seconds": 60, "storage_bytes": 1000000},
            "used": {"wall_seconds": 0, "storage_bytes": 0},
            "remaining": {"wall_seconds": 60, "storage_bytes": 1000000},
            "unresolved_costs": [],
        }
        atomic_json(self.phase, phase)
        core = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": str(self.scope_ref),
            "protocol_ref": str(protocol_ref),
            "base_bundle_id": "engineering",
            "phase_ref": str(self.phase),
            "operations": [operation],
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "job",
                "nodes": {
                    "job": {
                        "operation_id": "cpu-job",
                        "transitions": [{"next": "end"}],
                    },
                    "end": {"terminal": "engineering_complete"},
                },
            },
        }
        atomic_json(core_ref, core)
        atomic_json(inputs_ref, {})
        host_ref = self.root / "host.json"
        config = {
            "schema": "rsi-midterm-host-v1",
            "scope_id": self.scope["scope_id"],
            "scope_authorization_ref": str(self.scope_ref),
            "continuation_configuration_ref": str(core_ref),
            "continuation_root": str(self.root / "continuation"),
            "inputs_ref": str(inputs_ref),
            "repository_root": str(self.root),
            "python": os.sys.executable,
            "script_ref": str(
                Path(__file__).resolve().parents[2] / "scripts/run_midterm_host.py"
            ),
            "operations": {
                "cpu-job": {
                    "budget": {"storage_bytes": 1000000},
                    "supervisor_limits": {"wall_seconds": 5, "storage_bytes": 1000000},
                    "command": [
                        os.sys.executable,
                        "-c",
                        "import pathlib,time,json; "
                        f"pathlib.Path({str(count_ref)!r}).open('a').write('run\\n'); "
                        "time.sleep(.4); "
                        f"pathlib.Path({str(result_ref)!r}).write_text(json.dumps({{'operation_id':'cpu-job'}}))",
                    ],
                    "result_ref": str(result_ref),
                }
            },
        }
        atomic_json(host_ref, config)
        import fcntl

        lease_ref = self.root / "engineering-lease.lock"
        config["operations"]["cpu-job"]["admission_lease_ref"] = str(lease_ref)
        atomic_json(host_ref, config)
        with lease_ref.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            self.assertEqual(
                MidtermHost(host_ref).advance()["status"], "waiting_resources"
            )
            time.sleep(0.02)
            self.assertEqual(
                MidtermHost(host_ref).advance()["status"], "waiting_resources"
            )
            self.assertGreater(self.ledger.read()["used"]["resource_wait_seconds"], 0)
        first = MidtermHost(host_ref).advance()
        self.assertEqual(first["status"], "running")
        restarted = MidtermHost(host_ref)
        self.assertEqual(restarted.advance()["status"], "running")
        completion = restarted.root / "jobs/cpu-job/completion.json"
        deadline = time.monotonic() + 8
        while not completion.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(
            completion.exists(),
            (restarted.root / "jobs/cpu-job/backend.log").read_text(),
        )
        self.assertEqual(
            json.loads(completion.read_text())["status"],
            "completed",
            (completion.parent / "backend.log").read_text(),
        )
        self.assertEqual(restarted.advance()["status"], "completed")
        self.assertEqual(count_ref.read_text(), "run\n")
        self.assertEqual(
            restarted.ledger.read()["reservations"]["cpu-job"]["status"], "settled"
        )


class PreparationRepairTests(unittest.TestCase):
    def setUp(self):
        LedgerTests.setUp(self)
        self.scope["outer_limits"].update(runtime_starts=2, model_requests=10)
        atomic_json(self.scope_ref, self.scope)
        self.ledger.path.unlink()
        self.ledger = MidtermLedger(self.scope_ref)
        self.operation = {
            "operation_id": "rejected-runtime",
            "kind": "runtime",
            "stage": "development",
            "dynamic": True,
            "payload": {"run_id": "logical-slot"},
            "resource_requirements": {"simulation_episodes": 1},
        }
        self.core_ref = self.root / "core.json"
        self.host_ref = self.root / "host.json"
        self.source_ref = self.root / "old-source/manifest.json"
        import rsi_vla.runtime

        repository = Path(rsi_vla.runtime.__file__).resolve().parents[3]
        files = ["core/src/rsi_vla/runtime.py", "core/src/rsi_vla/task_completion.py"]
        for name in files:
            target = self.source_ref.parent / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, target)
        atomic_json(self.source_ref, {"version": "5a-assets-offline", "files": files})
        protocol_ref = self.root / "protocol.json"
        atomic_json(
            protocol_ref,
            {
                "protocol_id": "test",
                "budget": {
                    "runtime_per_start": {
                        "simulation_episodes": 1,
                        "wall_seconds": 5,
                        "gpu_seconds": 5,
                        "model_requests": 1,
                        "storage_bytes": 1000000,
                    },
                    "per_attempt_proposal": {"planner_timeout_seconds": 5},
                },
            },
        )
        inputs_ref = self.root / "inputs.json"
        atomic_json(
            inputs_ref,
            {
                "authorization": {
                    "binding": {"registration_ref": "old"},
                    "operation_ids": [self.operation["operation_id"]],
                }
            },
        )
        self.slot = {
            "run_id": "logical-slot",
            "bundle_id": "base",
            "task_key": "salad",
            "seed": 2,
            "split": "development",
            "cuda_device": 0,
            "output_ref": str(self.root / "runtime/logical-slot"),
            "scratch_ref": str(self.root / "scratch/logical-slot"),
            "permit_ref": str(self.root / "old-permit.json"),
        }
        self.core = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": str(self.scope_ref),
            "protocol_ref": str(protocol_ref),
            "phase_ref": str(self.phase),
            "repository_root": str(repository),
            "source_manifest_ref": str(self.source_ref),
            "isolation_task_root": str(self.root),
            "runtime_slots": [self.slot],
            "operations": [self.operation],
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "run",
                "nodes": {
                    "run": {
                        "operation_id": self.operation["operation_id"],
                        "transitions": [{"next": "done"}],
                    },
                    "done": {"terminal": "bounded_complete"},
                },
            },
        }
        atomic_json(self.core_ref, self.core)
        self.host = {
            "schema": "rsi-midterm-host-v1",
            "scope_id": self.scope["scope_id"],
            "scope_authorization_ref": str(self.scope_ref),
            "accounting_ref": str(self.ledger.path),
            "continuation_configuration_ref": str(self.core_ref),
            "continuation_root": str(self.root / "continuation"),
            "inputs_ref": str(inputs_ref),
            "configuration_ref": str(self.host_ref),
            "hub_configuration_ref": str(self.root / "hub.json"),
            "repository_root": str(repository),
            "python": os.sys.executable,
            "script_ref": "/script",
            "node": "/node",
            "autoloop_root": "/autoloop",
            "autoloop_home": str(self.root / "hub"),
            "poll_seconds": 2,
            "operations": {self.operation["operation_id"]: {"budget": {}}},
        }
        atomic_json(self.host_ref, self.host)
        self.job = (
            self.root / "operator/midterm-host/jobs" / self.operation["operation_id"]
        )
        atomic_json(
            self.job / "submission.json",
            {"operation": self.operation, "owner_uid": os.getuid()},
        )
        atomic_json(
            self.job / "supervisor.json",
            {
                "identity": {"run_id": self.operation["operation_id"]},
                "status": "completed",
                "exit_code": 1,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
            },
        )
        role_ref = (
            self.root
            / "runtime/role_receipts"
            / (self.operation["operation_id"] + "-execute-revision-00001.json")
        )
        atomic_json(
            role_ref,
            {
                "error": "ValueError: authorization collection binding missing or different",
                "role": "runtime",
                "request": {
                    "configuration": self.core,
                    "task": {"operation": self.operation},
                },
            },
        )
        self.ledger.reserve(
            self.operation["operation_id"],
            budget={"runtime_starts": 1, "model_requests": 1, "gpu_seconds": 5},
            phase_ref=self.phase,
            gpu_devices=[0],
            receipt_ref=self.job / "completion.json",
        )
        self.ledger.mark_submitted(self.operation["operation_id"])
        self.ledger.settle(
            self.operation["operation_id"],
            actual={"runtime_starts": 0, "gpu_seconds": 0.2},
            cleanup_confirmed=True,
        )

    def test_proven_unsubmitted_repair_preserves_original_and_same_slot(self):
        before = self.core_ref.read_bytes()
        proof = prove_unsubmitted_preparation(
            self.host_ref, self.operation["operation_id"]
        )
        self.assertTrue(proof["submission_absent"])
        result = prepare_preparation_repair(
            self.host_ref, self.operation["operation_id"], "repair1"
        )
        self.assertEqual(self.core_ref.read_bytes(), before)
        successor = MidtermHost(result["configuration_ref"])
        self.assertNotEqual(
            successor.config["continuation_root"], self.host["continuation_root"]
        )
        self.assertEqual(
            successor.core_config["runtime_slots"][0]["run_id"], self.slot["run_id"]
        )
        self.assertEqual(successor.coordinator().status()["receipts"], {})
        record = self.ledger.read()["reservations"][self.operation["operation_id"]]
        self.assertEqual(record["status"], "settled")
        self.assertEqual(
            record["actual"],
            {"runtime_starts": 0, "gpu_seconds": 0.2, "model_requests": 0},
        )
        self.assertFalse((self.root / "runtime/logical-slot").exists())
        self.assertEqual(
            prepare_preparation_repair(
                self.host_ref, self.operation["operation_id"], "repair1"
            ),
            result,
        )

    def test_existing_runtime_output_blocks_unsubmitted_classification(self):
        atomic_json(
            Path(self.slot["output_ref"]) / "runtime_context.json",
            {"run_id": self.slot["run_id"]},
        )
        with self.assertRaisesRegex(PermissionError, "Runtime output"):
            prove_unsubmitted_preparation(self.host_ref, self.operation["operation_id"])

    def test_extended_owned_process_metadata_retains_birth_matching(self):
        path = self.root / "process.json"
        atomic_json(path, {**process_identity(os.getpid()), "command": "engineering"})
        self.assertTrue(process_matches(path))


class RuntimePermitBindingTests(unittest.TestCase):
    def setUp(self):
        import test_runtime_plan

        test_runtime_plan.RuntimePlanTests.setUp(self)
        self.repository = test_runtime_plan.ROOT

    def test_real_planner_derives_collection_and_rejects_missing_binding(self):
        from rsi_vla.runtime import prepare_runtime

        profile_ref = self.artifacts / "research/profiles/initial-medium.json"
        profile = json.loads(profile_ref.read_text())
        profile["collection"] = {
            "version": "aligned-batches-v1",
            "enabled": True,
            "relative_root": "data",
            "flush_steps": 25,
            "required_for_release": False,
            "required_for_development": True,
        }
        atomic_json(profile_ref, profile)
        configuration = {
            "repository_root": str(self.repository),
            "artifacts_root": str(self.artifacts),
            "protocol_ref": str(self.repository / "config/pilot_v2.json"),
        }
        slot = {
            "run_id": "bounded-medium",
            "bundle_id": "initial-medium",
            "task_key": "position_swap_salad",
            "seed": 0,
            "split": "smoke",
            "output_ref": str(self.artifacts / "run"),
            "scratch_ref": str(self.artifacts / "scratch"),
            "cuda_device": 0,
        }
        kwargs = {
            "protocol_path": configuration["protocol_ref"],
            "run_id": slot["run_id"],
            "bundle_id": slot["bundle_id"],
            "task_key": slot["task_key"],
            "seed": slot["seed"],
            "split": slot["split"],
            "output_dir": slot["output_ref"],
            "storage_write_roots": [slot["scratch_ref"]],
        }
        plan = prepare_runtime(self.repository, self.artifacts, **kwargs)
        permit = {
            "authorized": True,
            "limits": plan["context"]["limits"],
            "run_id": slot["run_id"],
            "bundle_id": slot["bundle_id"],
            "protocol_id": plan["context"]["protocol_id"],
            "task_key": slot["task_key"],
            "seed": slot["seed"],
            "simulation_episodes": 1,
            "gpu_devices": [0],
            "storage_write_roots": plan["context"]["storage_write_roots"],
        }
        permit_ref = self.artifacts / "permit.json"
        atomic_json(permit_ref, permit)
        with self.assertRaisesRegex(ValueError, "collection binding"):
            prepare_runtime(
                self.repository, self.artifacts, authorization_path=permit_ref, **kwargs
            )
        binding = runtime_permit_binding(configuration, slot)
        self.assertEqual(
            binding["collection"]["root"], str(self.artifacts / "run/data")
        )
        manifest_ref = self.artifacts / "source/manifest.json"
        source_name = "core/src/rsi_vla/runtime.py"
        target = manifest_ref.parent / "source" / source_name
        target.parent.mkdir(parents=True)
        shutil.copyfile(self.repository / source_name, target)
        atomic_json(
            manifest_ref, {"version": "5a-assets-offline", "files": [source_name]}
        )
        permit.update(binding, source_manifest_ref=str(manifest_ref))
        atomic_json(permit_ref, permit)
        self.assertEqual(
            runtime_permit_binding(configuration, slot, authorization_path=permit_ref),
            binding,
        )
        self.assertFalse(Path(slot["output_ref"]).exists())


class EndedRuntimeTests(unittest.TestCase):
    def setUp(self):
        LedgerTests.setUp(self)
        self.scope["outer_limits"].update(runtime_starts=2, model_requests=10)
        atomic_json(self.scope_ref, self.scope)
        self.ledger.path.unlink()
        self.ledger = MidtermLedger(self.scope_ref)
        import rsi_vla.runtime

        self.repository = Path(rsi_vla.runtime.__file__).resolve().parents[3]
        self.operation = {
            "operation_id": "ended-run",
            "stage": "development",
            "kind": "runtime",
            "dynamic": True,
            "payload": {"run_id": "native-run"},
            "resource_requirements": {"simulation_episodes": 1},
        }
        self.audit_op = {
            "operation_id": "independent-audit",
            "stage": "development",
            "kind": "development_base_collect",
            "dynamic": False,
            "payload": {},
        }
        self.output = self.root / "runtime/native-run"
        self.slot = {
            "run_id": "native-run",
            "bundle_id": "D",
            "task_key": "salad",
            "seed": 2,
            "output_ref": str(self.output),
            "scratch_ref": str(self.root / "scratch"),
            "permit_ref": str(self.root / "permit.json"),
        }
        protocol_ref = self.root / "protocol.json"
        atomic_json(protocol_ref, {"protocol_id": "test"})
        source_ref = self.root / "source/manifest.json"
        atomic_json(source_ref, {"version": "5a-assets-offline", "files": []})
        core = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": str(self.scope_ref),
            "protocol_ref": str(protocol_ref),
            "phase_ref": str(self.phase),
            "source_manifest_ref": str(source_ref),
            "base_bundle_id": "D",
            "runtime_slots": [self.slot],
            "operations": [self.operation, self.audit_op],
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "runtime",
                "nodes": {
                    "runtime": {
                        "operation_id": "ended-run",
                        "transitions": [{"next": "audit"}],
                    },
                    "audit": {
                        "operation_id": "independent-audit",
                        "transitions": [
                            {
                                "when": {"path": ["passed"], "equals": True},
                                "next": "unexpected",
                            },
                            {"next": "budget-stop"},
                        ],
                    },
                    "unexpected": {"terminal": "unexpected_validity"},
                    "budget-stop": {"terminal": "budget_stop_unknown"},
                },
            },
        }
        core_ref = self.root / "core.json"
        atomic_json(core_ref, core)
        inputs_ref = self.root / "inputs.json"
        atomic_json(inputs_ref, {})
        host_ref = self.root / "host.json"
        atomic_json(
            host_ref,
            {
                "schema": "rsi-midterm-host-v1",
                "scope_id": self.scope["scope_id"],
                "scope_authorization_ref": str(self.scope_ref),
                "continuation_configuration_ref": str(core_ref),
                "continuation_root": str(self.root / "continuation"),
                "repository_root": str(self.repository),
                "inputs_ref": str(inputs_ref),
                "operations": {
                    "ended-run": {
                        "budget": {
                            "runtime_starts": 1,
                            "model_requests": 1,
                            "gpu_seconds": 5,
                            "storage_bytes": 1000000,
                            "current_disk_bytes": 1000000,
                        },
                        "output_ref": str(self.output),
                    },
                    "independent-audit": {"budget": {"storage_bytes": 1000000}},
                },
            },
        )
        self.host = MidtermHost(host_ref)
        self.calls = []

        def invoke(operation, action, inputs):
            self.calls.append(operation["kind"])
            if operation["kind"] == "runtime":
                raise RuntimeError(
                    "fixture outer supervisor stopped after native dispatch"
                )
            from rsi_vla.adapters.runtime_acceptance import inspect_attempt

            return inspect_attempt(self.output, require_vla=False)

        self.host.invoke = invoke
        backend = self.host.backend()
        self.inputs = {
            "now": datetime.now(timezone.utc).isoformat(),
            "authorization": {
                "dynamic_authorized": True,
                "binding": backend.registration,
                "expires_at": self.scope["expires_at"],
                "operation_ids": ["ended-run"],
            },
            "resources": {
                "available": {"simulation_episodes": 2},
                "required": {"simulation_episodes": 1},
                "reserve": {},
            },
        }
        self.host.coordinator().advance(self.inputs)
        self.native_receipt = self.root / "continuation/receipts/ended-run.json"
        self.before_native = self.native_receipt.read_bytes()
        self.job = self.host.root / "jobs/ended-run"
        self.report = {
            "identity": {"run_id": "ended-run"},
            "status": "interrupted",
            "stop_reason": "storage_bytes",
            "cleanup_confirmed": True,
            "remaining_process_ids": [],
            "wall_seconds": 3,
            "gpu_device_seconds_envelope": 3,
            "storage_bytes": 12,
            "storage_accounting": {"current_new_bytes": 12},
        }
        atomic_json(
            self.job / "submission.json",
            {"operation": self.operation, "owner_uid": os.getuid()},
        )
        atomic_json(self.job / "supervisor.json", self.report)
        self.original_completion = {
            "status": "unknown",
            "actual": {"runtime_starts": 1},
        }
        atomic_json(self.job / "completion.json", self.original_completion)
        identity = {
            "run_id": "native-run",
            "episode_id": "salad-s2",
            "bundle_id": "D",
            "protocol_id": "test",
        }
        self.context = {
            **identity,
            "execution_arm": {"arm": "D"},
            "planner": {"backend": "flash"},
            "execution_information": {
                "online_planner": False,
                "model_requests_expected": 0,
            },
            "code_snapshot_ref": str(self.output / "code_snapshot/manifest.json"),
        }
        atomic_json(self.output / "runtime_context.json", self.context)
        atomic_json(
            self.output / "request_gate_live.json",
            {"identity": identity, "requests_forwarded": 0},
        )
        atomic_json(self.output / "code_snapshot/manifest.json", {"identity": identity})
        target = self.output / "code_snapshot/core/src/rsi_vla/adapters/request_gate.py"
        target.parent.mkdir(parents=True)
        shutil.copyfile(
            self.repository / "core/src/rsi_vla/adapters/request_gate.py", target
        )
        (self.output / "action_events.jsonl").write_text(
            json.dumps(
                {**identity, "kind": "tool_call", "sequence": 0, "tool": "vla_execute"}
            )
            + "\n"
        )
        atomic_json(
            self.output / "environment_service/runtime_costs.json",
            {
                "identity": identity,
                "pending_action": False,
                "environment_steps_unknown": 0,
                "environment_steps_reserved": 10,
                "environment_steps_confirmed": 10,
            },
        )
        budget = self.host.config["operations"]["ended-run"]["budget"]
        self.ledger.reserve(
            "ended-run",
            budget=budget,
            phase_ref=self.phase,
            gpu_devices=[0],
            receipt_ref=self.job / "completion.json",
        )
        self.ledger.mark_submitted("ended-run")
        self.ledger.settle(
            "ended-run",
            actual={
                "runtime_starts": 1,
                "gpu_seconds": 3,
                "storage_bytes": 12,
                "current_disk_bytes": 12,
            },
            cleanup_confirmed=True,
        )
        limits = {
            "wall_seconds": 60,
            "gpu_seconds": 100,
            "storage_bytes": 1000000,
            "model_requests": 10,
            "tool_calls": 20,
            "environment_steps": 20,
            "simulation_episodes": 2,
        }
        used = {key: 0 for key in limits}
        used.update(
            wall_seconds=3, gpu_seconds=3, storage_bytes=12, simulation_episodes=1
        )
        atomic_json(
            self.phase,
            {
                "limits": limits,
                "used": used,
                "remaining": {key: value - used[key] for key, value in limits.items()},
                "protocol10_trial_charges": [
                    {"run_id": "native-run", "costs": {"simulation_episodes": 1}}
                ],
                "host_command_charges": [
                    {
                        "operation_id": "ended-run",
                        "wall_seconds": 3,
                        "gpu_seconds": 3,
                        "storage_bytes": 12,
                    }
                ],
                "unresolved_costs": [
                    {
                        "run_id": "native-run",
                        "output_ref": str(self.output),
                        "missing": [
                            "model_requests",
                            "tool_calls",
                            "environment_steps",
                            "whole_job_wall_seconds",
                            "gpu_device_seconds_envelope",
                            "artifact_bytes",
                        ],
                    }
                ],
            },
        )
        self.process_evidence = {
            "host_pid_namespace_confirmed": True,
            "live_processes": [],
            "unreadable_owned_processes": [],
        }

    def test_ended_unknown_reconciles_original_then_independently_audits_without_rerun(
        self,
    ):
        with patch(
            "rsi_vla.midterm_host.inspect_owned_runtime_processes",
            return_value=self.process_evidence,
        ):
            closure = self.host.close_ended_runtime("ended-run", self.report)
        self.assertEqual(closure["value"]["native_outcome"], "unknown")
        self.assertEqual(closure["value"]["model_requests"], 0)
        self.assertEqual(self.native_receipt.read_bytes(), self.before_native)
        self.assertEqual(
            json.loads((self.job / "completion.json").read_text()),
            self.original_completion,
        )
        phase = json.loads(self.phase.read_text())
        self.assertEqual(phase["used"]["gpu_seconds"], 3)
        self.assertEqual(phase["used"]["wall_seconds"], 3)
        self.assertEqual(phase["used"]["storage_bytes"], 12)
        self.assertEqual(phase["used"]["tool_calls"], 1)
        self.assertEqual(phase["used"]["environment_steps"], 10)
        self.assertEqual(self.host.advance()["status"], "ready")
        audit_state = self.host.coordinator().advance(self.inputs)
        self.assertFalse(
            audit_state["receipts"]["independent-audit"]["value"]["passed"]
        )
        terminal = self.host.coordinator().advance(self.inputs)
        self.assertEqual(terminal["terminal"], "budget_stop_unknown")
        self.assertEqual(self.calls, ["runtime", "development_base_collect"])
        self.assertFalse((self.output / "independent_result.json").exists())

    def test_live_forwarder_or_counter_contradiction_never_becomes_zero(self):
        evidence = {**self.process_evidence, "live_processes": [{"pid": 123}]}
        self.assertIsNone(ended_request_count(self.output, self.context, evidence)[0])
        live = json.loads((self.output / "request_gate_live.json").read_text())
        live["requests_forwarded"] = 1
        atomic_json(self.output / "request_gate_live.json", live)
        self.assertIsNone(
            ended_request_count(self.output, self.context, self.process_evidence)[0]
        )

    def test_unknown_costs_allow_only_reserved_cpu_evidence_closure(self):
        live = json.loads((self.output / "request_gate_live.json").read_text())
        live["requests_forwarded"] = 1
        atomic_json(self.output / "request_gate_live.json", live)
        with patch(
            "rsi_vla.midterm_host.inspect_owned_runtime_processes",
            return_value=self.process_evidence,
        ):
            closure = self.host.close_ended_runtime("ended-run", self.report)
        self.assertIsNone(closure["value"]["model_requests"])
        self.assertEqual(
            self.ledger.read()["reservations"]["ended-run"]["status"], "unknown"
        )
        self.ledger.reserve(
            "evidence-audit",
            budget={"storage_bytes": 1000},
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.root / "evidence-audit.json",
            evidence_closure=True,
        )
        self.ledger.validate_reservation("evidence-audit")
        with self.assertRaises(PermissionError):
            self.ledger.reserve(
                "not-evidence",
                budget={"gpu_seconds": 1},
                phase_ref=self.phase,
                gpu_devices=[0],
                receipt_ref=self.root / "not-evidence.json",
                evidence_closure=True,
            )

    def test_actual_owned_context_process_is_detected_before_cleanup(self):
        anchor = self.root / "host-anchor.json"
        atomic_json(anchor, process_identity(os.getpid()))
        process = subprocess.Popen(
            [os.sys.executable, "-c", "import time;time.sleep(10)"],
            env={
                **os.environ,
                "RSI_RUNTIME_CONTEXT": str(self.output / "runtime_context.json"),
            },
        )
        try:
            observed = inspect_owned_runtime_processes(anchor, "ended-run", self.output)
            self.assertIn(
                process.pid, [item["pid"] for item in observed["live_processes"]]
            )
        finally:
            process.terminate()
            process.wait(timeout=3)


class ProcessOwnershipScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.host_anchor = self.root / "host.json"
        self.operation_anchor = self.root / "operation.json"
        identity = process_identity(os.getpid())
        atomic_json(self.host_anchor, identity)
        atomic_json(
            self.operation_anchor, {**identity, "pid": 900002, "start_ticks": "200"}
        )
        self.output = self.root / "runtime"
        self.environment_reads = []

    def inspect(self, *, birth, argv, fresh=True, pid=900001):
        directory = self.root / str(pid)
        directory.mkdir()
        fields = ["0"] * 20
        fields[0], fields[1], fields[19] = "S", "1", str(birth)
        (directory / "stat").write_text(f"{pid} (opaque-process) " + " ".join(fields))
        (directory / "cmdline").write_bytes(argv)
        (directory / "environ").write_bytes(b"PRIVATE_FIXTURE=must-not-be-read")
        original_glob, original_read = Path.glob, Path.read_bytes

        def glob(path, pattern):
            return (
                iter([directory])
                if str(path) == "/proc"
                else original_glob(path, pattern)
            )

        def read_bytes(path):
            if path == directory / "environ":
                self.environment_reads.append(pid)
                raise PermissionError("fixture reproduces inaccessible environment")
            return original_read(path)

        with (
            patch.object(Path, "glob", glob),
            patch.object(Path, "read_bytes", read_bytes),
            patch(
                "rsi_vla.midterm_host.fresh_runtime_actor_evidence",
                return_value={"established": fresh},
            ),
        ):
            return inspect_owned_runtime_processes(
                self.host_anchor,
                "operation",
                self.output,
                operation_anchor_ref=self.operation_anchor,
            )

    def test_old_unassociated_service_is_excluded_without_reading_environment(self):
        observed = self.inspect(birth=100, argv=b"unrelated-service\0")
        self.assertEqual(observed["unreadable_owned_processes"], [])
        self.assertEqual(observed["live_processes"], [])
        self.assertEqual(
            [row["pid"] for row in observed["excluded_preexisting_processes"]], [900001]
        )
        self.assertEqual(self.environment_reads, [])
        self.assertFalse(
            observed["excluded_preexisting_processes"][0]["environment_read"]
        )

    def test_new_unreadable_actor_remains_unknown(self):
        observed = self.inspect(birth=300, argv=b"opaque-new-actor\0")
        self.assertEqual(observed["unreadable_owned_processes"], [900001])
        self.assertEqual(observed["excluded_preexisting_processes"], [])

    def test_known_argument_or_owner_marker_is_never_excluded_by_age(self):
        observed = self.inspect(birth=100, argv=(str(self.output) + "\0").encode())
        self.assertEqual(observed["live_processes"][0]["pid"], 900001)
        self.assertEqual(observed["excluded_preexisting_processes"], [])
        observed = self.inspect(birth=100, argv=b"opaque-owner\0", pid=900002)
        self.assertEqual(observed["live_processes"][0]["via"], "operation_owner_marker")
        self.assertEqual(observed["excluded_preexisting_processes"], [])

    def test_age_alone_cannot_exclude_actor_without_fresh_run_proof(self):
        observed = self.inspect(birth=100, argv=b"opaque-old-actor\0", fresh=False)
        self.assertEqual(observed["unreadable_owned_processes"], [900001])
        self.assertEqual(observed["excluded_preexisting_processes"], [])


if __name__ == "__main__":
    unittest.main()
