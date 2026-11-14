"""A failed process can close ownership without establishing technical validity."""

import unittest

import test_protocol10_integration as integration
from rsi_vla.protocol10_backend import read
from rsi_vla.runtime_cleanup import inspect_cleanup
from rsi_vla.storage import atomic_json


class FailedRuntimeClosureTests(unittest.TestCase):
    def test_nonzero_exit_settles_actual_owner_receipts_then_allows_failure_evaluation(
        self,
    ):
        f = integration.Protocol10IntegrationTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        run = f.slots[0]
        f.technical_fault_run = run["run_id"]
        f.coordinator.advance(f.inputs())
        output = run["output_ref"]
        from pathlib import Path

        output = Path(output)
        identity = read(output / "launcher_costs.json")["identity"]
        atomic_json(
            output / "supervisor.json",
            {
                "identity": identity,
                "status": "completed",
                "exit_code": 1,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
            },
        )
        atomic_json(
            output / "request_gate.json",
            {
                "identity": identity,
                "cleanup_complete": True,
                "active_requests_at_close": 0,
                "unfinished_request_numbers": [],
            },
        )
        original = f.audit

        def real_cleanup(output, **kwargs):
            report = original(output, **kwargs)
            report["checks"].pop("owner_cleanup")
            return report

        f.backend.audit = real_cleanup
        self.assertTrue(inspect_cleanup(output, identity))
        state = f.coordinator.advance(f.inputs())
        settlement = state["receipts"][run["run_id"] + "-settlement"]
        self.assertTrue(settlement["cleanup_complete"])
        self.assertFalse(settlement["audit"]["checks"]["clean_completion"])
        f.backend.progress_inspector = lambda *args: {"value": None}
        state = f.coordinator.advance(f.inputs())
        result = state["receipts"][run["run_id"] + "-attempt_evaluation"]
        self.assertEqual(
            (result["validity"], result["failure_class"]), ("invalid", "technical")
        )
        self.assertIsNone(result["critical_progress"])
        self.assertEqual(read(f.phase_ref)["used"]["simulation_episodes"], 1)
        gate = read(output / "request_gate.json")
        gate["active_requests_at_close"] = 1
        atomic_json(output / "request_gate.json", gate)
        self.assertFalse(inspect_cleanup(output, identity))
        gate["active_requests_at_close"] = 0
        gate["identity"]["run_id"] += "-other"
        atomic_json(output / "request_gate.json", gate)
        self.assertFalse(inspect_cleanup(output, identity))


class UnknownRoleLaunchTests(unittest.TestCase):
    def test_unreturned_dynamic_role_marker_never_relaunches(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from rsi_vla.protocol10_roles import run_role

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"isolation_task_root": str(root), "offline_fixture": True}
            operation = {
                "operation_id": "one",
                "kind": "runtime",
                "dynamic": True,
                "payload": {},
            }
            task = {
                "backend_root": str(root / "operator/protocol10"),
                "action": "execute",
                "operation": operation,
                "inputs": {},
            }
            token = "one-execute-revision-00001.json"
            receipt = root / "runtime/role_receipts" / token
            request = {
                "configuration": config,
                "task": task,
                "role": "runtime",
                "receipt_ref": str(receipt),
            }
            atomic_json(root / "operator/role_requests" / token, request)
            with (
                patch(
                    "rsi_vla.protocol10_roles.subprocess.run",
                    side_effect=AssertionError("duplicate process"),
                ),
                self.assertRaisesRegex(RuntimeError, "never resubmit"),
            ):
                run_role(config, task)
