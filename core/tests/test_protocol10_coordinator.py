"""Offline recovery and admission behavior for the protocol10 coordinator."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.runtime import dynamic_job_lock

BINDING = {
    "registration_ref": "registered.json",
    "protocol_id": "protocol10",
    "versions": {"execution": "e1", "evaluation": "v1", "control": "c1"},
}


def operation(key="baseline-01", *, dynamic=True, kind="runtime"):
    return {
        "operation_id": key,
        "stage": "baseline",
        "kind": kind,
        "dynamic": dynamic,
        "payload": {"run_id": key},
        "resource_requirements": {"model_requests": 1},
    }


def inputs():
    return {
        "now": "2026-11-13T00:00:00Z",
        "authorization": {
            "dynamic_authorized": True,
            "binding": deepcopy(BINDING),
            "expires_at": "2026-11-14T00:00:00Z",
            "operation_ids": ["baseline-01"],
        },
        "resources": {
            "available": {"model_requests": 3},
            "required": {"model_requests": 1},
            "reserve": {"model_requests": 1},
        },
    }


class Backend:
    def __init__(self, operations=None):
        self.operations = operations or [operation()]
        self.calls = []
        self.completed = {}
        self.reconciled = None
        self.execute_hook = None

    def observe(self, progress, values):
        self.calls.append("observe")
        for op in self.operations:
            if op["operation_id"] not in progress["receipts"]:
                return {"next_operation": deepcopy(op)}
        return {"terminal": "baseline_ceiling_stop"}

    def reconcile(self, op, values):
        self.calls.append("reconcile:" + op["operation_id"])
        if self.reconciled is not None:
            return deepcopy(self.reconciled)
        return self.completed.get(
            op["operation_id"],
            {
                "operation_id": op["operation_id"],
                "status": "not_submitted",
                "submission_absent": True,
                "receipt_ref": "native-directory-check",
            },
        )

    def execute(self, op, values):
        self.calls.append("execute:" + op["operation_id"])
        if self.execute_hook:
            self.execute_hook(op)
        receipt = {
            "operation_id": op["operation_id"],
            "status": "completed",
            "receipt_ref": "native-receipt.json",
        }
        self.completed[op["operation_id"]] = receipt
        return receipt


class CoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = Backend()
        self.coordinator = Coordinator(self.root, BINDING, self.backend)

    def event(self, kind, key, op="baseline-01", **extra):
        return {"type": kind, "event_id": key, "operation_id": op, **extra}

    def test_one_operation_per_advance_and_terminal_receipt_reuse(self):
        self.backend.operations.append(operation("baseline-02", dynamic=False))
        self.backend.operations[-1]["stage"] = "candidate"
        first = self.coordinator.advance(inputs())
        self.assertEqual(list(first["receipts"]), ["baseline-01"])
        self.assertEqual(
            self.backend.calls,
            ["observe", "reconcile:baseline-01", "execute:baseline-01"],
        )
        self.coordinator.advance(inputs())
        terminal = self.coordinator.advance(inputs())
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(
            (first["current_stage"], terminal["current_stage"]),
            ("baseline", "candidate"),
        )
        calls = list(self.backend.calls)
        self.assertEqual(self.coordinator.advance(inputs()), terminal)
        self.assertEqual(self.backend.calls, calls)

    def test_intent_durable_before_backend_and_no_dynamic_lock_nesting(self):
        def check(op):
            persisted = json.loads((self.root / "progress.json").read_text())
            self.assertEqual(persisted["pending"], op)
            self.assertEqual(persisted["pending_receipt"]["status"], "unknown")
            with dynamic_job_lock():
                pass

        self.backend.execute_hook = check
        self.assertIsNone(self.coordinator.advance(inputs())["pending"])

    def test_crash_after_native_effect_recovers_before_any_replay(self):
        def crash(op):
            self.backend.completed[op["operation_id"]] = {
                "operation_id": op["operation_id"],
                "status": "completed",
                "receipt_ref": "saved-runtime-receipt",
            }
            raise SystemExit("process died after submission")

        self.backend.execute_hook = crash
        with self.assertRaises(SystemExit):
            self.coordinator.advance(inputs())
        restored = Coordinator(self.root, BINDING, self.backend)
        self.assertEqual(restored.status()["pending"]["operation_id"], "baseline-01")
        self.backend.calls.clear()
        state = restored.advance({})
        self.assertEqual(self.backend.calls, ["reconcile:baseline-01"])
        self.assertEqual(
            state["receipts"]["baseline-01"]["receipt_ref"], "saved-runtime-receipt"
        )

    def test_unknown_never_changes_id_or_reexecutes_without_absence_evidence(self):
        self.backend.reconciled = {"operation_id": "baseline-01", "status": "unknown"}
        for _ in range(2):
            state = self.coordinator.advance(inputs())
            self.assertEqual(state["pending"]["operation_id"], "baseline-01")
            self.assertEqual(state["wait"]["reason"], "submission_unknown")
        self.backend.reconciled = {
            "operation_id": "baseline-01",
            "status": "not_submitted",
        }
        self.assertEqual(
            self.coordinator.advance(inputs())["wait"]["reason"], "submission_unknown"
        )
        self.assertFalse(any(c.startswith("execute:") for c in self.backend.calls))

    def test_execution_exception_is_unknown_and_reconciled_on_resume(self):
        def fail(op):
            raise RuntimeError("transport disconnected")

        self.backend.execute_hook = fail
        state = self.coordinator.advance(inputs())
        self.assertEqual(state["pending_receipt"]["status"], "unknown")
        self.backend.reconciled = {
            "operation_id": "baseline-01",
            "status": "waiting",
            "reason": "native_submission_unresolved",
        }
        before = len(self.backend.calls)
        self.coordinator.advance(inputs())
        self.assertEqual(self.backend.calls[before:], ["reconcile:baseline-01"])

    def test_reconcile_only_does_not_create_or_execute_new_intent(self):
        self.coordinator.advance(inputs(), reconcile_only=True)
        self.assertIsNone(self.coordinator.status()["pending"])
        self.backend.reconciled = {"operation_id": "baseline-01", "status": "unknown"}
        self.coordinator.advance(inputs())
        self.backend.reconciled = None
        state = self.coordinator.advance(inputs(), reconcile_only=True)
        self.assertIsNotNone(state["pending"])
        self.assertFalse(any(c.startswith("execute:") for c in self.backend.calls))

    def test_events_deduplicate_conflict_and_reject_authority_fields(self):
        event = self.event("pause", "event-1", None)
        first = self.coordinator.event(event)
        self.assertEqual(self.coordinator.event(event), first)
        with self.assertRaises(ValueError):
            self.coordinator.event(self.event("resume", "event-1", None))
        for field in ("command", "permission", "budget", "authorization"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.coordinator.event(
                    self.event("timer", "injected", None, **{field: True})
                )
        self.assertEqual(self.coordinator.status()["status"], "paused")

    def test_timer_cannot_release_pause_and_resume_cannot_grant_authority(self):
        self.coordinator.event(self.event("pause", "pause", None))
        self.coordinator.event(self.event("timer", "tick", None))
        self.assertEqual(self.coordinator.advance(inputs())["status"], "paused")
        self.assertFalse(self.backend.calls)
        self.coordinator.event(self.event("resume", "resume", None))
        self.assertEqual(
            self.coordinator.advance({})["wait"]["reason"], "authorization_required"
        )

    def test_late_event_cannot_pause_other_operation(self):
        self.coordinator.advance(inputs())
        self.backend.operations.append(operation("baseline-02", dynamic=False))
        self.backend.reconciled = {"operation_id": "baseline-02", "status": "unknown"}
        self.coordinator.advance(inputs())
        state = self.coordinator.event(self.event("pause", "late"))
        self.assertEqual(state["pending"]["operation_id"], "baseline-02")
        self.assertFalse(state["paused"])

    def test_pause_cancel_and_terminate_persist_with_evidence_only_recovery(self):
        for kind, expected in (
            ("pause", "paused"),
            ("cancel", "cancelled"),
            ("terminate", "terminated"),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                backend = Backend()
                backend.reconciled = {
                    "operation_id": "baseline-01",
                    "status": "unknown",
                }
                coordinator = Coordinator(directory, BINDING, backend)
                coordinator.advance(inputs())
                coordinator.event(self.event(kind, "control"))
                backend.reconciled = {
                    "operation_id": "baseline-01",
                    "status": "completed",
                    "receipt_ref": "late-native-receipt",
                }
                restored = Coordinator(directory, BINDING, backend)
                before = len(backend.calls)
                self.assertEqual(restored.advance(inputs())["status"], expected)
                self.assertEqual(len(backend.calls), before)
                closed = restored.advance(inputs(), reconcile_only=True)
                self.assertEqual(closed["status"], expected)
                self.assertIsNone(closed["pending"])
                self.assertEqual(backend.calls[before:], ["reconcile:baseline-01"])

    def test_authorization_and_budget_waits_keep_same_intent(self):
        cases = [
            (lambda v: v.pop("authorization"), "authorization_required"),
            (
                lambda v: v["authorization"].update(binding={"protocol_id": "wrong"}),
                "authorization_binding_mismatch",
            ),
            (
                lambda v: v["authorization"].update(operation_ids=[]),
                "operation_not_authorized",
            ),
            (lambda v: v.update(now="2026-11-14T00:00:00Z"), "authorization_expired"),
            (lambda v: v.pop("now"), "authorization_time_required"),
            (
                lambda v: v["resources"]["available"].update(model_requests=1),
                "budget_wait",
            ),
            (
                lambda v: v["resources"].update(unresolved_costs=["unknown-cost"]),
                "resource_evidence_required",
            ),
        ]
        for modify, reason in cases:
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as directory,
            ):
                backend = Backend()
                coordinator = Coordinator(directory, BINDING, backend)
                values = inputs()
                modify(values)
                state = coordinator.advance(values)
                self.assertEqual(state["wait"]["reason"], reason)
                self.assertEqual(
                    Coordinator(directory, BINDING, backend).status()["wait"],
                    state["wait"],
                )
                self.assertFalse(any(c.startswith("execute:") for c in backend.calls))
                self.assertIn("baseline-01", coordinator.advance(inputs())["receipts"])

    def test_evaluation_failure_does_not_restart_completed_runtime(self):
        self.coordinator.advance(inputs())
        self.backend.operations.append(
            operation("evaluate-01", dynamic=False, kind="evaluation")
        )

        def fail(op):
            raise ValueError("evaluation requires repaired consumer")

        self.backend.execute_hook = fail
        state = self.coordinator.advance(inputs())
        self.assertEqual(state["wait"]["reason"], "evaluation_repair")
        self.backend.reconciled = {
            "operation_id": "evaluate-01",
            "status": "completed",
            "receipt_ref": "evaluation-v2.json",
        }
        state = self.coordinator.advance(inputs())
        self.assertEqual(set(state["receipts"]), {"baseline-01", "evaluate-01"})
        self.assertEqual(self.backend.calls.count("execute:baseline-01"), 1)

    def test_native_evaluation_kinds_fail_as_repair_wait(self):
        for kind in ("attempt_evaluation", "matrix_evaluation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                backend = Backend([operation(dynamic=False, kind=kind)])

                def fail(op):
                    raise RuntimeError("evaluation parser failure")

                backend.execute_hook = fail
                state = Coordinator(directory, BINDING, backend).advance({})
                self.assertEqual(state["wait"]["reason"], "evaluation_repair")
                self.assertEqual(state["pending"]["kind"], kind)

    def test_binding_and_operation_conflicts_rejected(self):
        self.assertEqual(
            Coordinator(self.root, deepcopy(BINDING), self.backend).status(),
            self.coordinator.status(),
        )
        with self.assertRaises(ValueError):
            Coordinator(self.root, {**BINDING, "protocol_id": "other"}, self.backend)
        self.coordinator.advance(inputs())
        self.backend.operations[0]["payload"]["run_id"] = "changed"

        def repeat(progress, values):
            return {"next_operation": self.backend.operations[0]}

        self.backend.observe = repeat
        with self.assertRaisesRegex(ValueError, "conflicting operation"):
            self.coordinator.advance(inputs())

    def test_two_controllers_cannot_execute_concurrently(self):
        second = Coordinator(self.root, BINDING, self.backend)
        entered, release = threading.Event(), threading.Event()

        def hold(op):
            entered.set()
            self.assertTrue(release.wait(5))

        self.backend.execute_hook = hold
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.coordinator.advance, inputs())
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, "another protocol10"):
                    second.advance(inputs())
            finally:
                release.set()
            self.assertIn("baseline-01", future.result(timeout=5)["receipts"])
        self.assertEqual(self.backend.calls.count("execute:baseline-01"), 1)


if __name__ == "__main__":
    unittest.main()
