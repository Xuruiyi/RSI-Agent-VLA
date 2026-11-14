"""Concurrent inspection must preserve a running broker and manual pause."""

import threading
import unittest
from tempfile import TemporaryDirectory

from rsi_vla.protocol10_coordinator import Coordinator


class ConcurrentControlTests(unittest.TestCase):
    def test_status_and_pause_during_execute_preserve_receipt_without_replay(self):
        started, finish = threading.Event(), threading.Event()

        class Backend:
            calls = 0

            def observe(self, state, inputs):
                return {
                    "next_operation": {
                        "operation_id": "one",
                        "stage": "baseline",
                        "kind": "runtime",
                        "dynamic": False,
                        "payload": {},
                    }
                }

            def reconcile(self, op, inputs):
                return {
                    "operation_id": "one",
                    "status": "not_submitted",
                    "submission_absent": True,
                }

            def execute(self, op, inputs):
                self.calls += 1
                started.set()
                if not finish.wait(5):
                    raise AssertionError("test completion was not released")
                return {
                    "operation_id": "one",
                    "status": "completed",
                    "receipt_ref": "receipt.json",
                }

        with TemporaryDirectory() as root:
            backend = Backend()
            coordinator = Coordinator(root, {"campaign": "test"}, backend)
            thread = threading.Thread(target=lambda: coordinator.advance({}))
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(coordinator.status()["pending"]["operation_id"], "one")
                paused = coordinator.event(
                    {"event_id": "pause-one", "type": "pause", "operation_id": "one"}
                )
                self.assertTrue(paused["paused"])
                coordinator.event(
                    {
                        "event_id": "ordinary",
                        "type": "receipt_available",
                        "operation_id": "one",
                    }
                )
            finally:
                finish.set()
                thread.join(5)
            state = coordinator.status()
            self.assertEqual(state["status"], "paused")
            self.assertIn("one", state["receipts"])
            self.assertEqual(coordinator.advance({})["status"], "paused")
            self.assertEqual(backend.calls, 1)

    def test_stable_trigger_survives_restart_without_advancing_another_operation(self):
        from test_protocol10_coordinator import Backend, BINDING, inputs, operation

        with TemporaryDirectory() as root:
            backend = Backend(
                [operation("baseline-01"), operation("baseline-02", dynamic=False)]
            )
            coordinator = Coordinator(root, BINDING, backend)
            coordinator.advance(inputs(), trigger_id="autoloop-one")
            calls = list(backend.calls)
            restored = Coordinator(root, BINDING, backend)
            state = restored.advance(inputs(), trigger_id="autoloop-one")
            self.assertEqual(backend.calls, calls)
            self.assertEqual(len(state["receipts"]), 1)
            state = restored.advance(inputs(), trigger_id="autoloop-two")
            self.assertEqual(len(state["receipts"]), 2)
