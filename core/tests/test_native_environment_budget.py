# Copyright 2026 RSI-Agent-VLA Authors.
"""Independent import retains native-owner total-step budget failure evidence."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rsi_vla.adapters.rpent import closed_native_failure


class NativeEnvironmentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {
            "run_id": "run",
            "episode_id": "episode",
            "bundle_id": "bundle",
            "protocol_id": "protocol",
        }
        self.marker = {
            "schema_version": "1.0",
            "identity": dict(self.identity),
            "reason": "environment_steps",
            "limit": 400,
            "observed": 400,
            "stop_after_inflight": True,
            "producer_events_ref": str(
                self.root / "environment_service/action_events.jsonl"
            ),
        }
        self.marker_ref = self.root / "budget_stop.json"
        self.marker_ref.write_text(json.dumps(self.marker))
        self.result = {
            **self.identity,
            "attempt_closed": True,
            "error": None,
            "validity": "valid",
            "execution_health": {"status": "clear"},
            "verified_budget_exhaustion": True,
            "episode_execution": {
                "arm": "D",
                "status": "budget_exhausted",
                "budget_stop_ref": str(self.marker_ref),
            },
            "environment_service": {
                "identity": self.identity,
                "service_closed": True,
                "pending_action": False,
                "environment_steps_unknown": 0,
                "environment_steps_reserved": 400,
                "environment_steps_confirmed": 400,
                "last_native_success": False,
                "last_feedback_sequence": 1354,
            },
            "runtime": {
                "action_events_ref": str(self.root / "action_events.jsonl"),
                "environment_service": {
                    "action_events_ref": str(
                        self.root / "environment_service/action_events.jsonl"
                    )
                },
            },
            "attempt_outcome_source": {
                "native_success": "native _check_success()",
                "budget_completion": "normal cleanup after bounded execution",
                "last_feedback_sequence": 1354,
            },
        }

    def test_bound_environment_physical_budget_is_closed_native_failure(self):
        self.assertTrue(closed_native_failure(self.result))
        self.assertIsNone(
            self.result["attempt_outcome_source"].get("planner_budget_stop")
        )

    def test_foreign_producer_incomplete_native_feedback_and_fault_remain_unconfirmed(
        self,
    ):
        for case in [
            "foreign",
            "count",
            "pending",
            "unknown",
            "native",
            "closed",
            "feedback",
            "fault",
        ]:
            with self.subTest(case=case):
                result = deepcopy(self.result)
                marker = deepcopy(self.marker)
                service = result["environment_service"]
                if case == "foreign":
                    marker["producer_events_ref"] = str(self.root / "foreign.jsonl")
                if case == "count":
                    service["environment_steps_confirmed"] = 395
                if case == "pending":
                    service["pending_action"] = True
                if case == "unknown":
                    service["environment_steps_unknown"] = 1
                if case == "native":
                    service["last_native_success"] = None
                if case == "closed":
                    service["service_closed"] = False
                if case == "feedback":
                    service["last_feedback_sequence"] = 1353
                if case == "fault":
                    result["execution_health"]["status"] = "unresolved"
                self.marker_ref.write_text(json.dumps(marker))
                self.assertFalse(closed_native_failure(result))


if __name__ == "__main__":
    unittest.main()
