# Copyright 2026 RSI-Agent-VLA Authors.
"""One explicitly named recovery never clears old or newly unknown usage."""

import os
import unittest
from copy import deepcopy

import test_scope_request_gate

OLD = "framework-batch4-20261117-p1-A-base_salad-s1-runtime"
NEW = "framework-batch4-20261117-p1-A-base_salad-s1-recovery1-runtime"


class OneOffRequestRecoveryTests(unittest.TestCase):
    gate = test_scope_request_gate.ScopeRequestGateTests.gate
    post = test_scope_request_gate.ScopeRequestGateTests.post
    save = test_scope_request_gate.ScopeRequestGateTests.save

    def setUp(self):
        test_scope_request_gate.ScopeRequestGateTests.setUp(self)
        self.original_scope = deepcopy(self.scope)
        self.original_ledger = deepcopy(self.ledger)
        self.original_context = deepcopy(self.context)

    def configure(self, *, authorized=True):
        self.scope = deepcopy(self.original_scope)
        self.ledger = deepcopy(self.original_ledger)
        self.context = deepcopy(self.original_context)
        limits = {
            **self.scope["outer_limits"],
            "runtime_starts": 2,
            "gpu_seconds": 7200,
            "wall_seconds": 86400,
            "storage_bytes": 128 * 1024**3,
            "current_disk_bytes": 128 * 1024**3,
        }
        self.scope["outer_limits"] = limits
        self.ledger["limits"] = deepcopy(limits)
        self.ledger["used"] = dict.fromkeys(limits, 0)
        actual = {
            "runtime_starts": 1,
            "wall_seconds": 129.545832901,
            "gpu_seconds": 129.545832901,
            "storage_bytes": 25239952,
            "model_requests": 1,
            "current_disk_bytes": 16913844,
        }
        self.ledger["used"].update(actual)
        self.scope["one_off_token_unknown_recovery"] = {
            "authorized": authorized,
            "old_token_values_stay_unknown": True,
            "failed_request_count_not_refunded": True,
            "original_operation_id": OLD,
            "recovery_operation_id": NEW,
        }
        missing = ["input_tokens", "output_tokens"]
        self.ledger["reservations"] = {
            OLD: {
                "scope_id": self.scope["scope_id"],
                "operation_id": OLD,
                "owner_uid": os.getuid(),
                "status": "unknown",
                "cleanup_confirmed": True,
                "missing": missing,
                "budget": {
                    "runtime_starts": 1,
                    "wall_seconds": 1200,
                    "gpu_seconds": 1200,
                    "model_requests": 3,
                    "storage_bytes": 1024**3,
                    "current_disk_bytes": 1024**3,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "actual": actual,
            },
            NEW: {
                "scope_id": self.scope["scope_id"],
                "operation_id": NEW,
                "status": "active",
                "budget": {"model_requests": 3, "input_tokens": 0, "output_tokens": 0},
            },
        }
        self.ledger["unresolved_costs"] = [
            {"operation_id": OLD, "missing": missing, "cleanup_confirmed": True}
        ]
        self.context["limits"]["model_requests"] = 3
        self.context["request_scope"]["operation_id"] = NEW
        self.context["run_id"] = (
            "framework-batch4-20261117-p1-A-base_salad-s1-recovery1"
        )
        self.save()

    def test_no_authorization_keeps_the_original_usage_pause(self):
        self.configure(authorized=False)
        with self.gate() as gate:
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])

    def test_only_named_recovery_gets_three_requests_and_old_unknown_is_unchanged(self):
        self.configure()
        original = self.accounting_ref.read_bytes()
        with self.gate() as gate:
            for _ in range(3):
                self.assertEqual(self.post(gate).status_code, 200)
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent + self.ledger["used"]["model_requests"], 4)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.accounting_ref.read_bytes(), original)
        self.assertEqual(self.ledger["reservations"][OLD]["status"], "unknown")
        self.assertNotIn("input_tokens", self.ledger["reservations"][OLD]["actual"])
        self.assertNotIn("output_tokens", self.ledger["reservations"][OLD]["actual"])

    def test_another_operation_cannot_consume_the_named_exception(self):
        self.configure()
        self.ledger["reservations"]["another-recovery"] = deepcopy(
            self.ledger["reservations"][NEW]
        )
        self.ledger["reservations"]["another-recovery"]["operation_id"] = (
            "another-recovery"
        )
        self.context["request_scope"]["operation_id"] = "another-recovery"
        self.save()
        with self.gate() as gate:
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])

    def test_unknown_hard_costs_unclean_old_run_and_other_unknowns_still_pause(self):
        for case in ("model_requests", "gpu_seconds", "unclean", "other_unknown"):
            with self.subTest(case=case):
                self.configure()
                self.context["output_dir"] = str(self.root / case)
                old = self.ledger["reservations"][OLD]
                row = self.ledger["unresolved_costs"][0]
                if case in {"model_requests", "gpu_seconds"}:
                    old["actual"].pop(case)
                    old["missing"].append(case)
                elif case == "unclean":
                    old["cleanup_confirmed"] = False
                    row["cleanup_confirmed"] = False
                else:
                    self.ledger["unresolved_costs"].append(
                        {
                            "operation_id": "another-old",
                            "missing": ["output_tokens"],
                            "cleanup_confirmed": True,
                        }
                    )
                self.save()
                with self.gate() as gate:
                    self.assertEqual(self.post(gate).status_code, 429)
                    self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])

    def test_new_attempt_missing_usage_cannot_reuse_the_old_exception(self):
        self.configure()
        with self.gate() as gate:
            self.assertEqual(self.post(gate, "missing").status_code, 200)
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 1)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
