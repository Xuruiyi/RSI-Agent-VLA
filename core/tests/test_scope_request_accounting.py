# Copyright 2026 RSI-Agent-VLA Authors.
# ruff: noqa: UP017 - Python 3.10 support
"""Original Host ledger settles observed tokens without inventing hard caps."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from rsi_vla.midterm_host import MidtermHost, MidtermLedger
from rsi_vla.storage import atomic_json


class ScopeRequestAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope_ref = self.root / "scope.json"
        self.phase_ref = self.root / "phase.json"
        self.limits = {
            "model_requests": 4,
            "research_requests": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "training_starts": 1,
            "concurrent_gpu_devices": 1,
        }
        self.scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": "offline-accounting",
            "authorized": True,
            "dynamic_authorized": True,
            "status": "active",
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "outer_limits": self.limits,
            "token_policy": {
                "mode": "observed_cumulative_stop_threshold",
                "input_tokens_threshold": 10,
                "output_tokens_including_reasoning_threshold": 5,
            },
        }
        atomic_json(self.scope_ref, self.scope)
        atomic_json(
            self.phase_ref,
            {
                "limits": self.limits,
                "used": dict.fromkeys(self.limits, 0),
                "remaining": self.limits,
                "unresolved_costs": [],
            },
        )
        self.ledger = MidtermLedger(self.scope_ref)
        self.budget = {"model_requests": 1, "input_tokens": 0, "output_tokens": 0}

    def reserve(self, identity, budget=None):
        return self.ledger.reserve(
            identity,
            budget=self.budget if budget is None else budget,
            phase_ref=self.phase_ref,
            gpu_devices=[],
            receipt_ref=self.root / (identity + ".json"),
        )

    def test_observed_threshold_can_be_crossed_once_and_only_next_model_is_refused(
        self,
    ):
        self.reserve("runtime")
        actual = {"model_requests": 1, "input_tokens": 12, "output_tokens": 8}
        self.ledger.settle("runtime", actual=actual, cleanup_confirmed=True)
        before = self.ledger.path.read_bytes()
        self.ledger.settle("runtime", actual=actual, cleanup_confirmed=True)
        self.assertEqual(self.ledger.path.read_bytes(), before)
        self.assertEqual(self.ledger.read()["used"]["output_tokens"], 8)
        with self.assertRaisesRegex(PermissionError, "stop threshold"):
            self.reserve("next-model")
        no_model = {
            "model_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "training_starts": 1,
        }
        self.assertEqual(self.reserve("native-training", no_model)["status"], "active")

    def test_observed_mode_refuses_token_reservation_and_old_scope_keeps_hard_limits(
        self,
    ):
        with self.assertRaises(PermissionError):
            self.reserve("estimated", {"model_requests": 1, "input_tokens": 1})
        self.scope.pop("token_policy")
        atomic_json(self.scope_ref, self.scope)
        with self.assertRaisesRegex(PermissionError, "outer task budget"):
            self.reserve("over-hard", {"input_tokens": 11})
        self.assertEqual(
            self.reserve("within-hard", {"input_tokens": 9})["status"], "active"
        )

    def host_fixture(self, usage, *, foreign=False):
        identity = "runtime"
        self.reserve(identity)
        output = self.root / "runtime-output"
        output.mkdir()
        native_identity = {
            "run_id": "native-run",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
        }
        binding = {
            "scope_authorization_ref": str(self.scope_ref),
            "accounting_ref": str(self.ledger.path),
            "operation_id": "foreign-operation" if foreign else identity,
        }
        atomic_json(
            output / "runtime_context.json",
            {
                **native_identity,
                "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
                "request_scope": binding,
            },
        )
        atomic_json(
            output / "request_gate.json",
            {
                "identity": native_identity,
                "requests_forwarded": 1,
                "cleanup_complete": True,
                "active_requests_at_close": 0,
            },
        )
        events = [
            {"kind": "request_admitted", "request_number": 1},
            {
                "kind": "response_terminal",
                "request_number": 1,
                "response_id": "offline-response",
                "backend_model": "gpt-6.1-sol",
                "usage": usage,
            },
        ]
        (output / "model_requests.jsonl").write_text("\n".join(map(json.dumps, events)))
        host = SimpleNamespace(
            ledger=self.ledger,
            root=self.root / "host",
            core_config={"operations": [{"operation_id": identity, "kind": "runtime"}]},
            config={
                "scope_id": self.scope["scope_id"],
                "operations": {
                    identity: {"budget": self.budget, "output_ref": str(output)}
                },
            },
            charge_phase=lambda *_: None,
        )
        report = {
            "identity": {"run_id": identity},
            "status": "completed",
            "exit_code": 0,
            "cleanup_confirmed": True,
        }
        return host, report

    def test_actual_host_finish_bills_complete_usage_once_and_reasoning_is_subset(self):
        usage = {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 20,
        }
        host, report = self.host_fixture(usage)
        MidtermHost.finish(host, "runtime", report)
        before = self.ledger.path.read_bytes()
        MidtermHost.finish(host, "runtime", report)
        self.assertEqual(self.ledger.path.read_bytes(), before)
        record = self.ledger.read()["reservations"]["runtime"]
        self.assertEqual(record["status"], "settled")
        self.assertEqual(
            record["actual"],
            {"model_requests": 1, "input_tokens": 12, "output_tokens": 8},
        )

    def test_actual_host_missing_usage_stays_unknown_and_never_fills_zero(self):
        host, report = self.host_fixture(
            {"input_tokens": 4, "input_tokens_details": {"cached_tokens": 0}}
        )
        MidtermHost.finish(host, "runtime", report)
        record = self.ledger.read()["reservations"]["runtime"]
        self.assertEqual(record["status"], "unknown")
        self.assertEqual(record["actual"], {"model_requests": 1, "input_tokens": 4})
        self.assertIn("output_tokens", record["missing"])
        with self.assertRaisesRegex(PermissionError, "unresolved"):
            self.reserve("next-model")
        self.ledger.settle(
            "runtime",
            actual={"model_requests": 1, "input_tokens": 4, "output_tokens": 3},
            cleanup_confirmed=True,
        )
        self.assertEqual(self.ledger.read()["used"]["input_tokens"], 4)
        self.assertEqual(self.ledger.read()["used"]["output_tokens"], 3)

    def test_native_response_without_response_id_remains_unknown(self):
        usage = {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 7,
        }
        host, report = self.host_fixture(usage)
        output = Path(host.config["operations"]["runtime"]["output_ref"])
        path = output / "model_requests.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[1].pop("response_id")
        path.write_text("\n".join(map(json.dumps, events)))
        MidtermHost.finish(host, "runtime", report)
        record = self.ledger.read()["reservations"]["runtime"]
        self.assertEqual(record["status"], "unknown")
        self.assertIn("input_tokens", record["missing"])
        self.assertIn("output_tokens", record["missing"])

    def test_native_duplicate_response_id_remains_unknown(self):
        self.budget["model_requests"] = 2
        usage = {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 7,
        }
        host, report = self.host_fixture(usage)
        output = Path(host.config["operations"]["runtime"]["output_ref"])
        gate = json.loads((output / "request_gate.json").read_text())
        gate["requests_forwarded"] = 2
        atomic_json(output / "request_gate.json", gate)
        path = output / "model_requests.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events.extend(
            [{**events[0], "request_number": 2}, {**events[1], "request_number": 2}]
        )
        path.write_text("\n".join(map(json.dumps, events)))
        MidtermHost.finish(host, "runtime", report)
        record = self.ledger.read()["reservations"]["runtime"]
        self.assertEqual(record["status"], "unknown")
        self.assertEqual(record["actual"]["model_requests"], 2)
        self.assertIn("input_tokens", record["missing"])
        self.assertIn("output_tokens", record["missing"])

    def test_foreign_scope_operation_never_changes_authoritative_costs(self):
        host, report = self.host_fixture({}, foreign=True)
        before = self.ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "request scope"):
            MidtermHost.finish(host, "runtime", report)
        self.assertEqual(self.ledger.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
