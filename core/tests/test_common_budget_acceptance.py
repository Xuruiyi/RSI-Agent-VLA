# Copyright 2026 RSI-Agent-VLA Authors.
"""Independent model-gate stop drains a real common execution contract."""

import json
import tempfile
import unittest
from pathlib import Path

from rsi_vla.adapters.runtime_acceptance import audit_sdk_tools
from rsi_vla.storage import atomic_json


class CommonBudgetAcceptanceTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        identity = {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
        }
        spec = {
            "schema": "libero-common-execution-v1",
            "observation_cadence": "prediction",
            "allowed_motion_tools": ["vla_execute"],
            "action_prefix_steps": 5,
        }
        marker = {
            "identity": identity,
            "reason": "model_requests",
            "limit": 3,
            "observed": 3,
            "stop_after_inflight": True,
            "sequence": 6,
            "monotonic_ns": 400,
            "producer_events_ref": str(root / "model_requests.jsonl"),
        }
        segment = {
            "segment_sequence": 2,
            "stop_reason": "chunk_allowance",
            "predictions_started": 1,
            "predictions_completed": 1,
            "executed_steps": 5,
            "unknown_steps": 0,
        }
        rows = [
            {"kind": "tool_call", "tool": "vla_execute"},
            {
                "kind": "common_execution_call_start",
                "common_execution": spec,
                "max_chunks": 24,
            },
            {"kind": "vla_segment_start", "max_chunks": 1},
            {"kind": "chunk_execution_end", **segment},
            {"kind": "vla_segment_end", **segment},
            {
                "kind": "tool_rejected",
                "tool": "vla_execute",
                "resource": "model_requests",
            },
            {
                "kind": "common_execution_call_end",
                **segment,
                "max_chunks": 24,
                "logical_call_sequence": 1,
                "common_execution_call_sequence": 1,
                "stop_reason": "budget",
                "runtime_budget": {"budget_stop": marker},
                "common_execution_segments": [segment],
            },
            {"kind": "tool_end", "tool": "vla_execute", "status": "interrupted"},
            {"kind": "planner_budget_stop", "budget_stop": marker},
            {"kind": "planner_usage", "stats": {"sdk_errors": []}},
        ]
        events = [
            {
                **identity,
                **row,
                "sequence": index,
                "monotonic_ns": (index + 1) * 100,
                "utc_ns": (1001 + index) * 1_000_000,
            }
            for index, row in enumerate(rows)
        ]
        atomic_json(
            root / "runtime_context.json",
            {**identity, "common_execution": spec, "limits": {"model_requests": 3}},
        )
        atomic_json(root / "budget_stop.json", marker)
        atomic_json(
            root / "request_gate.json",
            {
                "identity": identity,
                "requests_forwarded": 3,
                "active_requests_at_close": 0,
                "cleanup_complete": True,
            },
        )
        source = {
            **marker,
            "budget_stop_ref": str(root / "budget_stop.json"),
            "planner_events_ref": str(root / "action_events.jsonl"),
            "planner_event_sequence": 8,
        }
        atomic_json(
            root / "result.json",
            {
                **identity,
                "attempt_closed": True,
                "validity": "valid",
                "error": None,
                "verified_budget_exhaustion": True,
                "environment_service": {
                    "identity": identity,
                    "service_closed": True,
                    "pending_action": False,
                    "environment_steps_unknown": 0,
                    "last_native_success": False,
                    "last_feedback_sequence": 10,
                },
                "attempt_outcome_source": {
                    "native_success": "synthetic-native",
                    "budget_completion": "drained",
                    "last_feedback_sequence": 10,
                    "planner_budget_stop": source,
                },
            },
        )
        model = []
        for index in range(3):
            model.extend(
                [
                    {
                        "identity": identity,
                        "sequence": 2 * index,
                        "kind": "request_admitted",
                        "request_number": index + 1,
                    },
                    {
                        "identity": identity,
                        "sequence": 2 * index + 1,
                        "kind": "response_terminal",
                        "request_number": index + 1,
                    },
                ]
            )
        model.append(
            {
                "identity": identity,
                "sequence": 6,
                "kind": "budget_stop",
                "reason": "model_requests",
                "limit": 3,
                "observed": 3,
            }
        )
        (root / "model_requests.jsonl").write_text("\n".join(map(json.dumps, model)))
        sdk = [
            {
                "method": "item/started",
                "payload": {
                    "item": {
                        "id": "call",
                        "type": "mcpToolCall",
                        "server": "rpent",
                        "tool": "vla_execute",
                        "status": "inProgress",
                        "started_at_ms": 1000,
                    }
                },
            },
            {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "id": "call",
                        "type": "mcpToolCall",
                        "server": "rpent",
                        "tool": "vla_execute",
                        "status": "completed",
                        "started_at_ms": 1000,
                        "completed_at_ms": 1010,
                    }
                },
            },
        ]
        (root / "codex_fixture.stream.jsonl").write_text(
            "\n".join(map(json.dumps, sdk))
        )
        (root / "action_events.jsonl").write_text("\n".join(map(json.dumps, events)))
        return root, identity, events

    def test_model_stream_marker_and_later_planner_notice_close_interrupted_tool(self):
        root, identity, events = self.fixture()
        # Model seq6 is a budget marker; L0 seq6 is the common-call closure.
        # Their numbers refer to separate producer streams.
        report = audit_sdk_tools(root, identity, events)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["l0_tool_calls"], 1)
        self.assertEqual(report["tools"][0]["l0_status"], "interrupted")

    def test_marker_identity_source_or_sequence_mismatch_is_not_a_closed_tool(self):
        for case in ("identity", "producer", "sequence", "notice"):
            with self.subTest(case=case):
                root, identity, events = self.fixture()
                marker = json.loads((root / "budget_stop.json").read_text())
                if case == "identity":
                    marker["identity"]["bundle_id"] = "other"
                elif case == "producer":
                    marker["producer_events_ref"] = str(root / "action_events.jsonl")
                elif case == "sequence":
                    marker["sequence"] = 2
                else:
                    events[8]["budget_stop"] = {**marker, "observed": 2}
                atomic_json(root / "budget_stop.json", marker)
                self.assertTrue(audit_sdk_tools(root, identity, events)["issues"])

    def test_unclosed_unbound_or_unexecuted_common_call_does_not_pass(self):
        for case in (
            "unclosed",
            "logical",
            "no_prediction",
            "no_chunk",
            "unknown_feedback",
        ):
            with self.subTest(case=case):
                root, identity, events = self.fixture()
                if case == "unclosed":
                    events[6]["kind"] = "missing_common_closure"
                elif case == "logical":
                    events[6]["logical_call_sequence"] = 99
                elif case == "no_prediction":
                    events[6]["predictions_completed"] = 0
                    events[6]["executed_steps"] = 0
                elif case == "no_chunk":
                    events[3]["kind"] = "missing_chunk_evidence"
                else:
                    events[3]["unknown_steps"] = 1
                self.assertTrue(audit_sdk_tools(root, identity, events)["issues"])

    def test_sdk_or_execution_fault_is_not_cleared_by_the_budget_marker(self):
        for case in ("sdk_event", "sdk_stats", "sdk_tool", "inference"):
            with self.subTest(case=case):
                root, identity, events = self.fixture()
                path = root / "codex_fixture.stream.jsonl"
                sdk = [json.loads(line) for line in path.read_text().splitlines()]
                if case == "sdk_event":
                    sdk.append(
                        {
                            "method": "error",
                            "payload": {"error": {"message": "offline SDK fault"}},
                        }
                    )
                elif case == "sdk_stats":
                    events[9]["stats"]["sdk_errors"] = [{"error": "offline SDK fault"}]
                elif case == "sdk_tool":
                    sdk[-1]["payload"]["item"]["status"] = "failed"
                else:
                    events[3].update(kind="inference_end", completed=False)
                path.write_text("\n".join(map(json.dumps, sdk)))
                self.assertTrue(audit_sdk_tools(root, identity, events)["issues"])


if __name__ == "__main__":
    unittest.main()
