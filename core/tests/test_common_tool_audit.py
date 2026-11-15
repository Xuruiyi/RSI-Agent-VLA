"""Explicit common native calls retain cost without inventing SDK calls."""

import json
import unittest

from test_runtime_acceptance import ToolAuditTests

from rsi_vla.adapters.runtime_acceptance import audit_sdk_tools


class CommonToolAuditTests(unittest.TestCase):
    setUp = ToolAuditTests.setUp
    stream = ToolAuditTests.stream
    sdk = ToolAuditTests.sdk

    def fixture(self):
        common = dict(schema="libero-common-execution-v1", observation_cadence="prediction",
                      allowed_motion_tools=["vla_execute"], action_prefix_steps=5)
        (self.root / "runtime_context.json").write_text(json.dumps(
            dict(self.identity, common_execution=common)))
        self.stream([self.sdk(), self.sdk(method="item/completed", status="completed", completed_at_ms=1020)])
        rows = [dict(kind="tool_call", tool="vla_execute"),
                dict(kind="common_execution_call_start", common_execution=common, max_chunks=3),
                dict(kind="tool_call", tool="vla_execute"),
                dict(kind="common_internal_tool_admission", logical_call_sequence=1, tool_call_sequence=2),
                dict(kind="tool_end", tool="vla_execute", status="completed", logical_call_sequence=1, tool_call_sequence=2),
                dict(kind="common_execution_call_end", logical_call_sequence=1),
                dict(kind="tool_end", tool="vla_execute", status="completed")]
        return [dict(self.identity, **row, sequence=i, utc_ns=(1001+i)*1_000_000)
                for i, row in enumerate(rows)]

    def test_common_nested_prediction_has_one_sdk_call_but_two_counted_calls(self):
        result = audit_sdk_tools(self.root, self.identity, self.fixture())
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["counted_tool_calls"], 2)
        self.assertEqual(result["minimum_tool_calls"], 2)
        self.assertEqual(result["l0_tool_calls"], 2)
        self.assertEqual(result["sdk_tool_calls"], 1)
        self.assertEqual(result["common_internal_tool_calls"], 1)

    def test_registration_is_required_and_fake_internal_labels_do_not_hide_calls(self):
        events = self.fixture()
        (self.root / "runtime_context.json").unlink()
        self.assertTrue(audit_sdk_tools(self.root, self.identity, events)["issues"])
        self.assertGreaterEqual(audit_sdk_tools(self.root, self.identity, events)["counted_tool_calls"], 2)

    def test_unbound_or_failed_native_subcalls_remain_unresolved(self):
        for row, updates in [(3, {"logical_call_sequence": 0}), (4, {"status": "failed"}),
                             (4, {"bundle_id": "other"}), (1, {"max_chunks": 1})]:
            with self.subTest(row=row, updates=updates):
                events = self.fixture()
                events[row].update(updates)
                result = audit_sdk_tools(self.root, self.identity, events)
                self.assertTrue(result["issues"])
                self.assertGreaterEqual(result["counted_tool_calls"], 2)

    def test_missing_parent_terminal_and_unregistered_extra_call_cannot_pass(self):
        events = self.fixture()
        self.assertTrue(audit_sdk_tools(self.root, self.identity, events[:-1])["issues"])
        hidden = [dict(self.identity, kind="tool_call", tool="vla_execute", sequence=7, utc_ns=1008*1_000_000)]
        self.assertTrue(audit_sdk_tools(self.root, self.identity, events+hidden)["issues"])


if __name__ == "__main__":
    unittest.main()
