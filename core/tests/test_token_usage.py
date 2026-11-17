# Copyright 2026 RSI-Agent-VLA Authors.
"""Actual token accounting preserves reasoning subsets and incomplete receipts."""

import copy
import json
import unittest

import test_batch4
from rsi_vla.adapters.runtime_closure import request_usage


class TokenUsageTests(unittest.TestCase):
    def events(self):
        return [
            {"kind": "request_admitted", "request_number": 1},
            {
                "kind": "response_terminal",
                "request_number": 1,
                "usage": {
                    "input_tokens": 22592,
                    "input_tokens_details": {"cached_tokens": 20352},
                    "output_tokens": 173,
                    "output_tokens_details": {"reasoning_tokens": 23},
                    "total_tokens": 22765,
                },
            },
        ]

    def test_reasoning_is_counted_once_as_output_subset(self):
        usage, missing = request_usage(
            self.events(), expected_requests=1, require_reasoning=True
        )
        self.assertEqual(missing, [])
        self.assertEqual(usage["reasoning_tokens"], 23)
        self.assertEqual(usage["output_tokens"], 173)
        self.assertEqual(usage["total_tokens"], 22765)
        self.assertEqual(
            usage["total_tokens"], usage["input_tokens"] + usage["output_tokens"]
        )

    def test_missing_reasoning_is_unknown_without_changing_historical_contract(self):
        events = self.events()
        del events[1]["usage"]["output_tokens_details"]
        self.assertEqual(request_usage(events, expected_requests=1)[1], [])
        usage, missing = request_usage(
            events, expected_requests=1, require_reasoning=True
        )
        self.assertIsNone(usage["reasoning_tokens"])
        self.assertIn("reasoning_tokens", missing)
        self.assertEqual(usage["output_tokens"], 173)

    def test_missing_duplicate_and_invalid_receipts_never_establish_complete_usage(
        self,
    ):
        mutations = [
            lambda e: e.append({"kind": "request_admitted", "request_number": 2}),
            lambda e: e.append(copy.deepcopy(e[1])),
            lambda e: e[1]["usage"].update(total_tokens=1),
            lambda e: e[1]["usage"]["output_tokens_details"].update(
                reasoning_tokens=174
            ),
            lambda e: e[1]["usage"].update(output_tokens=-1),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                events = self.events()
                mutate(events)
                _, missing = request_usage(events, require_reasoning=True)
                self.assertTrue(missing)
        events = self.events()
        usage, missing = request_usage(
            events, expected_requests=2, require_reasoning=True
        )
        self.assertIsNone(usage["output_tokens"])
        self.assertEqual(usage["observed_output_tokens"], 173)
        self.assertIn("reasoning_tokens", missing)

    def test_closed_zero_request_usage_is_explicit_zero(self):
        usage, missing = request_usage([], expected_requests=0, require_reasoning=True)
        self.assertEqual(missing, [])
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            self.assertEqual(usage[name], 0)
        usage, missing = request_usage([], expected_requests=1, require_reasoning=True)
        self.assertIsNone(usage["reasoning_tokens"])
        self.assertIn("reasoning_tokens", missing)

    def test_actual_bounded_runner_reports_missing_reasoning_without_recharging(self):
        fixture = test_batch4.RunnerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.test_one_dispatch_reconcile_no_repeat_charge_and_no_applicable_no_dispatch()
        measurement = json.loads(
            (fixture.root / "dispatch/execution-costs.json").read_text()
        )
        self.assertTrue(measurement["usage_complete"])
        self.assertFalse(measurement["usage_complete_including_reasoning"])
        self.assertIn(
            "reasoning_tokens", measurement["missing_usage_including_reasoning"]
        )
        self.assertNotIn("reasoning_tokens", measurement["costs"])


if __name__ == "__main__":
    unittest.main()
