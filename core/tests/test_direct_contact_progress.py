"""Direct contact loops report native termination, without pick or holding claims."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import test_direct_pick_progress as pick_fixture
from rsi_vla.critical_progress import inspect_progress
from rsi_vla.primitive_progress import audit_direct_contacts, audit_primitives


class DirectContactProgressTests(unittest.TestCase):
    result_from_feedback = pick_fixture.DirectPickProgressTests.result_from_feedback
    sdk_call = pick_fixture.DirectPickProgressTests.sdk_call
    write = pick_fixture.DirectPickProgressTests.write

    def setUp(self):
        pick_fixture.DirectPickProgressTests.setUp(self)
        self.args = dict(
            prompt="contact the observed drawer handle and pull outward", max_chunks=1
        )
        self.result = dict(
            name="pi0_doubled",
            instruction=self.args["prompt"],
            success=False,
            task_success=False,
            contact_skill_executed=True,
            chunks_used=1,
            max_chunks=1,
            terminated=False,
            truncated=False,
            diagnostics=dict(
                mode="contact_skill_success_by_termination",
                success_meaning="`success` mirrors official LIBERO task termination only; for intermediate contact skills, inspect image/state evidence.",
            ),
        )
        for event in self.events:
            if event.get("tool") == "pi0_pick":
                event["tool"] = "pi0_doubled"
            if event["kind"] in {"inference_start", "action_proposal"}:
                event["instruction"] = self.args["prompt"]
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] == "pi0_pick":
                item["tool"] = "pi0_doubled"
        self.sync()

    def sync(self):
        row = self.states["steps"][1]
        row.update(command={"action": "pi0_doubled", **self.args}, result=self.result)
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] != "pi0_doubled":
                continue
            item["arguments"] = deepcopy(self.args)
            if "result" in item:
                item["result"]["content"][0]["text"] = json.dumps(
                    dict(
                        step=1,
                        state=row["state"],
                        terminated=row["terminated"],
                        truncated=row["truncated"],
                        log=dict(command=row["command"], result=self.result),
                    )
                )

    def inspect(self):
        self.write()
        return audit_direct_contacts(self.root, self.identity, self.events, self.states)

    def test_failed_native_contact_is_a_closed_execution(self):
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIs(self.result["success"], False)
        self.assertIs(self.result["task_success"], False)
        self.assertIs(self.result["contact_skill_executed"], True)
        self.assertNotIn("direct_pick_recorded_heuristic_result", report["checks"])
        self.assertIn(
            "no registered route", report["not_applicable"]["route_and_B_feedback"]
        )
        self.assertIn(
            "not an independently measured contact effect",
            report["not_applicable"]["contact_effect_or_holding"],
        )

    def test_projection_and_critical_bind_contact_without_fabricated_segment(self):
        self.write()
        primitive = audit_primitives(self.root, self.identity, self.events, self.states)
        self.assertIs(primitive["value"], True, primitive)
        self.assertIs(primitive["checks"]["complete_state_trace"], True)
        checks = {
            name: True
            for name in [
                "ordered_l0",
                "execution_health",
                "action_mapping",
                "setup_coverage",
                "no_mid_attempt_reset",
                "vla_actual_loading_and_prediction",
            ]
        }
        with patch(
            "rsi_vla.adapters.runtime_acceptance.inspect_attempt",
            return_value={"checks": checks},
        ):
            report = inspect_progress(self.root, self.identity)
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["actual_direct_contact_audit"], True)
        self.assertIs(report["checks"]["proposal_to_native_send"], True)
        self.assertNotIn("direct_pick_progress", report)
        self.assertFalse(
            any(event["kind"] == "vla_segment_start" for event in self.events)
        )
        self.assertIs(self.result["task_success"], False)

    def test_contact_has_no_pick_heuristic_observation_requirement(self):
        for step in self.states["steps"]:
            step["state"] = {}
        del self.events[2]["feedback"]["observed_state"]
        del self.events[10]["feedback"]["observed_state"]
        self.sync()
        self.assertIs(self.inspect()["value"], True)

    def test_default_twenty_chunk_allowance_requires_native_stop(self):
        self.args.pop("max_chunks")
        self.result["max_chunks"] = 20
        self.events[10]["feedback"]["truncated"] = [False, True, False, False, False]
        self.states["steps"][1]["truncated"] = self.result["truncated"] = True
        self.sync()
        self.assertIs(self.inspect()["value"], True)
        self.events[10]["feedback"]["truncated"] = [False] * 5
        self.states["steps"][1]["truncated"] = self.result["truncated"] = False
        self.sync()
        report = self.inspect()
        self.assertIs(report["value"], False, report)
        self.assertIs(report["checks"]["direct_contact_existing_stop_rule"], False)

    def test_contact_execution_and_native_success_flags_cannot_be_self_confirmed(self):
        original = deepcopy(self.result)
        for key, value in [
            ("contact_skill_executed", False),
            ("success", True),
            ("task_success", True),
        ]:
            with self.subTest(key=key):
                self.result = {**deepcopy(original), key: value}
                self.sync()
                report = self.inspect()
                self.assertIs(report["value"], False, report)
                self.assertIs(report["checks"]["direct_contact_recorded_result"], False)

    def test_missing_native_or_action_closure_remains_unknown(self):
        self.events[10]["feedback"]["terminated"] = None
        self.assertIsNone(self.inspect()["value"])
        self.events[10]["feedback"]["terminated"] = [False] * 5
        self.events[10]["kind"] = "diagnostic"
        self.assertIsNone(self.inspect()["value"])

    def test_pick_threshold_arguments_are_not_contact_arguments(self):
        self.args["lift_thresh"] = 0.02
        self.sync()
        self.assertIsNone(self.inspect()["value"])

    def test_changed_contact_inference_instruction_is_negative(self):
        self.events[6]["instruction"] = "another contact goal"
        report = self.inspect()
        self.assertIs(report["value"], False, report)
        self.assertIs(report["checks"]["direct_contact_count_and_arguments"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
