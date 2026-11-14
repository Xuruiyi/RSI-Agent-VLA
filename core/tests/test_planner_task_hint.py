"""Finite Research-generated task text through real rule and preflight consumers."""

import unittest
from dataclasses import replace
from pathlib import Path

import test_research_preflight
import test_research_tools
import test_task_policy
from rsi_vla.research_preflight import operator_preflight
from rsi_vla.research_tools import BoundedResearchAdapter, Protocol10ResearchAdapter
from rsi_vla.research_transport import unpack
from rsi_vla.rule_updates import (
    RULE_SCHEMAS,
    TASK_HINT_RULE_SCHEMA,
    load_runtime_rules,
    rule_update,
    validate_runtime_rules,
)
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskResearchAdapter

ROOT = Path(test_research_tools.__file__).resolve().parents[2]


class TaskHintRulesTests(unittest.TestCase):
    def test_v1_through_v4_upgrade_preserves_every_existing_rule(self):
        before = {
            "schema": RULE_SCHEMAS[0],
            "chunk_steps": 5,
            "baseline_chunk_steps": 5,
        }
        for rank in range(4):
            with self.subTest(schema=before["schema"]):
                after, changes = rule_update(
                    before,
                    planner_task_hint="Identify the named target before choosing a grasp.",
                )
                self.assertEqual(after["schema"], TASK_HINT_RULE_SCHEMA)
                for key, value in before.items():
                    if key != "schema":
                        self.assertEqual(after[key], value)
                self.assertEqual(set(changes), {"planner_task_hint"})
            if rank < 3:
                name = (
                    "finish_requires_native_success",
                    "planner_finish_reserve_calls",
                    "planner_segment_handoff_hint",
                )[rank]
                before, _ = rule_update(
                    before,
                    **{name: 1 if name == "planner_finish_reserve_calls" else True},
                )

    def test_invalid_text_and_noop_reject_without_redefining_other_operators(self):
        base = {"schema": RULE_SCHEMAS[0], "chunk_steps": 5, "baseline_chunk_steps": 5}
        for text in ("", " \n\t", "x" * 1201, True, 7, {}):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                rule_update(base, planner_task_hint=text)
        current, _ = rule_update(base, planner_task_hint="x" * 1200)
        self.assertEqual(len(current["planner_task_hint"]), 1200)
        with self.assertRaisesRegex(ValueError, "actual behavior change"):
            rule_update(current, planner_task_hint="x" * 1200)
        for name in (
            "finish_requires_native_success",
            "planner_finish_reserve_calls",
            "planner_segment_handoff_hint",
        ):
            changed, _ = rule_update(
                current, **{name: 1 if name == "planner_finish_reserve_calls" else True}
            )
            self.assertEqual(changed["planner_task_hint"], "x" * 1200)
            self.assertEqual(changed["schema"], TASK_HINT_RULE_SCHEMA)
            validate_runtime_rules(changed)


class TaskHintResearchTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    configure = test_task_policy.ExplorationTests.configure

    def prepare(self, request_id, adapter=None):
        self.configure()
        self.tools.domain_policy["rule_operators"] = ["planner_task_hint"]
        self.protocol["task_policy"]["exploration"]["operators"] = ["planner_task_hint"]
        adapter = adapter or TaskResearchAdapter(
            {
                "reserved_bundle_id": request_id + "-trial",
                "development_slots": [{"task_key": "salad", "seed": 2}],
            }
        )
        transport = adapter.prepare_request(
            self.tools,
            request_id=request_id,
            result_ids=[],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        return adapter, transport

    def test_two_arbitrary_research_texts_reach_distinct_v5_artifacts(self):
        hints = [
            "Locate the salad dressing by its label and shape; compare it with the soup can before the first grasp.",
            "先核对当前目标物体及容器位置，再选择抓取点；若观察有歧义，补充观察后再行动。",
        ]
        for index, hint in enumerate(hints):
            request_id = "text-" + str(index)
            adapter, transport = self.prepare(request_id)
            request = unpack(transport)
            self.assertIn(
                "Planner startup hint",
                request["operator_semantics"]["planner_task_hint"],
            )
            self.assertEqual(request["allowed_operators"], ["planner_task_hint"])
            schema = adapter.response_schema(request_id, tools=self.tools)
            self.assertEqual(
                schema["properties"]["source"],
                {"type": "string", "const": "exploration"},
            )
            alternatives = schema["properties"]["intervention"]["anyOf"]
            self.assertEqual(len(alternatives), 2)
            hint_spec = next(
                row
                for row in alternatives
                if "planner_task_hint" in row.get("properties", {})
            )
            self.assertEqual(
                hint_spec["properties"]["planner_task_hint"]["maxLength"], 1200
            )
            self.assertNotIn("const", hint_spec["properties"]["planner_task_hint"])
            response = {
                "schema": "rsi-bounded-research-response-v2",
                "request_id": request_id,
                "decision": "prepare_rule_trial",
                "source": "exploration",
                "result_ids": [],
                "reason": "test the registered object selection hypothesis",
                "external_value": "correct target selection",
                "competing_explanations": self.explanations[:1],
                "intervention": {"planner_task_hint": hint},
                "distinguishing_prediction": "target object is selected more reliably",
                "stop_condition": "one development slot",
                "missing_evidence": ["independent trial result"],
                "development_slots": [{"task_key": "salad", "seed": 2}],
            }
            trial = adapter.apply_response(
                self.tools,
                response,
                evidence_level="offline",
                cost_limit={"offline_calls": 1},
                synthetic=True,
            )
            rules = load_runtime_rules(trial["runtime_rules_ref"])
            self.assertEqual(rules["planner_task_hint"], hint)
            self.assertEqual(rules["schema"], TASK_HINT_RULE_SCHEMA)
            self.assertFalse(rules["planner_segment_handoff_hint"])
            self.assertFalse(trial["requires_new_timing_baseline"])
            self.assertEqual(rules["chunk_steps"], 5)
            self.assertEqual(self.job.state["charges"], [])
        self.assertEqual(len(self.store.all("rule_trials")), 2)
        self.assertEqual(self.store.all("candidates"), [])

    def test_old_protocol_cannot_authorize_new_hint_or_reanimate_handoff(self):
        self.assertNotIn("planner_task_hint", Protocol10ResearchAdapter.operators)
        schema = Protocol10ResearchAdapter().response_schema("old")
        self.assertFalse(any("planner_task_hint" in item.get("properties", {})
                             for item in schema["properties"]["intervention"]["anyOf"]))
        self.tools.domain_policy["rule_operators"] = ["planner_task_hint"]
        with self.assertRaisesRegex(ValueError, "new-task policy"):
            BoundedResearchAdapter().prepare_request(
                self.tools,
                request_id="old",
                result_ids=["base-result"],
                max_input_tokens=100,
                max_output_tokens=10,
            )


class TaskHintPreflightTests(unittest.TestCase):
    setUp = test_research_preflight.PreflightTests.setUp

    def test_hint_reaches_actual_runtime_preparation_with_no_dynamic_dispatch(self):
        self.protocol.update(protocol_id="new-hint-protocol", scope_id="new-hint-scope")
        self.protocol["task_policy"] = {
            "schema": "rsi-task-policy-v1",
            "exploration": {
                "hypothesis": "a startup hint improves target selection",
                "criterion_ref": self.protocol["protocol_id"],
                "stop_condition": "one development slot",
                "operators": ["planner_task_hint"],
                "assignments": [{"task_key": "position_swap_salad", "seed": 2}],
                "cost_limit": {"offline_calls": 1},
            },
        }
        self.tools.base = replace(self.base, protocol_id=self.protocol["protocol_id"])
        self.tools.domain_policy.update(
            research_mode="exploratory", rule_operators=["planner_task_hint"]
        )
        atomic_json(self.protocol_ref, self.protocol)
        rows = operator_preflight(
            self.tools, ROOT, protocol_path=self.protocol_ref, result_ids=[]
        )
        hint = next(row for row in rows if row["operator"] == "planner_task_hint")
        self.assertTrue(hint["currently_trialable"])
        self.assertEqual(hint["artifact"]["schema"], TASK_HINT_RULE_SCHEMA)
        self.assertFalse(hint["execution_authorized"])
        self.assertEqual(hint["evidence_basis"], "registered_hypothesis")
        self.assertFalse(
            next(
                row for row in rows if row["operator"] == "planner_segment_handoff_hint"
            )["currently_trialable"]
        )


if __name__ == "__main__":
    unittest.main()
