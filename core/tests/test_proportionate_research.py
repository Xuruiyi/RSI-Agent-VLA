# Copyright 2026 RSI-Agent-VLA Authors.
"""New-task hypotheses stay testable without imposing comparison-only paperwork."""

import importlib.util
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_batch4
import test_research_tools
import test_task_policy
from rsi_vla.research_tools import BoundedResearchAdapter, Protocol10ResearchAdapter
from rsi_vla.research_transport import unpack
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class ProportionateResearchTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    configure = test_task_policy.ExplorationTests.configure

    def prepare(self, adapter=None):
        self.configure()
        adapter = adapter or BoundedResearchAdapter()
        transport = adapter.prepare_request(
            self.tools,
            request_id="one-hypothesis",
            result_ids=[],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        response = {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": "one-hypothesis",
            "decision": "prepare_rule_trial",
            "source": "exploration",
            "result_ids": [],
            "reason": "collect absent evidence",
            "external_value": "reduce wasted calls",
            "competing_explanations": self.explanations[:1],
            "intervention": {"finish_requires_native_success": True},
            "distinguishing_prediction": "closure differs",
            "stop_condition": "one trial",
            "missing_evidence": ["no trigger yet"],
        }
        return adapter, transport, response

    def test_single_testable_hypothesis_reaches_artifact_without_dispatch(self):
        adapter, transport, response = self.prepare()
        self.assertEqual(unpack(transport)["minimum_hypotheses"], 1)
        self.assertEqual(
            adapter.response_schema("one-hypothesis", tools=self.tools)["properties"][
                "competing_explanations"
            ]["minItems"],
            1,
        )
        # Schemas consume the saved request even if the next task policy changes.
        policy = self.protocol.pop("task_policy")
        self.assertEqual(
            adapter.response_schema("one-hypothesis", tools=self.tools)["properties"][
                "competing_explanations"
            ]["minItems"],
            1,
        )
        self.protocol["task_policy"] = policy
        trial = adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )
        self.assertTrue(Path(trial["runtime_rules_ref"]).is_file())
        self.assertEqual(self.job.state["charges"], [])
        self.assertEqual(self.store.all("candidates"), [])

    def test_formal_protocol10_keeps_two_explanations_with_new_task_tools(self):
        adapter, transport, response = self.prepare(Protocol10ResearchAdapter())
        self.assertNotIn("minimum_hypotheses", unpack(transport))
        self.assertEqual(
            adapter.response_schema("one-hypothesis", tools=self.tools)["properties"][
                "competing_explanations"
            ]["minItems"],
            2,
        )
        response["development_slots"] = [{"task_key": "base_salad", "seed": 2}]
        with self.assertRaisesRegex(ValueError, "at least 2"):
            adapter.apply_response(
                self.tools,
                response,
                evidence_level="offline",
                cost_limit={"offline_calls": 1},
                synthetic=True,
            )
        self.assertEqual(self.store.all("rule_trials"), [])

    def test_missing_prediction_or_duplicate_explanations_still_reject(self):
        adapter, _, response = self.prepare()
        for hypotheses in (
            [],
            [{"explanation": "claim without test"}],
            [self.explanations[0], self.explanations[0]],
        ):
            with (
                self.subTest(hypotheses=hypotheses),
                self.assertRaisesRegex(ValueError, "testable"),
            ):
                adapter.apply_response(
                    self.tools,
                    {**response, "competing_explanations": hypotheses},
                    evidence_level="offline",
                    cost_limit={"offline_calls": 1},
                    synthetic=True,
                )
        self.assertEqual(self.store.all("opportunities"), [])

    def test_diagnosis_accepts_one_explanation_without_becoming_trial_evidence(self):
        result = replace(
            self.result("technical", self.base.bundle_id, outcome="unknown"),
            invalid_count=1,
            data_refs=["fault.json"],
            assignments=[
                {
                    "task_key": "salad",
                    "seed": 1,
                    "validity": "invalid",
                    "outcome": "unknown",
                    "evidence_ref": "fault.json",
                }
            ],
        )
        self.store.save_development_result(result)
        self.tools.discover_opportunity(
            "diagnose",
            source="technical_diagnostic",
            result_ids=["technical"],
            external_value="repair transport",
            competing_explanations=self.explanations[:1],
            expected_cost={},
        )
        with self.assertRaisesRegex(ValueError, "does not establish"):
            self.tools.plan_rule_experiment(
                "not-a-trial",
                opportunity_id="diagnose",
                bundle_id="bad",
                finish_requires_native_success=True,
                criterion_ref="criterion",
                distinguishing_prediction="test",
                cost_limit={"offline_calls": 1},
            )


class PreparationSchemaTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    prepared = test_batch4.RunnerTest.prepared

    def test_real_preparation_cli_writes_matching_single_hypothesis_schema(self):
        self.protocol["task_policy"] = {"schema": "rsi-task-policy-v1"}
        self.prepared()
        protocol, policy, output = (
            self.root / "protocol.json",
            self.root / "policy.json",
            self.root / "cli",
        )
        atomic_json(protocol, self.protocol)
        atomic_json(policy, self.tools.domain_policy)
        spec = importlib.util.spec_from_file_location(
            "prepare_proportionate", ROOT / "scripts/prepare_bounded_research.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        argv = [
            "prepare_bounded_research.py",
            "--artifacts",
            str(self.root),
            "--job",
            "new-job",
            "--protocol",
            str(protocol),
            "--policy",
            str(policy),
            "--request",
            "cli-request",
            "--result-id",
            "base-result",
            "--output",
            str(output),
        ]
        with patch("sys.argv", argv):
            self.assertEqual(module.main(), 0)
        self.assertEqual(
            json.loads((output / "response-schema.json").read_text())["properties"][
                "competing_explanations"
            ]["minItems"],
            1,
        )
        self.assertEqual(
            unpack(json.loads((output / "request.json").read_text()))[
                "minimum_hypotheses"
            ],
            1,
        )
        self.assertEqual(self.tools.job.state["charges"], [])
