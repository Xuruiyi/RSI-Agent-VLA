"""Prospective finite efficiency criteria preserve success and all attempt costs."""

import copy
import unittest
from unittest.mock import patch

import test_evaluator
import test_gate_tristate
import test_upgrade_bridge
from rsi_vla.evaluation import Gate, apply_comparison_attempt_evidence
from rsi_vla.storage import atomic_json
from rsi_vla.upgrade_bridge import CRITERIA_FIELDS, _mapping


class ConditionalEfficiencyTests(unittest.TestCase):
    setUpBase = test_gate_tristate.GateTristateTests.setUp
    resultBase = test_gate_tristate.GateTristateTests.result

    def setUp(self):
        self.setUpBase()
        self.p["task_sets"]["release"]["seeds"] = [0, 1]
        self.p["cost_weights"] = {}
        self.p["comparison_only"] = True
        self.p["conditional_objective"] = {
            "schema": "rsi-success-ceiling-efficiency-v1",
            "minimum_relative_request_reduction": 0.20,
            "minimum_absolute_request_reduction": 8,
            "minimum_improved_tasks": 3,
            "maximum_task_request_increase": 1,
            "request_score_weight": 0.0025,
        }
        self.keys = self.p["task_sets"]["release"]["task_keys"]

    def result(self, bundle, per_task):
        result = self.resultBase(bundle)
        result.data_refs.append("independent-results")
        plan = {"slots": []}
        for key, count in zip(self.keys, per_task):
            for seed in (0, 1):
                requests = count // 2 if seed == 0 else count - count // 2
                plan["slots"].append(
                    {
                        "side": "side",
                        "task_key": key,
                        "attempts": [
                            {
                                "costs": {
                                    "model_requests": requests,
                                    "tool_calls": 1,
                                    "environment_steps": 5,
                                    "whole_job_wall_seconds": 10,
                                    "gpu_device_seconds_envelope": 10,
                                    "artifact_bytes": 20,
                                }
                            }
                        ],
                    }
                )
        apply_comparison_attempt_evidence(plan, "side", result)
        return result, plan

    def gate(self, costs=(16, 16, 16, 16), baseline_costs=(20, 20, 20, 20)):
        candidate, _ = self.result("candidate-bundle", costs)
        baseline, _ = self.result(self.candidate.base_bundle_id, baseline_costs)
        return self.evaluate(candidate, baseline)

    def evaluate(self, candidate, baseline):
        return Gate(self.p).evaluate(
            "conditional-gate",
            self.candidate,
            current_bundle_id=self.candidate.base_bundle_id,
            candidate_result=candidate,
            baseline_result=baseline,
            anchor_result=baseline,
        )

    def test_ceiling_accepts_preserved_success_with_distributed_twenty_percent_reduction(
        self,
    ):
        decision = self.gate()
        self.assertEqual(decision.verdict, "accept", decision.reasons)
        self.assertIn("objective branch: efficiency", decision.reasons)
        self.assertFalse(decision.release_eligible)
        self.assertIn("conditional_objective", CRITERIA_FIELDS)
        self.assertEqual(self.gate((8, 8, 8, 8), (10, 10, 10, 10)).verdict, "accept")

    def test_efficiency_rejects_any_task_success_regression(self):
        candidate, _ = self.result("candidate-bundle", (10, 10, 10, 10))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        next(
            row
            for row in candidate.assignments
            if row["task_key"] == "position_swap_salad"
        )["outcome"] = "failure"
        candidate.metrics["success_controlled"] = 0.5
        decision = self.evaluate(candidate, baseline)
        self.assertEqual(decision.verdict, "reject")
        self.assertTrue(
            any("task success regression" in reason for reason in decision.reasons)
        )

    def test_single_task_improvement_or_task_increase_does_not_pass(self):
        decision = self.gate((0, 20, 20, 20))
        self.assertEqual(decision.verdict, "reject")
        self.assertTrue(any("too few tasks" in reason for reason in decision.reasons))
        decision = self.gate((22, 14, 14, 14))
        self.assertEqual(decision.verdict, "reject")
        self.assertTrue(
            any("per-task request increase" in reason for reason in decision.reasons)
        )

    def test_eight_saved_requests_but_less_than_twenty_percent_rejects(self):
        decision = self.gate((18, 18, 18, 18))
        self.assertEqual(decision.verdict, "reject")
        self.assertTrue(
            any("absolute/relative" in reason for reason in decision.reasons)
        )

    def test_unknown_task_cost_or_missing_attempt_identity_is_inconclusive(self):
        candidate, plan = self.result("candidate-bundle", (16, 16, 16, 16))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        plan["slots"][0].pop("task_key")
        apply_comparison_attempt_evidence(plan, "side", candidate)
        self.assertEqual(self.evaluate(candidate, baseline).verdict, "inconclusive")

    def test_unknown_total_request_cost_and_unestablished_constraint_are_inconclusive(
        self,
    ):
        candidate, _ = self.result("candidate-bundle", (16, 16, 16, 16))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        candidate.costs.pop("model_requests")
        self.assertEqual(self.evaluate(candidate, baseline).verdict, "inconclusive")
        candidate, _ = self.result("candidate-bundle", (16, 16, 16, 16))
        baseline.hard_constraints["safety"] = None
        self.assertEqual(self.evaluate(candidate, baseline).verdict, "inconclusive")

    def test_technical_recovery_cost_cannot_be_discarded_for_cheaper_selected_attempt(
        self,
    ):
        candidate, plan = self.result("candidate-bundle", (16, 16, 16, 16))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        prior = copy.deepcopy(plan["slots"][0]["attempts"][0])
        prior["costs"]["model_requests"] = 16
        plan["slots"][0]["attempts"].insert(0, prior)
        apply_comparison_attempt_evidence(plan, "side", candidate)
        self.assertEqual(candidate.costs["model_requests"], 80)
        self.assertEqual(
            candidate.intervention["attempt_model_requests_by_task"][self.keys[0]], 32
        )
        self.assertEqual(self.evaluate(candidate, baseline).verdict, "reject")

    def test_old_protocol_retains_its_previous_score_acceptance(self):
        self.p.pop("conditional_objective")
        self.p["cost_weights"] = {"model_requests": 0.0025}
        decision = self.gate((0, 20, 20, 20))
        self.assertEqual(decision.verdict, "accept")
        self.assertFalse(
            any("objective branch" in reason for reason in decision.reasons)
        )

    def test_non_ceiling_baseline_keeps_the_success_gain_branch(self):
        candidate, _ = self.result("candidate-bundle", (60, 60, 60, 60))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        for row in baseline.assignments:
            if row["task_key"] == "position_swap_salad":
                row["outcome"] = "failure"
        baseline.metrics["success_controlled"] = 0
        decision = self.evaluate(candidate, baseline)
        self.assertEqual(decision.verdict, "accept", decision.reasons)
        self.assertIn("objective branch: success_gain", decision.reasons)
        self.assertEqual(decision.scores["candidate"], 1)
        self.assertEqual(self.p["cost_weights"], {})

    def test_non_ceiling_same_success_rate_cannot_accept_by_saving_requests(self):
        candidate, _ = self.result("candidate-bundle", (0, 0, 0, 0))
        baseline, _ = self.result(self.candidate.base_bundle_id, (20, 20, 20, 20))
        for result in (candidate, baseline):
            for row in result.assignments:
                if row["task_key"] == "position_swap_salad":
                    row["outcome"] = "failure"
            result.metrics["success_controlled"] = 0
        decision = self.evaluate(candidate, baseline)
        self.assertEqual(decision.verdict, "reject")
        self.assertEqual(decision.scores["candidate"], decision.scores["baseline"])
        self.assertIn("objective branch: success_gain", decision.reasons)
        self.assertEqual(self.p["cost_weights"], {})

    def test_efficiency_score_selection_does_not_mutate_frozen_protocol(self):
        original = copy.deepcopy(self.p)
        self.assertEqual(self.gate().verdict, "accept")
        self.assertEqual(self.p, original)

    def test_upgrade_mapping_rejects_conditional_criterion_drift(self):
        fixture = test_upgrade_bridge.UpgradeBridgeTests()
        fixture.setUp()
        try:
            fixture.registration["source_protocol_snapshot"][
                "conditional_objective"
            ] = copy.deepcopy(self.p["conditional_objective"])
            fixture.registration["target_protocol_snapshot"][
                "conditional_objective"
            ] = copy.deepcopy(self.p["conditional_objective"])
            fixture.registration["target_protocol_snapshot"]["conditional_objective"][
                "minimum_relative_request_reduction"
            ] = 0.1
            atomic_json(
                fixture.registration["source_protocol_ref"],
                fixture.registration["source_protocol_snapshot"],
            )
            atomic_json(
                fixture.registration["target_protocol_ref"],
                fixture.registration["target_protocol_snapshot"],
            )
            with self.assertRaisesRegex(
                ValueError, "frozen criterion: conditional_objective"
            ):
                _mapping(fixture.registration)
        finally:
            fixture.doCleanups()


class AuditedAssignmentCostsTests(unittest.TestCase):
    setUp = test_evaluator.EvaluatorTests.setUp
    native = test_evaluator.EvaluatorTests.native
    evaluate = test_evaluator.EvaluatorTests.evaluate

    def test_independent_audit_adds_requests_without_replacing_native_assignment(self):
        manifest = self.native()
        from pathlib import Path

        (Path(manifest[0]["result_path"]).parent / "runtime_context.json").write_text(
            "{}"
        )
        report = {
            "passed": True,
            "costs": {"model_requests": 7},
            "notes": [],
            "checks": {},
            "effective_rules": {},
        }
        with patch("rsi_vla.evaluator.inspect_attempt", return_value=report):
            result = self.evaluate(manifest)
        row = next(row for row in result.assignments if row["evidence_ref"])
        self.assertEqual(row["model_requests"], 7)
        self.assertEqual(row["outcome"], "success")
        self.assertEqual(row["outcome_source"], "native_benchmark")
        self.assertEqual(result.unknown_count, 7)


if __name__ == "__main__":
    unittest.main()
