# Copyright 2026 RSI-Agent-VLA contributors.

"""Six-state midterm regression guards against the historical one-hit floor."""

import unittest
from dataclasses import replace

from rsi_vla.midterm_evaluation import (
    compare_midterm_theta,
    critical_old_report,
    evaluate_midterm_gate,
    paired_comparison,
)
from test_gate_tristate import GateTristateTests


class MidtermEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = GateTristateTests()
        fixture.setUp()
        self.p = fixture.p
        self.p["midterm_policy"] = {
            "schema": "rsi-midterm-evaluation-v1",
            "critical_old_min_fraction": 0.5,
        }
        self.p["task_sets"]["release"]["seeds"] = list(range(10, 16))
        self.candidate = fixture.candidate
        self.make_result = fixture.result

    def result(self, bundle, *, old=6, controlled=6, incomplete=False):
        result = self.make_result(bundle)
        rows = []
        for row in result.assignments:
            row = dict(row)
            count = (
                old
                if row["task_key"].startswith("old_")
                else controlled
                if row["task_key"] == "position_swap_salad"
                else 6
            )
            row["outcome"] = "success" if row["seed"] - 10 < count else "failure"
            rows.append(row)
        if incomplete:
            rows[0].update(outcome="unknown", evidence_ref=None, outcome_source=None)
        groups = {task["task_key"]: task["group"] for task in self.p["tasks"]}
        metrics = {
            "success_" + group: sum(
                row["outcome"] == "success"
                for row in rows
                if groups[row["task_key"]] == group
            )
            / sum(groups[row["task_key"]] == group for row in rows)
            for group in ("base", "controlled", "old")
        }
        return replace(
            result,
            assignments=rows,
            metrics=metrics,
            assigned_count=len(rows),
            unknown_count=int(incomplete),
        )

    def gate(self, candidate, baseline, anchor=None):
        return evaluate_midterm_gate(
            self.p,
            self.candidate,
            candidate_result=candidate,
            baseline_result=baseline,
            anchor_result=anchor or baseline,
            current_bundle_id=self.candidate.base_bundle_id,
            decision_id="midterm-gate",
        )

    def test_one_success_out_of_six_rejects_even_when_historical_gate_floor_passed(
        self,
    ):
        baseline = self.result(self.candidate.base_bundle_id, old=1, controlled=0)
        candidate = self.result(self.candidate.bundle_id, old=1)
        result = self.gate(candidate, baseline)
        self.assertEqual(result["decision"]["verdict"], "reject")
        self.assertFalse(result["critical_old_report"]["value"])
        self.assertEqual(
            result["critical_old_report"]["tasks"]["old_milk"]["minimum_successes"], 3
        )
        self.assertFalse(result["deployment_changed"])

    def test_half_old_success_can_pass_comparison_but_is_not_initial_qualification(
        self,
    ):
        baseline = self.result(self.candidate.base_bundle_id, old=3, controlled=0)
        candidate = self.result(self.candidate.bundle_id, old=3)
        result = self.gate(candidate, baseline)
        self.assertEqual(result["decision"]["verdict"], "accept")
        self.assertTrue(
            result["candidate_result"]["hard_constraints"]["critical_old_preserved"]
        )
        self.assertIn("success_old=1", result["qualification_scope"])

    def test_meeting_half_does_not_allow_loss_against_current_or_fixed_anchor(self):
        candidate = self.result(self.candidate.bundle_id, old=3)
        baseline = self.result(self.candidate.base_bundle_id, old=3, controlled=0)
        anchor = self.result(self.candidate.base_bundle_id, old=4, controlled=0)
        self.assertEqual(
            self.gate(candidate, baseline, anchor)["decision"]["verdict"], "reject"
        )

    def test_missing_data_keeps_whole_denominator_and_no_complete_case_p_value(self):
        candidate = self.result(self.candidate.bundle_id, incomplete=True)
        baseline = self.result(self.candidate.base_bundle_id, controlled=0)
        result = self.gate(candidate, baseline)
        self.assertEqual(result["decision"]["verdict"], "inconclusive")
        comparison = result["comparison"]
        self.assertEqual(comparison["assigned_pairs"], 24)
        self.assertEqual(comparison["tasks"]["base_salad"]["assigned_pairs"], 6)
        self.assertIsNone(
            comparison["tasks"]["base_salad"]["exact_mcnemar_two_sided_p"]
        )
        self.assertFalse(comparison["equivalence_established"])

    def test_six_paired_wins_report_exact_large_effect_without_weighted_p_claim(self):
        candidate = self.result(self.candidate.bundle_id)
        baseline = self.result(self.candidate.base_bundle_id, controlled=0)
        result = paired_comparison(self.p, candidate, baseline)
        target = result["tasks"]["position_swap_salad"]
        self.assertEqual((target["wins"], target["losses"], target["ties"]), (6, 0, 0))
        self.assertEqual(target["exact_mcnemar_two_sided_p"], 0.03125)
        self.assertAlmostEqual(result["utility_difference"], 0.35)
        self.assertLess(target["candidate_wilson95"][0], 1)

    def test_missing_slot_and_reduced_old_threshold_are_not_accepted(self):
        candidate = self.result(self.candidate.bundle_id)
        with self.assertRaisesRegex(ValueError, "assignment matrix"):
            critical_old_report(
                self.p, replace(candidate, assignments=candidate.assignments[:-1])
            )
        self.p["midterm_policy"]["critical_old_min_fraction"] = 0.49
        with self.assertRaisesRegex(ValueError, "half-success"):
            critical_old_report(self.p, candidate)

    def test_theta0_results_cannot_substitute_for_a_missing_theta1(self):
        initial = self.result(self.candidate.base_bundle_id)
        with self.assertRaisesRegex(ValueError, "both actual cell"):
            compare_midterm_theta(self.p, initial, initial)
        binding = {
            "value": True,
            "cell": {
                "theta_id": "theta0",
                "arm": "D",
                "b0_ref": "B0",
                "theta_ref": "theta0",
            },
        }
        initial = replace(initial, intervention={"midterm_binding_evidence": binding})
        with self.assertRaisesRegex(ValueError, "distinct theta0/theta1"):
            compare_midterm_theta(self.p, initial, initial)


if __name__ == "__main__":
    unittest.main()
