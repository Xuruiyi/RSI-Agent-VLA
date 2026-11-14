"""Offline comparison decisions never confer deployment qualification."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from rsi_vla.contracts import Candidate, ExperimentResult, RunBundle
from rsi_vla.evaluation import EvaluationStore, Gate
from rsi_vla.release import (ReleaseManager, inspect_initial_comparison_result,
                             inspect_initial_result)
from rsi_vla.storage import ResearchStore, initialize


ROOT = Path(__file__).resolve().parents[2]


class ReleaseComparisonTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT / "config/pilot_5a_release07.json").read_text())
        self.p.update(comparison_only=True, dynamic_authorized=True)
        self.bundle = RunBundle(self.p["anchor_bundle_id"], "theta", "processor", "phi",
            "rho", "psi", "memory", self.p["protocol_id"], "backend", "code")
        self.candidate = Candidate("candidate", self.bundle.bundle_id, "candidate-bundle",
            self.p["protocol_id"], self.p["benchmark_id"], {"phi_ref": "new-phi"},
            ["new-phi"], [], {}, {}, self.p["scope"])

    def result(self, *, candidate=False, failures=()):
        rows = [dict(task_key=key, seed=seed,
            outcome="failure" if key in failures else "success", validity="valid",
            evidence_ref="synthetic-independent-result", outcome_source="native_benchmark")
            for key in self.p["task_sets"]["release"]["task_keys"]
            for seed in self.p["task_sets"]["release"]["seeds"]]
        groups = {task["task_key"]: task["group"] for task in self.p["tasks"]}
        metrics = {"success_" + group: sum(row["outcome"] == "success" for row in rows
            if groups[row["task_key"]] == group) / sum(groups[row["task_key"]] == group for row in rows)
            for group in ("base", "controlled", "old")}
        constraints = dict.fromkeys(self.p["hard_constraints"], True)
        constraints["critical_old_preserved"] = metrics["success_old"] == 1
        return ExperimentResult("candidate-result" if candidate else "initial-result",
            self.candidate.bundle_id if candidate else self.bundle.bundle_id,
            self.p["protocol_id"], self.p["benchmark_id"], "release", self.p["scope"],
            {}, ["synthetic-independent-evidence"], "criteria", metrics, constraints, {},
            candidate_id=self.candidate.candidate_id if candidate else None,
            assigned_count=len(rows), assignments=rows)

    def inspect(self, result):
        return inspect_initial_comparison_result(self.p, self.bundle.__dict__, result)

    def gate(self, candidate, baseline=None, anchor=None):
        return Gate(self.p).evaluate("decision", self.candidate,
            current_bundle_id=self.bundle.bundle_id, candidate_result=candidate,
            baseline_result=baseline, anchor_result=anchor)

    def test_native_old_failure_is_comparable_but_remains_unqualified(self):
        result = self.result(failures=("old_milk",))
        before = copy.deepcopy(result)
        self.assertTrue(self.inspect(result)["passed"])
        qualification = inspect_initial_result(self.p, self.bundle.__dict__, result)
        self.assertFalse(qualification["passed"])
        self.assertEqual(qualification["failure_kind"], "capability")
        self.assertIn("minimum", qualification["reason"])
        self.assertFalse(qualification["qualification_established"])
        self.assertFalse(qualification["activation_performed"])
        self.assertEqual(result, before)
        self.assertEqual(self.p["min_gain"], .02)
        self.assertEqual(self.p["anchor_max_drop"], .02)
        self.assertEqual([self.p["metrics"]["success_" + group]["weight"]
            for group in ("base", "controlled", "old")], [.4, .35, .25])

    def test_complete_native_base_failure_also_remains_comparable(self):
        result = self.result(failures=("base_salad",))
        self.assertTrue(self.inspect(result)["passed"])
        self.assertEqual(inspect_initial_result(self.p, self.bundle.__dict__, result)
            ["failure_kind"], "capability")

    def test_missing_technical_evidence_never_passes_initial_check(self):
        for key in self.p["hard_constraints"]:
            if key == "critical_old_preserved":
                continue
            for value in (None, False):
                with self.subTest(key=key, value=value):
                    result = self.result(failures=("old_milk",))
                    result.hard_constraints[key] = value
                    inspection = self.inspect(result)
                    self.assertFalse(inspection["passed"])
                    self.assertEqual(inspection["failure_kind"], "technical")
                    self.assertEqual(inspect_initial_result(self.p, self.bundle.__dict__, result)
                        ["failure_kind"], "technical")
        result = self.result()
        for changed in (replace(result, data_refs=[]), replace(result, protocol_id="other"),
                        replace(result, bundle_id="other"), replace(result, candidate_id="other"),
                        replace(result, unknown_count=1), replace(result, invalid_count=1)):
            with self.subTest(changed=changed):
                self.assertFalse(self.inspect(changed)["passed"])
        result.assignments[0]["evidence_ref"] = None
        self.assertFalse(self.inspect(result)["passed"])

    def test_old_capability_unknown_or_conflicting_label_cannot_pass_comparison(self):
        for failures, value in (((), None), ((), False), (("old_milk",), True)):
            with self.subTest(failures=failures, value=value):
                baseline = self.result(failures=failures)
                baseline.hard_constraints["critical_old_preserved"] = value
                self.assertFalse(self.inspect(baseline)["passed"])
                self.assertEqual(inspect_initial_result(self.p, self.bundle.__dict__, baseline)
                    ["failure_kind"], "technical")
                self.assertEqual(self.gate(self.result(candidate=True), baseline, baseline).verdict,
                    "inconclusive")

    def test_comparison_accepts_gain_from_unqualified_current_without_deployment(self):
        baseline = self.result(failures=("old_milk",))
        candidate = self.result(candidate=True)
        decision = self.gate(candidate, baseline, baseline)
        self.assertEqual(decision.verdict, "accept")
        self.assertAlmostEqual(decision.scores["candidate"], 1)
        self.assertAlmostEqual(decision.scores["baseline"], .875)
        self.assertFalse(decision.release_eligible)

    def test_candidate_base_and_each_old_capability_remain_required(self):
        baseline = self.result(failures=("base_salad", "position_swap_salad", "old_milk", "old_drawer"))
        for key in ("base_salad", "old_milk", "old_drawer"):
            with self.subTest(key=key):
                candidate = self.result(candidate=True, failures=(key,))
                self.assertEqual(self.gate(candidate, baseline, baseline).verdict, "reject")

    def test_missing_comparison_technical_checks_are_inconclusive_and_violation_wins(self):
        baseline = self.result(failures=("old_milk",))
        candidate = self.result(candidate=True)
        baseline.hard_constraints["resource_limits"] = None
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, "inconclusive")
        candidate.hard_constraints["safety"] = False
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, "reject")
        self.assertEqual(self.gate(replace(candidate, candidate_id="other"), baseline, baseline).verdict,
            "inconclusive")

    def test_unknown_base_contribution_cannot_establish_capability_regression(self):
        baseline = self.result()
        candidate = self.result(candidate=True)
        row = next(row for row in candidate.assignments if row["task_key"] == "base_salad")
        row.update(outcome="unknown", evidence_ref=None, outcome_source=None)
        candidate.metrics["success_base"] = 0
        candidate.hard_constraints["reference_coverage"] = None
        candidate = replace(candidate, unknown_count=1)
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, "inconclusive")

    def test_complete_equal_scores_reject_fixed_gain(self):
        initial = self.result()
        decision = self.gate(self.result(candidate=True), initial, initial)
        self.assertEqual(decision.verdict, "reject")
        self.assertIn("gain below fixed minimum", decision.reasons)

    def test_historical_protocol_keeps_initial_check_shape_and_eligibility(self):
        self.p.pop("comparison_only")
        result = self.result()
        qualification = inspect_initial_result(self.p, self.bundle.__dict__, result)
        self.assertTrue(qualification["passed"])
        self.assertNotIn("failure_kind", qualification)
        self.assertFalse(self.inspect(result)["passed"])
        baseline = self.result(failures=("position_swap_salad",))
        self.assertTrue(self.gate(self.result(candidate=True), baseline, baseline).release_eligible)

    def test_trusted_consumers_never_qualify_or_activate_comparison_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            research = ResearchStore(root / "research")
            initialize(research, self.bundle, psi={},
                memory={"provenance": "synthetic", "legal_information": []})
            research.save_bundle(replace(self.bundle, bundle_id=self.candidate.bundle_id,
                parent_bundle_id=self.bundle.bundle_id, phi_ref="new-phi"))
            research.save_candidate(self.candidate)
            evaluations = EvaluationStore(root / "evaluation")
            evaluations.put("protocols", self.p["protocol_id"], self.p)
            initial = self.result(failures=("old_milk",))
            candidate = self.result(candidate=True)
            evaluations.save_result(initial)
            evaluations.save_result(candidate)
            manager = ReleaseManager(root / "release", research, evaluations)
            manager.bootstrap(self.bundle.bundle_id, scope=self.p["scope"])
            before = manager.active()
            decision = manager.evaluate("comparison", self.candidate.candidate_id,
                candidate_result=candidate.result_id, baseline_result=initial.result_id,
                anchor_result=initial.result_id)
            self.assertEqual(decision.verdict, "accept")
            self.assertFalse(manager.records.all("qualifications"))
            with self.assertRaisesRegex(ValueError, "comparison-only"):
                manager.qualify_initial("initial", result_id=initial.result_id)
            with self.assertRaisesRegex(ValueError, "comparison-only"):
                manager.qualification(self.bundle.bundle_id, scope=self.p["scope"])
            with self.assertRaisesRegex(ValueError, "comparison-only"):
                manager.activate("comparison", safe_point="task_boundary", queues_empty=True,
                    scope=self.p["scope"])
            forged = replace(decision, decision_id="forged", release_eligible=True)
            manager.records.put("decisions", "forged", forged)
            with self.assertRaisesRegex(ValueError, "comparison-only"):
                manager.activate("forged", safe_point="task_boundary", queues_empty=True,
                    scope=self.p["scope"])
            with self.assertRaises(ValueError):
                manager.deployment_bundle(scope=self.p["scope"], safe_point="task_boundary")
            self.assertEqual(manager.active(), before)
            self.assertFalse(manager.records.all("qualifications"))
            self.assertFalse(manager.records.all("activations"))


if __name__ == "__main__":
    unittest.main()
