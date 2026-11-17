"""Finite objectives use complete independent matrices and all-attempt costs."""

import unittest
from dataclasses import replace

from rsi_vla.contracts import Candidate, ExperimentResult
from rsi_vla.evaluation import Gate
from rsi_vla.objective_evaluation import exact_paired, feedback_projection


def fixture(objective="success_gain"):
    protocol = {
        "protocol_id": "objective-v1",
        "benchmark_id": "benchmark",
        "status": "frozen",
        "scope": "offline",
        "anchor_bundle_id": "base",
        "metrics": {},
        "hard_constraints": ["safety"],
        "min_gain": 0.1,
        "anchor_max_drop": 0,
        "tasks": [
            {
                "task_key": "target",
                "group": "controlled",
                "variation_condition": "object pose",
            },
            {"task_key": "old", "group": "old"},
        ],
        "task_sets": {
            "release": {
                "task_keys": ["target", "old"],
                "seeds": [0, 1],
                "benchmark_id": "benchmark",
            }
        },
        "objective_comparison": {
            "schema": "rsi-objective-comparison-v1",
            "objective": objective,
            "task_keys": ["target"],
            "minimum_gain": 0.1,
            "anchor_max_drop": 0,
        },
    }
    candidate = Candidate(
        "candidate",
        "base",
        "changed",
        "objective-v1",
        "benchmark",
        {"phi_ref": "changed"},
        [],
        [],
        {},
        {},
        "offline",
    )

    def result(changed=False, outcomes=(False, False), cost=4):
        rows = [
            {
                "task_key": key,
                "seed": seed,
                "outcome": "success" if key == "old" or outcomes[seed] else "failure",
                "validity": "valid",
                "evidence_ref": f"native-{changed}-{key}-{seed}",
                "outcome_source": "native_benchmark",
            }
            for key in ["target", "old"]
            for seed in [0, 1]
        ]
        return ExperimentResult(
            "candidate-result" if changed else "baseline-result",
            "changed" if changed else "base",
            "objective-v1",
            "benchmark",
            "release",
            "offline",
            {
                "attempt_costs_by_slot": [
                    {
                        "task_key": row["task_key"],
                        "seed": row["seed"],
                        "attempts": [{"model_requests": cost}],
                    }
                    for row in rows
                ]
            },
            ["independent-source"],
            "objective-v1",
            {"success_controlled": sum(outcomes) / 2, "success_old": 1},
            {"safety": True},
            {"model_requests": cost * 4},
            candidate_id="candidate" if changed else None,
            assigned_count=4,
            assignments=rows,
        )

    return protocol, candidate, result


class ObjectiveTests(unittest.TestCase):
    def test_exact_paired_hand_calculation_and_zero_pairs(self):
        test = exact_paired([1, 1, 0])
        self.assertEqual((test["p_numerator"], test["p_denominator"]), (2, 8))
        self.assertAlmostEqual(test["mean"], 2 / 3)
        self.assertEqual(exact_paired([-1, 1])["p_value"], 0.75)

    def evaluate(self, p, c, changed, base):
        return Gate(p).evaluate(
            "decision",
            c,
            current_bundle_id="base",
            candidate_result=changed,
            baseline_result=base,
            anchor_result=base,
        )

    def test_all_supported_finite_objectives_reach_real_gate(self):
        for objective in ["success_gain", "robustness", "new_capability"]:
            p, c, result = fixture(objective)
            if objective == "new_capability":
                p["tasks"][0].update(
                    materialization_ref="independent-materialization",
                    native_success="bddl_goal",
                )
            decision = self.evaluate(p, c, result(True, (True, True)), result())
            self.assertEqual(decision.verdict, "accept")
            self.assertTrue(decision.release_eligible)
            self.assertEqual(decision.scores["paired_p_value"], 0.25)
            p["objective_comparison"]["statistics"] = {
                "method": "exact_paired",
                "alpha": 0.05,
            }
            self.assertEqual(
                self.evaluate(p, c, result(True, (True, True)), result()).verdict,
                "inconclusive",
            )

    def test_missing_duplicate_forged_metric_and_wrong_identity_are_inconclusive(self):
        p, c, result = fixture()
        base = result()
        for update in [
            lambda r: r.assignments.pop(),
            lambda r: r.assignments.append(r.assignments[0]),
            lambda r: r.metrics.update(success_controlled=0.5),
        ]:
            changed = result(True, (True, True))
            update(changed)
            self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")
        changed = replace(result(True, (True, True)), candidate_id="forged")
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")

    def test_unknown_safety_and_known_violation(self):
        p, c, result = fixture()
        changed = result(True, (True, True))
        base = result()
        changed.hard_constraints["safety"] = None
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")
        changed.hard_constraints["safety"] = False
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "reject")
        changed = replace(changed, candidate_id="forged")
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")

    def test_critical_old_loss_rejects(self):
        p, c, result = fixture()
        changed = result(True, (True, True))
        for row in changed.assignments:
            if row["task_key"] == "old":
                row["outcome"] = "failure"
        changed.metrics["success_old"] = 0
        self.assertEqual(self.evaluate(p, c, changed, result()).verdict, "reject")

    def test_synthetic_task_source_requires_both_explicit_offline_flags(self):
        from rsi_vla.evaluation import assignment_metrics

        p, _, result = fixture()
        measured = result(True, (True, True))
        for row in measured.assignments:
            row["outcome_source"] = "synthetic_task"
        with self.assertRaises(ValueError):
            assignment_metrics(p, measured)
        p["synthetic_engineering"] = True
        with self.assertRaises(ValueError):
            assignment_metrics(p, measured)
        measured.intervention["synthetic_engineering"] = True
        self.assertEqual(assignment_metrics(p, measured)["success_controlled"], 1)
        p["scope"] = "simulation"
        with self.assertRaises(ValueError):
            assignment_metrics(p, measured)

    def test_candidate_can_restore_an_independently_known_old_failure(self):
        p, c, result = fixture()
        p["hard_constraints"].append("critical_old_preserved")
        base = result()
        for row in base.assignments:
            if row["task_key"] == "old":
                row["outcome"] = "failure"
        base.metrics["success_old"] = 0
        base.hard_constraints["critical_old_preserved"] = False
        changed = result(True, (True, True))
        changed.hard_constraints["critical_old_preserved"] = True
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "accept")
        base.hard_constraints["critical_old_preserved"] = True
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")

    def test_efficiency_counts_failed_attempts_and_preserves_outcomes(self):
        p, c, result = fixture("efficiency")
        p["objective_comparison"]["cost_dimension"] = "model_requests"
        base = result(outcomes=(True, True))
        changed = result(True, (True, True), 2)
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "accept")
        # A failed first attempt costs more than the selected successful retry.
        changed.intervention["attempt_costs_by_slot"][0]["attempts"].append(
            {"model_requests": 8}
        )
        changed.costs["model_requests"] += 8
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "reject")
        changed.intervention["attempt_costs_by_slot"].pop()
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "inconclusive")
        changed = result(True, (False, True), 1)
        self.assertEqual(self.evaluate(p, c, changed, base).verdict, "reject")

    def test_sealed_projection_never_returns_scores_or_result_refs(self):
        _, _, result = fixture()
        for scope in ["sealed", "meta_exam"]:
            projected = feedback_projection(result(), information_scope=scope)
            self.assertEqual(set(projected), {"schema", "information_scope", "status"})
        self.assertIn(
            "metrics", feedback_projection(result(), information_scope="release")
        )


if __name__ == "__main__":
    unittest.main()


class PublicFeedbackTests(unittest.TestCase):
    def test_frozen_policy_controls_actual_feedback_file(self):
        import tempfile

        from rsi_vla.evaluation import EvaluationStore

        p, _, result = fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(directory)
            measured = result(True, (True, True))
            measured.intervention["information_scope"] = "release"
            store.save_result(measured)
            p["feedback_policy"] = {"release": "aggregate"}
            value = store.public_feedback(measured.result_id, protocol=p)
            self.assertEqual(
                store.get("public_feedback", measured.result_id), value["feedback"]
            )
            self.assertEqual(
                set(value["feedback"]["feedback"]),
                {
                    "schema",
                    "information_scope",
                    "result_id",
                    "metrics",
                    "unknown_count",
                    "invalid_count",
                },
            )
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(directory)
            measured.intervention["information_scope"] = "meta_exam"
            store.save_result(measured)
            p["feedback_policy"]["meta_exam"] = "aggregate"
            value = store.public_feedback(measured.result_id, protocol=p)
            self.assertEqual(value["feedback"]["feedback"]["status"], "withheld")
            self.assertNotIn("metrics", value["feedback"]["feedback"])
