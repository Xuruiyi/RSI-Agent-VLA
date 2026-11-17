"""Independent meta scoring preserves frozen controls, denominators and costs."""

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from rsi_vla.evaluation import EvaluationStore
from rsi_vla.meta_scoring import score_exam
from test_objective_evaluation import fixture


def meta_fixture(root, mode="milestones"):
    p, c, result = fixture()
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps(p))
    candidate = root / "candidate.json"
    candidate.write_text(json.dumps(asdict(c)))
    facts = root / "facts.json"
    facts.write_text(
        json.dumps({"legal_scope": "development_only", "role": "facts_not_policy"})
    )
    late = root / "late.json"
    late.write_text(
        json.dumps(
            {
                "legal_scope": "development_only",
                "role": "facts_not_policy",
                "later": True,
            }
        )
    )
    store = EvaluationStore(root / "evaluation")
    arms = {}
    for arm in ["psi0", "psi1", "psi0_late"]:
        snapshot = {
            "snapshot_id": arm,
            "components": {"policy": "changed" if arm == "psi1" else "base"},
            "problem_partition": {
                "development_families": ["development"],
                "development_mechanisms": ["dev"],
                "reserved_families": ["heldout"],
                "reserved_mechanisms": ["held"],
            },
        }
        ref = root / (arm + ".json")
        ref.write_text(json.dumps(snapshot))
        arms[arm] = {
            "policy_snapshot_id": arm,
            "policy_snapshot_ref": str(ref),
            "facts_ref": str(late if arm == "psi0_late" else facts),
        }
    points = [1, 2] if mode == "milestones" else [2]
    registration = {
        "schema": "rsi-meta-exam-v1",
        "exam_id": "exam",
        "scope": "offline",
        "mode": mode,
        "arms": arms,
        "problems": [
            {
                "problem_id": "problem",
                "family": "heldout",
                "mechanism": "held",
                "protocol_ref": str(protocol),
            }
        ],
        "repetitions": [0],
        "budget_points": points,
        "budget_dimension": "wall_seconds",
        "common": {
            "theta_ref": "theta0",
            "phi_ref": "phi0",
            "non_test_policy_ref": "rest",
        },
        "shared_facts_ref": str(facts),
        "missing_score": 0,
        "minimum_gain": 0.1,
        "retention_max_drop": 0,
        "policy_generation": {
            "generation_id": "generated-once",
            "costs": {"wall_seconds": 3},
            "independent_generations": 1,
        },
        "target_score": 0.9,
    }
    submissions = []
    for arm in arms:
        for point in points:
            # Every arm submits a concrete candidate independently scored on the
            # same task matrix. The fixture supplies synthetic native facts.
            measured = result(True, (True, True) if arm == "psi1" else (False, False))
            record = asdict(measured)
            record["result_id"] = f"{arm}-{point}"
            store.put("results", record["result_id"], record)
            row = {
                "problem_id": "problem",
                "repeat": 0,
                "arm": arm,
                "budget_point": point,
                "candidate_ref": str(candidate),
                "result_id": record["result_id"],
                "attempt_costs": [{"wall_seconds": 0.5}],
                "status": "submitted",
            }
            identity = {
                key: row[key] for key in ["problem_id", "repeat", "arm", "budget_point"]
            }
            cost = root / f"cost-{arm}-{point}.json"
            cost.write_text(
                json.dumps(
                    dict(
                        schema="rsi-meta-attempt-cost-v1",
                        exam_id="exam",
                        **identity,
                        attempt_costs=row["attempt_costs"],
                    )
                )
            )
            submission = root / f"submission-{arm}-{point}.json"
            submission.write_text(
                json.dumps(
                    dict(
                        schema="rsi-meta-submission-v1",
                        exam_id="exam",
                        **identity,
                        candidate_ref=row["candidate_ref"],
                        result_id=row["result_id"],
                        status=row["status"],
                        common=registration["common"],
                        policy_snapshot_id=arm,
                        facts_ref=arms[arm]["facts_ref"],
                    )
                )
            )
            row.update(
                cost_receipt_ref=str(cost), submission_receipt_ref=str(submission)
            )
            submissions.append(row)
    # Score uses this protocol's fixed success weights rather than trusting any
    # research supplied endpoint. Keep old competence independently evaluated.
    p["metrics"] = {
        "success_controlled": {"direction": "max", "min": 0, "max": 1, "weight": 0.75},
        "success_old": {"direction": "max", "min": 0, "max": 1, "weight": 0.25},
    }
    protocol.write_text(json.dumps(p))
    registration["protocol_snapshots"] = {"problem": deepcopy(p)}
    return registration, submissions, store


class MetaScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_registered_curve_uses_submitted_points_and_reports_all_costs(self):
        registration, submissions, store = meta_fixture(self.root)
        value = score_exam(registration, submissions, store)
        self.assertEqual(value["verdict"], "accept")
        self.assertEqual(value["metrics"]["endpoint_gain"], 0.75)
        self.assertEqual(value["metrics"]["auc"]["psi1"], 1)
        self.assertEqual(value["metrics"]["cost_to_target"]["psi1"], 1)
        self.assertEqual(value["costs"]["examination_all_attempts"]["wall_seconds"], 3)
        self.assertEqual(value["costs"]["policy_generation"]["wall_seconds"], 3)
        self.assertEqual(
            value["feedback"], {"information_scope": "meta_exam", "status": "withheld"}
        )
        self.assertIn("single_policy_snapshot", value["limitations"])

    def test_endpoint_does_not_invent_curve_or_time_to_target(self):
        registration, submissions, store = meta_fixture(self.root, "endpoint")
        value = score_exam(registration, submissions, store)
        self.assertIsNone(value["metrics"]["auc"])
        self.assertIsNone(value["metrics"]["cost_to_target"])
        self.assertNotIn("curves", value["metrics"])

    def test_duplicate_selection_and_start_contamination_refused(self):
        registration, submissions, store = meta_fixture(self.root)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_exam(registration, submissions + [submissions[0]], store)
        registration["arms"]["psi1"]["facts_ref"] = registration["arms"]["psi0_late"][
            "facts_ref"
        ]
        with self.assertRaisesRegex(ValueError, "same legal facts"):
            score_exam(registration, submissions, store)

    def test_missing_point_penalty_not_best_candidate_and_unknown_costs(self):
        registration, submissions, store = meta_fixture(self.root)
        submissions = [
            row
            for row in submissions
            if not (row["arm"] == "psi1" and row["budget_point"] == 2)
        ]
        value = score_exam(registration, submissions, store)
        self.assertEqual(value["metrics"]["endpoint"]["psi1"], 0)
        self.assertEqual(value["verdict"], "inconclusive")
        self.assertEqual(value["metrics"]["auc"]["psi1"], 0.5)

    def test_overbudget_and_forged_common_binding_rejected(self):
        registration, submissions, store = meta_fixture(self.root)
        row = submissions[0]
        row["attempt_costs"] = [{"wall_seconds": 4}]
        cost = Path(row["cost_receipt_ref"])
        record = json.loads(cost.read_text())
        record["attempt_costs"] = row["attempt_costs"]
        cost.write_text(json.dumps(record))
        self.assertEqual(
            score_exam(registration, submissions, store)["verdict"], "reject"
        )
        submission = Path(row["submission_receipt_ref"])
        record = json.loads(submission.read_text())
        record["common"]["theta_ref"] = "changed"
        submission.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "common start"):
            score_exam(registration, submissions, store)

    def test_no_update_is_not_meta_improvement(self):
        registration, submissions, store = meta_fixture(self.root)
        candidate = Path(registration["arms"]["psi1"]["policy_snapshot_ref"])
        record = json.loads(candidate.read_text())
        record["components"]["policy"] = "base"
        candidate.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "no policy update"):
            score_exam(registration, submissions, store)


if __name__ == "__main__":
    unittest.main()
