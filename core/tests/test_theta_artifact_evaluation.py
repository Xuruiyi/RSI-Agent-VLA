"""Theta publication closes actual Host serialization costs without rewriting history."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rsi_vla.contracts import Candidate
from rsi_vla.storage import atomic_json
from rsi_vla.theta_artifact_evaluation import (
    close_theta_candidate_costs,
    evaluate_theta_artifact_matrix,
)


class ThetaArtifactCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs = {}
        for operation, wall, storage, disk in [
            ("train", 1, 10, 8),
            ("bundle", 0.5, 4, 5),
        ]:
            report = self.root / (operation + "-supervisor.json")
            atomic_json(
                report,
                {
                    "identity": {"run_id": operation},
                    "status": "completed",
                    "exit_code": 0,
                    "cleanup_confirmed": True,
                    "remaining_process_ids": [],
                },
            )
            receipt = self.root / (operation + "-completion.json")
            actual = {
                "wall_seconds": wall,
                "storage_bytes": storage,
                "current_disk_bytes": disk,
                "model_requests": 0,
                "gpu_seconds": 0,
            }
            value = {
                "scope_id": "scope",
                "operation_id": operation,
                "status": "completed",
                "cleanup_confirmed": True,
                "unknown_dimensions": [],
                "actual": actual,
                "supervisor_ref": str(report),
            }
            atomic_json(receipt, value)
            self.refs[operation] = (receipt, value)
        self.accounting = self.root / "accounting.json"
        self.reservations = {
            name: {
                "status": "settled",
                "cleanup_confirmed": True,
                "unknown_dimensions": [],
                "receipt_ref": str(ref),
                "actual": value["actual"],
            }
            for name, (ref, value) in self.refs.items()
        }
        atomic_json(
            self.accounting, {"scope_id": "scope", "reservations": self.reservations}
        )
        source_ref = self.root / "source.json"
        atomic_json(
            source_ref,
            {
                "schema": "rsi-training-candidate-sources-v1",
                "candidate_id": "candidate",
                "host_cost_receipts": [{"receipt_ref": str(self.refs["train"][0])}],
                "pending_cost_operations": ["bundle"],
                "accounting_ref": str(self.accounting),
            },
        )
        self.source_ref = source_ref
        self.candidate = Candidate(
            "candidate",
            "base",
            "theta1",
            "protocol",
            "benchmark",
            {},
            [],
            [],
            {
                "whole_job_wall_seconds": 1,
                "artifact_bytes": 10,
                "peak_current_disk_bytes": 8,
                "model_requests": 0,
                "gpu_device_seconds_envelope": 0,
            },
            {
                "source_receipt_ref": str(source_ref),
                "pending_serialization_cost_operations": ["bundle"],
            },
            "offline",
        )

    def test_settled_serialization_added_once_without_changing_original_candidate(self):
        original = deepcopy(self.candidate.costs)
        raw = self.source_ref.read_bytes()
        value = close_theta_candidate_costs(self.candidate, scope_id="scope")
        self.assertEqual(value["all_attempt_costs"]["whole_job_wall_seconds"], 1.5)
        self.assertEqual(value["all_attempt_costs"]["artifact_bytes"], 14)
        self.assertEqual(value["all_attempt_costs"]["peak_current_disk_bytes"], 8)
        self.assertEqual(value["closed_serialization_operations"], ["bundle"])
        self.assertEqual(value["pending_cost_operations"], [])
        self.assertEqual(self.candidate.costs, original)
        self.assertEqual(self.source_ref.read_bytes(), raw)
        self.assertEqual(
            close_theta_candidate_costs(self.candidate, scope_id="scope"), value
        )

    def test_pending_or_unknown_serialization_blocks_publication_consumer(self):
        for defect in ["active", "unknown_cost", "cleanup"]:
            reservations = deepcopy(self.reservations)
            if defect == "active":
                reservations["bundle"]["status"] = "active"
            elif defect == "unknown_cost":
                reservations["bundle"]["unknown_dimensions"] = ["wall_seconds"]
            else:
                reservations["bundle"]["cleanup_confirmed"] = False
            atomic_json(
                self.accounting, {"scope_id": "scope", "reservations": reservations}
            )
            with (
                self.subTest(defect=defect),
                self.assertRaisesRegex(PermissionError, "unresolved"),
            ):
                close_theta_candidate_costs(self.candidate, scope_id="scope")

    def test_synthetic_interface_evidence_cannot_qualify_real_model_scope(self):
        for protocol in [
            {"scope": "simulation"},
            {"scope": "offline", "synthetic_engineering": False},
            {
                "scope": "offline",
                "synthetic_engineering": True,
                "theta_artifact_engineering": "rsi-theta-artifact-engineering-v1",
                "hard_constraints": ["safety", "compatibility"],
            },
        ]:
            with self.subTest(protocol=protocol), self.assertRaises(PermissionError):
                evaluate_theta_artifact_matrix(
                    protocol, {}, result_id="r", candidate_id="c", problem_ref="unused"
                )


if __name__ == "__main__":
    unittest.main()
