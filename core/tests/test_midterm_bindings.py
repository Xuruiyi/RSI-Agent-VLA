# Copyright 2026 RSI-Agent-VLA contributors.

"""Frozen behavior permits only separately registered theta asset changes."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rsi_vla.midterm_evaluation import freeze_midterm_b0, inspect_midterm_bindings


class MidtermBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = {}
        for arm in ("D", "S", "R", "A"):
            spec = {"schema": "libero-execution-arm-v1", "arm": arm, "max_chunks": 2}
            if arm in {"S", "R"}:
                spec["stages"] = [
                    {"id": "first", "instruction": "registered goal", "max_chunks": 2}
                ]
                if arm == "R":
                    spec["stages"][0].update(
                        on_complete="stop", on_timeout="stop", on_unknown="stop"
                    )
            self.bindings[arm] = {
                "bundle": {
                    "phi_ref": "program",
                    "rho_ref": "rho",
                    "psi_ref": "psi",
                    "memory_ref": "memory",
                },
                "references": {"phi_ref": {"chunk_steps": 5}},
                "records": {
                    "rho_ref": {"policy": "fixed"},
                    "psi_ref": {"prompt": "research"},
                    "memory_ref": {"retrieval": "fixed"},
                },
                "profile": {
                    "execution_arm": spec,
                    "model": {"checkpoint": "theta0"},
                    "planner": {"backend": "codex" if arm == "A" else "flash"},
                    "common_execution": {
                        "schema": "libero-common-execution-v1",
                        "observation_cadence": "prediction",
                        "allowed_motion_tools": ["vla_execute"],
                        "action_prefix_steps": 5,
                    },
                },
            }
        self.b0 = freeze_midterm_b0(self.bindings)
        self.b0_path = self.root / "B0.json"
        self.b0_path.write_text(json.dumps(self.b0))
        model_path = self.root / "model.json"
        model_path.write_text(
            json.dumps({"model": {"checkpoint": "theta0"}, "processor": "norm"})
        )
        self.theta_ref, self.processor_ref = (
            str(model_path) + "#/model",
            str(model_path) + "#/processor",
        )
        binding = deepcopy(self.bindings["D"])
        binding["bundle"].update(
            theta_ref=self.theta_ref, processor_ref=self.processor_ref
        )
        binding["references"]["theta_ref"] = {"checkpoint": "theta0"}
        self.profile = binding["profile"]
        self.profile["midterm_cell"] = {
            "cell_id": "theta0-D",
            "theta_id": "theta0",
            "arm": "D",
            "theta_ref": self.theta_ref,
            "processor_ref": self.processor_ref,
            "b0_ref": str(self.b0_path),
        }
        self.context = {
            "theta_ref": self.theta_ref,
            "processor_ref": self.processor_ref,
            "execution_arm": deepcopy(self.profile["execution_arm"]),
            "common_execution": self.profile["common_execution"],
            "execution_binding": binding,
        }
        self.loaded = {
            "identity": {
                "theta_ref": self.theta_ref,
                "processor_ref": self.processor_ref,
            },
            "loaded": True,
            "prediction_count": 2,
            "last_actions_finite": True,
            "action_chunk": 5,
        }
        self.manifest = [
            {
                "run_id": "run",
                "task_key": "base_salad",
                "result_path": str(self.root / "result.json"),
            }
        ]

    def inspect(self):
        (self.root / "runtime_context.json").write_text(json.dumps(self.context))
        (self.root / "model_load.json").write_text(json.dumps(self.loaded))
        return inspect_midterm_bindings(self.profile, self.manifest)

    def test_four_arm_snapshot_separates_theta_assets_and_preserves_undeveloped_B1(
        self,
    ):
        changed = deepcopy(self.bindings)
        for binding in changed.values():
            binding["profile"]["model"]["checkpoint"] = "theta1"
        self.assertEqual(freeze_midterm_b0(changed), self.b0)
        self.assertIsNone(self.b0["B1"])
        self.assertEqual(self.b0["B1_status"], "not_developed")
        self.assertTrue(self.inspect()["value"])
        self.bindings["D"]["records"]["rho_ref"]["policy"] = "changed"
        self.assertEqual(self.b0["arms"]["D"]["relations"]["policy"], "fixed")

    def test_actual_loader_identity_and_B0_program_mutation_are_rejected(self):
        self.loaded["identity"]["theta_ref"] = "theta1-displayed-only"
        self.assertFalse(self.inspect()["value"])
        self.loaded["identity"]["theta_ref"] = self.theta_ref
        self.context["execution_binding"]["references"]["phi_ref"]["chunk_steps"] = 3
        self.assertFalse(self.inspect()["value"])

    def test_missing_loader_record_is_unknown_and_never_zero_evidence(self):
        (self.root / "runtime_context.json").write_text(json.dumps(self.context))
        self.assertIsNone(
            inspect_midterm_bindings(self.profile, self.manifest)["value"]
        )

    def test_whole_task_mapping_is_frozen_and_wrong_selected_program_rejects(self):
        profile = self.context["execution_binding"]["profile"]
        profile["execution_arm_by_task"] = {
            "base_salad": {**profile["execution_arm"], "max_chunks": 2},
            "old_milk": {**profile["execution_arm"], "max_chunks": 3},
        }
        self.bindings["D"]["profile"] = deepcopy(profile)
        self.b0 = freeze_midterm_b0(self.bindings)
        self.b0_path.write_text(json.dumps(self.b0))
        self.assertTrue(self.inspect()["value"])
        self.context["execution_arm"] = profile["execution_arm_by_task"]["old_milk"]
        report = self.inspect()
        self.assertFalse(report["value"])
        self.assertFalse(report["runs"][0]["checks"]["task_program"])


if __name__ == "__main__":
    unittest.main()
