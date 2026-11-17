"""An unlisted conjunction is physically materialized without running a simulator."""

import json
import tempfile
import unittest
from pathlib import Path

from rsi_vla.goal_value import _atoms, _scene, evaluate_goal_value
from rsi_vla.trusted_goal import materialize_goal


def goal_fixture(root):
    scene = root / "original.bddl"
    scene.write_text("""(define (problem Libero_Tabletop_Manipulation)
    (:domain robosuite) (:language put first into basket)
    (:fixtures table_1 - table) (:objects first second - box basket_1 - basket)
    (:regions (contain_region (:target basket_1)))
    (:init (On first table_1) (On second table_1))
    (:goal (And (In first basket_1_contain_region))))""")
    registry = root / "predicates.py"
    registry.write_text("VALIDATE_PREDICATE_FN_DICT = {'in': In()}\n")
    initial = root / "scene.pruned_init"
    initial.write_bytes(b"isolated synthetic initial states")
    policy_ref = root / "policy.json"
    policy = {
        "policy_ref": str(policy_ref),
        "domain": "libero",
        "allowed_goal_kinds": ["bddl_conjunction"],
        "value_criterion_ref": "fixed-functional-rule",
        "independent_value_test_ref": "fixed-test",
        "metric": "domain_functional_value",
        "minimum": 1,
        "catalog_root": str(root),
        "predicate_registry_ref": str(registry),
        "composition_rules": {
            "schema": "rsi-bddl-composition-rules-v1",
            "functional_value_basis": "externally specified joint packing objective",
            "scene_task_keys": ["original"],
            "max_atoms": 2,
            "predicates": ["in"],
            "allowed_symbols": {
                "original": ["first", "second", "basket_1_contain_region"]
            },
            "destination_capacity": {"basket_1_contain_region": 2},
        },
        "materialization": {
            "backend": "libero_bddl",
            "initial_states": {"original": str(initial)},
        },
    }
    policy_ref.write_text(json.dumps(policy))
    protocol = {
        "protocol_id": "original-v1",
        "status": "frozen",
        "scope": "offline",
        "benchmark_id": "original-release",
        "tasks": [
            {
                "task_key": "original",
                "group": "base",
                "suite": "catalog",
                "task_id": 0,
                "bddl_relative_path": "original.bddl",
            }
        ],
        "task_sets": {
            role: {
                "task_keys": ["original"],
                "seeds": [0],
                "benchmark_id": "original-" + role,
            }
            for role in ["development", "release"]
        },
    }
    description = {
        "scene_task_key": "original",
        "all_of": [
            ["in", "first", "basket_1_contain_region"],
            ["in", "second", "basket_1_contain_region"],
        ],
    }
    opportunity = {
        "opportunity_id": "joint-packing",
        "protocol_id": "original-v1",
        "source": "new_goal",
        "target": {
            "domain": "libero",
            "kind": "bddl_conjunction",
            "description": json.dumps(description),
            "value_criterion_ref": "fixed-functional-rule",
            "independent_value_test_ref": "fixed-test",
        },
        "goal_review": {"policy_ref": str(policy_ref)},
    }
    return protocol, opportunity, policy


class GoalMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p, self.o, self.policy = goal_fixture(self.root)

    def materialize(self, **kwargs):
        return materialize_goal(
            self.p,
            self.o,
            policy=kwargs.get("policy", self.policy),
            output_root=self.root / "output",
            new_protocol_id=kwargs.get("new_protocol_id", "new-v1"),
        )

    def test_unlisted_combination_materializes_bddl_init_identity_and_new_matrix(self):
        evidence = evaluate_goal_value(self.p, self.o, policy=self.policy)
        self.assertEqual(evidence["metrics"]["domain_functional_value"], 1)
        receipt = self.materialize()
        task = receipt["task"]
        self.assertEqual(
            _atoms(_scene(Path(receipt["bddl_ref"]).read_text())["goal"]),
            _atoms(json.loads(self.o["target"]["description"])["all_of"]),
        )
        self.assertEqual(
            Path(task["initial_state_ref"]).read_bytes(),
            (self.root / "scene.pruned_init").read_bytes(),
        )
        self.assertEqual(
            receipt["protocol"]["task_sets"]["release"]["task_keys"],
            ["original", "joint-packing"],
        )
        self.assertEqual(task["native_success"], "bddl_goal")
        self.assertEqual(self.materialize(), receipt)
        self.assertEqual(self.p["task_sets"]["release"]["task_keys"], ["original"])
        self.assertEqual(receipt["physical_feasibility"], "unmeasured")

    def test_public_task_receipt_omits_hidden_protocol_and_private_evidence(self):
        self.p["hidden_exam_labels"] = {"sealed": "PRIVATE-MATERIALIZATION-LABEL"}
        materialized = self.materialize()
        public = json.loads(Path(materialized["public_receipt_ref"]).read_text())
        private = json.loads(Path(materialized["private_receipt_ref"]).read_text())
        self.assertNotIn("protocol", public)
        self.assertNotIn("protocol_ref", public)
        self.assertNotIn("evidence", public)
        self.assertNotIn("PRIVATE-MATERIALIZATION-LABEL", json.dumps(public))
        self.assertEqual(
            private["protocol"]["hidden_exam_labels"], self.p["hidden_exam_labels"]
        )
        self.assertEqual(
            public["task"]["materialization_ref"], materialized["public_receipt_ref"]
        )

    def test_rule_tamper_and_missing_backend_cannot_admit(self):
        self.policy["composition_rules"]["max_atoms"] = 3
        self.assertIsNone(
            evaluate_goal_value(self.p, self.o, policy=self.policy)["metrics"][
                "domain_functional_value"
            ]
        )
        self.policy["composition_rules"]["max_atoms"] = 2
        del self.policy["materialization"]
        Path(self.policy["policy_ref"]).write_text(json.dumps(self.policy))
        with self.assertRaisesRegex(ValueError, "backend"):
            self.materialize()

    def test_protocol_reuse_out_of_scope_and_capacity_rejected(self):
        with self.assertRaisesRegex(ValueError, "new protocol"):
            self.materialize(new_protocol_id="original-v1")
        self.policy["composition_rules"]["destination_capacity"][
            "basket_1_contain_region"
        ] = 1
        Path(self.policy["policy_ref"]).write_text(json.dumps(self.policy))
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.materialize()


if __name__ == "__main__":
    unittest.main()
