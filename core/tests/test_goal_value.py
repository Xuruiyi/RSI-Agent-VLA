"""Independent static goal review uses typed catalog semantics, not atom counts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.goal_value import evaluate_goal_value
from rsi_vla.task_policy import review_goal_value


def bddl(objects, goal, *, container_type="basket"):
    return f"""(define (problem fixture)
    (:domain robosuite)
    (:fixtures table_1 - table)
    (:objects {objects} container_1 - {container_type})
    (:regions (contain_region (:target container_1)))
    (:init)
    (:goal (And {goal})))"""


class StaticGoalValueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "bddl"
        suite = self.catalog / "public"
        suite.mkdir(parents=True)
        self.original = suite / "original.bddl"
        self.original.write_text(
            bddl("item_a item_b - box", "(In item_a container_1_contain_region)")
        )
        self.witness = suite / "independent.bddl"
        self.witness.write_text(
            bddl(
                "first second - box",
                "(In first container_1_contain_region) (In second container_1_contain_region)",
            )
        )
        self.index = self.root / "catalog.py"
        self.index.write_text(
            "libero_task_map = {'public': ['original', 'independent']}\n"
        )
        self.registry = self.root / "predicates.py"
        self.registry.write_text(
            "VALIDATE_PREDICATE_FN_DICT = {'in': In(), 'on': On(), 'open': Open(), 'close': Close()}\n"
        )
        self.primitives = self.root / "tools.py"
        self.primitives.write_text(
            "class LiberoPrimitives:\n"
            + "".join(
                "    @tool\n    def " + name + "(self): pass\n"
                for name in ["move_to", "set_gripper", "pi0_pick", "release"]
            )
        )
        self.protocol = dict(
            protocol_id="new-protocol",
            task_policy=dict(schema="rsi-task-policy-v1", opportunity_discovery=True),
            tasks=[dict(task_key="current", bddl_relative_path="public/original.bddl")],
        )
        self.policy = dict(
            policy_ref=str(self.root / "policy.json"),
            domain="libero_tabletop",
            allowed_goal_kinds=["bddl_conjunction"],
            value_criterion_ref="functional-value-v1",
            independent_value_test_ref="catalog-test-v1",
            metric="domain_functional_value",
            minimum=1,
            catalog_root=str(self.catalog),
            catalog_index_ref=str(self.index),
            predicate_registry_ref=str(self.registry),
            catalog_suites=["public"],
            capacity_basis="exact_catalog_joint_goal",
            runtime_primitives_ref=str(self.primitives),
        )
        self.description = dict(
            scene_task_key="current",
            all_of=[
                ["in", "item_a", "container_1_contain_region"],
                ["in", "item_b", "container_1_contain_region"],
            ],
            value_claim="proposed utility, not evidence",
        )
        self.target = dict(
            domain="libero_tabletop",
            kind="bddl_conjunction",
            description=json.dumps(self.description),
            value_criterion_ref="functional-value-v1",
            independent_value_test_ref="catalog-test-v1",
        )
        self.opportunity = dict(
            opportunity_id="unlisted",
            protocol_id="new-protocol",
            source="new_goal",
            target=self.target,
            goal_review=dict(policy_ref=self.policy["policy_ref"]),
        )

    def evaluate(self, description=None, *, policy=None):
        opportunity = deepcopy(self.opportunity)
        if description is not None:
            opportunity["target"]["description"] = json.dumps(description)
        return evaluate_goal_value(
            self.protocol, opportunity, policy=policy or self.policy
        )

    def metric(self, evidence):
        return evidence["metrics"]["domain_functional_value"]

    def test_complete_typed_catalog_graph_confirms_functional_value_only(self):
        before = deepcopy(self.opportunity)
        with patch(
            "subprocess.run", side_effect=AssertionError("no external execution")
        ):
            evidence = self.evaluate()
        self.assertEqual(self.opportunity, before)
        self.assertEqual(evidence["validity"], "valid")
        self.assertEqual(self.metric(evidence), 1)
        self.assertEqual(evidence["catalog_witness"]["ref"], str(self.witness))
        self.assertEqual(
            {
                evidence["catalog_witness"]["symbol_mapping"][key]
                for key in ["item_a", "item_b"]
            },
            {"first", "second"},
        )
        self.assertIn(str(self.witness), evidence["evidence_refs"])
        self.assertIsNone(evidence["capability_gain"])
        self.assertIsNone(evidence["task_success"])
        self.assertEqual(evidence["physical_feasibility"], "unmeasured")
        self.assertFalse(evidence["release_eligible"])
        reviewed = review_goal_value(
            self.protocol, self.opportunity, evidence, policy=self.policy
        )
        self.assertEqual(reviewed["value_status"], "confirmed")
        self.assertIsNone(reviewed["capability_gain"])
        self.assertFalse(reviewed["release_eligible"])

    def test_package_root_with_explicit_bddl_directory_preserves_scene_metadata(self):
        package = self.root / "package"
        package.mkdir()
        self.catalog.rename(package / "bddl_files")
        protocol = deepcopy(self.protocol)
        protocol["tasks"][0]["bddl_relative_path"] = "bddl_files/public/original.bddl"
        before = deepcopy(protocol)
        policy = {
            **self.policy,
            "catalog_root": str(package),
            "catalog_directory": "bddl_files",
        }
        evidence = evaluate_goal_value(protocol, self.opportunity, policy=policy)
        self.assertEqual(protocol, before)
        self.assertEqual(self.metric(evidence), 1)
        self.assertEqual(
            evidence["catalog_witness"]["ref"],
            str(package / "bddl_files/public/independent.bddl"),
        )
        self.assertIn(
            str(package / "bddl_files/public/original.bddl"), evidence["evidence_refs"]
        )

    def test_extra_predicates_without_joint_witness_are_inconclusive(self):
        self.witness.write_text(
            bddl("first second - box", "(In first container_1_contain_region)")
        )
        evidence = self.evaluate()
        self.assertIsNone(self.metric(evidence))
        self.assertEqual(evidence["validity"], "unknown")
        self.assertNotIn("catalog_witness", evidence)
        self.assertEqual(
            review_goal_value(
                self.protocol, self.opportunity, evidence, policy=self.policy
            )["value_status"],
            "inconclusive",
        )

    def test_missing_grounding_or_capacity_basis_is_inconclusive(self):
        missing = deepcopy(self.description)
        missing["all_of"][1][1] = "invented_object"
        self.assertIsNone(self.metric(self.evaluate(missing)))
        policy = {k: v for k, v in self.policy.items() if k != "capacity_basis"}
        evidence = self.evaluate(policy=policy)
        self.assertIsNone(self.metric(evidence))
        self.assertIn("capacity", evidence["reason"])
        policy = {k: v for k, v in self.policy.items() if k != "runtime_primitives_ref"}
        evidence = self.evaluate(policy=policy)
        self.assertIsNone(self.metric(evidence))
        self.assertIn("actionability", evidence["reason"])

    def test_shared_support_surface_also_needs_joint_capacity_basis(self):
        self.original.write_text(
            bddl(
                "item_a item_b - box", "(On item_a container_1)", container_type="plate"
            )
        )
        self.witness.write_text(
            bddl(
                "first second - box",
                "(On first container_1) (On second container_1)",
                container_type="plate",
            )
        )
        description = {
            **self.description,
            "all_of": [
                ["on", "item_a", "container_1"],
                ["on", "item_b", "container_1"],
            ],
        }
        policy = {k: v for k, v in self.policy.items() if k != "capacity_basis"}
        self.assertIsNone(self.metric(self.evaluate(description, policy=policy)))
        self.assertEqual(self.metric(self.evaluate(description)), 1)

    def test_object_types_and_shared_container_roles_must_match(self):
        self.witness.write_text(
            bddl(
                "first - box second - bottle",
                "(In first container_1_contain_region) (In second container_1_contain_region)",
            )
        )
        self.assertIsNone(self.metric(self.evaluate()))
        self.witness.write_text("""(define (problem fixture)
            (:fixtures table_1 - table)
            (:objects first second - box container_1 container_2 - basket)
            (:regions (contain_region (:target container_1)) (contain_region (:target container_2)))
            (:goal (And (In first container_1_contain_region) (In second container_2_contain_region))))""")
        self.assertIsNone(self.metric(self.evaluate()))

    def test_repeated_original_goal_or_out_of_scope_scene_is_rejected(self):
        repeated = deepcopy(self.description)
        repeated["all_of"] = [repeated["all_of"][0], repeated["all_of"][0]]
        self.assertEqual(self.metric(self.evaluate(repeated)), 0)
        outside = {**self.description, "scene_task_key": "outside"}
        self.assertEqual(self.metric(self.evaluate(outside)), 0)

    def test_conflicting_actuator_states_and_self_containment_are_rejected(self):
        conflicting = {
            **self.description,
            "all_of": [
                ["open", "container_1_contain_region"],
                ["close", "container_1_contain_region"],
            ],
        }
        self.assertEqual(self.metric(self.evaluate(conflicting)), 0)
        self_goal = {
            **self.description,
            "all_of": [["in", "container_1", "container_1_contain_region"]],
        }
        self.assertEqual(self.metric(self.evaluate(self_goal)), 0)

    def test_unused_orphan_region_does_not_block_grounded_goal(self):
        text = self.original.read_text().replace(
            "(:regions (contain_region (:target container_1)))",
            "(:regions (contain_region (:target container_1)) (unused_region (:target absent_fixture)))",
        )
        self.original.write_text(text)
        evidence = self.evaluate()
        self.assertEqual(self.metric(evidence), 1)
        self.assertEqual(evidence["checks"]["grounded"], True)
        self.assertEqual(evidence["catalog_witness"]["ref"], str(self.witness))

    def test_referenced_orphan_region_remains_unknown(self):
        text = self.original.read_text().replace(
            "(:regions (contain_region (:target container_1)))",
            "(:regions (contain_region (:target container_1)) (unused_region (:target absent_fixture)))",
        )
        self.original.write_text(text)
        description = {
            **self.description,
            "all_of": [["in", "item_a", "absent_fixture_unused_region"]],
        }
        evidence = self.evaluate(description)
        self.assertIsNone(self.metric(evidence))
        self.assertEqual(evidence["validity"], "unknown")
        self.assertIsNone(evidence["checks"]["grounded"])
        self.assertNotIn("catalog_witness", evidence)

    def test_malformed_metadata_never_falls_back_to_another_scene(self):
        self.original.write_text("(define (problem incomplete)")
        evidence = self.evaluate()
        self.assertIsNone(self.metric(evidence))
        self.assertEqual(evidence["validity"], "unknown")
        self.assertNotIn("catalog_witness", evidence)

    def test_description_and_registry_reject_unsupported_predicates(self):
        for atom in [["true"], ["stack", "item_a", "item_b"], ["in", "item_a"]]:
            with self.subTest(atom=atom):
                self.assertEqual(
                    self.metric(self.evaluate({**self.description, "all_of": [atom]})),
                    0,
                )
        self.registry.write_text("VALIDATE_PREDICATE_FN_DICT = {}\n")
        self.assertEqual(self.metric(self.evaluate()), 0)

    def test_only_exact_registered_representation_policy_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.evaluate(policy={**self.policy, "allowed_goal_kinds": ["preset_goal"]})
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.evaluate(policy={**self.policy, "metric": "predicate_count"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
