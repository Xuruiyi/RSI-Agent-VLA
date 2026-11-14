"""CPU contracts for finite production skill routes and independent handoff facts."""

import copy
import unittest

from rsi_vla.cooperation import prepare_skill_routes, review_skill_handoff
from rsi_vla.relation_learning import SCHEMA


SIGNATURE = {"theta_ref": "theta", "processor_ref": "processor", "phi_ref": "phi", "code_ref": "code"}
CRITERION = {"schema": "rsi-skill-lift-criterion-v1", "criterion_ref": "registered-object-lift-v1",
             "object_position_key": "salad_dressing_1_pos", "min_lift_m": .03,
             "max_eef_object_distance_m": .1, "min_gripper_opening": .005,
             "max_gripper_opening": .06}
ROUTE = {"route_id": "pick-salad", "state": "salad:pick", "requested_worker": "vla_execute",
         "trigger_workers": ["pi0_pick", "vla_execute"], "instruction": "pick up the salad dressing",
         "allowed_workers": ["pi0_pick", "vla_execute", "handoff", "defer"], "criterion": CRITERION}


class SkillRouteConsumerTests(unittest.TestCase):
    def snapshot(self):
        return {"rhoB": {"type": SCHEMA, "version_signatures": {"salad:pick": SIGNATURE},
                         "values": {"salad:pick": {"pi0_pick": {"count": 1, "total": 1.},
                                                   "vla_execute": {"count": 1, "total": 0.}}}}}

    def project(self, snapshot=None, route=None, signature=None):
        return prepare_skill_routes(self.snapshot() if snapshot is None else snapshot,
            rho_ref="rho-measured", version_signature=signature or SIGNATURE,
            registered_routes=[copy.deepcopy(ROUTE if route is None else route)])

    def test_learned_choice_projects_only_same_goal_existing_skill(self):
        control = self.project()
        self.assertEqual(control["routes"][0]["selected_worker"], "pi0_pick")
        self.assertEqual(control["routes"][0]["instruction"], ROUTE["instruction"])
        self.assertEqual(control["version_signature"], SIGNATURE)
        self.assertNotIn("selected_worker", ROUTE)

    def test_fixed_and_incompatible_versions_preserve_requested_skill(self):
        fixed = {"rhoB": {"type": "fixed", "route": "existing_authorized_worker"}}
        self.assertEqual(self.project(fixed)["routes"][0]["selected_worker"], "vla_execute")
        self.assertEqual(self.project(signature={**SIGNATURE, "phi_ref": "changed"})["routes"][0]["selected_worker"], "vla_execute")

    def test_permission_and_measured_handoff_choices_reach_projection(self):
        route = {**ROUTE, "allowed_workers": []}
        self.assertEqual(self.project(route=route)["routes"][0]["selected_worker"], "defer")
        snapshot = self.snapshot()
        snapshot["rhoB"]["values"]["salad:pick"]["handoff"] = {"count": 1, "total": 1.}
        snapshot["rhoB"]["values"]["salad:pick"]["pi0_pick"]["total"] = 0.
        self.assertEqual(self.project(snapshot)["routes"][0]["selected_worker"], "handoff")
        for change in ({"requested_worker": "pi0_doubled"}, {"trigger_workers": ["move_to"]},
                       {"allowed_workers": ["new-expert"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.project(route={**ROUTE, **change})

    def observations(self):
        common = {"kind": "skill_route_observation", "clock_domain": "environment_service",
                  "run_id": "r", "episode_id": "e", "bundle_id": "b", "protocol_id": "p",
                  "route_id": "pick-salad", "state": "salad:pick", "worker": "pi0_pick",
                  "client_route_sequence": 8, "criterion_ref": CRITERION["criterion_ref"],
                  "object_position_key": CRITERION["object_position_key"], "version_signature": SIGNATURE}
        before = {**common, "sequence": 5, "phase": "start",
                  "measurements": {"object_pos": [0., 0., .1], "eef_pos": [0., 0., .2], "gripper_opening": .08}}
        after = {**common, "sequence": 20, "phase": "handoff",
                 "measurements": {"object_pos": [0., 0., .2], "eef_pos": [0., 0., .21], "gripper_opening": .02}}
        return before, after

    def test_independent_geometry_success_failure_and_missing_ignore_task_labels(self):
        before, after = self.observations()
        after.update(terminated=False, success=False)
        report = review_skill_handoff(before, after, CRITERION)
        self.assertIs(report["success"], True)
        after["measurements"]["object_pos"][2] = .1
        after.update(terminated=True, success=True)
        self.assertIs(review_skill_handoff(before, after, CRITERION)["success"], False)
        after["measurements"]["object_pos"] = None
        self.assertIsNone(review_skill_handoff(before, after, CRITERION)["success"])
        self.assertIn("handoff:object_pos", review_skill_handoff(before, after, CRITERION)["missing"])

    def test_independent_review_rejects_cross_identity_and_fabricated_source(self):
        before, after = self.observations()
        for change in ({"worker": "vla_execute"}, {"client_route_sequence": 99},
                       {"criterion_ref": "other"}, {"clock_domain": "runtime_client"},
                       {"sequence": 4}, {"version_signature": {**SIGNATURE, "phi_ref": "other"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                review_skill_handoff(before, {**after, **change}, CRITERION)
