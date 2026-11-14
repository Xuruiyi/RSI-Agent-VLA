"""Registered drawer routing, private joint feedback and original control closure."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_relation_evidence
import test_routed_progress
from rsi_vla.cooperation import prepare_skill_routes
from rsi_vla.relation_evidence import attach_skill_feedback, review_skill_feedback
from rsi_vla.relation_learning import SCHEMA, learn_relations
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.critical_progress import audit_skill_routes, inspect_progress

DRAWER = {"schema": "rsi-skill-drawer-open-criterion-v1", "criterion_ref": "drawer-criterion",
          "region": "wooden_cabinet_1_middle_region", "fixture": "wooden_cabinet_1",
          "joint": "wooden_cabinet_1_middle_level", "joint_type": "slide",
          "joint_range": [-.16, .01], "open_threshold": -.14}
GOAL = "Open the middle layer of the drawer"


def drawer_route(criterion=None):
    return {"route_id": "drawer", "task_key": "old_drawer", "state": "old_drawer",
            "requested_worker": "vla_execute", "trigger_workers": ["vla_execute", "pi0_doubled"],
            "allowed_workers": ["vla_execute", "pi0_doubled", "handoff", "defer"],
            "instruction": GOAL, "max_chunks": 24, "criterion": deepcopy(criterion or DRAWER)}


class DrawerFeedbackTests(unittest.TestCase):
    write_streams = test_relation_evidence.SkillFeedbackTests.write_streams

    def setUp(self):
        test_relation_evidence.SkillFeedbackTests.setUp(self)
        self.criterion = {**deepcopy(DRAWER), "criterion_ref": self.criterion["criterion_ref"]}
        atomic_json(self.criterion["criterion_ref"], self.criterion)
        route = {**drawer_route(self.criterion), "selected_worker": "pi0_doubled"}
        self.context["relation_control"]["routes"] = [route]
        atomic_json(self.output / "runtime_context.json", self.context)
        self.client[0]["route_ids"] = ["drawer"]
        for event in self.client[1:]:
            event.update(route_id="drawer", state="old_drawer", requested_worker="vla_execute",
                         selected_worker="pi0_doubled", trigger_worker="vla_execute")
        for event in self.environment:
            event.pop("object_position_key", None)
            event.update(route_id="drawer", state="old_drawer", worker="pi0_doubled")
            event["measurements"] = {"joint": self.observation(0 if event["phase"] == "start" else -.15)}
        self.write_streams()
        self.result = replace(self.result, result_id="drawer-result", assignments=[{
            "task_key": "old_drawer", "seed": 1, "validity": "valid", "outcome": "success"}])
        self.store.save_development_result(self.result)

    def observation(self, qpos):
        return {"schema": "rsi-private-drawer-joint-observation-v1", "status": "recorded",
                **{key: self.criterion[key] for key in ("region", "fixture", "joint", "joint_type", "joint_range")},
                "qpos": qpos, "missing": []}

    def test_strict_open_threshold_and_already_open_start_are_not_task_labels(self):
        self.assertIs(review_skill_feedback(self.output)["reviews"][0]["success"], True)
        for qpos, expected in ((-.14, False), (-.139, False), (-.141, True)):
            self.environment[1]["measurements"]["joint"]["qpos"] = qpos
            self.environment[1].update(success=True, terminated=True)
            self.write_streams()
            self.assertIs(review_skill_feedback(self.output)["reviews"][0]["success"], expected)
        self.environment[0]["measurements"]["joint"]["qpos"] = -.15
        self.write_streams()
        review = review_skill_feedback(self.output)["reviews"][0]
        self.assertIsNone(review["success"])
        self.assertEqual(review["eligibility"], "not_eligible_start_already_open")

    def test_missing_or_wrong_joint_mapping_is_unknown_without_boolean_training(self):
        original = deepcopy(self.environment[1]["measurements"]["joint"])
        for update in ({"joint": "wooden_cabinet_1_top_level"}, {"region": "another_region"},
                       {"joint_type": "hinge"}, {"joint_range": [0, 1]}, {"qpos": None},
                       {"status": "unknown"}, {"missing": ["ambiguous_joint"]}):
            self.environment[1]["measurements"]["joint"] = {**original, **update}
            self.write_streams()
            self.assertIsNone(review_skill_feedback(self.output)["reviews"][0]["success"])
        report = review_skill_feedback(self.output)
        enriched = attach_skill_feedback(self.store, self.result, report, self.root / "joint-review.json")
        view = self.store.get("evidence_views", enriched.intervention["evidence_view_id"])
        self.assertFalse(any("route_feedback" in event["flags"] for event in view["l0"]))

    def test_independent_joint_feedback_reaches_existing_learner_with_original_prefix(self):
        report = review_skill_feedback(self.output)
        enriched = attach_skill_feedback(self.store, self.result, report, self.root / "joint-review.json")
        original = self.store.get("evidence_views", "view")
        view = self.store.get("evidence_views", enriched.intervention["evidence_view_id"])
        self.assertEqual(view["l0"][:len(original["l0"])], original["l0"])
        protocol = {"protocol_id": "protocol", "benchmark_id": "bench", "scope": "offline"}
        tools = ResearchTools(self.store, ResearchJob(self.store, "joint-job", "base", {"steps": 1}), protocol, domain_policy={})
        learned = learn_relations(tools, ResearchStore(self.root / "learned"), "drawer", [enriched.result_id], relations=["rhoB"])
        self.assertEqual(learned["snapshot"]["rhoB"]["values"]["old_drawer"]["pi0_doubled"], {"count": 1, "total": 1.})

    def test_drawer_group_and_new_executable_keep_old_signatures_frozen(self):
        old = {**self.signature, "code_ref": "old-code"}
        rho = {"rhoB": {"type": SCHEMA, "version_signatures": {"old_drawer": old},
            "values": {"old_drawer": {"pi0_doubled": {"count": 1, "total": 1.},
                                      "vla_execute": {"count": 1, "total": 0.}}}}}
        before = deepcopy(rho)
        control = prepare_skill_routes(rho, rho_ref="learned", version_signature=self.signature,
                                        registered_routes=[drawer_route()])
        self.assertEqual(control["routes"][0]["selected_worker"], "vla_execute")
        self.assertEqual(rho, before)
        for change in ({"requested_worker": "pi0_pick"}, {"max_chunks": 12},
                       {"allowed_workers": ["pi0_pick"]}, {"trigger_workers": ["pi0_pick"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_skill_routes(rho, rho_ref="learned", version_signature=self.signature,
                                     registered_routes=[{**drawer_route(), **change}])


class DrawerRoutedProgressTests(unittest.TestCase):
    write = test_routed_progress.RoutedProgressTests.write

    def setUp(self):
        test_routed_progress.RoutedProgressTests.setUp(self)
        source_root = Path(__file__).resolve().parents[2] / "RPent/robots/libero"
        (self.root / "code_snapshot/RPent/robots/libero/drawer_observation.py").write_bytes(
            (source_root / "drawer_observation.py").read_bytes())
        context = json.loads((self.root / "runtime_context.json").read_text())
        context["relation_control"]["routes"] = [{**drawer_route(), "selected_worker": "pi0_doubled"}]
        self.write("runtime_context.json", context)
        for event in self.events:
            if "route_id" in event:
                event.update(route_id="drawer", state="old_drawer", selected_worker="pi0_doubled",
                             criterion_ref=DRAWER["criterion_ref"])
            if "actual_arguments" in event:
                event["actual_arguments"] = {"prompt": GOAL, "max_chunks": 24}
            if "instruction" in event:
                event["instruction"] = GOAL
        self.states["steps"][0]["command"] = {"action": "vla_execute", "instruction": GOAL, "max_chunks": 24}
        self.states["steps"][0]["result"]["relation_route"].update(
            route_id="drawer", state="old_drawer", selected_worker="pi0_doubled", criterion_ref=DRAWER["criterion_ref"])
        private = [json.loads(line) for line in (self.root / "environment_service/action_events.jsonl").read_text().splitlines()]
        for event in private:
            event.update(route_id="drawer", worker="pi0_doubled", criterion_ref=DRAWER["criterion_ref"])
        self.write("environment_service/action_events.jsonl", private, lines=True)

    def test_registered_contact_uses_real_prompt_no_segment_and_existing_control_mapping(self):
        self.write("states.json", self.states)
        self.write("action_events.jsonl", self.events, lines=True)
        checks = {name: True for name in ("ordered_l0", "execution_health", "action_mapping", "setup_coverage",
                                         "no_mid_attempt_reset", "vla_actual_loading_and_prediction")}
        with patch("rsi_vla.adapters.runtime_acceptance.inspect_attempt", return_value={"checks": checks}):
            report = inspect_progress(self.root, self.identity)
        self.assertIs(report["value"], True, report)
        self.assertTrue(report["checks"]["proposal_to_native_send"])
        self.assertFalse(any(event["kind"] == "vla_segment_start" for event in self.events))
        self.assertEqual(len(report["skill_routes"]["missing"]), 0)

    def test_contact_argument_allowance_segment_and_chunk_gaps_stay_unknown(self):
        for index, field, value in ((5, "actual_arguments", {"instruction": GOAL, "max_chunks": 24}),
                                    (5, "actual_arguments", {"prompt": GOAL, "max_chunks": 12}),
                                    (6, "segment_sequence", 4), (11, "kind", "vla_segment_start"),
                                    (10, "unknown_steps", 1)):
            with self.subTest(field=field):
                events = deepcopy(self.events)
                events[index][field] = value
                self.assertIsNone(audit_skill_routes(self.root, self.identity, events, self.states)["value"])


if __name__ == "__main__":
    unittest.main()
