"""Independent private-observation feedback and immutable development attachment."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from rsi_vla.contracts import ExecutionEvent, ExperimentResult, RunBundle
from rsi_vla.evidence import EventLedger
from rsi_vla.relation_evidence import attach_skill_feedback, review_skill_feedback
from rsi_vla.relation_learning import learn_relations, version_signature
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json, initialize


class SkillFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "runtime"
        self.store = ResearchStore(self.root / "research")
        self.bundle = RunBundle("base", "theta", "processor", "phi", "rho", "psi", "memory", "protocol", "backend", "code")
        initialize(self.store, self.bundle, psi={}, memory={"provenance": "fixture", "legal_information": ["development"]})
        self.signature = version_signature(self.bundle)
        self.identity = {"run_id": "run", "episode_id": "episode", "bundle_id": "base", "protocol_id": "protocol"}
        criterion_path = self.root / "criterion.json"
        self.criterion = {"schema": "rsi-skill-lift-criterion-v1", "criterion_ref": str(criterion_path),
            "object_position_key": "target_pos", "min_lift_m": .05, "max_eef_object_distance_m": .05,
            "min_gripper_opening": .01, "max_gripper_opening": .08}
        atomic_json(criterion_path, self.criterion)
        route = {"route_id": "pick", "state": "task", "requested_worker": "pi0_pick", "selected_worker": "pi0_pick",
                 "trigger_workers": ["pi0_pick"], "allowed_workers": ["pi0_pick", "vla_execute"],
                 "instruction": "pick the registered target", "criterion": self.criterion}
        self.context = {**self.identity, **self.signature, "rho_ref": "rho", "information_scope": "development",
            "relation_control": {"schema": "rsi-skill-routing-v1", "rho_ref": "rho",
                "version_signature": self.signature, "routes": [route]}}
        atomic_json(self.output / "runtime_context.json", self.context)
        binding = {"rho_ref": "rho", "route_id": "pick", "state": "task", "requested_worker": "pi0_pick",
                   "selected_worker": "pi0_pick", "trigger_worker": "pi0_pick", "criterion_ref": str(criterion_path)}
        self.client = [{**self.identity, "sequence": 0, "kind": "relation_policy_loaded", "relation": "rhoB",
                        "rho_ref": "rho", "version_signature": self.signature, "route_ids": ["pick"]},
                       {**self.identity, **binding, "sequence": 1, "kind": "relation_route_decision"}]
        for index, phase in enumerate(("start", "handoff")):
            self.client.append({**self.identity, **binding, "sequence": index + 2, "kind": "skill_route_" + phase,
                "decision_sequence": 1, "client_route_sequence": 1,
                "environment_observation": {"source_sequence": index + 10, "status": "recorded"}})
        self.environment = []
        for index, phase in enumerate(("start", "handoff")):
            z = .1 + index * .1
            self.environment.append({**self.identity, "kind": "skill_route_observation", "sequence": index + 10,
                "clock_domain": "environment_service", "phase": phase, "route_id": "pick", "state": "task",
                "worker": "pi0_pick", "client_route_sequence": 1, "criterion_ref": str(criterion_path),
                "object_position_key": "target_pos", "version_signature": self.signature,
                "measurements": {"object_pos": [0, 0, z], "eef_pos": [0, 0, z + .01], "gripper_opening": .03}})
        self.write_streams()
        ledger = EventLedger(self.store)
        ledger.append(ExecutionEvent("original", "run", "episode", 0, "base", "observation", "fixture", media=["frame.png"]))
        self.original_view = ledger.view("view", "base")
        self.store.put("evidence_views", "view", self.original_view)
        self.result = ExperimentResult("result", "base", "protocol", "bench", "development", "offline",
            {"information_scope": "development", "identity": self.identity, "evidence_view_id": "view"},
            ["independent-original"], "criterion", {}, {}, {}, assigned_count=1, outcome="success",
            assignments=[{"task_key": "task", "seed": 1, "validity": "valid", "outcome": "success"}])
        self.store.save_development_result(self.result)

    def write_streams(self):
        for path, rows in ((self.output / "action_events.jsonl", self.client),
                           (self.output / "environment_service/action_events.jsonl", self.environment)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_success_and_failure_are_recomputed_from_private_geometry(self):
        self.assertIs(review_skill_feedback(self.output)["reviews"][0]["success"], True)
        self.environment[1]["measurements"]["object_pos"][2] = .1
        self.environment[1]["measurements"]["eef_pos"][2] = .11
        self.write_streams()
        review = review_skill_feedback(self.output)["reviews"][0]
        self.assertIs(review["success"], False)
        self.assertIs(review["checks"]["object_lift"], False)

    def test_unknown_measurements_and_missing_events_remain_null(self):
        self.environment[1]["measurements"]["object_pos"] = None
        self.write_streams()
        self.assertIsNone(review_skill_feedback(self.output)["reviews"][0]["success"])
        self.environment = self.environment[:1]
        self.write_streams()
        review = review_skill_feedback(self.output)["reviews"][0]
        self.assertIsNone(review["success"])
        self.assertIn("environment handoff", review["missing"])

    def test_identity_sequence_version_and_criterion_conflicts_are_rejected(self):
        baseline = deepcopy(self.environment)
        for change in ({"run_id": "other"}, {"version_signature": {"phi_ref": "other"}},
                       {"client_route_sequence": 999}, {"worker": "vla_execute"}):
            self.environment = deepcopy(baseline)
            self.environment[1].update(change)
            self.write_streams()
            with self.assertRaises(ValueError):
                review_skill_feedback(self.output)
        self.environment = baseline
        self.write_streams()
        atomic_json(self.criterion["criterion_ref"], {**self.criterion, "min_lift_m": .001})
        with self.assertRaisesRegex(ValueError, "criterion"):
            review_skill_feedback(self.output)

    def test_attachment_preserves_original_facts_and_exposes_independent_learning(self):
        report = review_skill_feedback(self.output)
        report_ref = self.root / "evaluation/review.json"
        atomic_json(report_ref, report)
        enriched = attach_skill_feedback(self.store, self.result, report, report_ref)
        self.assertEqual(enriched.result_id, "result-relations")
        original = self.store.get("evidence_views", "view")
        derived = self.store.get("evidence_views", "view-relations")
        self.assertEqual(derived["l0"][:len(original["l0"])], original["l0"])
        self.assertEqual(self.store.get("development_results", "result"), asdict(self.result))
        self.assertEqual(self.store.get("evidence_views", "view"), asdict(self.original_view))
        feedback = derived["l0"][-1]["flags"]["route_feedback"]
        self.assertTrue(feedback["success"])
        self.assertEqual(derived["l0"][-1]["flags"]["environment_source_sequences"], {"start": 10, "handoff": 11})
        protocol = {"protocol_id": "protocol", "benchmark_id": "bench", "scope": "offline"}
        tools = ResearchTools(self.store, ResearchJob(self.store, "job", "base", {"steps": 1}), protocol, domain_policy={})
        state = learn_relations(tools, ResearchStore(self.root / "learned"), "learn", [enriched.result_id], relations=["rhoB"])
        self.assertEqual(state["snapshot"]["rhoB"]["values"]["task"]["pi0_pick"], {"count": 1, "total": 1.0})
        self.assertEqual(attach_skill_feedback(self.store, self.result, report, report_ref), enriched)

    def test_same_raw_handoff_is_not_counted_again_for_a_result_alias(self):
        report = review_skill_feedback(self.output)
        report_ref = self.root / "evaluation/review.json"
        atomic_json(report_ref, report)
        original = attach_skill_feedback(self.store, self.result, report, report_ref)
        alias = replace(self.result, result_id="result-reparsed")
        self.store.save_development_result(alias)
        reparsed = attach_skill_feedback(self.store, alias, report, report_ref)
        protocol = {"protocol_id": "protocol", "benchmark_id": "bench", "scope": "offline"}
        tools = ResearchTools(self.store, ResearchJob(self.store, "job", "base", {"steps": 1}), protocol, domain_policy={})
        state = learn_relations(tools, ResearchStore(self.root / "learned"), "learn",
                                [original.result_id, reparsed.result_id], relations=["rhoB"])
        self.assertEqual(state["snapshot"]["rhoB"]["values"]["task"]["pi0_pick"], {"count": 1, "total": 1.0})
        self.assertEqual(len(state["feedback"]), 1)

    def test_unknown_feedback_does_not_create_boolean_training_labels(self):
        self.environment[1]["measurements"]["eef_pos"] = None
        self.write_streams()
        report = review_skill_feedback(self.output)
        enriched = attach_skill_feedback(self.store, self.result, report, self.root / "review.json")
        view = self.store.get("evidence_views", enriched.intervention["evidence_view_id"])
        self.assertTrue(all("route_feedback" not in event["flags"] for event in view["l0"]))
        report["identity"] = {**self.identity, "bundle_id": "other"}
        with self.assertRaisesRegex(ValueError, "match"):
            attach_skill_feedback(self.store, self.result, report, self.root / "wrong.json")


if __name__ == "__main__":
    unittest.main()
