"""Finite rho proposals through the existing Research and Candidate consumers."""

from dataclasses import asdict, replace
import unittest

import test_relation_learning
from rsi_vla.contracts import RunBundle
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import BoundedResearchAdapter, ResearchTools
from rsi_vla.storage import atomic_json


class RelationResearchTests(unittest.TestCase):
    make_tools = test_relation_learning.RelationLearningTest.make_tools

    def setUp(self):
        test_relation_learning.RelationLearningTest.setUp(self)
        phi_ref = self.root / "base-phi-v5.json"
        atomic_json(phi_ref, {"schema": "rsi-runtime-rules-v5"})
        self.base = replace(self.base, phi_ref=str(phi_ref))
        self.tools.base = self.base
        atomic_json(self.source.path("bundles", self.base.bundle_id), asdict(self.base))
        view = self.source.get("evidence_views", "view")
        for event in view["l0"]:
            if event["flags"].get("route_feedback"):
                event["flags"]["route_feedback"]["version_signature"]["phi_ref"] = str(phi_ref)
        atomic_json(self.source.path("evidence_views", "view"), view)
        self.protocol["task_policy"] = {"schema": "rsi-task-policy-v1", "relation_learning": {
            "enabled": True, "relations": ["rhoA", "rhoB", "rhoC"], "max_feedback": 64}}
        self.tools.domain_policy["rule_operators"] = ["relation_feedback_update"]
        self.adapter = BoundedResearchAdapter()

    def response(self, key="rho-request", selected=None):
        self.adapter.prepare_request(self.tools, request_id=key, result_ids=["result"],
            max_input_tokens=100, max_output_tokens=10)
        return {"schema": "rsi-bounded-research-response-v2", "request_id": key,
            "decision": "prepare_rule_trial", "source": "failure", "result_ids": ["result"],
            "reason": "test observed relation feedback", "external_value": "reduce repeated evidence",
            "competing_explanations": [{"explanation": "reference duplication", "prediction": "coverage retained",
                                         "distinguishing_test": "matched development evidence and task screening"}],
            "intervention": {"relation_feedback_update": selected or ["rhoA", "rhoC"]},
            "distinguishing_prediction": "coverage retained with fewer references", "stop_condition": "one development slot",
            "missing_evidence": []}

    def apply(self, response):
        return self.adapter.apply_response(self.tools, response, evidence_level="offline",
                                            cost_limit={"input_tokens": 1}, synthetic=True)

    def successful_result(self, response, trial, *, consumption=True):
        identity = {"run_id": "new-run", "episode_id": "new-episode", "bundle_id": trial["bundle_id"],
                    "protocol_id": self.protocol["protocol_id"]}
        checks = {name: True for name in ("native_success", "technical_valid", "runtime_intervention_loaded",
                                         "critical_progress", "collection_complete")}
        intervention = {"information_scope": "development", **response["intervention"],
                        "runtime_loaded": True, "identity": identity, "screening_checks": checks}
        if consumption:
            intervention["relation_consumption"] = {"schema": "rsi-relation-consumption-v1", "identity": identity,
                "rho_ref": trial["rho_ref"], "relations": {name: {"value": True,
                    "source_refs": ["independent-offline-fixture"]} for name in response["intervention"]["relation_feedback_update"]}}
        return replace(self.result, result_id="new-development", bundle_id=trial["bundle_id"],
            intervention=intervention, criteria_ref=self.protocol["protocol_id"], outcome="success",
            costs={"input_tokens": 1}, assignments=[{"task_key": "task", "seed": 1,
                                                     "outcome": "success", "validity": "valid"}])

    def test_relation_schema_uses_supported_api_fields_and_program_rejects_duplicates(self):
        response = self.response("schema-repair")
        schema = self.adapter.response_schema("schema-repair", tools=self.tools)
        def visit(value):
            if isinstance(value, dict):
                self.assertNotIn("uniqueItems", value)
                for child in value.values(): visit(child)
            elif isinstance(value, list):
                for child in value: visit(child)
        visit(schema)
        response["intervention"] = {"relation_feedback_update": ["rhoA", "rhoA"]}
        with self.assertRaisesRegex(ValueError, "supported"):
            self.apply(response)
        self.assertEqual(self.source.all("rule_trials"), [])

    def test_relation_only_keeps_registered_denominator_and_defers_goal_stage(self):
        self.protocol["task_policy"]["opportunity_discovery"] = True
        self.tools.domain_policy.update(relation_only=True, fixed_relation_result_ids=["result"])
        response = self.response("fixed")
        schema = self.adapter.response_schema("fixed", tools=self.tools)
        self.assertNotIn("propose_goal", schema["properties"]["decision"]["enum"])
        self.assertEqual(schema["properties"]["result_ids"]["minItems"], 1)
        response["result_ids"] = []
        with self.assertRaises(ValueError):
            self.apply(response)
        self.assertEqual(self.source.all("rule_trials"), [])

    def test_model_selects_only_supported_relation_and_program_freezes_trial(self):
        response = self.response()
        request = self.source.get("bounded_requests", response["request_id"])
        self.assertEqual(set(request["relation_learning"]["supported_relations"]), {"rhoA", "rhoB", "rhoC"})
        trial = self.apply(response)
        self.assertEqual(trial["runtime_rule_schema"], "rsi-runtime-rules-v5")
        target = RunBundle(**self.source.get("bundles", trial["bundle_id"]))
        self.assertNotEqual(target.rho_ref, self.base.rho_ref)
        self.assertEqual((target.phi_ref, target.psi_ref, target.theta_ref),
                         (self.base.phi_ref, self.base.psi_ref, self.base.theta_ref))
        rho = self.source.get("rho", target.rho_ref)
        self.assertEqual(rho["rhoB"], self.source.get("rho", "rho")["rhoB"])
        self.assertTrue(rho["rhoA"]["values"])
        self.assertFalse(self.source.all("candidates"))
        self.assertFalse(trial["runtime_loaded"])

    def test_opt_in_feedback_boundary_and_model_values_are_enforced(self):
        self.protocol["task_policy"].pop("relation_learning")
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.response()
        self.protocol["task_policy"]["relation_learning"] = {
            "enabled": True, "relations": ["rhoA", "rhoC"], "max_feedback": 64}
        response = self.response("restricted")
        response["intervention"] = {"relation_feedback_update": ["rhoB"]}
        with self.assertRaisesRegex(ValueError, "supported"):
            self.apply(response)
        response = self.response("injected")
        response["intervention"] = {"relation_feedback_update": [{"rhoA": {"total": 999}}]}
        with self.assertRaises(ValueError):
            self.apply(response)
        self.assertFalse(self.source.all("candidates"))

    def test_candidate_cannot_use_old_development_or_missing_consumer_receipt(self):
        response = self.response()
        trial = self.apply(response)
        experiment = response["request_id"] + "-experiment"
        with self.assertRaises(FileNotFoundError):
            self.tools.submit_candidate("premature", experiment_id=experiment)
        result = self.successful_result(response, trial, consumption=False)
        self.tools.record_experiment_result(experiment, result)
        with self.assertRaisesRegex(ValueError, "actual consumer receipts"):
            self.tools.submit_candidate("unconsumed", experiment_id=experiment)
        self.assertFalse(self.source.all("candidates"))

    def test_new_development_and_receipts_allow_actual_rho_candidate_changes(self):
        response = self.response()
        trial = self.apply(response)
        experiment = response["request_id"] + "-experiment"
        self.tools.record_experiment_result(experiment, self.successful_result(response, trial))
        candidate = self.tools.submit_candidate("offline-rho-candidate", experiment_id=experiment)
        self.assertEqual(set(candidate["changes"]), {"rho_ref", "diff_ref"})
        self.assertEqual(candidate["changes"]["rho_ref"], trial["rho_ref"])
        self.assertEqual(candidate["approval_scope"], "offline")
        self.assertEqual(candidate["development_results"], ["new-development"])
        self.assertEqual(candidate["compatibility"]["relation_consumption"]["rho_ref"], trial["rho_ref"])

    def test_additional_development_cannot_bypass_relation_screening(self):
        response = self.response()
        trial = self.apply(response)
        experiment = response["request_id"] + "-experiment"
        good = self.successful_result(response, trial)
        self.tools.record_experiment_result(experiment, good)
        for kind in ("research_experiments", "rule_trials"):
            original = self.tools._owned(kind, experiment)
            self.tools._save(kind, "second-experiment", {**original, "experiment_id": "second-experiment"})
        bad_intervention = {**good.intervention,
            "screening_checks": {**good.intervention["screening_checks"], "native_success": False}}
        bad = replace(good, result_id="second-development", intervention=bad_intervention, outcome="failure")
        self.tools.record_experiment_result("second-experiment", bad)
        with self.assertRaisesRegex(ValueError, "screening"):
            self.tools.submit_candidate("cannot-drop-failed-slot", experiment_id=experiment,
                                        additional_experiment_ids=["second-experiment"])
        self.assertFalse(self.source.all("candidates"))

    def test_learned_rhoC_is_consumed_before_another_trial(self):
        response = self.response()
        trial = self.apply(response)
        target = RunBundle(**self.source.get("bundles", trial["bundle_id"]))
        next_tools = ResearchTools(self.source, ResearchJob(self.source, "next-job", target.bundle_id,
                                    {"input_tokens": 1000, "output_tokens": 100}), self.protocol,
                                  domain_policy=self.tools.domain_policy)
        decision = next_tools.research_relation_admission(["result"], states=["task"], legal_repairs=["relation_feedback_update"])
        self.assertEqual(decision["action"], "defer")
        self.assertFalse(decision["dynamic_execution_authorized"])
        self.assertEqual(next_tools.job.state["used"], {"input_tokens": 0, "output_tokens": 0})

    def test_goal_only_request_has_no_trial_or_candidate_path(self):
        self.protocol["task_policy"]["opportunity_discovery"] = True
        self.tools.domain_policy = {**self.tools.domain_policy, "rule_operators": [], "opportunity_only": True,
            "domain": "tabletop", "allowed_goal_kinds": ["manipulation"], "policy_ref": "independent-domain-policy"}
        self.adapter.prepare_request(self.tools, request_id="goal-only", result_ids=["result"],
                                     max_input_tokens=100, max_output_tokens=10)
        schema = self.adapter.response_schema("goal-only", tools=self.tools)
        self.assertEqual(schema["properties"]["decision"]["enum"], ["propose_goal", "defer"])
        self.assertEqual(schema["properties"]["intervention"], {"anyOf": [{"type": "null"}]})
        response = {"schema": "rsi-bounded-research-response-v3", "request_id": "goal-only",
            "decision": "propose_goal", "source": "new_goal", "result_ids": ["result"],
            "reason": "test an unlisted relation-sensitive goal", "external_value": "independent domain value",
            "competing_explanations": [{"explanation": "new target needs sensing", "prediction": "new criterion differs",
                                         "distinguishing_test": "independent goal review"}],
            "intervention": None, "distinguishing_prediction": "registered independent criterion",
            "stop_condition": "independent goal handoff", "missing_evidence": [],
            "target": {"domain": "tabletop", "kind": "manipulation", "description": "unlisted two-object handoff goal",
                       "value_criterion_ref": "domain-criterion", "independent_value_test_ref": "independent-value-test"}}
        result = self.apply(response)
        self.assertEqual(result["action"], "defer")
        self.assertEqual(self.source.get("opportunities", "goal-only-opportunity")["target"], response["target"])
        self.assertFalse(self.source.all("rule_trials"))
        self.assertFalse(self.source.all("candidates"))
        response["decision"] = "prepare_rule_trial"
        response["source"] = "failure"
        response["target"] = None
        response["intervention"] = {"relation_feedback_update": ["rhoA"]}
        with self.assertRaisesRegex(ValueError, "decision"):
            self.apply(response)


if __name__ == "__main__":
    unittest.main()
