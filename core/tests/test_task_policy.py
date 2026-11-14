"""CPU regressions for new-task scope, staged admission and exploration consumers."""

import copy
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_batch4
import test_improvement_protocol
import test_research_preflight
import test_research_tools
from rsi_vla.contracts import EvidenceView
from rsi_vla.improvement_protocol import admit_resources, baseline_readiness
from rsi_vla.research_tools import BoundedResearchAdapter, ResearchToolkit
from rsi_vla.research_transport import model_prompt, pack, select_input, unpack
from rsi_vla.storage import atomic_json
from rsi_vla.task_policy import resource_wait, select_gpu, task_policy

ROOT = Path(__file__).resolve().parents[2]
POLICY = {"schema": "rsi-task-policy-v1"}


class AdmissionTests(unittest.TestCase):
    setUp = test_improvement_protocol.ImprovementProtocolTest.setUp
    phase = test_improvement_protocol.ImprovementProtocolTest.phase
    baseline = test_improvement_protocol.ImprovementProtocolTest.baseline

    def new_protocol(self):
        p = copy.deepcopy(self.p)
        p.update(
            protocol_id="new-objective", scope_id="new-task", task_policy=dict(POLICY)
        )
        return p

    def test_staged_admits_necessary_prefix_without_full_conditional_envelope(self):
        phase = self.phase()
        phase["limits"]["simulation_episodes"] = phase["remaining"][
            "simulation_episodes"
        ] = 1
        original = copy.deepcopy(phase)
        self.assertFalse(
            admit_resources(
                self.p, phase, offline_used_seconds=0, offline_used_bytes=0
            )["resource_fit"]
        )
        p = self.new_protocol()
        p["task_policy"]["resource_admission"] = "staged"
        report = admit_resources(
            p,
            phase,
            offline_used_seconds=0,
            offline_used_bytes=0,
            next_required={"simulation_episodes": 1, "wall_seconds": 900},
            cleanup={"wall_seconds": 10},
            reserve={},
        )
        self.assertTrue(report["resource_fit"])
        self.assertEqual(report["required_now"]["wall_seconds"], 910)
        self.assertFalse(report["scope_completion_guaranteed"])
        self.assertEqual(phase, original)
        phase["remaining"]["wall_seconds"] = phase["limits"]["wall_seconds"] = 909
        self.assertFalse(
            admit_resources(
                p,
                phase,
                offline_used_seconds=0,
                offline_used_bytes=0,
                next_required={"wall_seconds": 900},
                cleanup={"wall_seconds": 10},
                reserve={},
            )["resource_fit"]
        )
        phase["unresolved_costs"] = ["unknown submitted request"]
        with self.assertRaisesRegex(ValueError, "unresolved"):
            admit_resources(
                p,
                phase,
                offline_used_seconds=0,
                offline_used_bytes=0,
                next_required={"wall_seconds": 1},
                cleanup={},
                reserve={},
            )

    def test_frozen_protocol_and_projection_reject_opt_in_policy(self):
        for p in (copy.deepcopy(self.p), {**self.p, "runtime_projection": {}}):
            p["task_policy"] = {**POLICY, "resource_admission": "staged"}
            with self.assertRaisesRegex(ValueError, "new protocol identity"):
                task_policy(p)
        p = self.new_protocol()
        p["task_policy"]["resource_admission"] = "staged"
        with self.assertRaisesRegex(ValueError, "explicit next stage"):
            admit_resources(
                p, self.phase(), offline_used_seconds=0, offline_used_bytes=0
            )

    def test_ceiling_closes_success_branch_but_efficiency_and_repair_have_own_entry(
        self,
    ):
        p = self.new_protocol()
        p["task_policy"]["research_branches"] = [
            {
                "objective": obj,
                "criterion_ref": "declared-" + obj,
                "evaluation_ref": "independent-" + obj,
            }
            for obj in ("success_gain", "efficiency", "repair")
        ]
        complete = replace(self.baseline(), protocol_id=p["protocol_id"])
        report = baseline_readiness(p, complete)
        self.assertTrue(report["candidate_generation_allowed"])
        self.assertEqual(
            [r["research_allowed"] for r in report["research_branches"]],
            [False, True, True],
        )
        weak = replace(
            self.baseline(failed=(("base_salad", 0),)), protocol_id=p["protocol_id"]
        )
        report = baseline_readiness(p, weak)
        self.assertEqual(
            [r["research_allowed"] for r in report["research_branches"]],
            [False, False, True],
        )
        unsafe = replace(
            complete, hard_constraints={**complete.hard_constraints, "safe": False}
        )
        p["hard_constraints"]["safe"] = "independent safety evidence"
        self.assertFalse(baseline_readiness(p, unsafe)["candidate_generation_allowed"])
        self.assertEqual(
            baseline_readiness(self.p, self.baseline())["action"], "freeze_stop_ceiling"
        )
        del p["task_policy"]["research_branches"][1]["evaluation_ref"]
        report = baseline_readiness(
            p,
            replace(
                complete, hard_constraints={**complete.hard_constraints, "safe": True}
            ),
        )
        self.assertTrue(report["research_branches"][1]["research_allowed"])
        self.assertIsNone(report["research_branches"][1]["evaluation_ref"])
        self.assertIn("independent", report["release_qualification"])
        del p["task_policy"]["research_branches"][1]["criterion_ref"]
        with self.assertRaisesRegex(ValueError, "testable criterion"):
            baseline_readiness(
                p,
                replace(
                    complete,
                    hard_constraints={**complete.hard_constraints, "safe": True},
                ),
            )


class ExplorationTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def configure(self):
        self.protocol["task_policy"] = {
            **POLICY,
            "exploration": {
                "hypothesis": "finish supervision may reduce unnecessary calls",
                "criterion_ref": self.protocol["protocol_id"],
                "stop_condition": "one development trial",
                "assignments": [{"task_key": "salad", "seed": 2}],
                "operators": ["finish_requires_native_success"],
                "cost_limit": {"offline_calls": 1},
            },
        }
        self.tools.domain_policy.update(
            research_mode="exploratory",
            rule_operators=["finish_requires_native_success"],
        )

    def test_no_evidence_can_prepare_hypothesis_trial_without_dynamic_authorization(
        self,
    ):
        self.configure()
        adapter = BoundedResearchAdapter()
        transport = adapter.prepare_request(
            self.tools,
            request_id="explore",
            result_ids=[],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        request = unpack(transport)
        self.assertEqual(request["research_mode"], "exploratory")
        self.assertEqual(request["evidence"], [])
        response = {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": "explore",
            "decision": "prepare_rule_trial",
            "source": "exploration",
            "result_ids": [],
            "reason": "collect missing evidence",
            "external_value": "reduce wasted calls",
            "competing_explanations": self.explanations,
            "intervention": {"finish_requires_native_success": True},
            "distinguishing_prediction": "mismatch changes",
            "stop_condition": "one trial",
            "missing_evidence": ["no observed mismatch yet"],
        }
        trial = adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )
        plan = self.store.get("research_experiments", "explore-experiment")
        self.assertEqual(plan["evidence_basis"], "registered_hypothesis")
        self.assertEqual(plan["trigger_evidence"], [])
        self.assertFalse(plan["dynamic_execution_authorized"])
        self.assertTrue(Path(trial["runtime_rules_ref"]).exists())
        self.assertEqual(self.job.state["charges"], [])
        with self.assertRaises(FileNotFoundError):
            self.tools.submit_candidate(
                "not-evaluated", experiment_id="explore-experiment"
            )

    def test_missing_scope_wrong_assignment_and_excess_cost_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered new-protocol scope"):
            self.tools.discover_opportunity(
                "bad",
                source="exploration",
                result_ids=[],
                external_value="value",
                competing_explanations=self.explanations,
                expected_cost={},
            )
        self.configure()
        adapter = BoundedResearchAdapter()
        self.protocol["task_policy"]["exploration"]["assignments"][0]["seed"] = 5
        with self.assertRaisesRegex(ValueError, "outside development"):
            adapter.prepare_request(
                self.tools,
                request_id="bad",
                result_ids=[],
                max_input_tokens=100,
                max_output_tokens=10,
            )
        self.protocol["task_policy"]["exploration"]["assignments"][0]["seed"] = 2
        self.protocol["task_policy"]["exploration"]["cost_limit"]["offline_calls"] = 11
        with self.assertRaisesRegex(ValueError, "original job budget"):
            adapter.prepare_request(
                self.tools,
                request_id="bad",
                result_ids=[],
                max_input_tokens=100,
                max_output_tokens=10,
            )

    def test_selected_input_and_exact_read_leave_authoritative_l0_unchanged(self):
        events = [
            {"event_id": "one", "payload": {"unknown": None}},
            {"event_id": "two", "payload": "large"},
        ]
        view = EvidenceView("view", self.base.bundle_id, events, [], "all")
        self.store.put("evidence_views", "view", view)
        result = replace(
            self.result("with-view", self.base.bundle_id),
            intervention={
                "information_scope": "development",
                "evidence_view_id": "view",
            },
        )
        self.store.save_development_result(result)
        evidence = self.tools.query_development(["with-view"])
        request = {
            "evidence": evidence,
            "evidence_views": [self.store.get("evidence_views", "view")],
        }
        self.assertIs(select_input(self.tools, request), request)
        self.protocol["task_policy"] = {
            **POLICY,
            "evidence_input": {
                "mode": "selected_l0",
                "reason": "inspect the first transition",
                "event_ids": {"with-view": ["one"]},
            },
        }
        selected = select_input(self.tools, request)
        self.assertEqual(selected["evidence_views"][0]["l0"], events[:1])
        self.assertEqual(self.store.get("evidence_views", "view")["l0"], events)
        self.assertEqual(
            selected["evidence_input"]["views"][0]["available_event_count"], 2
        )
        self.assertIn("Selection is not complete L0", model_prompt(pack(selected)))
        read = ResearchToolkit(self.tools).call(
            "read_development_events", {"result_id": "with-view", "event_ids": ["two"]}
        )
        self.assertEqual(read["events"], events[1:])
        index = self.tools.read_development_events("with-view")
        self.assertEqual([e["event_id"] for e in index["event_index"]], ["one", "two"])
        with self.assertRaisesRegex(ValueError, "unknown development event"):
            self.tools.read_development_events("with-view", ["not-in-view"])
        self.store.put(
            "development_results",
            "release",
            replace(result, result_id="release", role="release"),
        )
        with self.assertRaisesRegex(ValueError, "matching development"):
            self.tools.read_development_events("release", ["two"])


class GPUAndStageConsumerTests(unittest.TestCase):
    def test_wait_exhaustion_saves_a_resource_checkpoint_not_a_closed_campaign(self):
        policy = {"max_wait_seconds": 75, "check_interval_seconds": 12}
        first = resource_wait(policy, waited_seconds=70, offline_remaining_seconds=30)
        self.assertEqual(first["next_check_seconds"], 5)
        blocked = resource_wait(policy, waited_seconds=75, offline_remaining_seconds=30)
        self.assertTrue(blocked["wait_budget_exhausted"])
        self.assertFalse(blocked["task_closed"])
        self.assertEqual(blocked["state"], "waiting_resources")

    def test_memory_policy_uses_free_memory_capacity_and_compatibility(self):
        policy = {
            "required_memory_mib": 12000,
            "safety_margin_mib": 2000,
            "minimum_compute_capability": 8.0,
        }
        cards = [
            {
                "physical_gpu": 0,
                "free_memory_mib": 13000,
                "compute_capability": "8.9",
                "utilization_percent": 0,
            },
            {
                "physical_gpu": 1,
                "free_memory_mib": 40000,
                "compute_capability": "8.0",
                "utilization_percent": 10,
            },
            {
                "physical_gpu": 2,
                "free_memory_mib": 80000,
                "compute_capability": "7.0",
                "utilization_percent": 0,
            },
        ]
        self.assertEqual(select_gpu(cards, policy), 1)
        self.assertIsNone(select_gpu(cards[:1], policy))
        policy["required_memory_mib"] = float("nan")
        with self.assertRaises(ValueError):
            select_gpu(cards, policy)

    def test_gpu_query_passes_telemetry_to_capacity_policy(self):
        spec = importlib.util.spec_from_file_location(
            "policy_gpu_driver", ROOT / "scripts/run_release_comparison.py"
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        policy = {
            "required_memory_mib": 20000,
            "safety_margin_mib": 2000,
            "minimum_compute_capability": 8,
        }
        outputs = [
            "0, a\n1, b",
            "a, 10, 13000\nb, 11, 1000",
            "0, a, Big, 80000, 67000, 0, 8.0\n1, b, Small, 24000, 21000, 0, 8.9",
        ]
        with patch.object(
            driver.subprocess,
            "run",
            side_effect=[SimpleNamespace(stdout=s) for s in outputs],
        ):
            report = driver.gpu_admission(policy=policy)
        self.assertEqual(report["selected"], 0)
        self.assertIsNone(report["threshold_mib"])
        self.assertFalse(report["other_jobs_signalled"])

    def test_stage_cli_uses_runtime_limits_and_never_dispatches(self):
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "task_stage", ROOT / "scripts/prepare_task_stage.py"
        )
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, phase = root / "protocol.json", root / "phase.json"
            atomic_json(
                protocol,
                {
                    "protocol_id": "new",
                    "task_policy": {**POLICY, "resource_admission": "staged"},
                },
            )
            cap = {"simulation_episodes": 1, "wall_seconds": 910}
            atomic_json(
                phase,
                {
                    "limits": cap,
                    "used": dict.fromkeys(cap, 0),
                    "remaining": cap,
                    "unresolved_costs": [],
                },
            )
            request = {
                "repository_root": str(ROOT),
                "artifacts": tmp,
                "run_id": "next",
                "bundle_id": "b",
                "task_key": "salad",
                "seed": 2,
                "split": "development",
                "output_dir": str(root / "run"),
                "cuda_device": 3,
                "cleanup": {"wall_seconds": 10},
                "reserve": {},
            }
            plan = {
                "context": {"limits": {"wall_seconds": 900, "planner_seconds": 600}},
                "blockers": ["permit missing"],
            }
            with patch.object(stage, "prepare_runtime", return_value=plan) as runtime:
                report = stage.prepare(protocol, phase, request)
            runtime.assert_called_once()
            self.assertTrue(report["admission"]["resource_fit"])
            self.assertFalse(report["dispatch"])
            self.assertEqual(report["plan"]["blockers"], ["permit missing"])


class ExplorationRuntimeTests(unittest.TestCase):
    setUp = test_research_preflight.PreflightTests.setUp

    def test_registered_hypothesis_reaches_actual_preflight_and_assignment_boundary(
        self,
    ):
        from rsi_vla.research_cycle import prepare_trial
        from rsi_vla.research_preflight import operator_preflight

        p = self.protocol
        p.update(protocol_id="exploration-new-protocol", scope_id="new-exploration")
        p["task_policy"] = {
            **POLICY,
            "exploration": {
                "hypothesis": "finish guard may prevent incorrect closure",
                "criterion_ref": p["protocol_id"],
                "stop_condition": "one seed2 slot",
                "operators": ["finish_requires_native_success"],
                "assignments": [{"task_key": "position_swap_salad", "seed": 2}],
                "cost_limit": {"offline_calls": 1},
            },
        }
        self.tools.base = replace(self.base, protocol_id=p["protocol_id"])
        self.tools.domain_policy.update(research_mode="exploratory")
        atomic_json(self.protocol_ref, p)
        rows = operator_preflight(
            self.tools, ROOT, protocol_path=self.protocol_ref, result_ids=[]
        )
        guard = next(
            r for r in rows if r["operator"] == "finish_requires_native_success"
        )
        self.assertTrue(guard["currently_trialable"])
        self.assertFalse(guard["evidence_supports_trial"])
        self.assertEqual(guard["evidence_basis"], "registered_hypothesis")
        self.assertFalse(guard["execution_authorized"])
        self.assertEqual(guard["development_assignment"]["seed"], 2)
        explanations = [
            {
                "explanation": "premature finish",
                "prediction": "guard helps",
                "distinguishing_test": "native result",
            },
            {
                "explanation": "perception error",
                "prediction": "guard cannot help",
                "distinguishing_test": "native result",
            },
        ]
        self.tools.discover_opportunity(
            "explore",
            source="exploration",
            result_ids=[],
            external_value="correct closure",
            competing_explanations=explanations,
            expected_cost={"offline_calls": 1},
        )
        self.tools.plan_rule_experiment(
            "trial",
            opportunity_id="explore",
            bundle_id="new-trial",
            finish_requires_native_success=True,
            criterion_ref=p["protocol_id"],
            distinguishing_prediction="native closure changes",
            cost_limit={"offline_calls": 1},
            evidence_level="simulation",
        )
        self.tools.prepare_rule_trial("trial")
        with self.assertRaisesRegex(ValueError, "outside registered exploration slots"):
            prepare_trial(
                self.tools,
                "trial",
                root=ROOT,
                artifacts=self.root,
                protocol_path=self.protocol_ref,
                output=self.root / "no-run",
                run_id="never-start",
                task_key="old_milk",
                seed=2,
            )
        self.assertFalse((self.root / "no-run").exists())


class ResearchLimitTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    prepared = test_batch4.RunnerTest.prepared

    def test_per_request_policy_cannot_expand_job_budget_or_change_prepared_input(self):
        from rsi_vla.bounded_runner import validate_permit

        self.protocol["task_policy"] = {
            **POLICY,
            "research_per_request": {"wall_seconds": 600, "storage_bytes": 20000000},
        }
        _, transport, output, permit, _ = self.prepared()
        self.assertEqual(
            validate_permit(
                self.tools, transport, permit, root=self.root, output=output
            )["task_policy"],
            self.protocol["task_policy"],
        )
        permit["limits"]["wall_seconds"] = 400
        with self.assertRaisesRegex(PermissionError, "original job budget"):
            validate_permit(
                self.tools, transport, permit, root=self.root, output=output
            )
        permit["limits"]["wall_seconds"] = 300
        self.protocol["task_policy"]["research_per_request"]["wall_seconds"] = 900
        with self.assertRaisesRegex(PermissionError, "prepared task policy changed"):
            validate_permit(
                self.tools, transport, permit, root=self.root, output=output
            )


class SC3OpportunityTests(unittest.TestCase):
    """Model-output fixtures test consumers, without claiming autonomous research."""

    def setUp(self):
        test_research_tools.ResearchToolsTest.setUp(self)
        self.protocol["task_policy"] = {**POLICY, "opportunity_discovery": True}
        self.tools.domain_policy["rule_operators"] = ["finish_requires_native_success"]
        self.adapter = BoundedResearchAdapter()

    result = test_research_tools.ResearchToolsTest.result

    def prepare_response(self, request_id="normal"):
        self.adapter.prepare_request(
            self.tools, request_id=request_id, result_ids=["base-result"],
            max_input_tokens=100, max_output_tokens=10,
        )
        return {
            "schema": "rsi-bounded-research-response-v3", "request_id": request_id,
            "decision": "prepare_rule_trial", "source": "normal",
            "result_ids": ["base-result"], "reason": "successful run may close sooner",
            "external_value": "reduce calls while preserving native success",
            "competing_explanations": self.explanations,
            "intervention": {"finish_requires_native_success": True},
            "distinguishing_prediction": "closure overhead falls", "stop_condition": "one development slot",
            "missing_evidence": [], "target": None,
        }

    def apply(self, response, cost=1):
        return self.adapter.apply_response(
            self.tools, response, evidence_level="offline",
            cost_limit={"offline_calls": cost}, synthetic=True,
        )

    def goal(self):
        response = self.prepare_response("goal")
        response.update(
            source="new_goal", decision="propose_goal", intervention=None,
            target={"domain": "tabletop", "kind": "manipulation",
                    "description": "stack an unlisted pair of objects after successful placement",
                    "value_criterion_ref": "independent-domain-value-v1",
                    "independent_value_test_ref": "development-goal-value-test"},
        )
        return response

    def test_successful_run_supports_normal_opportunity_and_finite_preparation(self):
        trial = self.apply(self.prepare_response())
        opportunity = self.store.get("opportunities", "normal-opportunity")
        self.assertEqual(opportunity["source"], "normal")
        self.assertEqual(opportunity["admission"]["action"], "prepare")
        self.assertEqual(self.store.get("bounded_requests", "normal")["remaining_job_budget"],
                         self.job.remaining())
        self.assertTrue(Path(trial["runtime_rules_ref"]).is_file())
        self.assertEqual(self.job.state["charges"], [])
        self.assertFalse(opportunity["experiment_authorized"])
        self.assertEqual(self.store.get("bounded_proposals", "normal")["origin"],
                         "synthetic_engineering_fixture")

    def test_insufficient_budget_defers_success_opportunity_without_trial_or_charge(self):
        response = self.prepare_response("budget")
        before = copy.deepcopy(self.job.state)
        decision = self.apply(response, cost=11)
        self.assertEqual(decision["action"], "defer")
        opportunity = self.store.get("opportunities", "budget-opportunity")
        self.assertEqual(opportunity["result_ids"], ["base-result"])
        self.assertEqual(opportunity["admission"]["additional_required"]["offline_calls"], 1)
        self.assertFalse(opportunity["admission"]["resource_fit"])
        self.assertEqual(self.store.all("rule_trials"), [])
        self.assertEqual(self.store.all("candidates"), [])
        self.assertEqual(self.job.state, before)

    def test_unlisted_target_is_retained_for_independent_review_without_rule_trial(self):
        response = self.goal()
        schema = self.adapter.response_schema("goal", tools=self.tools)
        self.assertIn("new_goal", schema["properties"]["source"]["enum"])
        target_schema = schema["properties"]["target"]["anyOf"][1]
        self.assertNotIn("enum", target_schema["properties"]["description"])
        decision = self.apply(response)
        opportunity = self.store.get("opportunities", "goal-opportunity")
        self.assertEqual(opportunity["target"], response["target"])
        self.assertEqual(opportunity["goal_review"]["value_status"], "unverified")
        self.assertIsNone(opportunity["goal_review"]["capability_gain"])
        self.assertEqual(decision["action"], "defer")
        self.assertIn("independent domain value", decision["missing_evidence"][0])
        self.assertEqual(self.store.all("rule_trials"), [])
        self.assertEqual(self.store.all("candidates"), [])
        self.assertEqual(self.job.state["charges"], [])

    def test_target_cannot_escape_domain_or_smuggle_a_rule_trial(self):
        response = self.goal()
        response["target"]["domain"] = "surgery"
        with self.assertRaisesRegex(ValueError, "domain eligibility"):
            self.apply(response)
        response["target"]["domain"] = "tabletop"
        response["decision"] = "prepare_rule_trial"
        response["intervention"] = {"finish_requires_native_success": True}
        with self.assertRaisesRegex(ValueError, "no rule intervention"):
            self.apply(response)
        self.assertEqual(self.store.all("opportunities"), [])
        self.assertEqual(self.store.all("candidates"), [])

    def test_existing_v2_schema_stays_unchanged_and_rejects_new_goal_fields(self):
        self.protocol["task_policy"].pop("opportunity_discovery")
        response = self.prepare_response("legacy")
        response["schema"] = "rsi-bounded-research-response-v2"
        response.pop("target")
        schema = self.adapter.response_schema("legacy", tools=self.tools)
        self.assertNotIn("target", schema["properties"])
        self.assertNotIn("new_goal", schema["properties"]["source"]["enum"])
        self.assertTrue(Path(self.apply(response)["runtime_rules_ref"]).is_file())

    def test_fixture_cannot_claim_real_transport_provenance(self):
        response = self.prepare_response("not-real")
        with self.assertRaises(FileNotFoundError):
            self.adapter.apply_response(self.tools, response, evidence_level="offline",
                                        cost_limit={"offline_calls": 1}, synthetic=False)
        self.assertEqual(self.store.all("opportunities"), [])

    def test_independent_value_recomputes_confirmed_rejected_and_missing_states(self):
        from rsi_vla.task_policy import review_goal_value

        response = self.goal()
        self.apply(response)
        opportunity = self.store.get("opportunities", "goal-opportunity")
        target = response["target"]
        policy = {"policy_ref": self.tools.domain_policy["policy_ref"],
                  "domain": "tabletop", "allowed_goal_kinds": ["manipulation"],
                  "value_criterion_ref": target["value_criterion_ref"],
                  "independent_value_test_ref": target["independent_value_test_ref"],
                  "metric": "useful_compositions", "minimum": 2}
        evidence = {"schema": "rsi-goal-value-evidence-v1",
                    "protocol_id": self.protocol["protocol_id"],
                    "opportunity_id": opportunity["opportunity_id"], "target": target,
                    "criterion_ref": policy["value_criterion_ref"],
                    "test_ref": policy["independent_value_test_ref"],
                    "information_scope": "development", "validity": "valid",
                    "metrics": {"useful_compositions": 3}, "evidence_refs": ["independent-cpu-fixture"]}
        for observed, validity, expected in [(3, "valid", "confirmed"), (1, "valid", "rejected"),
                                              (None, "valid", "inconclusive"), (3, "invalid", "inconclusive")]:
            with self.subTest(observed=observed, validity=validity):
                report = review_goal_value(self.protocol, opportunity,
                    {**evidence, "metrics": {"useful_compositions": observed}, "validity": validity},
                    policy=policy)
                self.assertEqual(report["value_status"], expected)
                self.assertIsNone(report["capability_gain"])
                self.assertTrue(report["requires_new_evaluation_protocol"])
                self.assertFalse(report["release_eligible"])
        for change in [{"opportunity_id": "another"}, {"information_scope": "release"},
                       {"target": {**target, "description": "different target"}},
                       {"test_ref": "wrong-test"}, {"evidence_refs": []},
                       {"metrics": {"useful_compositions": float("nan")}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                review_goal_value(self.protocol, opportunity, {**evidence, **change}, policy=policy)
