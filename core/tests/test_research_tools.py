"""Offline checks for callable research, finite phi updates and information limits."""
import json
import importlib.util
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.research import BudgetExhausted, ResearchJob
from rsi_vla.research_tools import ResearchLLMAdapter, ResearchToolkit, ResearchTools
from rsi_vla.rule_updates import load_runtime_rules, rule_update, validate_runtime_rules, write_rule_trial
from rsi_vla.storage import ResearchStore, initialize


class ResearchToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResearchStore(self.root / "research")
        profile = self.root / "profile.json"
        profile.write_text(json.dumps({"control": {"action_chunk": 5}}))
        self.base = RunBundle("initial-medium", "theta-fixed", "processor-fixed",
            str(profile) + "#/control", "rho-fixed", "psi-fixed", "memory-fixed",
            "pilot-medium", "backend-fixed", "code-fixed")
        initialize(self.store, self.base, psi={"learned": False},
                   memory={"provenance": "fixture", "legal_information": "development only"})
        self.protocol = {"protocol_id": "pilot-medium", "scope": "simulation",
            "benchmark_id": "four-tasks", "dynamic_authorized": False,
            "task_sets": {"development": {"task_keys": ["salad"], "seeds": [1, 2],
                                          "benchmark_id": "four-tasks-development"},
                          "release": {"task_keys": ["salad"], "seeds": [3, 4]},
                          "sealed": {"task_keys": ["salad"], "seeds": [5, 6]}}}
        self.job = ResearchJob(self.store, "job", self.base.bundle_id,
                               {"offline_calls": 10, "input_tokens": 1000, "output_tokens": 100})
        self.tools = ResearchTools(self.store, self.job, self.protocol,
            domain_policy={"domain": "tabletop", "allowed_goal_kinds": ["manipulation"],
                           "policy_ref": "external-domain-rule-v1"})
        self.store.save_development_result(self.result("base-result", self.base.bundle_id))
        self.explanations = [
            {"explanation": "prefix overshoots", "prediction": "shorter prefix reduces overshoot",
             "distinguishing_test": "matched new control timing baseline and trial"},
            {"explanation": "observation error", "prediction": "prefix change leaves error unchanged",
             "distinguishing_test": "compare legal observations before action"}]

    def result(self, result_id, bundle_id="trial-bundle", *, outcome="success", seed=1,
               level="offline", costs=None, chunk_steps=None):
        intervention = {"information_scope": "development"}
        if chunk_steps is not None:
            intervention["chunk_steps"] = chunk_steps
        return ExperimentResult(result_id, bundle_id, "pilot-medium", "four-tasks-development",
            "development", level, intervention, ["fixture-data"], "criterion-v1", {}, {},
            {"offline_calls": 1} if costs is None else costs,
            assigned_count=1, outcome=outcome,
            assignments=[{"task_key": "salad", "seed": seed, "outcome": outcome,
                          "validity": "valid", "outcome_source": "offline_fixture"}])

    def opportunity(self, **changes):
        arguments = {"source": "normal", "result_ids": ["base-result"],
            "external_value": "reduce costly recovery while preserving task outcome",
            "competing_explanations": self.explanations, "expected_cost": {"offline_calls": 1}}
        arguments.update(changes)
        return self.tools.discover_opportunity("opportunity", **arguments)

    def prepare(self, *, level="offline"):
        self.opportunity()
        self.tools.plan_rule_experiment("experiment", opportunity_id="opportunity",
            bundle_id="trial-bundle", chunk_steps=2, criterion_ref="criterion-v1",
            distinguishing_prediction="overshoot decreases only if prefix is causal",
            cost_limit={"offline_calls": 1}, evidence_level=level)
        return self.tools.prepare_rule_trial("experiment")

    def test_real_artifact_difference_and_frozen_theta(self):
        trial = self.prepare()
        self.assertEqual(load_runtime_rules(trial["runtime_rules_ref"]),
                         {"schema": "rsi-runtime-rules-v1", "chunk_steps": 2, "baseline_chunk_steps": 5})
        target = RunBundle(**self.store.get("bundles", "trial-bundle"))
        for name in ("theta_ref", "processor_ref", "rho_ref", "psi_ref", "memory_ref",
                     "backend_ref", "code_ref", "protocol_id"):
            self.assertEqual(getattr(target, name), getattr(self.base, name))
        diff = json.loads(Path(trial["diff_ref"]).read_text())
        self.assertEqual(diff["changes"], {"chunk_steps": {"before": 5, "after": 2}})
        self.assertIn("new matched", diff["comparison_requirement"])
        self.assertFalse(trial["runtime_loaded"])

    def test_finite_rules_reject_widening_noop_bool_and_extra_fields(self):
        for steps in (0, 6, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                validate_runtime_rules({"schema": "rsi-runtime-rules-v1", "chunk_steps": steps,
                                        "baseline_chunk_steps": 5})
        with self.assertRaises(ValueError):
            validate_runtime_rules({"schema": "rsi-runtime-rules-v1", "chunk_steps": 2,
                                    "baseline_chunk_steps": 5, "disable_safety": True})
        with self.assertRaises(ValueError):
            write_rule_trial(self.store, self.base, bundle_id="noop", artifact_id="noop", chunk_steps=5)
        with self.assertRaises(ValueError):
            load_runtime_rules("relative.json")

    def test_candidate_uses_target_evidence_and_has_no_release_side_effect(self):
        self.prepare()
        self.tools.run_offline_experiment("experiment", lambda plan, trial: self.result(
            "trial-result", chunk_steps=2), cost={"offline_calls": 1})
        candidate = self.tools.submit_candidate("candidate", experiment_id="experiment")
        self.assertEqual(candidate["development_results"], ["trial-result"])
        self.assertEqual(set(candidate["changes"]), {"phi_ref", "diff_ref"})
        self.assertEqual(candidate["approval_scope"], "offline")
        self.assertEqual(candidate["benchmark_id"], "four-tasks")
        self.assertEqual(candidate["costs"], self.job.state["used"])
        self.assertTrue(candidate["compatibility"]["requires_new_timing_baseline"])
        self.assertFalse(candidate["compatibility"]["runtime_loaded"])
        self.assertEqual(self.store.get("bundles", self.base.bundle_id)["theta_ref"], "theta-fixed")
        self.assertFalse((self.store.root / "active").exists())
        self.assertFalse(hasattr(self.tools, "release"))

    def test_costs_prepaid_once_and_recovery_retains_job_binding(self):
        self.prepare()
        self.tools.run_offline_experiment("experiment", lambda p, t: self.result(
            "trial-result", chunk_steps=2), cost={"offline_calls": 1})
        self.tools.record_experiment_result("experiment", self.result("trial-result", chunk_steps=2))
        restored = ResearchJob(self.store, "job")
        self.assertEqual(restored.state["used"]["offline_calls"], 1)
        self.assertEqual(restored.state["bundle_id"], "initial-medium")
        with self.assertRaises(ValueError):
            self.tools.run_offline_experiment("experiment", lambda p, t: None,
                                              cost={"offline_calls": 1})

    def test_direct_record_accounts_for_cost_and_retains_negative_finding(self):
        self.prepare()
        finding = self.tools.record_experiment_result("experiment", self.result(
            "negative", outcome="failure", chunk_steps=2))
        self.assertEqual(finding["outcome"], "failure")
        self.assertFalse(finding["hypotheses_resolved"])
        self.assertEqual(self.job.state["used"]["offline_calls"], 1)
        candidate = self.tools.submit_candidate("negative-candidate", experiment_id="experiment")
        self.assertEqual(candidate["development_results"], ["negative"])

    def test_job_exhaustion_prevents_evaluator_dispatch(self):
        self.prepare()
        self.job.consume({"offline_calls": 10}, "another action")
        called = []
        with self.assertRaises(BudgetExhausted):
            self.tools.run_offline_experiment("experiment", lambda p, t: called.append(True),
                                              cost={"offline_calls": 1})
        self.assertEqual(called, [])

    def test_unresolved_dispatch_is_not_automatically_retried(self):
        self.prepare()
        def interrupted(plan, trial):
            raise RuntimeError("fixture interrupted")
        with self.assertRaises(RuntimeError):
            self.tools.run_offline_experiment("experiment", interrupted, cost={"offline_calls": 1})
        self.assertEqual(self.job.state["used"]["offline_calls"], 1)
        with self.assertRaisesRegex(ValueError, "already dispatched"):
            self.tools.run_offline_experiment("experiment", interrupted, cost={"offline_calls": 1})

    def test_dynamic_plan_cannot_use_offline_executor(self):
        self.prepare(level="simulation")
        with self.assertRaisesRegex(ValueError, "authorized backend"):
            self.tools.run_offline_experiment("experiment", lambda p, t: None,
                                              cost={"offline_calls": 1})
        self.assertEqual(self.job.state["used"]["offline_calls"], 0)

    def test_sealed_or_release_information_cannot_select_candidates(self):
        sealed = self.result("sealed", self.base.bundle_id, seed=5)
        self.store.save_development_result(sealed)
        with self.assertRaisesRegex(ValueError, "nondevelopment"):
            self.tools.query_development(["sealed"])
        release = replace(self.result("release", self.base.bundle_id), role="release")
        self.store.put("development_results", "release", release)
        with self.assertRaisesRegex(ValueError, "development"):
            self.tools.query_development(["release"])
        hidden = replace(self.result("hidden", self.base.bundle_id),
                         intervention={"information_scope": "sealed"})
        self.store.save_development_result(hidden)
        with self.assertRaisesRegex(ValueError, "information scope"):
            self.tools.query_development(["hidden"])

    def test_mismatched_trial_result_is_rejected_before_charge(self):
        self.prepare()
        for result in (self.result("wrong-bundle", self.base.bundle_id, chunk_steps=2),
                       self.result("wrong-change", chunk_steps=3),
                       replace(self.result("wrong-criterion", chunk_steps=2), criteria_ref="other")):
            with self.subTest(result=result.result_id), self.assertRaises(ValueError):
                self.tools.record_experiment_result("experiment", result)
        self.assertEqual(self.job.state["used"]["offline_calls"], 0)

    def test_failure_opportunity_competing_explanations_and_defer(self):
        self.store.save_development_result(self.result("unknown", self.base.bundle_id, outcome="unknown"))
        opportunity = self.opportunity(source="failure", result_ids=["unknown"])
        self.assertEqual(len(opportunity["hypotheses"]), 2)
        decision = self.tools.defer("defer", opportunity_id="opportunity", reason="missing legal observation",
                                    missing_evidence=["holding-target reference"])
        self.assertEqual(decision["action"], "defer")
        self.assertEqual(decision["remaining_budget"], self.job.remaining())
        with self.assertRaises(ValueError):
            self.opportunity(competing_explanations=[self.explanations[0]])

    def test_new_goal_has_domain_screen_and_unverified_independent_value(self):
        target = {"domain": "tabletop", "kind": "manipulation", "description": "stack two objects",
                  "value_criterion_ref": "external-value-rule", "independent_value_test_ref": "held-out-task"}
        opportunity = self.opportunity(source="new_goal", target=target)
        self.assertEqual(opportunity["goal_review"]["value_status"], "unverified")
        self.assertIsNone(opportunity["goal_review"]["capability_gain"])
        with self.assertRaisesRegex(ValueError, "cannot establish"):
            self.tools.plan_rule_experiment("new-goal", opportunity_id="opportunity", bundle_id="goal-trial",
                chunk_steps=2, criterion_ref="criterion", distinguishing_prediction="prediction",
                cost_limit={"offline_calls": 1})
        with self.assertRaises(ValueError):
            self.opportunity(source="new_goal", target={**target, "domain": "surgery"})

    def test_toolkit_dispatches_real_tools_and_blocks_release_and_arbitrary_methods(self):
        toolkit = ResearchToolkit(self.tools)
        self.assertEqual(toolkit.call("query_development", {"result_ids": ["base-result"]})[0]["outcome"], "success")
        for name in ("release", "run_offline_experiment", "_save", "__class__"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                toolkit.call(name, {})

    def test_llm_factory_is_explicit_medium_persisted_and_uses_legal_evidence(self):
        adapter = ResearchLLMAdapter(model="gpt-6.1-sol", reasoning_effort="medium")
        request = adapter.prepare_request(self.tools, request_id="request", result_ids=["base-result"],
                                          max_input_tokens=1000, max_output_tokens=100)
        self.assertEqual(self.store.get("llm_requests", "request"), request)
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertIsNone(request["execution"]["effective_model"])
        self.assertEqual(self.job.state["used"]["input_tokens"], 0)
        calls = [{"name": "query_development", "arguments": {"result_ids": ["base-result"]}}]
        response = {"effective_model": "gpt-6.1-sol",
            "effective_reasoning_effort": "medium", "request_id": "request", "tool_calls": calls}
        self.assertEqual(adapter.validate_response(self.tools, response), calls)
        self.assertEqual(self.store.get("llm_responses", "request"), response)
        with self.assertRaises(ValueError):
            ResearchLLMAdapter(model="gpt-6.1-sol", reasoning_effort="high")
        with self.assertRaises(ValueError):
            adapter.validate_response(self.tools, {"effective_model": "gpt-6.1-sol", "request_id": "request",
                                       "effective_reasoning_effort": "low", "tool_calls": calls})
        with self.assertRaises(ValueError):
            adapter.prepare_request(self.tools, request_id="oversized", result_ids=["base-result"],
                                    max_input_tokens=1001, max_output_tokens=100)

    def test_real_v2_protocol_development_split_consumer(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "config" / "pilot_v2.json"
        if not path.exists():
            self.skipTest("v2 protocol not present in this snapshot")
        protocol = json.loads(path.read_text())
        split = protocol["task_sets"]["development"]
        base = replace(self.base, bundle_id="v2-start", protocol_id=protocol["protocol_id"])
        self.store.save_bundle(base)
        job = ResearchJob(self.store, "v2-job", base.bundle_id, {"offline_calls": 10})
        tools = ResearchTools(self.store, job, protocol, domain_policy=self.tools.domain_policy)
        row = {"task_key": split["task_keys"][0], "seed": split["seeds"][0],
               "outcome": "success", "validity": "valid", "outcome_source": "native_benchmark",
               "evidence_ref": "offline-native-record-fixture"}
        result = replace(self.result("v2-development", base.bundle_id),
            protocol_id=protocol["protocol_id"], benchmark_id=split["benchmark_id"],
            evidence_level="simulation", assignments=[row], outcome="unknown")
        self.store.save_development_result(result)
        self.assertEqual(tools.query_development([result.result_id])[0]["benchmark_id"], split["benchmark_id"])
        opportunity = tools.discover_opportunity("v2-opportunity", source="normal",
            result_ids=[result.result_id], external_value="reduce execution cost",
            competing_explanations=self.explanations, expected_cost={"offline_calls": 1})
        self.assertEqual(opportunity["source"], "normal")
        self.store.save_development_result(replace(result, result_id="v2-wrong-release-benchmark",
                                                   benchmark_id=protocol["benchmark_id"]))
        with self.assertRaisesRegex(ValueError, "development"):
            tools.query_development(["v2-wrong-release-benchmark"])

    def test_rule_factory_to_actual_runtime_journal_consumer(self):
        root = Path(__file__).resolve().parents[2]
        runtime_root = Path(os.environ.get("RSI_RPENT_ROOT", root / "RPent"))
        path = runtime_root / "rpent" / "session" / "runtime.py"
        if not path.exists():
            self.skipTest("runtime journal not present in this snapshot")
        spec = importlib.util.spec_from_file_location("rsi_test_runtime_consumer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trial = self.prepare()
        journal = module.RuntimeJournal({"runtime_rules_ref": trial["runtime_rules_ref"],
            "output_dir": str(self.root / "runtime-output"), "run_id": "offline-handoff",
            "episode_id": "episode", "bundle_id": trial["bundle_id"],
            "protocol_id": self.protocol["protocol_id"], "profile_ref": "offline-fixture"})
        self.assertEqual(journal.chunk_steps, 2)
        start = json.loads(journal.events_path.read_text().splitlines()[0])
        self.assertEqual(start["runtime_rules_ref"], trial["runtime_rules_ref"])
        self.assertEqual(start["bundle_id"], trial["bundle_id"])


class NativeFinishRuleTests(unittest.TestCase):
    """Only the newly authorized guard extension; all records here are fixtures."""
    def setUp(self):
        self.f = ResearchToolsTest("test_real_artifact_difference_and_frozen_theta")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def prepare_guard(self, *, level="offline"):
        self.f.opportunity()
        self.f.tools.plan_rule_experiment("guard-experiment", opportunity_id="opportunity",
            bundle_id="guard-bundle", finish_requires_native_success=True, criterion_ref="criterion-v1",
            distinguishing_prediction="native guard avoids premature completion within unchanged budgets",
            cost_limit={"offline_calls": 1}, evidence_level=level)
        return self.f.tools.prepare_rule_trial("guard-experiment")

    def raw_medium_development(self, **changes):
        f = self.f
        raw = {"planner_finish": {"status": "success"}, "benchmark_success": False,
               "attempt_closed": True, "native_result_source": {"terminated": "native LIBERO callback"},
               "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
               "bundle_id": f.base.bundle_id, "protocol_id": f.protocol["protocol_id"]}
        raw.update(changes)
        path = f.root / "medium-result.json"
        path.write_text(json.dumps(raw))
        result = f.result("medium-result", f.base.bundle_id, outcome="failure", level="simulation")
        row = {**result.assignments[0], "evidence_ref": str(path), "outcome_source": "native_benchmark"}
        result = replace(result, data_refs=[str(path)], assignments=[row])
        f.store.save_development_result(result)
        return result

    def test_versioned_schema_accepts_boolean_baseline_and_preserves_v1(self):
        baseline = {"schema": "rsi-runtime-rules-v2", "chunk_steps": 5,
                    "baseline_chunk_steps": 5, "finish_requires_native_success": False}
        self.assertEqual(validate_runtime_rules(baseline), baseline)
        for invalid in (0, 1, "true", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_runtime_rules({**baseline, "finish_requires_native_success": invalid})
        v1 = {"schema": "rsi-runtime-rules-v1", "chunk_steps": 5, "baseline_chunk_steps": 5}
        after, changes = rule_update(v1, chunk_steps=2)
        self.assertEqual(after, {**v1, "chunk_steps": 2})
        self.assertEqual(set(changes), {"chunk_steps"})

    def test_guard_factory_changes_only_guard_and_freezes_control(self):
        trial = self.prepare_guard()
        rules = load_runtime_rules(trial["runtime_rules_ref"])
        self.assertEqual(rules, {"schema": "rsi-runtime-rules-v2", "chunk_steps": 5,
                               "baseline_chunk_steps": 5, "finish_requires_native_success": True})
        diff = json.loads(Path(trial["diff_ref"]).read_text())
        self.assertEqual(diff["changes"], {"finish_requires_native_success": {"before": False, "after": True}})
        self.assertEqual(diff["frozen_control"], {"chunk_steps": 5, "baseline_chunk_steps": 5})
        self.assertFalse(diff["requires_new_timing_baseline"])
        self.assertEqual(diff["frozen"]["theta_ref"], self.f.base.theta_ref)
        self.assertEqual(diff["frozen"]["protocol_id"], self.f.base.protocol_id)
        self.assertFalse(trial["runtime_loaded"])

    def test_v2_baseline_factory_and_single_intervention_scope(self):
        f = self.f
        baseline = {"schema": "rsi-runtime-rules-v2", "chunk_steps": 5,
                    "baseline_chunk_steps": 5, "finish_requires_native_success": False}
        path = f.root / "guard-baseline.json"
        path.write_text(json.dumps(baseline))
        base = replace(f.base, phi_ref=str(path))
        target, diff = write_rule_trial(f.store, base, bundle_id="guard-only", artifact_id="guard-only",
                                        finish_requires_native_success=True)
        self.assertTrue(load_runtime_rules(target.phi_ref)["finish_requires_native_success"])
        self.assertEqual(set(diff["changes"]), {"finish_requires_native_success"})
        for kwargs in ({"chunk_steps": 2, "finish_requires_native_success": True},
                       {"finish_requires_native_success": False}, {}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rule_update(baseline, **kwargs)

    def test_guard_plan_result_candidate_preserve_actual_intervention_and_offline_scope(self):
        f = self.f
        self.prepare_guard()
        result = replace(f.result("guard-result", "guard-bundle"),
            intervention={"information_scope": "development", "finish_requires_native_success": True})
        f.tools.record_experiment_result("guard-experiment", result)
        candidate = f.tools.submit_candidate("guard-candidate", experiment_id="guard-experiment")
        self.assertEqual(candidate["compatibility"]["runtime_rule_schema"], "rsi-runtime-rules-v2")
        self.assertFalse(candidate["compatibility"]["requires_new_timing_baseline"])
        self.assertEqual(candidate["approval_scope"], "offline")
        self.assertEqual(candidate["benchmark_id"], f.protocol["benchmark_id"])
        self.assertFalse(hasattr(f.tools, "release"))

    def test_guard_result_rejects_unverified_or_mistyped_intervention(self):
        f = self.f
        self.prepare_guard()
        for value in (False, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                f.tools.record_experiment_result("guard-experiment", replace(
                    f.result("guard-result", "guard-bundle"), intervention={
                        "information_scope": "development", "finish_requires_native_success": value}))
        self.assertEqual(f.job.state["used"]["offline_calls"], 0)

    def test_medium_observed_mismatch_required_for_simulation_plan(self):
        f = self.f
        f.opportunity()
        with self.assertRaisesRegex(ValueError, "observed medium"):
            f.tools.plan_rule_experiment("guard-simulation", opportunity_id="opportunity",
                bundle_id="guard-bundle", finish_requires_native_success=True,
                criterion_ref="criterion-v1", distinguishing_prediction="guard avoids early finish",
                cost_limit={"offline_calls": 1}, evidence_level="simulation")
        self.assertEqual(f.tools.inspect_native_finish_opportunity(["base-result"]), [])
        self.assertFalse(f.store.path("rule_trials", "guard-simulation").exists())

    def test_observed_medium_mismatch_is_evidence_not_automatic_candidate(self):
        f = self.f
        result = self.raw_medium_development()
        observed = f.tools.inspect_native_finish_opportunity([result.result_id])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["cause"], "unresolved")
        self.assertEqual(observed[0]["guard_benefit"], "unverified")
        f.opportunity(source="failure", result_ids=[result.result_id])
        plan = f.tools.plan_rule_experiment("guard-simulation", opportunity_id="opportunity",
            bundle_id="guard-bundle", finish_requires_native_success=True,
            criterion_ref="criterion-v1", distinguishing_prediction="guard avoids early finish",
            cost_limit={"offline_calls": 1}, evidence_level="simulation")
        self.assertEqual(plan["trigger_evidence"], observed)
        self.assertFalse(plan["dynamic_execution_authorized"])
        self.assertEqual(f.store.all("candidates"), [])

    def test_low_or_unknown_native_records_do_not_become_medium_opportunity(self):
        self.raw_medium_development()
        path = self.f.root / "medium-result.json"
        baseline = json.loads(path.read_text())
        for changes in ({"benchmark_success": None}, {"attempt_closed": False},
                        {"native_result_source": {}},
                        {"planner": {"model": "gpt-6.1-sol", "reasoning_effort": "low"}}):
            path.write_text(json.dumps({**baseline, **changes}))
            self.assertEqual(self.f.tools.inspect_native_finish_opportunity(["medium-result"]), [])

    def test_simulation_candidate_scope_requires_authorized_protocol_and_actual_rule_evidence(self):
        f = self.f
        source = self.raw_medium_development()
        f.protocol["dynamic_authorized"] = True
        f.opportunity(source="failure", result_ids=[source.result_id])
        f.tools.plan_rule_experiment("authorized-guard", opportunity_id="opportunity",
            bundle_id="guard-bundle", finish_requires_native_success=True, criterion_ref="criterion-v1",
            distinguishing_prediction="guard prevents early finish", cost_limit={"offline_calls": 1},
            evidence_level="simulation")
        f.tools.prepare_rule_trial("authorized-guard")
        result = f.result("guard-native-fixture", "guard-bundle", level="simulation")
        result = replace(result, intervention={"information_scope": "development",
            "finish_requires_native_success": True}, assignments=[{**result.assignments[0],
            "evidence_ref": "guard-native-fixture.json", "outcome_source": "native_benchmark"}])
        f.tools.record_experiment_result("authorized-guard", result)
        candidate = f.tools.submit_candidate("simulation-guard-candidate", experiment_id="authorized-guard")
        self.assertEqual(candidate["approval_scope"], "simulation")
        self.assertFalse(candidate["compatibility"]["runtime_loaded"])
        self.assertEqual(f.store.all("decisions"), [])


class DynamicObservedCostTests(unittest.TestCase):
    """Fake backend reports only; no LLM, GPU or simulator is dispatched."""
    def setUp(self):
        self.f = ResearchToolsTest("test_real_artifact_difference_and_frozen_theta")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.protocol["budget"] = {"token_policy": "observed_only"}
        self.hard_limits = {"whole_job_wall_seconds": 100, "gpu_device_seconds_envelope": 100,
            "model_requests": 10, "tool_calls": 20, "environment_steps": 100, "artifact_bytes": 10000}
        f.job = ResearchJob(f.store, "observed-cost-job", f.base.bundle_id,
                            {**self.hard_limits, "other_operation_calls": 10})
        f.tools = ResearchTools(f.store, f.job, f.protocol, domain_policy=f.tools.domain_policy)
        self.adapter = ResearchLLMAdapter(model="gpt-6.1-sol", reasoning_effort="medium")
        self.transport = {"effective_model": "gpt-6.1-sol", "effective_reasoning_effort": "medium",
                          "evidence_ref": "fake-supervised-transport-metadata.json"}

    def prepare(self, experiment="cost-experiment", bundle="cost-bundle", *, limits=None, level="simulation"):
        f = self.f
        f.opportunity()
        f.tools.plan_rule_experiment(experiment, opportunity_id="opportunity", bundle_id=bundle,
            chunk_steps=2, criterion_ref="criterion-v1", distinguishing_prediction="bounded fake comparison",
            cost_limit=({key: min(cap, f.job.remaining()[key]) for key, cap in self.hard_limits.items()}
                        if limits is None else limits), evidence_level=level)
        return f.tools.prepare_rule_trial(experiment)

    def costs(self):
        return {"whole_job_wall_seconds": 10, "gpu_device_seconds_envelope": 8, "model_requests": 1,
                "tool_calls": 3, "environment_steps": 12, "artifact_bytes": 1000,
                "input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 10,
                "reset_calls": 1, "planner_turns": 2}

    def result(self, result_id="cost-result", bundle="cost-bundle", *, costs=None, outcome="success"):
        f = self.f
        result = f.result(result_id, bundle, outcome=outcome, level="simulation", chunk_steps=2,
                          costs=self.costs() if costs is None else costs)
        return replace(result, assignments=[{**result.assignments[0], "evidence_ref": "fake-native-result.json",
                                            "outcome_source": "native_benchmark"}])

    def request(self, request_id="cost-request"):
        return self.adapter.prepare_request(self.f.tools, request_id=request_id, result_ids=["base-result"],
            max_input_tokens=64000, max_output_tokens=4000)

    def test_dynamic_report_preserves_observed_tokens_reset_and_only_charges_plan_projection(self):
        f = self.f
        self.prepare()
        result = self.result()
        finding = f.tools.record_experiment_result("cost-experiment", result)
        expected = {key: result.costs[key] for key in self.hard_limits}
        self.assertEqual(finding["hard_costs"], expected)
        self.assertEqual(finding["costs"], result.costs)
        self.assertEqual(finding["token_policy"], "observed_only")
        self.assertEqual(f.store.get("development_results", result.result_id)["costs"], result.costs)
        self.assertEqual(f.job.state["charges"][0]["costs"], expected)
        self.assertEqual(f.job.state["used"]["other_operation_calls"], 0)
        self.assertNotIn("input_tokens", f.job.state["used"])

    def test_missing_plan_hard_dimension_rejects_without_zero_fill_or_charge(self):
        f = self.f
        self.prepare()
        costs = self.costs()
        del costs["environment_steps"]
        with self.assertRaisesRegex(ValueError, "environment_steps"):
            f.tools.record_experiment_result("cost-experiment", self.result(costs=costs))
        self.assertEqual(f.job.state["charges"], [])
        self.assertFalse(f.store.path("research_findings", "cost-experiment").exists())

    def test_job_resources_for_other_operations_are_not_required_for_this_plan(self):
        f = self.f
        self.prepare(limits={"whole_job_wall_seconds": 100, "model_requests": 10})
        report = {"whole_job_wall_seconds": 5, "model_requests": 1,
                  "input_tokens": 9999999, "cached_input_tokens": 8000000, "output_tokens": 12000}
        finding = f.tools.record_experiment_result("cost-experiment", self.result(costs=report))
        self.assertEqual(finding["hard_costs"], {"whole_job_wall_seconds": 5, "model_requests": 1})
        self.assertEqual(f.job.state["used"]["gpu_device_seconds_envelope"], 0)

    def test_prepaid_hard_projection_and_recovery_do_not_double_charge(self):
        f = self.f
        self.prepare()
        hard = {key: self.costs()[key] for key in self.hard_limits}
        f.job.consume(hard, "experiment cost-experiment")
        result = self.result()
        first = f.tools.record_experiment_result("cost-experiment", result)
        f.job = ResearchJob(f.store, "observed-cost-job")
        f.tools = ResearchTools(f.store, f.job, f.protocol, domain_policy=f.tools.domain_policy)
        self.assertEqual(f.tools.record_experiment_result("cost-experiment", result), first)
        self.assertEqual(len(f.job.state["charges"]), 1)

    def test_hard_limit_violation_rejects_while_observed_tokens_have_no_hard_limit(self):
        f = self.f
        self.prepare(limits={"whole_job_wall_seconds": 5, "model_requests": 2})
        with self.assertRaisesRegex(ValueError, "registered resource limit"):
            f.tools.record_experiment_result("cost-experiment", self.result(costs={
                "whole_job_wall_seconds": 6, "model_requests": 1, "input_tokens": 10000000}))
        self.assertEqual(f.job.state["charges"], [])

    def test_observed_only_plan_cannot_reintroduce_token_hard_caps(self):
        f = self.f
        f.opportunity()
        with self.assertRaisesRegex(ValueError, "observed-only tokens"):
            f.tools.plan_rule_experiment("false-token-limit", opportunity_id="opportunity", bundle_id="no-bundle",
                chunk_steps=2, criterion_ref="criterion-v1", distinguishing_prediction="prediction",
                cost_limit={"input_tokens": 100}, evidence_level="simulation")

    def test_observed_only_llm_prepare_uses_request_wall_admission_and_token_references(self):
        f = self.f
        request = self.request()
        self.assertEqual(request["token_parameters_role"], "reference_only")
        self.assertEqual(request["token_policy"], "observed_only")
        self.assertEqual(request["max_input_tokens"], 64000)
        self.assertEqual(f.job.state["charges"], [])
        f.job.consume({"model_requests": 10}, "other model requests")
        with self.assertRaisesRegex(ValueError, "remaining model_requests"):
            self.request("after-exhaustion")

    def test_llm_execution_retains_full_costs_and_transport_metadata_idempotently(self):
        f = self.f
        self.request()
        costs = {"whole_job_wall_seconds": 2, "model_requests": 1, "input_tokens": 200,
                 "cached_input_tokens": 150, "output_tokens": 20}
        execution = self.adapter.record_execution(f.tools, "cost-request", costs, self.transport)
        self.assertEqual(execution["costs"], costs)
        self.assertEqual(execution["hard_costs"], {"whole_job_wall_seconds": 2, "model_requests": 1})
        self.assertEqual(self.adapter.record_execution(f.tools, "cost-request", costs, self.transport), execution)
        self.assertEqual(len(f.job.state["charges"]), 1)
        calls = [{"name": "query_development", "arguments": {"result_ids": ["base-result"]}}]
        self.assertEqual(self.adapter.validate_response(f.tools, {"request_id": "cost-request",
            "effective_model": "model-text-is-not-transport", "tool_calls": calls}), calls)
        stored = f.store.get("llm_responses", "cost-request")
        self.assertEqual(stored["effective_model"], "gpt-6.1-sol")
        self.assertEqual(stored["effective_settings_source"], "llm_executions.transport_facts")

    def test_llm_text_cannot_replace_missing_or_mismatched_transport_evidence(self):
        f = self.f
        self.request()
        with self.assertRaises(FileNotFoundError):
            self.adapter.validate_response(f.tools, {"request_id": "cost-request", "effective_model": "gpt-6.1-sol",
                "effective_reasoning_effort": "medium", "tool_calls": []})
        with self.assertRaisesRegex(ValueError, "transport evidence"):
            self.adapter.record_execution(f.tools, "cost-request", {"whole_job_wall_seconds": 2, "model_requests": 1},
                {**self.transport, "effective_reasoning_effort": "low"})
        with self.assertRaisesRegex(ValueError, "wall/request"):
            self.adapter.record_execution(f.tools, "cost-request", {"model_requests": 1}, self.transport)
        self.assertEqual(f.job.state["charges"], [])

    def test_candidate_costs_include_all_findings_llm_usage_and_other_charges_once(self):
        f = self.f
        self.prepare()
        f.tools.record_experiment_result("cost-experiment", self.result())
        self.prepare("negative-experiment", "negative-bundle")
        f.tools.record_experiment_result("negative-experiment", self.result(
            "negative-result", "negative-bundle", outcome="failure"))
        self.request()
        self.adapter.record_execution(f.tools, "cost-request", {
            "whole_job_wall_seconds": 2, "model_requests": 1, "input_tokens": 200,
            "cached_input_tokens": 150, "output_tokens": 20}, self.transport)
        f.job.consume({"tool_calls": 1}, "separate research operation")
        candidate = f.tools.submit_candidate("cost-candidate", experiment_id="cost-experiment")
        costs = candidate["costs"]
        self.assertEqual(costs["whole_job_wall_seconds"], 22)
        self.assertEqual(costs["model_requests"], 3)
        self.assertEqual(costs["tool_calls"], 7)
        self.assertEqual(costs["input_tokens"], 400)
        self.assertEqual(costs["cached_input_tokens"], 270)
        self.assertEqual(costs["output_tokens"], 40)
        self.assertEqual(costs["reset_calls"], 2)
        self.assertEqual(candidate["compatibility"]["token_policy"], "observed_only")

    def test_offline_known_cost_recording_still_rejects_unregistered_observed_extras(self):
        f = self.f
        self.prepare(limits={"model_requests": 1}, level="offline")
        result = replace(self.result(costs={"model_requests": 1, "input_tokens": 100}), evidence_level="offline")
        with self.assertRaisesRegex(ValueError, "registered resource limit"):
            f.tools.record_experiment_result("cost-experiment", result)
        exact = replace(result, costs={"model_requests": 1})
        f.tools.record_experiment_result("cost-experiment", exact)
        self.assertEqual(f.job.state["used"]["model_requests"], 1)

    def test_budget_refused_finish_declaration_does_not_prove_accepted_tool_or_guard_gain(self):
        f = self.f
        fixture = NativeFinishRuleTests()
        fixture.f = f
        source = fixture.raw_medium_development(planner_finish={"status": "success", "budget_refused": True})
        observation = f.tools.inspect_native_finish_opportunity([source.result_id])[0]
        self.assertIsNone(observation["finish_intent_accepted"])
        self.assertEqual(observation["cause"], "unresolved")
        self.assertEqual(observation["guard_benefit"], "unverified")


class FinishReservePreparationTests(unittest.TestCase):
    """The new hint and structured proposal interface, using bounded fixtures."""
    def setUp(self):
        self.harness = DynamicObservedCostTests()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.f = self.harness.f
        self.f.protocol["budget"]["token_policy"] = "observed_only; input/cached/output retained, not a hard token/spend cap"

    def observed_rejection(self):
        f = self.f
        root = f.root / "finish-budget-fixture"
        root.mkdir()
        identity = {"run_id": "fixture-run", "episode_id": "fixture-episode",
                    "bundle_id": f.base.bundle_id, "protocol_id": f.protocol["protocol_id"]}
        raw = {**identity, "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
               "planner_finish": None, "benchmark_success": False, "attempt_closed": True,
               "native_result_source": {"terminated": "fixture-native-source"}}
        (root / "result.json").write_text(json.dumps(raw))
        (root / "runtime_context.json").write_text(json.dumps({**identity, "limits": {"tool_calls": 40}}))
        entries = [{"kind": "run_start"}] + [{"kind": "tool_call", "tool": "observe"} for _ in range(40)]
        entries += [{"kind": "tool_rejected", "tool": "finish", "reason": "tool_calls exhausted"},
                    {"kind": "tool_end", "tool": "finish", "status": "rejected"}]
        (root / "action_events.jsonl").write_text("".join(json.dumps({**identity, "sequence": index, **entry}) + "\n"
                                                    for index, entry in enumerate(entries)))
        result = f.result("finish-budget-result", f.base.bundle_id, outcome="failure", level="simulation")
        result = replace(result, data_refs=[str(root / "result.json")], assignments=[{
            **result.assignments[0], "evidence_ref": str(root / "result.json"), "outcome_source": "native_benchmark"}])
        f.store.save_development_result(result)
        return result, root

    def request(self):
        result, root = self.observed_rejection()
        adapter = self.harness.adapter
        request = adapter.prepare_request(self.f.tools, request_id="reserve-request", result_ids=[result.result_id],
                                          max_input_tokens=64000, max_output_tokens=4000)
        adapter.record_execution(self.f.tools, "reserve-request", {"model_requests": 1, "whole_job_wall_seconds": 1},
                                 self.harness.transport)
        return adapter, request, root

    def proposal(self, request, *, decision="propose_finish_reserve_hint"):
        return {"schema": "rsi-research-response-v1", "request_id": request["request_id"], "decision": decision,
                "reason": "Test a budget-awareness explanation against insufficient task capability",
                "evidence_refs": [request["finish_budget_observations"][0]["action_events_ref"]],
                "competing_explanations": self.f.explanations,
                "proposed_rule": {"planner_finish_reserve_calls": 1} if decision != "defer" else None,
                "experiment": {"criterion_ref": self.f.protocol["protocol_id"],
                               "distinguishing_prediction": "fewer rejected finishes at identical dispatch cap"}
                              if decision != "defer" else None, "tool_calls": []}

    def test_qualified_observed_policy_and_actual_audit_token_names_remain_observed(self):
        h, f = self.harness, self.f
        h.prepare()
        costs = h.costs()
        costs.update(llm_input_tokens=2000000, llm_cached_input_tokens_subset=1500000)
        finding = f.tools.record_experiment_result("cost-experiment", h.result(costs=costs))
        self.assertEqual(finding["token_policy"], "observed_only")
        self.assertEqual(finding["costs"]["llm_cached_input_tokens_subset"], 1500000)
        self.assertNotIn("llm_input_tokens", finding["hard_costs"])

    def test_counted_finish_rejection_is_distinct_from_native_finish_eligibility(self):
        result, _ = self.observed_rejection()
        observations = self.f.tools.inspect_finish_budget_opportunity([result.result_id])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["counted_dispatches"], 40)
        self.assertFalse(observations[0]["finish_intent_accepted"])
        self.assertEqual(observations[0]["hint_benefit"], "unverified")
        self.assertEqual(self.f.tools.inspect_native_finish_opportunity([result.result_id]), [])

    def test_hint_factory_preserves_theta_chunk_native_guard_and_protocol(self):
        f = self.f
        result, _ = self.observed_rejection()
        f.opportunity(source="failure", result_ids=[result.result_id])
        f.tools.plan_rule_experiment("reserve-trial", opportunity_id="opportunity", bundle_id="reserve-bundle",
            planner_finish_reserve_calls=1, criterion_ref="criterion-v1", distinguishing_prediction="budget-aware finish",
            cost_limit={"whole_job_wall_seconds": 50, "model_requests": 2}, evidence_level="simulation")
        trial = f.tools.prepare_rule_trial("reserve-trial")
        rules = load_runtime_rules(trial["runtime_rules_ref"])
        self.assertEqual(rules, {"schema": "rsi-runtime-rules-v3", "chunk_steps": 5, "baseline_chunk_steps": 5,
            "finish_requires_native_success": False, "planner_finish_reserve_calls": 1})
        diff = json.loads(Path(trial["diff_ref"]).read_text())
        self.assertEqual(diff["changes"], {"planner_finish_reserve_calls": {"before": 0, "after": 1}})
        self.assertEqual(diff["frozen"]["theta_ref"], f.base.theta_ref)
        self.assertEqual(diff["frozen"]["protocol_id"], f.base.protocol_id)
        self.assertFalse(diff["requires_new_timing_baseline"])

    def test_reserve_hint_needs_exact_single_rule_and_real_counted_rejection(self):
        base = {"schema": "rsi-runtime-rules-v1", "chunk_steps": 5, "baseline_chunk_steps": 5}
        for value in (True, 0, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rule_update(base, planner_finish_reserve_calls=value)
        with self.assertRaises(ValueError):
            rule_update(base, chunk_steps=2, planner_finish_reserve_calls=1)
        result, root = self.observed_rejection()
        events = root / "action_events.jsonl"
        entries = [json.loads(line) for line in events.read_text().splitlines()]
        entries[-2]["reason"] = "wall_seconds exhausted"
        events.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        self.assertEqual(self.f.tools.inspect_finish_budget_opportunity([result.result_id]), [])

    def test_structured_proposal_eligible_or_defer_and_schema_is_serializable(self):
        adapter, request, _ = self.request()
        schema = adapter.response_schema(request["request_id"])
        self.assertEqual(schema["properties"]["request_id"]["const"], request["request_id"])
        json.dumps(schema)
        proposal = self.proposal(request)
        self.assertEqual(adapter.validate_response(self.f.tools, proposal), [])
        self.assertEqual(self.f.store.all("candidates"), [])

    def test_structured_defer_is_valid_without_manufacturing_candidate(self):
        adapter, request, _ = self.request()
        self.assertEqual(adapter.validate_response(self.f.tools, self.proposal(request, decision="defer")), [])
        self.assertEqual(self.f.store.all("candidates"), [])

    def test_structured_refs_accept_only_prepared_resolved_result_ids(self):
        adapter, request, _ = self.request()
        proposal = self.proposal(request)
        proposal["evidence_refs"] = [request["evidence"][0]["result_id"]]
        self.assertEqual(adapter.validate_response(self.f.tools, proposal), [])
        with self.assertRaisesRegex(ValueError, "legal evidence references"):
            adapter.validate_response(self.f.tools, {**proposal, "evidence_refs": ["unresolved-result-id"]})

    def test_structured_proposal_rejects_sealed_refs_unsupported_rule_and_hidden_tools(self):
        adapter, request, _ = self.request()
        proposal = self.proposal(request)
        invalid = [dict(proposal, evidence_refs=["sealed-result.json"]),
                   dict(proposal, proposed_rule={"finish_requires_native_success": True}),
                   dict(proposal, tool_calls=[{"name": "submit_candidate", "arguments": {}}]),
                   dict(proposal, decision="defer")]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                adapter.validate_response(self.f.tools, response)


if __name__ == "__main__":
    unittest.main()
