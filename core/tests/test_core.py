import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rsi_vla import (BudgetExhausted, Candidate, EventLedger, EvaluationStore,
                     ExecutionEvent, ExperimentResult, Gate, ReleaseManager,
                     ResearchJob, ResearchStore, RunBundle, TaskState, initialize,
                     load_protocol, fixed_route, fixed_research_step)
from rsi_vla.adapters.rpent import read_trajectory
from rsi_vla.evaluation import assignment_metrics


def offline_protocol():
    return {"protocol_id": "offline-test", "status": "frozen", "scope": "offline",
            "benchmark_id": "fixture-v1", "anchor_bundle_id": "initial",
            "metrics": {"success": {"min": 0, "max": 1, "direction": "max", "weight": 1}},
            "hard_constraints": {"safety": "fixture safe", "compatibility": "fixture compatible"},
            "min_gain": .1, "anchor_max_drop": .05, "cost_weights": {},
            "initial_requirements":{"metric_minima":{"success":.5},"required_old_tasks":[]}}


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = ResearchStore(self.root / "research")
        self.bundle = RunBundle("initial", "theta-v0", "processor-v0", "phi-v0", "rho-v0",
                                "psi-v0", "memory-v0", "offline-test", "backend-v0", "code-v0")
        initialize(self.store, self.bundle, psi={"learned": False},
                   memory={"provenance": "external_prior", "legal_information": ["public"]})
        self.next = replace(self.bundle, bundle_id="candidate-bundle", parent_bundle_id="initial", phi_ref="phi-v1")
        self.store.save_bundle(self.next)
        self.candidate = Candidate("candidate", "initial", "candidate-bundle", "offline-test", "fixture-v1",
                                   {"phi_ref": "phi-v1"}, ["phi-v1"], [], {}, {}, "offline")
        self.store.save_candidate(self.candidate)
        self.p = offline_protocol()
        self.evals = EvaluationStore(self.root / "evaluation")
        self.evals.put("protocols", self.p["protocol_id"], self.p)
        self.release = ReleaseManager(self.root / "release", self.store, self.evals)
        self.release.bootstrap("initial")
        self.evals.save_result(self.result("initial-qualified", "initial", .5))
        self.release.qualify_initial("initial-qualified", result_id="initial-qualified")
        self.release.establish_initial("initial-qualified",safe_point="task_boundary",queues_empty=True)

    def result(self, key="candidate-result", bundle="candidate-bundle", success=.9, **changes):
        result = ExperimentResult(key, bundle, "offline-test", "fixture-v1", "release", "offline",
                                  {}, ["fixture-evidence"], "fixture-criteria", {"success": success},
                                  {"safety": True, "compatibility": True}, {"seconds": 1},
                                  "candidate" if bundle == "candidate-bundle" else None, assigned_count=10)
        return replace(result, **changes)

    def evaluate(self, candidate=None, baseline=None, anchor=None):
        return Gate(self.p).evaluate("decision", self.candidate, current_bundle_id="initial",
             candidate_result=candidate or self.result(),
             baseline_result=baseline or self.result("baseline", "initial", .5),
             anchor_result=anchor or self.result("anchor", "initial", .5))

    def event(self, key="e0", sequence=0, **changes):
        return ExecutionEvent(key, "run", "episode", sequence, "initial", "observation", "fixture", **changes)

    def test_initial_snapshot_reloads_and_is_immutable(self):
        reopened = ResearchStore(self.store.root)
        self.assertEqual(reopened.get("memory", "memory-v0")["provenance"], "external_prior")
        self.assertFalse(reopened.get("rho", "rho-v0")["learned"])
        self.assertEqual(set(reopened.get("rho", "rho-v0")), {"rhoA", "rhoB", "rhoC", "learned"})
        with self.assertRaises(ValueError):
            reopened.put("psi", "psi-v0", {"learned": True})

    def test_rho_missing_key_with_extra_rejected(self):
        with self.assertRaises(ValueError):
            initialize(self.store, self.bundle, psi={"learned": False},
                       memory={"provenance": "external_prior", "legal_information": ["public"]},
                       rho={"rhoA": {}, "rhoB": {}, "extra": {}})

    def test_events_idempotent_and_conflict_checked(self):
        ledger = EventLedger(self.store)
        self.assertTrue(ledger.append(self.event()))
        self.assertFalse(ledger.append(self.event()))
        with self.assertRaises(ValueError):
            ledger.append(self.event(outcome="success"))
        with self.assertRaises(ValueError):
            ledger.append(self.event("other", 0))
        self.assertEqual(len(self.store.all("events")), 1)

    def test_l0_cannot_be_hidden_by_media_selection(self):
        ledger = EventLedger(self.store)
        ledger.append(self.event(media=["frame.png"], missing=["sent_action"]))
        view = ledger.view("view", "initial", selected_media=[])
        self.assertEqual(view.selected_media, [])
        self.assertEqual(len(view.l0), 1)
        self.assertEqual(view.l0[0]["missing"], ["sent_action"])
        self.assertEqual(view.l0[0]["outcome"], "unknown")
        self.assertEqual(view.l0[0]["probabilities"]["media_retention"], None)

    def test_completion_deduplicates_and_invalidation_preserves_history(self):
        ledger = EventLedger(self.store)
        ledger.append(self.event())
        ledger.append(self.event("e1", 1))
        state = TaskState(self.store, "episode")
        self.assertTrue(state.update("holding", True, "e0", criterion_ref="grasp-visible", evidence_kind="legal_observation"))
        self.assertFalse(state.update("holding", True, "e0", criterion_ref="grasp-visible", evidence_kind="legal_observation"))
        with self.assertRaises(ValueError):
            state.update("holding", False, "e0", criterion_ref="grasp-visible", evidence_kind="legal_observation")
        state.update("holding", None, "e1", criterion_ref="occluded", evidence_kind="legal_observation")
        restored = TaskState(self.store, "episode")
        self.assertFalse(restored.state["relations"]["holding"]["valid"])
        self.assertEqual(len(restored.state["history"]), 2)
        self.assertEqual(len(self.store.all("events")), 2)

    def test_tool_completion_and_heuristic_cannot_commit_state(self):
        EventLedger(self.store).append(self.event(tool_status="completed"))
        with self.assertRaises(ValueError):
            TaskState(self.store, "episode").update("holding", True, "e0", criterion_ref="tool-success", evidence_kind="heuristic")

    def test_budget_exhausts_and_recovery_does_not_reset(self):
        job = ResearchJob(self.store, "job", "initial", {"steps": 2, "seconds": 10})
        job.consume({"steps": 1}, "first")
        resumed = ResearchJob(self.store, "job")
        self.assertEqual(resumed.remaining()["steps"], 1)
        resumed.consume({"steps": 1}, "recovery")
        self.assertEqual(resumed.state["status"], "budget_exhausted")
        self.assertEqual(resumed.unknown_action(allowed_observation=True), "stop")
        with self.assertRaises(BudgetExhausted):
            resumed.run(lambda: self.fail("must not execute"), cost={"steps": 1}, reason="too late")
        with self.assertRaises(ValueError):
            ResearchJob(self.store, "job", "initial", {"steps": 3})

    def test_bounded_job_action_and_zero_budget(self):
        job = ResearchJob(self.store, "one", "initial", {"calls": 1})
        calls = []
        self.assertEqual(job.run(lambda: calls.append(1), cost={"calls": 1}, reason="allowed"), None)
        self.assertEqual(calls, [1])
        self.assertEqual(job.state["status"], "budget_exhausted")
        self.assertEqual(ResearchJob(self.store, "zero", "initial", {"gpu": 0}).unknown_action(), "stop")

    def test_unknown_action_permissions_and_over_budget_no_dispatch(self):
        job = ResearchJob(self.store, "job", "initial", {"calls": 1})
        self.assertEqual(job.unknown_action(), "stop")
        self.assertEqual(job.unknown_action(allowed_observation=True), "observe")
        self.assertEqual(job.unknown_action(allowed_recovery=True), "recover")
        with self.assertRaises(BudgetExhausted):
            job.run(lambda: self.fail("oversized action dispatched"), cost={"calls": 2}, reason="oversize")
        self.assertEqual(job.state["status"], "budget_exhausted")

    def test_fixed_rhoB_routes_only_authorized_existing_worker(self):
        rho = self.store.get("rho", self.bundle.rho_ref)
        self.assertEqual(fixed_route(rho, "vla", {"vla", "observe"}), "vla")
        self.assertEqual(fixed_route(rho, "new-worker", {"vla", "observe"}), "defer")

    def test_fixed_rhoC_inspects_experiments_or_defers_under_budget(self):
        job = ResearchJob(self.store, "rhoC-job", "initial", {"steps": 1})
        rho = self.store.get("rho", self.bundle.rho_ref)
        self.assertEqual(fixed_research_step(rho, job, evidence_available=False), "defer")
        self.assertEqual(fixed_research_step(rho, job, evidence_available=True), "inspect")
        self.assertEqual(fixed_research_step(rho, job, evidence_available=True, inspected=True), "defer")
        self.assertEqual(fixed_research_step(rho, job, evidence_available=True, inspected=True, experiment_permitted=True), "defer")
        self.assertEqual(fixed_research_step(rho, job, evidence_available=True, legal_repairs=["repair-phi"], inspected=True, experiment_permitted=True), "bounded_experiment")
        job.consume({"steps": 1}, "inspection")
        self.assertEqual(fixed_research_step(rho, job, evidence_available=True, legal_repairs=["repair-phi"], inspected=True, experiment_permitted=True), "defer")

    def test_fixed_policies_do_not_silently_ignore_changed_snapshot(self):
        rho = self.store.get("rho", self.bundle.rho_ref)
        rho["rhoB"]["route"] = "unimplemented-policy"
        with self.assertRaises(ValueError):
            fixed_route(rho, "vla", {"vla"})
        job = ResearchJob(self.store, "changed-policy-job", "initial", {"steps": 1})
        rho["rhoC"]["order"] = ["unbounded-search"]
        with self.assertRaises(ValueError):
            fixed_research_step(rho, job, evidence_available=True)

    def test_candidate_actual_diff_required(self):
        with self.assertRaises(ValueError):
            self.store.save_candidate(replace(self.candidate, candidate_id="false-diff", changes={"theta_ref": "theta-v1"}))

    def test_gate_accept_reject_unknown(self):
        self.assertEqual(self.evaluate().verdict, "accept")
        self.assertEqual(self.evaluate(candidate=self.result(success=.51)).verdict, "reject")
        self.assertEqual(self.evaluate(candidate=self.result(unknown_count=1)).verdict, "inconclusive")
        self.assertEqual(self.evaluate(candidate=self.result(metrics={"success": None})).verdict, "inconclusive")
        self.assertEqual(self.evaluate(candidate=self.result(hard_constraints={"safety": None})).verdict, "inconclusive")

    def test_known_hard_failure_rejects_even_without_comparators(self):
        d = Gate(self.p).evaluate("d", self.candidate, current_bundle_id="initial",
             candidate_result=self.result(hard_constraints={"safety": False}))
        self.assertEqual(d.verdict, "reject")
        d = Gate(self.p).evaluate("d", self.candidate, current_bundle_id="initial",
             candidate_result=self.result(candidate_id="other", hard_constraints={"safety": False}))
        self.assertEqual(d.verdict, "inconclusive")

    def test_gate_result_link_checks(self):
        for changes in ({"bundle_id": "wrong"}, {"protocol_id": "wrong"},
                        {"benchmark_id": "wrong"}, {"role": "development"},
                        {"evidence_level": "simulation"}, {"candidate_id": "wrong"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.evaluate(candidate=self.result(**changes)).verdict, "inconclusive")
        self.assertEqual(self.evaluate(anchor=self.result("anchor", "wrong", .5)).verdict, "inconclusive")

    def test_anchor_regression_and_stale_baseline(self):
        self.assertEqual(self.evaluate(anchor=self.result("anchor", "initial", 1)).verdict, "reject")
        d = Gate(self.p).evaluate("d", self.candidate, current_bundle_id="other")
        self.assertEqual(d.verdict, "inconclusive")

    def test_independent_write_boundary(self):
        with self.assertRaises(ValueError):
            self.store.save_development_result(self.result())
        with self.assertRaises(ValueError):
            self.evals.save_result(self.result(role="development"))
        self.assertFalse(hasattr(self.store, "activate"))

    def test_safe_activation_restart_and_rollback(self):
        for result in (self.result(), self.result("baseline", "initial", .5), self.result("anchor", "initial", .5)):
            self.evals.save_result(result)
        d = self.release.evaluate("accept", "candidate", candidate_result="candidate-result", baseline_result="baseline", anchor_result="anchor")
        self.assertTrue(d.release_eligible)
        with self.assertRaises(ValueError):
            self.release.activate("accept", safe_point="during_task", queues_empty=True)
        with self.assertRaises(ValueError):
            self.release.activate("accept", safe_point="task_boundary", queues_empty=False)
        self.release.activate("accept", safe_point="task_boundary", queues_empty=True)
        restored = ReleaseManager(self.root / "release", self.store, self.evals)
        self.assertEqual(restored.active()["bundle_id"], "candidate-bundle")
        restored.rollback("initial", safe_point="task_boundary", queues_empty=True, reason="fixture regression")
        self.assertEqual(restored.active()["bundle_id"], "initial")
        with self.assertRaises(ValueError):
            restored.rollback("never-active", safe_point="task_boundary", queues_empty=True, reason="bad")

    def test_reject_and_inconclusive_never_change_active(self):
        self.evals.save_result(self.result(success=.5))
        self.evals.save_result(self.result("baseline", "initial", .5))
        self.evals.save_result(self.result("anchor", "initial", .5))
        for name, refs in (("unknown", {}), ("reject", {"candidate_result": "candidate-result", "baseline_result": "baseline", "anchor_result": "anchor"})):
            d = self.release.evaluate(name, "candidate", **refs)
            self.assertIn(d.verdict, {"reject", "inconclusive"})
            with self.assertRaises(ValueError):
                self.release.activate(name, safe_point="task_boundary", queues_empty=True)
            self.assertEqual(self.release.active()["bundle_id"], "initial")

    def trajectory(self, environment_success=None):
        states = self.root / "states.json"
        transcript = self.root / "transcript.json"
        states.write_text(json.dumps({"steps": [{"step_idx": 0, "state": {}, "terminated": True,
            "command": {"action": "pi0_pick"}, "result": {"success": True}, "artifacts": ["agentview.png"]}]}))
        transcript.write_text(json.dumps({"suite": "suite", "task": 2, "seed": 0,
            "environment_success": environment_success, "finish": {"status": "success"}}))
        (self.root / "agentview.png").mkdir(exist_ok=True)
        (self.root / "agentview.png" / "00.png").write_bytes(b"fixture")
        return states, transcript

    def test_adapter_preserves_unknown_and_read_only(self):
        states, transcript = self.trajectory()
        before = [p.read_bytes() for p in (states, transcript)]
        events = read_trajectory(states, transcript, bundle_id="initial", run_id="run", episode_id="episode")
        self.assertEqual(events[-1].outcome, "unknown")
        self.assertEqual(events[0].subgoal_outcome, "unknown")
        self.assertEqual(events[0].flags["heuristic_pick_success"], True)
        self.assertEqual(events[0].proposal, None)
        self.assertEqual(before, [p.read_bytes() for p in (states, transcript)])
        self.assertTrue(events[0].media[0].endswith("agentview.png/00.png"))

    def test_adapter_unsourced_legacy_boolean_remains_unknown(self):
        states, transcript = self.trajectory(False)
        events = read_trajectory(states, transcript, bundle_id="initial", run_id="run", episode_id="episode")
        self.assertEqual(events[-1].outcome, "unknown")
        self.assertIs(events[-1].flags["legacy_environment_success"], False)

    def test_adapter_new_independent_result_schema_and_identity(self):
        states, transcript = self.trajectory()
        path = self.root / "result.json"
        value = {"schema_version": "1.0", "robot": "libero", "task": {"suite": "suite", "task": 2, "seed": 0},
                 "benchmark_success": True, "benchmark_status": "success", "terminated": True,
                 "native_result_source": {"terminated": "installed_env._check_success"}, "process_labels": {"released": None}}
        path.write_text(json.dumps(value))
        events = read_trajectory(states, transcript, bundle_id="initial", run_id="run", episode_id="episode", independent_result_path=path)
        self.assertEqual(events[-1].outcome, "success")
        self.assertIn("bundle_id_not_recorded_by_runtime", events[-1].missing)
        value["task"]["seed"] = 1
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            read_trajectory(states, transcript, bundle_id="initial", run_id="run", episode_id="episode", independent_result_path=path)

    def test_adapter_native_conflict_and_active_are_unknown(self):
        states, transcript = self.trajectory()
        path = self.root / "result.json"
        value = {"schema_version": "1.0", "robot": "libero", "task": {"suite": "suite", "task": 2, "seed": 0},
                 "benchmark_success": False, "terminated": False, "truncated": False,
                 "native_result_source": {"terminated": "native-check"}}
        path.write_text(json.dumps(value))
        events = read_trajectory(states, transcript, bundle_id="initial", run_id="run", episode_id="episode", independent_result_path=path)
        self.assertEqual(events[-1].outcome, "unknown")
        self.assertIn("conflicting_states_independent_result", events[-1].missing)
        self.assertIn("inconsistent_independent_native_result", events[-1].missing)

    def test_cli_init_consumes_immutable_profile_snapshot(self):
        import contextlib
        import io
        from rsi_vla.__main__ import main
        profile = self.root / "profile.json"
        protocol = self.root / "protocol.json"
        profile.write_text(json.dumps({"memory": {"version": "external-v1"}, "model": {"checkpoint": "model-v1"}, "control": {}}))
        protocol.write_text(json.dumps(self.p))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--artifacts", str(self.root / "cli"), "--profile", str(profile), "--protocol", str(protocol)])
        store = ResearchStore(self.root / "cli" / "research")
        bundle = store.get("bundles", "initial")
        profile.write_text(json.dumps({"model": {"checkpoint": "changed"}}))
        snapshot = Path(bundle["theta_ref"].split("#")[0])
        self.assertEqual(json.loads(snapshot.read_text())["model"]["checkpoint"], "model-v1")
        self.assertNotEqual(str(snapshot), str(profile))


class AssignmentTest(unittest.TestCase):
    def setUp(self):
        self.p = offline_protocol()
        self.p.update(scope="simulation", dynamic_authorized=True,
            metrics={"success_" + g: {"min": 0, "max": 1, "direction": "max", "weight": 1/3} for g in ("base", "controlled", "old")},
            tasks=[{"task_key": key, "group": group} for key, group in (("a", "base"), ("b", "controlled"), ("c", "old"), ("d", "old"))],
            task_sets={"release": {"task_keys": ["a", "b", "c", "d"], "seeds": [3, 4]}})
        self.c = Candidate("c", "initial", "new", "offline-test", "fixture-v1", {"phi_ref": "new"}, ["new"], [], {}, {}, "simulation")

    def result(self, key="new", success=True):
        assignments = [{"task_key": task, "seed": seed, "outcome": "success" if success else "failure", "validity": "valid",
                        "evidence_ref": "result-" + task + str(seed), "outcome_source": "native_benchmark"} for task in "abcd" for seed in (3, 4)]
        return ExperimentResult(key, "new" if key == "new" else "initial", "offline-test", "fixture-v1", "release", "simulation", {}, [], "criteria",
             {"success_" + g: 1.0 if success else 0.0 for g in ("base", "controlled", "old")}, {"safety": True, "compatibility": True}, {},
             candidate_id="c" if key == "new" else None, assigned_count=8, assignments=assignments)

    def test_matrix_recomputes_metrics_and_requires_all_assignments(self):
        r = self.result()
        self.assertEqual(assignment_metrics(self.p, r)["success_old"], 1)
        for bad in (replace(r, assignments=r.assignments[:-1]), replace(r, assignments=r.assignments+[r.assignments[0]]),
                    replace(r, metrics={"success_base": .9}), replace(r, assigned_count=7)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                assignment_metrics(self.p, bad)

    def test_unknown_invalid_and_exhaustion_success_cannot_be_hidden(self):
        r = self.result()
        for field, value in (("outcome", "unknown"), ("validity", "invalid"), ("outcome_source", "verified_budget_exhaustion")):
            rows = copy.deepcopy(r.assignments)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                assignment_metrics(self.p, replace(r, assignments=rows))

    def test_no_dynamic_authorization_is_inconclusive(self):
        self.p["dynamic_authorized"] = False
        d = Gate(self.p).evaluate("d", self.c, current_bundle_id="initial", candidate_result=self.result(), baseline_result=self.result("base", False), anchor_result=self.result("anchor", False))
        self.assertEqual(d.verdict, "inconclusive")
        self.assertFalse(d.release_eligible)

    def test_critical_old_minimum_is_computed_not_declared(self):
        r = self.result(success=False)
        d = Gate(self.p).evaluate("d", self.c, current_bundle_id="initial", candidate_result=r, baseline_result=self.result("base", False), anchor_result=self.result("anchor", False))
        self.assertEqual(d.verdict, "reject")


if __name__ == "__main__":
    unittest.main()
