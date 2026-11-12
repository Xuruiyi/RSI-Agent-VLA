"""CPU synthetic response/release checks; no experiment or model dispatch."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
import test_research_tools
import test_core
from rsi_vla.adapters.rpent import action_origin
from rsi_vla.research_tools import BoundedResearchAdapter


class BoundedResponseTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def prepare(self, key="request"):
        self.tools.domain_policy["rule_operators"] = list(
            BoundedResearchAdapter.operators
        )
        self.adapter = BoundedResearchAdapter(
            model="gpt-6.1-sol", reasoning_effort="medium"
        )
        self.adapter.prepare_request(
            self.tools,
            request_id=key,
            result_ids=["base-result"],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        return {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": key,
            "decision": "prepare_rule_trial",
            "source": "normal",
            "result_ids": ["base-result"],
            "reason": "costly successful run",
            "external_value": "less overhead at equal success",
            "competing_explanations": self.explanations,
            "intervention": {"chunk_steps": 2},
            "distinguishing_prediction": "less overshoot versus unchanged perception error",
            "stop_condition": "stop at one prepared trial or missing matched baseline",
            "missing_evidence": ["matched timing baseline"],
        }

    def test_response_artifact_candidate_and_defer_without_dispatch(self):
        response = self.prepare()
        trial = self.adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )
        rules = json.loads(Path(trial["runtime_rules_ref"]).read_text())
        self.assertEqual(rules["chunk_steps"], 2)
        self.assertEqual(self.job.state["charges"], [])
        # Inject a clearly synthetic result, not an executed experiment.
        result = replace(
            self.result("synthetic", "request-trial", chunk_steps=2),
            criteria_ref=self.protocol["protocol_id"],
            notes=["synthetic fixture"],
        )
        self.tools.record_experiment_result("request-experiment", result)
        self.tools.record_experiment_result("request-experiment", result)
        candidate = self.tools.submit_candidate(
            "synthetic-candidate", experiment_id="request-experiment"
        )
        self.assertEqual(candidate["approval_scope"], "offline")
        self.assertEqual(len(self.job.state["charges"]), 1)
        other = self.prepare("deferred")
        other.update(decision="defer", intervention=None)
        decision = self.adapter.apply_response(
            self.tools,
            other,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )
        self.assertEqual(decision["action"], "defer")
        self.assertEqual(len(self.store.all("rule_trials")), 1)

    def test_policy_evidence_and_transport_boundaries(self):
        response = self.prepare()
        bad = copy.deepcopy(response)
        bad["intervention"] = {"evaluation_threshold": 0}
        with self.assertRaises(ValueError):
            self.adapter.apply_response(
                self.tools, bad, evidence_level="offline", cost_limit={}, synthetic=True
            )
        bad["intervention"] = {"chunk_steps": 2}
        bad["result_ids"] = ["sealed"]
        with self.assertRaises(ValueError):
            self.adapter.apply_response(
                self.tools, bad, evidence_level="offline", cost_limit={}, synthetic=True
            )
        with self.assertRaises(FileNotFoundError):
            self.adapter.apply_response(
                self.tools, response, evidence_level="simulation", cost_limit={}
            )

    def test_origins(self):
        for name in ("vla_execute", "pi0_pick", "pi0_doubled"):
            self.assertEqual(action_origin(name), "vla")
        for name in (
            "move_to",
            "rotate_wrist",
            "rotate_pitch",
            "move_pose",
            "release",
            "set_gripper",
        ):
            self.assertEqual(action_origin(name), "scripted")
        self.assertEqual(action_origin("unknown_extension"), "unknown")


class DeploymentTest(unittest.TestCase):
    setUp = test_core.CoreTest.setUp
    result = test_core.CoreTest.result

    def test_next_binding_stale_scope_and_rollback(self):
        old, _ = self.release.deployment_bundle(
            scope="offline", safe_point="task_boundary"
        )
        for result in (
            self.result(),
            self.result("baseline", "initial", 0.5),
            self.result("anchor", "initial", 0.5),
        ):
            self.evals.save_result(result)
        self.release.evaluate(
            "accept",
            "candidate",
            candidate_result="candidate-result",
            baseline_result="baseline",
            anchor_result="anchor",
        )
        self.release.activate("accept", safe_point="task_boundary", queues_empty=True)
        new, binding = self.release.deployment_bundle(
            scope="offline", safe_point="task_boundary"
        )
        self.assertEqual(new["phi_ref"], "phi-v1")
        self.assertEqual(old["bundle_id"], "initial")
        self.assertEqual(binding["bundle_id"], "candidate-bundle")
        with self.assertRaises(ValueError):
            self.release.deployment_bundle(
                scope="simulation", safe_point="task_boundary"
            )
        with self.assertRaises(ValueError):
            self.release.deployment_bundle(scope="offline", safe_point="during_task")
        self.release.rollback(
            "initial", safe_point="task_boundary", queues_empty=True, reason="fixture"
        )
        # Even an identical bundle after rollback cannot resurrect an old decision.
        with self.assertRaises(ValueError):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(
            self.release.deployment_bundle(scope="offline", safe_point="task_boundary")[
                0
            ],
            old,
        )

    def test_reject_unknown_leave_selection_and_bootstrap_unqualified(self):
        test_core.CoreTest.test_reject_and_inconclusive_never_change_active(self)
        self.assertEqual(
            self.release.deployment_bundle(scope="offline", safe_point="task_boundary")[
                0
            ]["bundle_id"],
            "initial",
        )
        from rsi_vla.release import ReleaseManager

        manager = ReleaseManager(self.root / "unqualified", self.store, self.evals)
        manager.bootstrap("initial", scope="simulation")
        with self.assertRaises(ValueError):
            manager.deployment_bundle(scope="simulation", safe_point="task_boundary")


class RuntimeConsumptionTest(unittest.TestCase):
    def test_selected_bundle_changes_all_runtime_consumers_and_explicit_history(self):
        from test_runtime_plan import RuntimePlanTests
        from rsi_vla.storage import ResearchStore
        from rsi_vla.contracts import RunBundle, Candidate
        from rsi_vla.release import ReleaseManager
        from rsi_vla.evaluation import EvaluationStore
        from test_core import offline_protocol

        fixture = RuntimePlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        store = ResearchStore(fixture.artifacts / "research")
        base = RunBundle(**store.get("bundles", "initial-medium"))
        model = fixture.artifacts / "model.json"
        model.write_text(
            json.dumps({"checkpoint": str(fixture.artifacts / "new-model")})
        )
        rules = fixture.artifacts / "rules.json"
        rules.write_text(
            json.dumps(
                {
                    "schema": "rsi-runtime-rules-v3",
                    "chunk_steps": 5,
                    "baseline_chunk_steps": 5,
                    "finish_requires_native_success": False,
                    "planner_finish_reserve_calls": 1,
                }
            )
        )
        memory = store.get("memory", base.memory_ref)
        memory["upstream"]["version"] = "synthetic-legal-memory"
        store.put("memory", "new-memory", memory)
        target = replace(
            base,
            bundle_id="new-runtime",
            parent_bundle_id=base.bundle_id,
            theta_ref=str(model),
            processor_ref=str(fixture.artifacts / "new-normalization"),
            phi_ref=str(rules),
            memory_ref="new-memory",
        )
        store.save_bundle(target)
        store.save_candidate(
            Candidate(
                "runtime-candidate",
                base.bundle_id,
                target.bundle_id,
                base.protocol_id,
                "fixture-v1",
                {
                    k: getattr(target, k)
                    for k in ("theta_ref", "processor_ref", "phi_ref", "memory_ref")
                },
                [str(model), str(rules)],
                [],
                {},
                {},
            )
        )
        # Isolated offline Gate, not a simulation release protocol or production state.
        p = offline_protocol()
        p.update(protocol_id=base.protocol_id, anchor_bundle_id=base.bundle_id)
        evaluation = EvaluationStore(fixture.artifacts / "eval")
        evaluation.put("protocols", p["protocol_id"], p)
        release = ReleaseManager(fixture.artifacts / "release", store, evaluation)
        release.bootstrap(base.bundle_id)
        from rsi_vla.contracts import ExperimentResult

        for key, bundle, score in (
            ("new", target.bundle_id, 0.9),
            ("current", base.bundle_id, 0.5),
            ("anchor", base.bundle_id, 0.5),
        ):
            evaluation.save_result(
                ExperimentResult(
                    key,
                    bundle,
                    base.protocol_id,
                    "fixture-v1",
                    "release",
                    "offline",
                    {},
                    ["synthetic"],
                    "fixture",
                    {"success": score},
                    {"safety": True, "compatibility": True},
                    {},
                    candidate_id="runtime-candidate" if key == "new" else None,
                    assigned_count=10,
                )
            )
        release.evaluate(
            "accept",
            "runtime-candidate",
            candidate_result="new",
            baseline_result="current",
            anchor_result="anchor",
        )
        old_plan = fixture.plan()
        release.activate("accept", safe_point="task_boundary", queues_empty=True)
        selected, _ = release.deployment_bundle(
            scope="offline", safe_point="task_boundary"
        )
        new_plan = fixture.plan(
            bundle_id=selected["bundle_id"],
            split="deployment",
            seed=1,
            deployment_binding=release.active(),
        )
        self.assertFalse(new_plan["counts_as_release"])
        self.assertIsNone(new_plan["benchmark_id"])
        with self.assertRaises(ValueError):
            fixture.plan(bundle_id=selected["bundle_id"], split="deployment", seed=1)
        self.assertEqual(
            new_plan["environment"]["PI05_CHECKPOINT_PATH"],
            str(fixture.artifacts / "new-model"),
        )
        self.assertEqual(
            new_plan["environment"]["PI05_NORM_STATS_PATH"],
            str(fixture.artifacts / "new-normalization"),
        )
        self.assertEqual(new_plan["context"]["runtime_rules_ref"], str(rules))
        self.assertIn("synthetic-legal-memory", new_plan["command"])
        self.assertEqual(
            new_plan["context"]["memory"]["version"], "synthetic-legal-memory"
        )
        self.assertEqual(fixture.plan()["command"], old_plan["command"])
        self.assertEqual(old_plan["context"]["bundle_id"], base.bundle_id)

    def test_four_task_input_draft_consumes_selected_task(self):
        import tempfile
        from rsi_vla.__main__ import main
        from rsi_vla.runtime import prepare_runtime
        from test_runtime_plan import ROOT

        with tempfile.TemporaryDirectory() as tmp:
            main(
                [
                    "init",
                    "--artifacts",
                    tmp,
                    "--profile",
                    str(ROOT / "config/run_profile_eval_draft.json"),
                    "--protocol",
                    str(ROOT / "config/pilot_v9.json"),
                    "--bundle",
                    "draft",
                ]
            )
            p = json.loads((ROOT / "config/pilot_v9.json").read_text())
            for task in p["tasks"]:
                plan = prepare_runtime(
                    ROOT,
                    tmp,
                    protocol_path=ROOT / "config/pilot_v9.json",
                    run_id="draft-" + task["task_key"],
                    bundle_id="draft",
                    task_key=task["task_key"],
                    seed=3,
                    split="release",
                    output_dir=Path(tmp) / "unstarted",
                )
                self.assertEqual(
                    plan["context"]["task_input"]["task_language"], task["language"]
                )
                self.assertEqual(plan["context"]["limits"]["tool_calls"], 60)
                self.assertEqual(plan["context"]["limits"]["model_requests"], 20)
                self.assertFalse((Path(tmp) / "unstarted").exists())


class DevelopmentImportTest(unittest.TestCase):
    def test_unknown_invalid_replay_conflict_and_release_seed_boundary(self):
        import tempfile
        from rsi_vla.development import import_development
        from rsi_vla.storage import ResearchStore, initialize
        from rsi_vla.contracts import RunBundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            store = ResearchStore(root / "research")
            bundle = RunBundle(
                "b",
                "t",
                "p",
                "f",
                "rho",
                "psi",
                "memory",
                "protocol",
                "backend",
                "code",
            )
            initialize(
                store,
                bundle,
                psi={},
                memory={"provenance": "fixture", "legal_information": "development"},
            )
            identity = {
                "run_id": "r",
                "episode_id": "e",
                "bundle_id": "b",
                "protocol_id": "protocol",
            }

            def write(name, value):
                (run / name).write_text(json.dumps(value))

            context = {**identity, "task": {"suite": "suite", "task": 2, "seed": 1}}
            write("runtime_context.json", context)
            write(
                "independent_result.json",
                {
                    **identity,
                    "schema_version": "1.0",
                    "robot": "libero",
                    "validity": "technical_invalid",
                    "task": context["task"],
                    "benchmark_success": None,
                },
            )
            write(
                "transcript.json",
                {
                    "suite": "suite",
                    "task": 2,
                    "seed": 1,
                    "runtime": {"identity": identity},
                },
            )
            write("states.json", {"steps": []})
            (run / "action_events.jsonl").write_text("")
            write("cost_breakdown.json", {"identity": identity, "missing_cost": None})
            write("independent-audit.json", {"identity": identity, "costs": {}})
            protocol = {
                "protocol_id": "protocol",
                "benchmark_id": "benchmark",
                "scope": "simulation",
                "tasks": [{"task_key": "task", "suite": "suite", "task_id": 2}],
                "task_sets": {"development": {"task_keys": ["task"], "seeds": [1]}},
            }
            result = import_development(store, protocol, run, result_id="version2")
            self.assertEqual(result.outcome, "unknown")
            self.assertEqual(result.invalid_count, 1)
            self.assertEqual(
                import_development(store, protocol, run, result_id="version2"), result
            )
            with self.assertRaises(ValueError):
                import_development(store, protocol, run, result_id="alias")
            view = store.get("evidence_views", "version2-view")
            self.assertIn("independent_task_outcome", view["l0"][-1]["missing"])
            context["task"]["seed"] = 3
            write("runtime_context.json", context)
            with self.assertRaisesRegex(ValueError, "development"):
                import_development(store, protocol, run, result_id="release")
