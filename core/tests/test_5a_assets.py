"""Changed asset/query/trial/qualification paths, in-process substitutes only."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
import test_research_tools
import test_core
from rsi_vla.research_tools import BoundedResearchAdapter
from rsi_vla.research_transport import unpack, model_prompt
from rsi_vla.research_cycle import execute_trial, reconcile_trial
from rsi_vla.release import ReleaseManager
from rsi_vla.storage import atomic_json


class ExperienceAndCycleTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def prepare(self):
        self.tools.domain_policy["rule_operators"] = ["planner_segment_handoff_hint"]
        # Isolated development-only L0 fixture, never introduced as historical facts.
        self.store.put(
            "evidence_views",
            "v",
            {
                "view_id": "v",
                "bundle_id": self.base.bundle_id,
                "l0": [
                    {
                        "event_id": "e1",
                        "kind": "vla_segment_end",
                        "flags": {
                            "raw_event": {
                                "stop_reason": "chunk_allowance",
                                "task_success": False,
                                "instruction": "whole task",
                                "max_chunks": 24,
                            }
                        },
                        "missing": [],
                    },
                    {
                        "event_id": "e2",
                        "kind": "vla_segment_start",
                        "flags": {
                            "raw_event": {
                                "instruction": "remaining placement",
                                "max_chunks": 12,
                            }
                        },
                        "missing": [],
                    },
                ],
                "selected_media": [],
                "policy_ref": "fixture",
            },
        )
        self.store.save_development_result(
            replace(
                self.result("with-view", self.base.bundle_id),
                intervention={
                    "information_scope": "development",
                    "evidence_view_id": "v",
                },
            )
        )
        adapter = BoundedResearchAdapter()
        transport = adapter.prepare_request(
            self.tools,
            request_id="asset-request",
            result_ids=["with-view"],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        response = {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": "asset-request",
            "decision": "prepare_rule_trial",
            "source": "normal",
            "result_ids": ["with-view"],
            "reason": "study recorded handoff",
            "external_value": "reduce avoidable handoff overhead",
            "competing_explanations": self.explanations,
            "intervention": {"planner_segment_handoff_hint": True},
            "distinguishing_prediction": "hint changes next instruction selection versus no change",
            "stop_condition": "one task or missing premise",
            "missing_evidence": ["independent task comparison"],
        }
        trial = adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )
        return adapter, transport, response, trial

    def test_query_prompt_actual_diff_and_scope(self):
        adapter, transport, response, trial = self.prepare()
        request = unpack(transport)
        self.assertIn("experience_overview", request)
        self.assertIn("Semantic overview", model_prompt(transport))
        facts = self.tools.query_experience(task_keys=["salad"], outcomes=["success"])[
            "facts"
        ]
        self.assertEqual(facts[1]["segments"][1]["instruction"], "remaining placement")
        rules = json.loads(Path(trial["runtime_rules_ref"]).read_text())
        self.assertTrue(rules["planner_segment_handoff_hint"])
        self.assertEqual(rules["chunk_steps"], 5)
        self.assertEqual(
            trial["changes"],
            {"planner_segment_handoff_hint": {"before": False, "after": True}},
        )
        self.tools.domain_policy["rule_operators"] = []
        with self.assertRaisesRegex(ValueError, "scope"):
            self.tools.plan_rule_experiment(
                "bad",
                opportunity_id="asset-request-opportunity",
                bundle_id="bad",
                planner_segment_handoff_hint=True,
                criterion_ref="criterion",
                distinguishing_prediction="prediction",
                cost_limit={"offline_calls": 1},
            )

    def test_cycle_candidate_stop_idempotent_and_independent_qualification(self):
        self.job.state["limits"]["artifact_bytes"] = 1000000
        self.job.state["used"]["artifact_bytes"] = 0
        self.job._save()
        _, _, _, trial = self.prepare()
        experiment = "asset-request-experiment"
        out = self.root / "execution"
        ctx = {
            "run_id": "trial-run",
            "bundle_id": trial["bundle_id"],
            "protocol_id": self.protocol["protocol_id"],
            "output_dir": str(out),
            "limits": dict(
                wall_seconds=10,
                gpu_seconds=10,
                model_requests=20,
                tool_calls=60,
                environment_steps=2000,
                storage_bytes=100000,
            ),
            "code_ref": "code-fixed",
        }
        plan = {"experiment_id": experiment, "context": ctx, "blockers": []}
        phase = self.root / "phase.json"
        phase_limits = {**ctx["limits"], "simulation_episodes": 2}
        atomic_json(
            phase,
            {
                "limits": phase_limits,
                "used": {k: 0 for k in phase_limits},
                "remaining": phase_limits,
            },
        )
        permit = self.root / "permit.json"
        atomic_json(
            permit,
            {
                "schema": "rsi-development-trial-permit-v1",
                "authorized": True,
                "experiment_id": experiment,
                "job_id": "job",
                "bundle_id": trial["bundle_id"],
                "protocol_id": self.protocol["protocol_id"],
                "intervention": {"planner_segment_handoff_hint": True},
                "runtime_rules": json.loads(
                    Path(trial["runtime_rules_ref"]).read_text()
                ),
                "output_dir": str(out.resolve()),
                "run_id": "trial-run",
                "limits": ctx["limits"],
                "collection": None,
                "code_ref": "code-fixed",
                "phase_snapshot_ref": str(phase),
            },
        )
        executions = []

        def executor(p):
            executions.append(p["context"]["bundle_id"])
            out.mkdir()
            atomic_json(out / "runtime_context.json", ctx)
            atomic_json(
                out / "independent_result.json",
                {"native": True, "validity": "synthetic_only"},
            )

        def collector(tools, key, output):
            raw = json.loads((Path(output) / "independent_result.json").read_text())
            result = replace(
                self.result("bound", trial["bundle_id"]),
                criteria_ref=self.protocol["protocol_id"],
                intervention={
                    "information_scope": "development",
                    "planner_segment_handoff_hint": True,
                    "runtime_loaded": True,
                },
            )
            return result, {
                "native_success": raw["native"],
                "synthetic_pipeline_valid": True,
            }

        decision = execute_trial(
            self.tools, plan, permit_path=permit, executor=executor, collector=collector
        )
        charged = json.loads(json.dumps(self.job.state["charges"]))
        self.assertIsNotNone(decision["candidate"])
        self.assertEqual(decision["candidate"]["approval_scope"], "offline")
        self.assertEqual(
            execute_trial(
                self.tools,
                plan,
                permit_path=permit,
                executor=executor,
                collector=collector,
            ),
            decision,
        )
        self.assertEqual(self.job.state["charges"], charged)
        self.assertEqual(len(executions), 1)
        self.assertEqual(
            sum(
                c["reason"].startswith("experiment ") for c in self.job.state["charges"]
            ),
            1,
        )
        self.assertEqual(
            len(json.loads(phase.read_text())["development_trial_charges"]), 1
        )
        self.assertFalse(decision["model_reanalysis"])
        queried = self.tools.query_experience(["with-view"])
        self.assertEqual(
            queried["candidates"][0]["candidate_id"],
            decision["candidate"]["candidate_id"],
        )
        self.assertEqual(queried["findings"][0]["result_id"], decision["result_id"])
        # A second fixture with insufficient native outcome stops, never mints a candidate.
        self.store.put(
            "rule_trials",
            "second",
            {**self.tools._owned("rule_trials", experiment), "experiment_id": "second"},
        )
        self.store.put(
            "research_experiments",
            "second",
            {
                **self.tools._owned("research_experiments", experiment),
                "experiment_id": "second",
            },
        )
        out2 = self.root / "execution-second"
        out2.mkdir()
        atomic_json(
            out2 / "runtime_context.json",
            {**ctx, "run_id": "trial-run2", "output_dir": str(out2)},
        )
        atomic_json(out2 / "independent_result.json", {"native": False})
        self.tools._save(
            "trial_dispatches",
            "second-dispatch",
            {"experiment_id": "second", "output": str(out2), "status": "dispatched"},
        )
        p = json.loads(permit.read_text())
        p.update(experiment_id="second", run_id="trial-run2", output_dir=str(out2))
        atomic_json(permit, p)

        def unknown(*args):
            result, checks = collector(*args)
            return replace(
                result,
                result_id="unknown",
                outcome="unknown",
                unknown_count=1,
                invalid_count=1,
                assignments=[
                    {
                        **result.assignments[0],
                        "outcome": "unknown",
                        "validity": "invalid",
                    }
                ],
            ), {"native_success": None}

        stopped = reconcile_trial(
            self.tools, "second", output=out2, permit_path=permit, collector=unknown
        )
        self.assertEqual(stopped["action"], "stop")
        self.assertIsNone(stopped["candidate"])

    def test_interrupted_native_fact_retained_and_no_training_admission(self):
        from rsi_vla.research_cycle import _collect

        _, _, _, trial = self.prepare()
        self.protocol["tasks"] = [{"task_key": "position_swap_salad",
                                   "suite": "libero_object_swap", "task_id": 2}]
        self.protocol["task_sets"]["development"]["task_keys"] = ["position_swap_salad"]
        out = self.root / "interrupted"
        out.mkdir()
        identity = {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": trial["bundle_id"],
            "protocol_id": self.protocol["protocol_id"],
        }
        atomic_json(
            out / "runtime_context.json",
            {**identity, "task": {"suite": "libero_object_swap", "task": 2, "seed": 1}},
        )
        atomic_json(
            out / "independent_result.json",
            {
                **identity,
                "validity": "technical_invalid",
                "outcome": "unknown",
                "native_task_result": {
                    "benchmark_success": True,
                    "source_ref": "native-source",
                },
            },
        )
        atomic_json(
            out / "cost_breakdown.json",
            {"actual_model_requests": 1, "tool_calls": 2, "environment_steps": None},
        )
        atomic_json(
            out / "launcher_costs.json",
            {"whole_job_wall_seconds": 3, "gpu_device_seconds_envelope": 3},
        )
        result, checks = _collect(self.tools, "asset-request-experiment", out)
        self.assertEqual(result.outcome, "unknown")
        self.assertEqual(result.invalid_count, 1)
        self.assertTrue(result.intervention["native_task_fact"]["benchmark_success"])
        self.assertNotIn("environment_steps", result.costs)
        self.assertIsNone(checks["collection_complete"])
        self.assertFalse(checks["technical_valid"])

    def test_historical_scope_and_diagnostic_request_without_behavior_operator(self):
        from rsi_vla.research import ResearchJob
        from rsi_vla.research_tools import ResearchTools

        newer = replace(self.base, bundle_id="new-engineering-base")
        self.store.save_bundle(newer)
        job = ResearchJob(
            self.store,
            "new-job",
            newer.bundle_id,
            {"offline_calls": 10, "input_tokens": 1000, "output_tokens": 100},
        )
        policy = {
            "rule_operators": [],
            "research_mode": "technical_diagnostic",
            "historical_development_bundle_ids": [self.base.bundle_id],
        }
        tools = ResearchTools(self.store, job, self.protocol, domain_policy=policy)
        self.assertEqual(
            tools.query_experience(["base-result"])["facts"][0]["bundle_id"],
            self.base.bundle_id,
        )
        adapter = BoundedResearchAdapter()
        document = adapter.prepare_request(
            tools,
            request_id="diagnostic",
            result_ids=["base-result"],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        self.assertEqual(unpack(document)["allowed_operators"], [])
        self.assertEqual(unpack(document)["research_mode"], "technical_diagnostic")


class QualificationTest(unittest.TestCase):
    setUp = test_core.CoreTest.setUp
    result = test_core.CoreTest.result

    def test_unqualified_initial_scope_accept_stale_and_rollback(self):
        unqualified = ReleaseManager(self.root / "unqualified", self.store, self.evals)
        unqualified.bootstrap("initial")
        with self.assertRaises(ValueError):
            unqualified.deployment_bundle(scope="offline", safe_point="task_boundary")
        self.evals.save_result(
            self.result("insufficient", "initial", 0.9, unknown_count=1)
        )
        with self.assertRaises(ValueError):
            unqualified.qualify_initial("bad", result_id="insufficient")
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
        bound, _ = self.release.deployment_bundle(
            scope="offline", safe_point="task_boundary"
        )
        self.release.activate("accept", safe_point="task_boundary", queues_empty=True)
        self.assertEqual(bound["bundle_id"], "initial")
        self.assertEqual(
            self.release.deployment_bundle(scope="offline", safe_point="task_boundary")[
                0
            ]["bundle_id"],
            "candidate-bundle",
        )
        self.release.rollback(
            "initial",
            safe_point="task_boundary",
            queues_empty=True,
            reason="qualified fixture",
        )
        with self.assertRaises(ValueError):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )
        # An ancestral record alone cannot grant a rollback target qualification.
        other = replace(self.bundle, bundle_id="unqualified-ancestor")
        self.store.save_bundle(other)
        self.release.records.put(
            "activations",
            "old",
            {"previous": "unqualified-ancestor", "bundle_id": "initial"},
        )
        with self.assertRaises(ValueError):
            self.release.rollback(
                "unqualified-ancestor",
                safe_point="task_boundary",
                queues_empty=True,
                reason="fixture",
            )
        with self.assertRaises(ValueError):
            self.release.deployment_bundle(
                scope="simulation", safe_point="task_boundary"
            )

    def test_stale_deployment_rejected_before_start_and_switch_uses_shared_lock(self):
        from rsi_vla.runtime import execute_runtime, dynamic_job_lock
        import test_runtime_plan

        fixture = test_runtime_plan.RuntimePlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        active = self.release.active()
        output = self.root / "never-launched"
        atomic_json(
            self.release.active_path,
            {**active, "revision": active.get("revision", 0) + 1},
        )
        # Freeze real Runtime inputs before exercising the independent release
        # revision check. No service may start for the stale effective version.
        plan = fixture.plan(output_dir=output)
        plan["blockers"] = []
        plan["context"].update(
            release_binding=active,
            deployment_release_root=str(self.release.records.root),
        )
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            execute_runtime(plan)
        self.assertFalse(output.exists())
        with dynamic_job_lock():
            with self.assertRaisesRegex(RuntimeError, "shared resource slot"):
                self.release.activate(
                    "not-dispatched", safe_point="task_boundary", queues_empty=True
                )

    def test_reject_and_pending_do_not_change_active(self):
        test_core.CoreTest.test_reject_and_inconclusive_never_change_active(self)


class CollectionEvaluatorTest(unittest.TestCase):
    def test_required_collection_changes_coverage_not_native_outcome(self):
        from unittest.mock import patch
        import test_evaluator

        fixture = test_evaluator.EvaluatorTests(
            "test_native_success_kept_with_incomplete_action_cost_evidence"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol["task_sets"]["release"]["seeds"] = [3]
        fixture.profile["collection"] = {"enabled": True, "required_for_release": True}
        template = json.loads(Path(fixture.native()[0]["result_path"]).read_text())
        manifest = []
        identities = []
        for task in fixture.protocol["tasks"]:
            folder = Path(fixture.temp.name) / task["task_key"]
            folder.mkdir()
            identity = {
                "run_id": task["task_key"],
                "episode_id": task["task_key"] + "-s3",
                "bundle_id": "initial-medium",
                "protocol_id": fixture.protocol["protocol_id"],
            }
            atomic_json(
                folder / "result.json",
                {
                    **template,
                    **identity,
                    "task": {
                        "suite": task["suite"],
                        "task": task["task_id"],
                        "seed": 3,
                    },
                },
            )
            (folder / "runtime_context.json").write_text("{}")
            manifest.append(
                {
                    "task_key": task["task_key"],
                    "seed": 3,
                    "run_id": identity["run_id"],
                    "episode_id": identity["episode_id"],
                    "result_path": str(folder / "result.json"),
                }
            )
            identities.append((folder, identity))
        checks = {
            key: True
            for key in [
                "identity",
                "model_reference",
                "vla_actual_loading_and_prediction",
                "tool_budget",
                "request_budget",
                "step_budget",
                "wall_gpu_storage_budget",
                "all_tools_use_counted_dispatcher",
                "no_mid_attempt_reset",
                "clean_completion",
                "execution_health",
                "complete_usage",
            ]
        }
        audit = {
            "passed": True,
            "costs": {},
            "notes": [],
            "checks": checks,
            "effective_rules": {},
        }
        for required, status in [
            (True, True),
            (True, False),
            (True, None),
            (False, False),
        ]:
            fixture.profile["collection"]["required_for_release"] = required
            for folder, identity in identities:
                path = folder / "dataset_manifest.json"
                if status is None:
                    path.unlink()
                else:
                    atomic_json(
                        path,
                        {
                            "identity": identity,
                            "alignment_complete": status,
                            "episodes": [{"path": "array-fixture"}],
                        },
                    )
            with (
                patch("rsi_vla.evaluator.inspect_attempt", return_value=audit),
                patch(
                    "rsi_vla.evaluator.inspect_progress",
                    return_value={"value": True, "evidence_refs": []},
                ),
                patch("rpent.flywheel.episode.validate_episode") as validate,
            ):
                result = fixture.evaluate(manifest)
            self.assertTrue(all(a["outcome"] == "success" for a in result.assignments))
            self.assertIs(
                result.hard_constraints["reference_coverage"],
                True if status is True or not required else None,
            )
            self.assertEqual(validate.call_count, 4 if status is True else 0)
        # Optional training arrays never waive the independent L0 audit.
        audit["passed"] = False
        with (
            patch("rsi_vla.evaluator.inspect_attempt", return_value=audit),
            patch(
                "rsi_vla.evaluator.inspect_progress",
                return_value={"value": True, "evidence_refs": []},
            ),
        ):
            result = fixture.evaluate(manifest)
        self.assertIsNone(result.hard_constraints["reference_coverage"])


class CollectionManifestTest(unittest.TestCase):
    def test_native_invalid_and_partial_collection_are_separate(self):
        import tempfile
        import numpy as np
        from robots.libero.flywheel import create_episode_writer
        from rsi_vla.collection import register_collection

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {
                "run_id": "r",
                "episode_id": "e",
                "bundle_id": "b",
                "protocol_id": "p",
            }
            config = {
                "root": str(root / "data"),
                "suite": "libero_object",
                "task_id": 2,
                "seed": 1,
                "collection_version": "aligned-batches-v1",
                "identity": identity,
                "flush_steps": 2,
            }
            obs = {
                "main_images": np.zeros((256, 256, 3), np.uint8),
                "wrist_images": np.zeros((256, 256, 3), np.uint8),
                "states": np.zeros(8, np.float64),
                "task_descriptions": "native task",
            }
            writer = create_episode_writer(config, obs)
            writer.begin_dispatch(np.zeros(7), source="unknown")
            writer.add_transition(np.zeros(7), obs, 1, True, False)
            writer.end_dispatch()
            writer.finalize()
            atomic_json(
                root / "independent_result.json",
                {
                    **identity,
                    "validity": "technical_invalid",
                    "outcome": "unknown",
                    "native_task_result": {"benchmark_success": True},
                    "runtime_stop_reason": "transport_failure",
                },
            )
            context = {
                **identity,
                "output_dir": str(root),
                "task": {"suite": "libero_object", "task": 2, "seed": 1},
                "information_scope": "development",
                "collection": {"root": str(root / "data")},
            }
            result = register_collection(context)
            self.assertTrue(result["alignment_complete"])
            self.assertEqual(
                result["episodes"][0]["technical_validity"], "technical_invalid"
            )
            self.assertTrue(
                result["episodes"][0]["native_task_fact"]["benchmark_success"]
            )
            self.assertEqual(
                result["episodes"][0]["training_admission"],
                "not_admitted_autonomous_execution",
            )
            pending = create_episode_writer(config, obs)
            pending.begin_dispatch(np.zeros((5, 7)), source="vla")
            result = register_collection(context)
            self.assertFalse(result["alignment_complete"])
            self.assertTrue(
                any(e["pending_dispatch"] is not None for e in result["episodes"])
            )
