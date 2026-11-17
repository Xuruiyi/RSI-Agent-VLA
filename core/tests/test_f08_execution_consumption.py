# Copyright 2026 RSI-Agent-VLA Authors.
"""F08 updates reach the existing Runtime, Planner and LIBERO dispatch chain."""

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_relation_runtime_wiring
import test_runtime_plan
from robots.libero.skill_routing import record_skill_observation
from robots.libero.task_input import GLOBAL_MEMORY, V2_EXCERPTS, build_task_input
from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session.runtime import RuntimeJournal
from rpent.tools import ToolResult
from rpent.utils import templates
from rsi_vla.contracts import ExecutionEvent, ExperimentResult, RunBundle
from rsi_vla.evidence import EventLedger
from rsi_vla.relation_learning import (
    freeze_relations,
    learn_relations,
    version_signature,
)
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.rule_updates import write_rule_trial
from rsi_vla.runtime import prepare_runtime
from rsi_vla.storage import ResearchStore, atomic_json
from tests.unit_tests.robots.libero.test_bounded_vla import driver

ROOT = Path(__file__).resolve().parents[2]


class ExecutionUpdateConsumersTests(unittest.TestCase):
    setUp = test_runtime_plan.RuntimePlanTests.setUp
    plan = test_runtime_plan.RuntimePlanTests.plan
    route = test_relation_runtime_wiring.RelationRuntimeProjectionTests.route

    def configure(self):
        self.store = ResearchStore(self.artifacts / "research")
        self.base = RunBundle(**self.store.get("bundles", "initial-medium"))
        self.profile = json.loads(Path(self.base.backend_ref).read_text())
        self.profile["relation_routes"] = [self.route()]
        self.profile["task_input"] = {
            "version": "libero-startup-v2-bounded-vla",
            "task_scope": {"suite": "libero_object_swap", "task": 2},
            "task_language": "Pick the salad dressing and place it in the basket",
        }
        atomic_json(self.base.backend_ref, self.profile)
        self.protocol = json.loads((ROOT / "config/pilot_v2.json").read_text())
        self.job = ResearchJob(
            self.store,
            "offline-learning",
            self.base.bundle_id,
            {"input_tokens": 100, "output_tokens": 10},
        )
        self.tools = ResearchTools(
            self.store, self.job, self.protocol, domain_policy={}
        )

    def feedback(self, *, known=True, technical_valid=True):
        ledger = EventLedger(self.store)
        signature = version_signature(self.base)
        for index in range(4):
            flags = {}
            if known and index >= 2:
                flags["route_feedback"] = {
                    "evidence_kind": "independent_criterion",
                    "criterion_ref": "fixture-independent-lift",
                    "state": self.route()["state"],
                    "worker": "pi0_pick" if index == 2 else "vla_execute",
                    "success": index == 3,
                    "version_signature": signature,
                    "start_event_id": "feedback-0",
                    "handoff_event_id": "feedback-1",
                }
            ledger.append(
                ExecutionEvent(
                    "feedback-" + str(index),
                    "feedback-run",
                    "feedback-episode",
                    index,
                    self.base.bundle_id,
                    "independent_skill_review" if index >= 2 else "observation_update",
                    "offline-independent-fixture",
                    flags=flags,
                    missing=[] if known else ["missing independent handoff criterion"],
                )
            )
        self.store.put(
            "evidence_views", "feedback", ledger.view("feedback", self.base.bundle_id)
        )
        result = ExperimentResult(
            "feedback-result",
            self.base.bundle_id,
            self.protocol["protocol_id"],
            self.protocol["task_sets"]["development"]["benchmark_id"],
            "development",
            "offline",
            {
                "information_scope": "development",
                "evidence_view_id": "feedback",
                "runtime_loaded": True,
                "identity": {
                    "run_id": "feedback-run",
                    "episode_id": "feedback-episode",
                },
                "screening_checks": {"technical_valid": technical_valid},
            },
            ["offline-independent-fixture"],
            "fixture-independent-lift",
            {},
            {},
            {},
            assigned_count=1,
            outcome="unknown",
            assignments=[
                {
                    "task_key": "position_swap_salad",
                    "seed": 1,
                    "outcome": "unknown",
                    "validity": "valid" if technical_valid else "invalid",
                }
            ],
        )
        self.store.save_development_result(result)
        return result

    def runtime_plan(self, artifacts, bundle_id, name):
        return prepare_runtime(
            ROOT,
            artifacts,
            protocol_path=ROOT / "config/pilot_v2.json",
            run_id=name,
            bundle_id=bundle_id,
            task_key="position_swap_salad",
            seed=1,
            split="development",
            output_dir=self.artifacts / name,
        )

    def toolkit(self, plan):
        context = deepcopy(plan["context"])
        journal = RuntimeJournal(context)
        primitive, env, model = driver(journal=journal)
        private = RuntimeJournal(
            {**context, "output_dir": str(journal.output_dir / "environment_service")}
        )
        env.rsi_skill_observation = lambda *, evidence: record_skill_observation(
            private,
            {
                "salad_dressing_1_pos": np.array([0.0, 0.0, env.steps / 100]),
                "robot0_eef_pos": np.array([0.0, 0.0, env.steps / 100]),
                "robot0_gripper_qpos": np.array([0.02, -0.02]),
            },
            evidence,
        )

        def init(toolkit, **kwargs):
            toolkit._primitives = primitive
            primitive._check_cancelled = toolkit.raise_if_cancelled
            toolkit._action_frame_cursor = 0

        with (
            patch.object(RuntimeJournal, "from_environment", return_value=journal),
            patch.object(LiberoToolkit, "init_primitives", init),
            patch.object(
                templates,
                "default_variables",
                return_value={"output_dir": str(journal.output_dir)},
            ),
        ):
            toolkit = LiberoToolkit(
                runtime_kwargs={},
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(journal.output_dir / "memory"),
                state_output_dir=journal.output_dir / "state",
            )
        toolkit._capture_env_state = lambda *, result, **kwargs: ToolResult(
            data={"execution_segment": result}
        )
        return toolkit, env, model, journal

    def invoke(self, toolkit, env, model):
        result = toolkit.execute_tool(
            "vla_execute", {"instruction": self.route()["instruction"], "max_chunks": 1}
        ).data
        self.assertEqual(result["tool_status"], "completed")
        self.assertEqual(result["native_outcome"], "unknown")
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual([len(value) for value in env.sent], [5])
        self.assertEqual(
            model.inputs[0]["task_descriptions"], self.route()["instruction"]
        )
        return result["execution_segment"]["relation_route"]["selected_worker"]

    def test_rhob_learning_freezing_and_new_runtime_change_actual_worker_only(self):
        self.configure()
        before_policy = self.store.get("rho", self.base.rho_ref)
        old_plan = self.runtime_plan(self.artifacts, self.base.bundle_id, "old-job")
        old_toolkit, old_env, old_model, old_journal = self.toolkit(old_plan)
        self.feedback()
        destination_root = self.artifacts / "derived"
        destination = ResearchStore(destination_root / "research")
        state = learn_relations(
            self.tools,
            destination,
            "learn-routing",
            ["feedback-result"],
            relations=["rhoB"],
        )
        self.assertEqual(len(state["feedback"]), 2)
        target = freeze_relations(
            self.tools,
            destination,
            "learn-routing",
            rho_ref="rho-learned",
            bundle_id="routing-child",
        )
        new_plan = self.runtime_plan(destination_root, target.bundle_id, "new-job")
        new_toolkit, new_env, new_model, new_journal = self.toolkit(new_plan)
        self.assertEqual(self.invoke(old_toolkit, old_env, old_model), "pi0_pick")
        self.assertEqual(self.invoke(new_toolkit, new_env, new_model), "vla_execute")
        self.assertEqual(self.store.get("rho", self.base.rho_ref), before_policy)
        self.assertEqual(old_journal.context["rho_ref"], self.base.rho_ref)
        self.assertEqual(new_journal.context["rho_ref"], target.rho_ref)
        self.assertFalse((destination_root / "active.json").exists())
        self.assertEqual(
            self.job.state["used"], {"input_tokens": 0, "output_tokens": 0}
        )
        # Changing an executable dependency cannot reuse measured route preference.
        changed = replace(
            target, bundle_id="new-code", code_ref=target.code_ref + "-changed"
        )
        destination.save_bundle(changed)
        incompatible = self.runtime_plan(
            destination_root, changed.bundle_id, "incompatible-job"
        )
        toolkit, env, model, _ = self.toolkit(incompatible)
        self.assertEqual(self.invoke(toolkit, env, model), "pi0_pick")

    def test_unknown_technical_feedback_keeps_original_route_and_cannot_freeze(self):
        self.configure()
        self.feedback(known=False, technical_valid=False)
        destination = ResearchStore(self.artifacts / "unknown-derived/research")
        state = learn_relations(
            self.tools,
            destination,
            "unknown-routing",
            ["feedback-result"],
            relations=["rhoB"],
        )
        self.assertEqual(state["feedback"], [])
        self.assertEqual(state["snapshot"]["rhoB"]["values"], {})
        with self.assertRaisesRegex(ValueError, "freezing needs feedback"):
            freeze_relations(
                self.tools,
                destination,
                "unknown-routing",
                rho_ref="not-learned",
                bundle_id="not-created",
            )
        self.assertFalse(destination.path("bundles", "not-created").exists())
        toolkit, env, model, _ = self.toolkit(
            self.runtime_plan(self.artifacts, self.base.bundle_id, "unchanged-job")
        )
        self.assertEqual(self.invoke(toolkit, env, model), "pi0_pick")

    def test_phi_trial_reaches_actual_planner_input_and_old_job_stays_frozen(self):
        self.configure()
        old_plan = self.runtime_plan(self.artifacts, self.base.bundle_id, "old-phi-job")
        old_toolkit, old_env, old_model, old = self.toolkit(old_plan)
        hint = (
            "Inspect the visible target label before selecting the next existing skill."
        )
        target, difference = write_rule_trial(
            self.store,
            self.base,
            bundle_id="phi-child",
            artifact_id="phi-child",
            planner_task_hint=hint,
        )
        new_plan = self.runtime_plan(self.artifacts, target.bundle_id, "new-phi-job")
        new_toolkit, new_env, new_model, new = self.toolkit(new_plan)
        memory = (
            self.artifacts
            / self.profile["memory"]["revision"]
            / self.profile["memory"]["version"]
        )
        for name in GLOBAL_MEMORY:
            path = memory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(V2_EXCERPTS[name]))
        before = build_task_input(
            old_plan["context"], memory, old_plan["context"]["task"]
        )
        after = build_task_input(
            new_plan["context"], memory, new_plan["context"]["task"]
        )
        self.assertEqual(after["system_prompt"], before["system_prompt"])
        self.assertEqual(
            after["user_message"],
            before["user_message"]
            + "\n\nAdditional Planner instruction from the runtime rule:\n"
            + hint,
        )
        self.assertEqual(old.planner_task_hint, "")
        self.assertEqual(new.planner_task_hint, hint)
        self.assertEqual(
            old.chunk_steps or self.profile["control"]["action_chunk"], new.chunk_steps
        )
        self.assertEqual(new.chunk_steps, 5)
        self.assertEqual(self.invoke(old_toolkit, old_env, old_model), "pi0_pick")
        self.assertEqual(self.invoke(new_toolkit, new_env, new_model), "pi0_pick")
        self.assertEqual(set(difference["changes"]), {"planner_task_hint"})
        self.assertEqual(new.context["limits"], old.context["limits"])
        self.assertEqual(target.rho_ref, self.base.rho_ref)
        self.assertEqual(target.theta_ref, self.base.theta_ref)
        self.assertFalse((self.artifacts / "active.json").exists())


if __name__ == "__main__":
    unittest.main()
