"""No-Planner arms use the actual producer/finalizer/independent acceptance chain.

Env/VLA, request gate and supervisor are local CPU facts. This proves wiring and
failure semantics, and does not claim simulator task success or model capability.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from robots.libero import tools
from robots.libero.result import finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.planner.flash import FlashPlanner
from rpent.session import EnvState
from rsi_vla.adapters.runtime_acceptance import inspect_attempt
from rsi_vla.adapters.runtime_closure import finalize_runtime
from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit


class ExecutionArmAcceptanceTests(unittest.TestCase):
    def fixture(self, directory, arm="D", outcome="success"):
        patch = pytest.MonkeyPatch()
        self.addCleanup(patch.undo)
        tk, env, model, journal = make_toolkit(directory, patch)
        root = journal.output_dir
        identity = journal.identity()
        context = journal.context
        context.update(
            task={"suite": "libero_object", "task": 0, "seed": 0},
            execution_arm={
                "schema": "libero-execution-arm-v1",
                "arm": arm,
                "max_chunks": 4,
            },
            planner={
                "backend": "flash",
                "model": "gpt-6.1-sol",
                "reasoning_effort": "medium",
            },
            token_policy="observed_only",
        )
        context["limits"].update(
            model_requests=20, gpu_seconds=900, storage_bytes=1048576
        )
        if arm != "D":
            stage = {"id": "pick", "instruction": "pick up the bowl", "max_chunks": 4}
            if arm == "R":
                stage.update(on_complete="stop", on_timeout="stop", on_unknown="stop")
            context["execution_arm"]["stages"] = [stage]
        if outcome in {"success", "truncation"}:
            env.terminal = outcome
        elif outcome == "budget":
            context["limits"]["environment_steps"] = 39
        env.raw_obs = lambda: {
            "robot0_eef_pos": np.array([env.steps, 0, 0]),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "agentview_image": env.obs()["main_images"],
            "robot0_eye_in_hand_image": env.obs()["wrist_images"],
        }
        env.get_camera_meta = lambda *args, **kwargs: None
        env.get_task_language = lambda: "native task"
        env.render_camera = lambda **kwargs: (
            env.obs()["main_images"],
            np.zeros((2, 2)),
        )
        tk._state = EnvState(root)
        tk._state_output_dir = root
        tools.dump_state(tk.primitives, tk._state)
        planner = FlashPlanner(recipe_tag="unused", robot_name="libero")
        solved = planner.solve(
            system_prompt="unused", user_message="unused", toolkit=tk, max_turns=80
        )
        self.assertIsNone(solved.error)

        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        events = [
            json.loads(line) for line in journal.events_path.read_text().splitlines()
        ]
        native = [{**identity, "kind": "reset_start", "sequence": 0}]
        for event in events:
            if event["kind"] == "action_send":
                native.append(
                    {
                        **identity,
                        "sequence": len(native),
                        "kind": "action_send",
                        "source": "rlinf.policy",
                    }
                )
                native.append(
                    {
                        **identity,
                        "sequence": len(native),
                        "kind": "rpc_dispatch_end",
                        "rpc_evidence": {"client_send_sequence": event["sequence"]},
                        "execution_mask": [True] * 5,
                        "unknown_proposal_indices": [],
                    }
                )
        (root / "environment_service/action_events.jsonl").write_text(
            "\n".join(map(json.dumps, native))
        )
        write(
            "environment_service/runtime_costs.json",
            {
                "identity": identity,
                "construction_complete": True,
                "reset_calls": 1,
                "reset_completed": 1,
                "setup_steps_confirmed": 30,
                "policy_steps_confirmed": env.steps,
                "environment_steps_confirmed": 30 + env.steps,
                "environment_steps_reserved": 30 + env.steps,
                "environment_steps_unknown": 0,
                "reset_wall_seconds": 0.01,
                "service_closed": True,
                "pending_action": False,
                "last_native_success": env.terminated,
                "last_feedback_sequence": len(native) - 1,
                "native_success_source": "CPU native worker fixture",
            },
        )
        tk.close()
        metrics = tk.evaluation_metrics(success=None)
        finalize_run(
            RunFinalizationContext(
                output_dir=root,
                robot_name="libero",
                task_desc=context["task"],
                environment_success=None,
                environment_metrics=metrics,
                agent_error=None,
                elapsed_s=1,
                planner="flash",
                model="gpt-6.1-sol",
                reasoning_effort="medium",
                max_turns=80,
                planner_timeout_s=600,
                finish_result=solved.finish_result,
                stats=solved.stats,
            )
        )
        write("runtime_context.json", context)
        write(
            "request_gate.json",
            {
                "identity": identity,
                "hard_request_cap_enforced": True,
                "requests_forwarded": 0,
                "active_requests_at_close": 0,
                "cleanup_complete": True,
            },
        )
        (root / "model_requests.jsonl").write_text("")
        write(
            "supervisor.json",
            {
                "identity": identity,
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "process_group_cleanup_seconds": 0.01,
            },
        )
        write(
            "launcher_costs.json",
            {
                "identity": identity,
                "whole_job_wall_seconds": 1,
                "gpu_device_seconds_envelope": 0,
                "training_seconds": 0,
                "manual_seconds": 0,
            },
        )
        write(
            "runner_cleanup.json",
            {"identity": identity, "daemon_cleanup_seconds": 0.01},
        )
        write(
            "model_load.json",
            {
                "identity": identity,
                "loaded": True,
                "embodiment": "libero",
                "action_chunk": 5,
                "prediction_count": len(model.inputs),
                "last_actions_finite": True,
                "last_action_shape": [5, 7],
                "model_path": "/cpu-fixture/model",
                "norm_stats_path": "/cpu-fixture/norm",
                "effective_config": {
                    "pi05": True,
                    "openpi": {"config_name": "pi05_libero"},
                },
                "model_load_seconds": 0,
            },
        )
        write(
            "launch_plan.json",
            {
                "context": context,
                "environment": {
                    "PI05_CHECKPOINT_PATH": "/cpu-fixture/model",
                    "PI05_NORM_STATS_PATH": "/cpu-fixture/norm",
                },
            },
        )
        return root, context, write

    def test_no_planner_producer_chain_is_independently_accepted_for_all_frozen_arms(
        self,
    ):
        for arm in ("D", "S", "R"):
            for outcome in ("success", "truncation"):
                with (
                    self.subTest(arm=arm, outcome=outcome),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root, context, _ = self.fixture(Path(directory), arm, outcome)
                    independent = finalize_runtime(context)
                    report = inspect_attempt(root)
                    self.assertEqual(
                        independent["outcome"],
                        "success" if outcome == "success" else "failure",
                    )
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["costs"]["model_requests"], 0)
                    self.assertEqual(report["costs"]["tool_calls"], 1)
                    self.assertEqual(report["token_usage"]["input_tokens"], 0)
                    self.assertFalse(list(root.glob("codex_*.stream.jsonl")))

    def test_direct_budget_native_failure_is_accepted_without_a_planner_stop_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.fixture(Path(directory), outcome="budget")
            raw = json.loads((root / "result.json").read_text())
            self.assertTrue(raw["verified_budget_exhaustion"])
            self.assertIsNone(raw["attempt_outcome_source"]["planner_budget_stop"])
            independent = finalize_runtime(context)
            self.assertEqual(independent["outcome"], "failure", independent)
            self.assertEqual(independent["runtime_status"], "budget_exhausted")
            report = inspect_attempt(root)
            self.assertTrue(report["passed"], report)

    def test_missing_request_stream_requires_a_bound_closed_zero_request_gate(self):
        for case in (
            "zero",
            "unknown",
            "nonzero",
            "wrong_identity",
            "active",
            "unclean",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root, context, write = self.fixture(Path(directory))
                stream = root / "model_requests.jsonl"
                stream.unlink()
                gate = json.loads((root / "request_gate.json").read_text())
                if case == "unknown":
                    gate["requests_forwarded"] = None
                elif case == "nonzero":
                    gate["requests_forwarded"] = 1
                elif case == "wrong_identity":
                    gate["identity"]["run_id"] = "another-run"
                elif case == "active":
                    gate["active_requests_at_close"] = 1
                elif case == "unclean":
                    gate["cleanup_complete"] = False
                write("request_gate.json", gate)
                independent = finalize_runtime(context)
                report = inspect_attempt(root)
                costs = json.loads((root / "cost_breakdown.json").read_text())
                self.assertFalse(
                    stream.exists(),
                    "acceptance must consume the gate receipt without fabricating a stream",
                )
                self.assertTrue(
                    json.loads((root / "result.json").read_text())["benchmark_success"]
                )
                if case == "zero":
                    self.assertTrue(report["passed"], report)
                    self.assertIs(independent["technical_valid"], True)
                    self.assertEqual(independent["outcome"], "success")
                    self.assertEqual(report["costs"]["model_requests"], 0)
                    self.assertEqual(costs["input_tokens"], 0)
                    self.assertNotIn("model_requests.jsonl", independent["missing"])
                else:
                    self.assertFalse(report["passed"], report)
                    self.assertIsNot(independent["technical_valid"], True)
                    self.assertEqual(independent["outcome"], "unknown")
                    self.assertIsNone(costs["input_tokens"])

    def test_disguised_direct_arm_with_forwarded_planner_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, write = self.fixture(Path(directory))
            write(
                "request_gate.json",
                {
                    "identity": {
                        key: context[key]
                        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                    },
                    "hard_request_cap_enforced": True,
                    "requests_forwarded": 1,
                    "active_requests_at_close": 0,
                    "cleanup_complete": True,
                },
            )
            events = [
                {
                    "kind": "request_admitted",
                    "request_number": 1,
                    "model": "gpt-6.1-sol",
                    "reasoning_effort": "medium",
                },
                {
                    "kind": "response_terminal",
                    "request_number": 1,
                    "event_type": "response.completed",
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                    },
                },
                {
                    "kind": "request_returned",
                    "request_number": 1,
                    "http_status": 200,
                    "elapsed_seconds": 0.1,
                },
            ]
            (root / "model_requests.jsonl").write_text(
                "\n".join(map(json.dumps, events))
            )
            finalize_runtime(context)
            self.assertFalse(inspect_attempt(root)["passed"])

    def test_disguised_direct_arm_with_sdk_tools_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.fixture(Path(directory))
            (root / "codex_unexpected.stream.jsonl").write_text(
                json.dumps(
                    {
                        "method": "item/started",
                        "payload": {
                            "item": {
                                "id": "hidden-planner",
                                "type": "mcpToolCall",
                                "server": "rpent",
                                "tool": "vla_execute",
                            }
                        },
                    }
                )
            )
            finalize_runtime(context)
            self.assertFalse(inspect_attempt(root)["passed"])

    def test_direct_instruction_substitution_in_actual_inference_events_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.fixture(Path(directory))
            events = [
                json.loads(line)
                for line in (root / "action_events.jsonl").read_text().splitlines()
            ]
            for event in events:
                if (
                    event["kind"] in {"inference_start", "action_proposal"}
                    and "instruction" in event
                ):
                    event["instruction"] = "hidden dynamically substituted subgoal"
            (root / "action_events.jsonl").write_text(
                "\n".join(map(json.dumps, events))
            )
            finalize_runtime(context)
            self.assertFalse(inspect_attempt(root)["passed"])


if __name__ == "__main__":
    unittest.main()
