"""Offline launcher projections bind the actual deterministic/dynamic consumers."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_runtime_plan

from robots.libero.robot_spec import get_robot_spec
from rpent.cli.main import _build_argparser


class ExecutionArmRuntimeTests(unittest.TestCase):
    setUp = test_runtime_plan.RuntimePlanTests.setUp
    plan = test_runtime_plan.RuntimePlanTests.plan

    def configure(self, arm, *, backend=None, routes=None):
        path = self.artifacts / "research/profiles/initial-medium.json"
        profile = json.loads(path.read_text())
        profile["planner"]["backend"] = backend or ("codex" if arm == "A" else "flash")
        spec = {"schema": "libero-execution-arm-v1", "arm": arm, "max_chunks": 80}
        if arm in {"S", "R"}:
            spec["stages"] = [
                {
                    "id": "pick",
                    "instruction": "pick up the salad bowl",
                    "max_chunks": 24,
                },
                {
                    "id": "place",
                    "instruction": "put the salad bowl in the basket",
                    "max_chunks": 56,
                },
            ]
            if arm == "R":
                for index, stage in enumerate(spec["stages"]):
                    next_id = "place" if index == 0 else "stop"
                    stage.update(
                        on_complete=next_id, on_timeout="stop", on_unknown="stop"
                    )
        profile["execution_arm"] = spec
        if routes is not None:
            profile["relation_routes"] = routes
        path.write_text(json.dumps(profile))
        return profile

    def parse_actual_cli(self, plan):
        parser = _build_argparser()
        spec = get_robot_spec()
        spec.add_cli_args(parser, use_dashboard=False)
        args = parser.parse_args(plan["command"][3:])
        # This is the actual pre-service robot validation, including local/HF
        # memory compatibility; no model, simulator or service is loaded.
        with patch(
            "rpent.session.runtime.runtime_context", return_value=plan["context"]
        ):
            spec.validate_args(args)
            config = spec.parse_config(args)
        return args, config

    def test_each_arm_projects_into_its_actual_cli_consumer(self):
        for arm in ("D", "S", "R", "A"):
            with self.subTest(arm=arm):
                profile = self.configure(arm)
                plan = self.plan()
                args, config = self.parse_actual_cli(plan)
                self.assertEqual(
                    plan["context"]["execution_arm"], profile["execution_arm"]
                )
                self.assertEqual(args.planner, "codex" if arm == "A" else "flash")
                self.assertEqual(args.memory_profile, "hf" if arm == "A" else "local")
                self.assertEqual(
                    plan["context"]["execution_information"]["online_planner"],
                    arm == "A",
                )
                self.assertIs(
                    plan["context"]["execution_information"][
                        "private_evaluation_truth"
                    ],
                    False,
                )
                self.assertEqual(
                    config.task_desc["suite"], plan["context"]["task"]["suite"]
                )
                if arm != "A":
                    self.assertIsNone(args.memory_repo)
                    self.assertEqual(args.memory_version, "auto")
                    self.assertNotIn("relation_control", plan["context"])
                    self.assertEqual(
                        plan["context"]["execution_information"][
                            "model_requests_expected"
                        ],
                        0,
                    )
                self.assertFalse((self.artifacts / "run").exists())

    def test_direct_keeps_the_common_physical_budget_model_and_processor(self):
        self.configure("A")
        dynamic = self.plan()
        self.configure("D")
        direct = self.plan()
        for field in ("theta_ref", "processor_ref", "phi_ref", "gpu_devices", "limits"):
            self.assertEqual(
                direct["context"].get(field), dynamic["context"].get(field)
            )
        self.assertEqual(direct["environment"], dynamic["environment"])
        for flag in (
            "--max-episode-steps",
            "--planner-timeout-s",
            "--am-backend",
            "--cuda-device",
        ):
            self.assertEqual(
                direct["command"][direct["command"].index(flag) + 1],
                dynamic["command"][dynamic["command"].index(flag) + 1],
            )
        self.assertEqual(direct["context"]["planner"]["model"], "gpt-6.1-sol")
        self.assertEqual(direct["context"]["planner"]["reasoning_effort"], "medium")

    def test_wrong_planner_and_hidden_routes_reject_before_output_creation(self):
        for arm, backend in (("D", "codex"), ("A", "flash")):
            with self.subTest(arm=arm):
                self.configure(arm, backend=backend)
                with self.assertRaisesRegex(ValueError, "actual planner consumer"):
                    self.plan()
        self.configure("D", routes=[{"route_id": "hidden-worker"}])
        with self.assertRaisesRegex(ValueError, "no rhoB routing"):
            self.plan()
        self.assertFalse((self.artifacts / "run").exists())

    def test_empty_route_list_does_not_create_an_empty_rhob_projection(self):
        self.configure("D", routes=[])
        self.assertNotIn("relation_control", self.plan()["context"])

    def test_frozen_hf_startup_input_is_only_projected_to_the_dynamic_arm(self):
        from rpent.cli import main as cli

        options = {
            "version": "libero-startup-v2-bounded-vla",
            "task_language": "Pick the salad dressing and place it in the basket",
            "source": "frozen HF startup fixture; Env remains authoritative",
            "task_scope": {"suite": "libero_object_swap", "task": 2},
        }
        for arm in ("D", "S", "R", "A"):
            with self.subTest(arm=arm):
                profile = self.configure(arm)
                profile["task_inputs"] = {"position_swap_salad": options}
                profile_path = self.artifacts / "research/profiles/initial-medium.json"
                profile_path.write_text(json.dumps(profile))
                plan = self.plan(output_dir=self.artifacts / ("startup-" + arm))
                args, _ = self.parse_actual_cli(plan)
                if arm == "A":
                    self.assertEqual(plan["context"]["task_input"], options)
                    self.assertEqual(args.memory_profile, "hf")
                    self.assertTrue(args.memory_repo)
                    continue
                self.assertNotIn("task_input", plan["context"])
                self.assertEqual(args.memory_profile, "local")
                runtime_boundary = []

                def stop_before_services(*args, **kwargs):
                    runtime_boundary.append(True)
                    raise SystemExit("reached-runtime-boundary")

                robot = replace(get_robot_spec(), init_runtime=stop_before_services)
                # Exercise the real CLI through prompt/memory/planner startup.
                # Stop at the first service boundary, before any GPU work.
                with (
                    patch(
                        "rpent.session.runtime.runtime_context",
                        return_value=plan["context"],
                    ),
                    patch.object(cli, "get_robot_spec", return_value=robot),
                    patch("sys.argv", plan["command"][2:]),
                    patch(
                        "robots.libero.task_input.build_task_input",
                        side_effect=AssertionError(
                            "frozen arms must not build Planner startup input"
                        ),
                    ) as startup,
                ):
                    with self.assertRaisesRegex(SystemExit, "reached-runtime-boundary"):
                        cli.main()
                    startup.assert_not_called()
                self.assertEqual(runtime_boundary, [True])

    def test_legacy_plan_without_arm_keeps_original_consumer_and_memory(self):
        plan = self.plan()
        self.assertNotIn("execution_arm", plan["context"])
        self.assertNotIn("execution_information", plan["context"])
        args, _ = self.parse_actual_cli(plan)
        self.assertEqual((args.planner, args.memory_profile), ("codex", "hf"))
        self.assertTrue(args.memory_repo)

    def test_common_execution_projects_after_effective_permit_bounds(self):
        profile = self.configure("D")
        profile["common_execution"] = {
            "schema": "libero-common-execution-v1",
            "observation_cadence": "prediction",
            "allowed_motion_tools": ["vla_execute"],
            "action_prefix_steps": 5,
        }
        path = self.artifacts / "research/profiles/initial-medium.json"
        path.write_text(json.dumps(profile))
        protocol = json.loads(
            (test_runtime_plan.ROOT / "config/pilot_v2.json").read_text()
        )
        protocol["budget"]["per_attempt_proposal"]["tool_calls"] = 512
        protocol_path = self.artifacts / "common-protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        plan = self.plan(protocol_path=protocol_path)
        self.assertEqual(
            plan["context"]["common_execution"], profile["common_execution"]
        )
        self.assertFalse((self.artifacts / "run").exists())
        permit = {
            key: plan["context"][key]
            for key in ("run_id", "bundle_id", "protocol_id", "gpu_devices")
        }
        permit.update(
            authorized=True,
            task_key="position_swap_salad",
            seed=0,
            simulation_episodes=1,
            limits={**plan["context"]["limits"], "tool_calls": 1},
        )
        permit_path = self.artifacts / "permit.json"
        permit_path.write_text(json.dumps(permit))
        with self.assertRaisesRegex(ValueError, "physical bound"):
            self.plan(protocol_path=protocol_path, authorization_path=permit_path)

    def test_legacy_plan_does_not_enable_common_execution(self):
        self.assertNotIn("common_execution", self.plan()["context"])

    def test_registered_task_program_selects_language_and_retains_whole_B0_mapping(
        self,
    ):
        from copy import deepcopy

        profile = self.configure("S")
        programs = {}
        for key, language in (
            ("base_salad", "pick salad dressing"),
            ("position_swap_salad", "pick salad dressing from its current position"),
            ("old_milk", "pick milk"),
            ("old_drawer", "open middle drawer"),
        ):
            spec = deepcopy(profile["execution_arm"])
            spec["stages"][0]["instruction"] = language
            programs[key] = spec
        profile["execution_arm_by_task"] = programs
        profile_path = self.artifacts / "research/profiles/initial-medium.json"
        profile_path.write_text(json.dumps(profile))
        protocol = json.loads(
            (test_runtime_plan.ROOT / "config/pilot_v2.json").read_text()
        )
        protocol_path = self.artifacts / "mapped-protocol.json"
        for task_key in programs:
            with self.subTest(task_key=task_key):
                protocol["next_step_proposal"]["task"] = task_key
                protocol_path.write_text(json.dumps(protocol))
                plan = self.plan(protocol_path=protocol_path, task_key=task_key)
                self.assertEqual(plan["context"]["execution_arm"], programs[task_key])
                self.assertEqual(
                    plan["context"]["execution_binding"]["profile"][
                        "execution_arm_by_task"
                    ],
                    programs,
                )
                args, _ = self.parse_actual_cli(plan)
                self.assertEqual(args.planner, "flash")
        self.assertFalse((self.artifacts / "run").exists())

    def test_task_mapping_never_falls_back_to_another_tasks_instruction(self):
        profile = self.configure("S")
        profile["execution_arm_by_task"] = {"old_milk": profile["execution_arm"]}
        path = self.artifacts / "research/profiles/initial-medium.json"
        path.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "no frozen execution program"):
            self.plan()
        profile["execution_arm_by_task"]["position_swap_salad"] = {
            "schema": "libero-execution-arm-v1",
            "arm": "D",
            "max_chunks": 80,
        }
        path.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "same arm"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
