# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Whole-episode consumers exercise actual toolkit, VLA and result contracts."""

import json

import numpy as np
import pytest

from robots.libero.execution_arms import run_execution_arm, validate_execution_arm
from robots.libero.result import finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.planner.flash import FlashPlanner
from rpent.tools import ToolResult
from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    tk.get_env_state = lambda *, result, **kwargs: ToolResult(
        data={"execution_segment": result}
    )
    return tk, env, model, journal


def direct(max_chunks=3):
    return {"schema": "libero-execution-arm-v1", "arm": "D", "max_chunks": max_chunks}


def test_direct_uses_native_instruction_latest_images_and_frozen_prefix(setup):
    tk, env, model, journal = setup
    journal.chunk_steps = 3
    result = run_execution_arm(tk, direct()).data
    assert [obs["task_descriptions"] for obs in model.inputs] == ["native task"] * 3
    assert [obs["states"][0] for obs in model.inputs] == [0, 3, 6]
    assert [int(obs["main_images"][0, 0, 0]) for obs in model.inputs] == [0, 3, 6]
    assert [len(chunk) for chunk in env.sent] == [3, 3, 3]
    assert result["executed_steps"] == 9
    assert result["native_success"] is None and result["status"] == "unknown"
    assert result["decisions"][0]["after"]["image_shapes"]["main_images"] == [2, 2, 3]
    assert journal.costs()["tool_calls"] == 3
    events = [json.loads(row) for row in journal.events_path.read_text().splitlines()]
    assert not any(
        row["kind"] in {"relation_route_decision", "planner_request"} for row in events
    )


@pytest.mark.parametrize("route", [{}, {"routes": []}])
def test_direct_refuses_even_empty_rhob_projection(setup, route):
    tk, env, model, journal = setup
    journal.context["relation_control"] = route
    with pytest.raises(ValueError, match="no rhoB"):
        run_execution_arm(tk, direct())
    assert not model.inputs and not env.sent


def stage(id, instruction, max_chunks=1, **kwargs):
    return {"id": id, "instruction": instruction, "max_chunks": max_chunks, **kwargs}


def test_sequence_freezes_order_instructions_and_observed_completion(setup):
    tk, env, model, journal = setup
    spec = {
        **direct(5),
        "arm": "S",
        "stages": [
            stage(
                "grasp",
                "pick bowl",
                max_chunks=4,
                completion={"signal": "gripper_opening", "op": "ge", "value": 0.07},
            ),
            stage("place", "put bowl in basket", max_chunks=2),
        ],
    }
    result = run_execution_arm(tk, spec).data
    assert [o["task_descriptions"] for o in model.inputs] == [
        "pick bowl",
        "put bowl in basket",
        "put bowl in basket",
    ]
    assert result["decisions"][0]["edge"] == "on_complete"
    assert result["decisions"][0]["after"]["gripper_opening"] == pytest.approx(0.08)
    assert (
        result["stop_reason"] == "program_exhausted"
        and result["native_success"] is None
    )
    assert env.steps == 15


@pytest.mark.parametrize(
    "gap,expected", [(0.02, "place bowl"), (0.08, "recover grasp")]
)
def test_conditional_policy_selects_frozen_branch_from_real_state(setup, gap, expected):
    tk, env, model, journal = setup
    original = tk.primitives.set_obs

    def perceive(obs):
        obs["states"][6:8] = [gap / 2, -gap / 2]
        original(obs)

    tk.primitives.set_obs = perceive
    spec = {
        **direct(4),
        "arm": "R",
        "stages": [
            stage(
                "grasp",
                "pick bowl",
                completion={"signal": "gripper_opening", "op": "le", "value": 0.04},
                on_complete="place",
                on_timeout="recover",
                on_unknown="stop",
            ),
            stage(
                "place",
                "place bowl",
                on_complete="stop",
                on_timeout="stop",
                on_unknown="stop",
            ),
            stage(
                "recover",
                "recover grasp",
                on_complete="grasp",
                on_timeout="stop",
                on_unknown="stop",
            ),
        ],
    }
    result = run_execution_arm(tk, spec).data
    assert model.inputs[1]["task_descriptions"] == expected
    assert result["decisions"][0]["next_stage"] == (
        "place" if gap < 0.04 else "recover"
    )
    assert len(model.inputs) <= 4


@pytest.mark.parametrize(
    "case,status,outcome",
    [
        ("success", "completed", True),
        ("truncation", "completed", False),
        ("technical", "technical_error", None),
        ("budget", "budget_exhausted", None),
        ("unknown", "unknown", None),
        ("rpc_unknown", "technical_error", None),
    ],
)
def test_hook_and_finalizer_preserve_native_failure_budget_technical_and_unknown(
    setup, case, status, outcome
):
    tk, env, model, journal = setup
    journal.context["execution_arm"] = direct()
    if case in {"success", "truncation"}:
        env.terminal = case
    elif case == "technical":
        model.predict = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("prediction failed")
        )
    elif case == "budget":
        journal.context["limits"]["environment_steps"] = 39
    elif case == "rpc_unknown":
        env.rsi_chunk_step = lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("receipt missing")
        )
    # A normal Flash planner calls the robot hook, with no SDK/model Planner.
    planner = FlashPlanner(recipe_tag="unused", robot_name="libero")
    result = planner.solve(
        system_prompt="unused", user_message="unused", toolkit=tk, max_turns=1
    )
    assert result.error is None
    episode = tk._episode_execution
    assert episode["status"] == status and episode["native_success"] is outcome
    if case == "rpc_unknown":
        assert episode["unknown_steps"] == 5
    service = journal.output_dir / "environment_service/runtime_costs.json"
    service.parent.mkdir(exist_ok=True)
    service.write_text(
        json.dumps(
            {
                "identity": journal.identity(),
                "service_closed": True,
                "pending_action": False,
                "environment_steps_unknown": 5 if case == "rpc_unknown" else 0,
                "last_native_success": bool(env.terminated),
                "last_feedback_sequence": 7,
                "native_success_source": "fake native worker",
            }
        )
    )
    final = json.loads(
        finalize_run(
            RunFinalizationContext(
                output_dir=journal.output_dir,
                robot_name="libero",
                task_desc={},
                environment_success=None,
                environment_metrics=tk.evaluation_metrics(success=None),
                agent_error=result.error,
                elapsed_s=1,
                planner="flash",
                model="gpt-6.1-sol",
                reasoning_effort="medium",
                max_turns=1,
                planner_timeout_s=600,
                finish_result=result.finish_result,
                stats=result.stats,
            )
        ).read_text()
    )
    assert final["episode_execution"]["status"] == status
    assert final["benchmark_success"] is (False if case == "budget" else outcome)
    assert final["verified_budget_exhaustion"] is (case == "budget")
    if case in {"unknown", "technical", "rpc_unknown"}:
        assert not final["attempt_closed"]


def test_dynamic_arm_delegates_to_actual_runner(setup):
    tk, *_ = setup
    called = []

    def agent(toolkit):
        called.append(toolkit)
        return toolkit.execute_tool(
            "vla_execute", {"instruction": "dynamic subgoal", "max_chunks": 1}
        )

    run_execution_arm(
        tk, {"schema": "libero-execution-arm-v1", "arm": "A"}, dynamic_runner=agent
    )
    assert (
        called == [tk]
        and tk.primitives.model.inputs[0]["task_descriptions"] == "dynamic subgoal"
    )


def test_r_unknown_condition_takes_registered_stop(setup):
    tk, env, model, journal = setup
    # Public state can be missing while VLA image input remains available.
    original = tk.primitives.set_obs

    def missing(obs):
        obs["states"] = np.asarray(obs["states"], dtype=float)
        obs["states"][2] = float("nan")
        original(obs)

    tk.primitives.set_obs = missing

    spec = {
        **direct(),
        "arm": "R",
        "stages": [
            stage(
                "observe",
                "look at bowl",
                completion={"signal": "eef_z", "op": "ge", "value": 0.2},
                on_complete="observe",
                on_timeout="observe",
                on_unknown="stop",
            )
        ],
    }
    result = run_execution_arm(tk, spec).data
    assert result["decisions"][0]["edge"] == "on_unknown" and len(model.inputs) == 1


@pytest.mark.parametrize(
    "spec",
    [
        {**direct(), "max_chunks": True},
        {**direct(), "stages": [stage("bad", "replacement")]},
        {**direct(), "arm": "S", "stages": [stage("bad", "x", on_timeout="stop")]},
        {**direct(), "arm": "R", "stages": [stage("bad", "x")]},
    ],
)
def test_invalid_programs_reject_before_execution(spec):
    with pytest.raises(ValueError):
        validate_execution_arm(spec)


@pytest.mark.parametrize("arm", ["D", "S", "R"])
def test_deterministic_cli_config_has_no_flash_corpus_or_molmo_dependency(
    tmp_path, monkeypatch, arm
):
    from robots.libero import robot_spec
    from rpent.cli.main import _build_argparser
    from rpent.session import runtime

    spec = direct()
    if arm != "D":
        spec.update(arm=arm, stages=[stage("first", "pick bowl")])
        if arm == "R":
            spec["stages"][0].update(
                on_complete="stop", on_timeout="stop", on_unknown="stop"
            )
    monkeypatch.setattr(runtime, "runtime_context", lambda: {"execution_arm": spec})
    parser = _build_argparser()
    robot_spec._add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--robot",
            "libero",
            "--planner",
            "flash",
            "--suite",
            "libero_object",
            "--task",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "empty"),
        ]
    )
    config = robot_spec._parse_config(args)
    # Ordinary Flash rejects this suite, requires a Molmo endpoint and reads
    # plan/anchors files. Deterministic arms need only Env + VLA + existing SAM3.
    assert config.prompt_vars["memory_dir"] == str(tmp_path / "empty")
    assert args.molmo_endpoint is None
    selected = []
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, component, starter: (
            selected.append(component),
            component,
        ),
    )
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, **kwargs: {})
    robot_spec._init_runtime(args, tmp_path, None, None)
    assert set(selected) == {"env", "vla", "sam3"}
