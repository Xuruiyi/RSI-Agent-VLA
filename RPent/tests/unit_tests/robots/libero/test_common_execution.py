# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Common recording and physical allowances through the actual Toolkit."""

import json

import numpy as np
import pytest

from robots.libero.execution_arms import run_execution_arm, validate_common_execution
from robots.libero.toolkit import LiberoToolkit
from rpent.tools import ToolResult
from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit


def common():
    return {
        "schema": "libero-common-execution-v1",
        "observation_cadence": "prediction",
        "allowed_motion_tools": ["vla_execute"],
        "action_prefix_steps": 5,
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    original = LiberoToolkit._register_libero_tools

    def register(self):
        self.runtime_journal.context["common_execution"] = common()
        self.runtime_journal.context["limits"]["tool_calls"] = 512
        original(self)

    monkeypatch.setattr(LiberoToolkit, "_register_libero_tools", register)
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    captures = []

    def capture(*, command, result, elapsed_s):
        captures.append((env.steps, command, dict(result)))
        return ToolResult(
            data={"observed_step": env.steps, "execution_segment": result}
        )

    # The shared native dump_state/media method is spied at its IO boundary;
    # the scheduler, primitive, latest observations and budgets remain real.
    monkeypatch.setattr(tk, "_capture_env_state", capture)
    return tk, env, model, journal, captures


@pytest.mark.parametrize("arm", ["D", "S", "R", "A"])
def test_all_arms_capture_every_prediction_and_charge_every_native_call(setup, arm):
    tk, env, model, journal, captures = setup
    if arm == "A":
        result = tk.execute_tool(
            "vla_execute", {"instruction": "native task", "max_chunks": 3}
        )
        assert result.data["execution_segment"]["predictions_completed"] == 3
    else:
        spec = {"schema": "libero-execution-arm-v1", "arm": arm, "max_chunks": 3}
        if arm != "D":
            spec["stages"] = [
                {"id": "first", "instruction": "native task", "max_chunks": 3}
            ]
            if arm == "R":
                spec["stages"][0].update(
                    on_complete="stop", on_timeout="stop", on_unknown="stop"
                )
        run_execution_arm(tk, spec)
    assert [row[0] for row in captures] == [5, 10, 15]
    assert [obs["states"][0] for obs in model.inputs] == [0, 5, 10]
    assert [len(actions) for actions in env.sent] == [5, 5, 5]
    assert journal.costs()["tool_calls"] == 3
    assert journal.active_tools == 0
    events = [json.loads(row) for row in journal.events_path.read_text().splitlines()]
    assert sum(row["kind"] == "common_prediction_call" for row in events) == 3
    assert sum(row["kind"] == "tool_call" for row in events) == 3
    assert sum(row["kind"] == "tool_end" for row in events) == 3


@pytest.mark.parametrize("terminal", ["success", "truncation"])
def test_native_terminal_stops_long_segment_without_extra_capture(setup, terminal):
    tk, env, model, journal, captures = setup
    env.terminal = terminal
    result = tk.execute_tool("vla_execute", {"instruction": "task", "max_chunks": 24})
    assert len(captures) == len(model.inputs) == journal.tool_calls == 1
    assert result.data["execution_segment"]["stop_reason"] == (
        "native_success" if terminal == "success" else "native_truncation"
    )
    assert journal.active_tools == 0


@pytest.mark.parametrize("resource", ["tool_calls", "environment_steps"])
def test_budget_stop_preserves_completed_prefix_and_closes_internal_calls(
    setup, resource
):
    tk, env, model, journal, captures = setup
    journal.context["limits"][resource] = 2 if resource == "tool_calls" else 40
    result = tk.execute_tool("vla_execute", {"instruction": "task", "max_chunks": 24})
    segment = result.data["execution_segment"]
    assert segment["predictions_completed"] == 2 and segment["executed_steps"] == 10
    assert segment["stop_reason"] == "budget"
    assert [row[0] for row in captures] == [5, 10]
    assert len(model.inputs) == 2 and journal.active_tools == 0


def test_prediction_failure_retains_previous_actions_and_failed_prediction(setup):
    tk, env, model, journal, captures = setup
    predict = model.predict

    def fail_second(*args, **kwargs):
        if len(model.inputs):
            raise ValueError("second prediction failed")
        return predict(*args, **kwargs)

    model.predict = fail_second
    result = tk.execute_tool("vla_execute", {"instruction": "task", "max_chunks": 3})
    segment = result.data["execution_segment"]
    assert segment["stop_reason"] == "technical_error"
    assert segment["predictions_started"] == 2 and segment["predictions_completed"] == 1
    assert segment["executed_steps"] == 5 and len(captures) == 2
    assert journal.tool_calls == 2 and journal.active_tools == 0
    assert "second prediction failed" in result.data["error"]


def test_disallowed_motion_is_unavailable_to_agent(setup):
    tk, env, model, journal, captures = setup
    names = {tool.name for tool in tk.list_tools()}
    assert {"vla_execute", "segment", "view_env_state"} <= names
    assert not ({"move_to", "release", "pi0_pick", "pi0_doubled"} & names)
    result = tk.execute_tool("pi0_pick", {"prompt": "pick"})
    assert result.data["error"] == "unknown tool: pi0_pick"
    assert not model.inputs and not env.sent


@pytest.mark.parametrize(
    "change",
    [
        {"action_prefix_steps": 3},
        {"allowed_motion_tools": ["pi0_pick"]},
        {"observation_cadence": "tool"},
    ],
)
def test_invalid_common_conditions_reject(change):
    with pytest.raises(ValueError, match="common LIBERO"):
        validate_common_execution({**common(), **change})


def test_tool_allowance_covers_whole_common_physical_budget():
    assert (
        validate_common_execution(common(), environment_steps=2000, tool_calls=512)
        == common()
    )
    with pytest.raises(ValueError, match="physical bound"):
        validate_common_execution(common(), environment_steps=2000, tool_calls=60)


@pytest.mark.parametrize("arm", ["D", "A"])
def test_native_state_and_image_writer_has_same_prediction_cadence(setup, arm):
    tk, env, model, journal, _captures = setup
    tk._capture_env_state = LiberoToolkit._capture_env_state.__get__(tk)
    env.get_task_language = lambda: "native task"
    env.get_camera_meta = lambda *args, **kwargs: {
        "intrinsic_K": np.eye(3).tolist(),
        "extrinsic_cam2world": np.eye(4).tolist(),
    }
    env.render_camera = lambda **kwargs: (
        env.obs()["main_images"],
        np.ones((2, 2), dtype=np.float32),
    )
    env.raw_obs = lambda: {
        "robot0_eef_pos": [env.steps, 0, 0],
        "robot0_eef_quat": [0, 0, 0, 1],
        "robot0_gripper_qpos": [0.04, -0.04],
        "agentview_image": env.obs()["main_images"],
        "robot0_eye_in_hand_image": env.obs()["wrist_images"],
    }
    if arm == "A":
        result = tk.execute_tool(
            "vla_execute", {"instruction": "native task", "max_chunks": 3}
        )
        assert result.data["execution_segment"]["executed_steps"] == 15
    else:
        run_execution_arm(
            tk, {"schema": "libero-execution-arm-v1", "arm": "D", "max_chunks": 3}
        )
    records = tk.state.records()
    assert [record.state["robot0_eef_pos"][0] for record in records] == [5, 10, 15]
    assert [record.result["predictions_completed"] for record in records] == [1, 1, 1]
    for record in records:
        assert {
            "agentview.png",
            "wrist.png",
            "agentview_high.png",
            "wrist_high.png",
        } <= set(record.artifacts)
    assert journal.active_tools == 0
