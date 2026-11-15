# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU fakes exercise real reset dispatch, native outcomes and tool accounting."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import toolkit as toolkit_module
from robots.libero.result import finalize_run
from robots.libero.runtime_evidence import observed_libero_class
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.memory import MemoryManager
from rpent.session.runtime import RuntimeBudgetExceeded, RuntimeJournal


def _context(tmp_path, steps=100):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile.json",
        "parent_decision_id": None,
        "output_dir": str(tmp_path),
        "limits": {"environment_steps": steps, "tool_calls": 10},
    }


class _Vector:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.disconnect = False

    def step(self, actions, *args, **kwargs):
        self.calls.append((actions.copy(), args, kwargs))
        if self.disconnect:
            raise ConnectionError("native feedback lost")
        obs = [
            {
                "robot0_eef_pos": np.array([0, 0, 0.1]),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
            }
        ]
        return obs, np.array([0.0]), np.array([False]), [{}]

    def close(self):
        self.closed = True


class _NativeShape:
    def __init__(self, **_kwargs):
        self._init_env()
        self.cfg = SimpleNamespace(
            reset_gripper_open=True, get=lambda _key, _default: 15
        )
        self.num_envs, self.is_start, self.current_raw_obs = 1, False, None
        self.policy_steps = 0

    def _init_env(self):
        self.env = _Vector()

    def _get_random_reset_state_ids(self, count):
        return np.arange(count)

    def _reconfigure(self, _reset_ids, _env_idx):
        pass

    def _wrap_obs(self, value):
        return value

    def _reset_metrics(self, _env_idx):
        self.policy_steps = 0

    def reset(self):
        # The optional installed-source check below replaces this fixture method
        # with the actual installed RLinf method, without importing the backend.
        for _ in range(15):
            actions = np.zeros((1, 7))
            actions[:, -1] = -1
            raw_obs, *_ = self.env.step(actions, np.array([0]))
        self._reset_metrics(None)
        return raw_obs, {}

    def step(self, actions):
        self.policy_steps += 1
        return self.env.step(actions)


def _records(env):
    return [
        json.loads(line)
        for line in env._rsi_journal.events_path.read_text().splitlines()
    ]


def test_reset_settling_is_observed_and_counted_without_changing_actions(tmp_path):
    baseline = _NativeShape()
    env = observed_libero_class(_NativeShape, _context(tmp_path))()
    baseline.reset()
    env.reset()
    assert len(baseline.env.calls) == len(env.env.calls) == 15
    for left, right in zip(baseline.env.calls, env.env.calls):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1][0], right[1][0])
        assert left[2] == right[2]
    assert env.policy_steps == 0
    env.step(np.ones((1, 7)))
    env.env.close()
    costs = json.loads(env._rsi_journal.costs_path.read_text())
    assert costs["environment_steps_confirmed"] == 16
    assert costs["setup_steps_confirmed"] == 15 and costs["policy_steps_confirmed"] == 1
    assert costs["reset_calls"] == costs["reset_completed"] == 1
    assert costs["construction_complete"] and costs["service_closed"]
    assert costs["last_native_success"] is False and costs["pending_action"] is False
    assert costs["identity"]["bundle_id"] == "b"
    assert costs["reset_wall_seconds"] > 0
    records = _records(env)
    sends = [row for row in records if row["kind"] == "action_send"]
    assert [row["source"] for row in sends] == ["rlinf.reset_settling"] * 15 + [
        "rlinf.policy"
    ]
    assert [row["sequence"] for row in records] == list(range(len(records)))
    assert all(row["clock_domain"] == "environment_service" for row in records)


def test_setup_spends_the_same_total_environment_budget_before_policy_dispatch(
    tmp_path,
):
    env = observed_libero_class(_NativeShape, _context(tmp_path, steps=15))()
    env.reset()
    with pytest.raises(RuntimeBudgetExceeded):
        env.step(np.ones((1, 7)))
    assert len(env.env.calls) == 15
    assert env._rsi_journal.reserved_steps == 15
    assert not env._rsi_journal.pending_action


def test_reset_feedback_loss_is_pending_and_never_refunded(tmp_path):
    env = observed_libero_class(_NativeShape, _context(tmp_path))()
    # Proxy attributes are read-only delegation; mutate the actual fake vector.
    env.env._vector.disconnect = True
    with pytest.raises(ConnectionError):
        env.reset()
    costs = json.loads(env._rsi_journal.costs_path.read_text())
    assert costs["pending_action"] is True
    assert (
        costs["environment_steps_reserved"] == costs["environment_steps_unknown"] == 1
    )
    assert costs["environment_steps_confirmed"] == costs["reset_completed"] == 0
    assert (
        next(row for row in _records(env) if row["kind"] == "action_feedback")[
            "execution_mask"
        ]
        is None
    )


def test_current_installed_reset_body_uses_the_same_observed_dispatch(tmp_path):
    source = os.environ.get("RPENT_TEST_LIBERO_ENV_SOURCE")
    if not source:
        pytest.skip(
            "set RPENT_TEST_LIBERO_ENV_SOURCE to inspect a specific installed RLinf reset"
        )
    module = ast.parse(Path(source).read_text())
    native = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv"
    )
    reset = next(
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef) and node.name == "reset"
    )
    scope = {"np": np, "Optional": Optional, "Union": Union}
    exec(compile(ast.Module(body=[reset], type_ignores=[]), source, "exec"), scope)
    base = type("InstalledResetShape", (_NativeShape,), {"reset": scope["reset"]})
    baseline, env = base(), observed_libero_class(base, _context(tmp_path))()
    baseline.reset()
    env.reset()
    assert len(baseline.env.calls) == len(env.env.calls) == 15
    for left, right in zip(baseline.env.calls, env.env.calls):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1][0], right[1][0])
    assert env._rsi_journal.setup_steps_confirmed == 15


def _closed_context(tmp_path, *, native=False, pending=False):
    context = _context(tmp_path)
    journal = RuntimeJournal(context)
    identity = {
        key: context[key]
        for key in (
            "run_id",
            "episode_id",
            "bundle_id",
            "protocol_id",
            "profile_ref",
            "parent_decision_id",
        )
    }
    service = {
        "identity": identity,
        "service_closed": True,
        "pending_action": pending,
        "environment_steps_unknown": 0,
        "last_native_success": native,
        "last_feedback_sequence": 5,
        "native_success_source": "native worker _check_success()",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(service))
    return RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={"suite": "demo", "task": 0, "seed": 0},
        environment_success=False,
        agent_error=None,
        elapsed_s=1.0,
        planner="codex",
        model="offline",
        reasoning_effort="medium",
        max_turns=1,
        planner_timeout_s=10,
        finish_result={"_finish": True, "status": "success"},
        stats={"planner_turn_status": "completed"},
        environment_metrics={
            "terminated": False,
            "truncated": False,
            "runtime": {
                "identity": identity,
                "action_events_ref": str(journal.events_path),
                "environment_service": {"costs_ref": str(path)},
            },
        },
    )


@pytest.mark.parametrize(
    "native,pending,error,status,expected",
    [
        (False, False, None, "completed", False),
        (True, False, None, "completed", True),
        (False, True, None, "completed", None),
        (False, False, "timeout", "completed", None),
        (False, False, None, "interrupted", None),
    ],
)
def test_native_false_is_final_failure_only_after_confirmed_normal_attempt_close(
    tmp_path, native, pending, error, status, expected
):
    context = _closed_context(tmp_path, native=native, pending=pending)
    context = replace(context, agent_error=error, stats={"planner_turn_status": status})
    result = json.loads(finalize_run(context).read_text())
    assert result["benchmark_success"] is expected
    assert result["attempt_closed"] is (expected is not None)
    if expected is not None:
        assert (
            result["attempt_outcome_source"]["native_success"]
            == "native worker _check_success()"
        )


@pytest.mark.parametrize(
    "enabled,native,status,allowed",
    [
        (False, False, "success", True),
        (True, False, "success", False),
        (True, None, "success", False),
        (True, True, "success", True),
        (True, False, "failure", True),
    ],
)
def test_native_finish_rule_is_consumed_by_actual_hybrid_tool_registration(
    tmp_path, monkeypatch, fake_single_arm_primitives, enabled, native, status, allowed
):
    journal = RuntimeJournal(_context(tmp_path))
    journal.finish_requires_native_success = enabled
    monkeypatch.setattr(RuntimeJournal, "from_environment", Mock(return_value=journal))
    fake = fake_single_arm_primitives.for_robot(
        toolkit_module.libero_tools.LiberoPrimitives
    )
    monkeypatch.setattr(toolkit_module.libero_tools, "LiberoPrimitives", fake)
    monkeypatch.setattr(
        toolkit_module.libero_tools,
        "dump_state",
        lambda *_args, **_kwargs: SimpleNamespace(step_idx=0),
    )
    monkeypatch.setattr(
        toolkit_module.LiberoToolkit, "_publish_step", lambda *_args: None
    )
    toolkit = toolkit_module.LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "states",
    )
    toolkit._primitives.env = SimpleNamespace(terminated=native, truncated=False)
    result = toolkit.execute_tool("finish", {"status": status, "summary": "offline"})
    assert bool(result.data.get("_finish")) is allowed
    records = [
        json.loads(line) for line in journal.events_path.read_text().splitlines()
    ]
    assert records[-1]["kind"] == "tool_end"
    assert records[-1]["status"] == ("completed" if allowed else "failed")
    assert records[-1]["elapsed_seconds"] >= 0
