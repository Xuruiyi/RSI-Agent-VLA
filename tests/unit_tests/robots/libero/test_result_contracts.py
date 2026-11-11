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

"""Offline contracts for independent LIBERO native task results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from robots.libero.result import finalize_run
from robots.libero.robot_spec import get_robot_spec
from robots.libero.toolkit import LiberoToolkit
from rpent.evaluation import RunFinalizationContext
from rpent.session.base import StepRecord


@pytest.mark.parametrize(
    "native,expected_success,expected_status",
    [
        ({"terminated": True, "truncated": False}, True, "success"),
        ({"terminated": True, "truncated": True}, True, "success"),
        ({"terminated": False, "truncated": True}, False, "truncated"),
        ({"terminated": False, "truncated": False}, None, "active"),
        ({"terminated": False}, None, "unknown"),
        ({"truncated": True}, None, "unknown"),
        ({}, None, "unknown"),
    ],
)
def test_native_result_does_not_infer_success_from_planner_or_solved(
    tmp_path, native, expected_success, expected_status
):
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={"suite": "libero_object_swap", "task": 2, "seed": 0},
        environment_success=True,
        environment_metrics=native,
        agent_error="RuntimeError: planner stopped",
        elapsed_s=1.2,
        planner="codex",
        model="offline",
        reasoning_effort="high",
        max_turns=20,
        planner_timeout_s=180,
        finish_result={"status": "success", "summary": "planner claim"},
        stats={},
    )
    assert get_robot_spec().finalize_run is finalize_run
    path = finalize_run(context)
    assert path == tmp_path / "result.json"
    result = json.loads(path.read_text())
    assert result["benchmark_success"] is expected_success
    assert result["benchmark_status"] == expected_status
    assert result["terminated"] is native.get("terminated")
    assert result["truncated"] is native.get("truncated")
    assert result["planner_finish"] == context.finish_result
    assert result["error"] == context.agent_error
    assert result["tool_successes"] == []
    assert all(value is None for value in result["process_labels"].values())
    if "terminated" not in native:
        assert result["native_result_source"]["terminated"] is None


def test_toolkit_keeps_pick_heuristic_separate_from_native_result():
    toolkit = LiberoToolkit.__new__(LiberoToolkit)
    toolkit._primitives = SimpleNamespace(
        env=SimpleNamespace(terminated=False, truncated=False)
    )
    toolkit._mode, toolkit._attempt = "evaluation", 1
    toolkit._state = SimpleNamespace(
        records=lambda: [
            StepRecord(
                step_idx=1,
                state={},
                command={"action": "pi0_pick"},
                result={"success": True},
            ),
            StepRecord(
                step_idx=2,
                state={},
                command={"action": "move_to"},
                result={"terminated": False},
            ),
        ]
    )
    metrics = toolkit.evaluation_metrics(success=True)
    assert metrics["terminated"] is False
    assert metrics["truncated"] is False
    assert metrics["tool_successes"] == [
        {
            "step_idx": 1,
            "action": "pi0_pick",
            "success": True,
            "source": "states.json:steps[].result.success",
            "meaning": "pick_heuristic_or_native_termination",
        }
    ]
    toolkit._primitives.env = SimpleNamespace()
    missing = toolkit.evaluation_metrics(success=False)
    assert missing["terminated"] is None and missing["truncated"] is None
