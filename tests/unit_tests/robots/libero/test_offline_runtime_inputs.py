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

"""Read-only localization and versioned runtime input contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.task_input import GLOBAL_MEMORY, build_task_input
from robots.libero.toolkit import LiberoToolkit
from robots.libero.tools import back_project, back_project_batch
from rpent.session.runtime import RuntimeBudgetExceeded, RuntimeJournal
from rpent.tools.base import ToolResult
from rpent.tools.toolkit import Toolkit


class State:
    def __init__(self):
        self.world = np.arange(48, dtype=float).reshape(4, 4, 3) / 10
        self.world[1, 1] = np.nan
        self.record = SimpleNamespace(
            step_idx=3, artifacts=["agentview_world_high.npz"]
        )

    def get(self, step):
        if step not in (-1, 3):
            raise KeyError(step)
        return self.record

    def load(self, name, *, step):
        return self.world


def test_batch_matches_each_original_point_and_preserves_errors():
    state = State()
    points = [[0, 1], [3, 3], [-1, 0], [1, 1], [0], None, [True, 2], [1.1, 0]]
    results = back_project_batch(points, state=state).data["results"]
    assert len(results) == len(points)
    for index, point in enumerate(points[:4]):
        original = back_project(row=point[0], col=point[1], step=3, state=state).data
        assert {
            k: v for k, v in results[index].items() if k not in {"index", "pixel"}
        } == {k: v for k, v in original.items() if k != "pixel"}
    assert all("error" in entry for entry in results[2:])
    assert back_project_batch([], state=state).is_error
    assert all(
        "error" in entry
        for entry in back_project_batch([[0, 0], [1, 2]], step=4, state=state).data[
            "results"
        ]
    )
    state.record.artifacts = []
    assert (
        "not recorded"
        in back_project_batch([[0, 0]], state=state).data["results"][0]["error"]
    )


def context(path, **limits):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile",
        "output_dir": str(path),
        "limits": {
            "tool_calls": 60,
            "model_requests": 20,
            "wall_seconds": 900,
            "environment_steps": 2000,
            **limits,
        },
    }


def test_budget_feedback_reads_owner_counts_without_charging(tmp_path, monkeypatch):
    journal = RuntimeJournal(context(tmp_path))
    identity = journal.identity()
    (tmp_path / "request_gate_live.json").write_text(
        json.dumps({"identity": identity, "requests_forwarded": 4})
    )
    service = tmp_path / "environment_service"
    service.mkdir()
    (service / "runtime_costs.json").write_text(
        json.dumps({"identity": identity, "environment_steps_reserved": 130})
    )
    toolkit = object.__new__(LiberoToolkit)
    toolkit.runtime_journal = journal
    monkeypatch.setattr(
        Toolkit, "execute_tool", lambda *a: ToolResult(data={"ok": True})
    )
    feedback = toolkit.execute_tool("read", {}).data["runtime_budget"]
    assert (
        feedback["tool_calls_remaining"],
        feedback["actual_model_requests"],
        feedback["environment_steps_remaining"],
    ) == (59, 4, 1870)
    assert journal.tool_calls == 1 and journal.active_tools == 0
    journal.budget_snapshot()
    assert journal.tool_calls == 1
    (tmp_path / "request_gate_live.json").unlink()
    assert journal.budget_snapshot()["actual_model_requests"] is None


def test_stop_rejects_actions_but_admitted_rpc_can_drain(tmp_path):
    journal = RuntimeJournal(context(tmp_path))
    journal.mark_budget_stop("tool_calls", limit=60, observed=60)

    def callback():
        return ({}, 0, False, False, {})

    with pytest.raises(RuntimeBudgetExceeded):
        journal.action(callback, [0] * 7, source="test")
    journal.action(callback, [0] * 7, source="test", already_admitted=True)
    assert journal.confirmed_steps == 1
    with pytest.raises(RuntimeBudgetExceeded):
        journal.admit_tool("move_to")


def test_startup_only_reads_explicit_prior_and_marks_missing_match(tmp_path):
    root = tmp_path / "revision" / "version"
    root.mkdir(parents=True)
    for relative in GLOBAL_MEMORY:
        p = root / relative
        p.parent.mkdir(exist_ok=True)
        p.write_text("legal prior " + relative)
    (root / "MEMORY.md").write_text("index must not be inlined")
    (root / "release").mkdir()
    (root / "release" / "secret").write_text("SEALED CANARY")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"memory": {"version": "version", "revision": "revision"}})
    )
    c = context(tmp_path / "run")
    c.update(
        profile_ref=str(profile),
        task={"suite": "libero_object_swap", "task": 2, "seed": 1},
        task_input={
            "version": "libero-startup-v1",
            "task_language": "Pick the salad dressing and place it in the basket",
        },
    )
    view = build_task_input(c, root, c["task"])
    text = view["system_prompt"] + view["user_message"]
    assert "SEALED CANARY" not in text and "index must not be inlined" not in text
    assert len(view["sources"]) == 3 and view["missing_sources"] == [
        "task-family/task-family_libero_object_swap_t2.md"
    ]
    assert view["project_learning"] is False and "back_project_batch" in text
    assert "terminated=true" in text and "no reset" in text.lower()
    with pytest.raises(ValueError):
        build_task_input(c, root, {**c["task"], "seed": 5})


def test_cli_passes_compact_view_to_planner_and_saves_it(tmp_path, monkeypatch):
    import sys

    import robots.libero.task_input as inputs
    import rpent.session.runtime as runtime
    from rpent.cli import main as cli
    from rpent.planner.base import PlannerResult
    from rpent.robots import PromptBundle, RobotSpec, RunConfig

    bound = context(tmp_path)
    bound["task_input"] = {"version": "libero-startup-v1"}
    view = {
        "identity": {
            key: bound[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        },
        "system_prompt": "compact rules",
        "user_message": "selected task and prior",
        "version": "libero-startup-v1",
    }
    monkeypatch.setattr(runtime, "runtime_context", lambda: bound)
    monkeypatch.setattr(inputs, "build_task_input", lambda *args: view)
    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda v: "full old system", user=lambda v: "full old user"
        ),
        add_cli_args=lambda *args, **kwargs: None,
        parse_config=lambda args: RunConfig(
            recipe_tag="offline",
            output_dir=tmp_path,
            prompt_vars={"memory_dir": tmp_path / "memory"},
            task_desc={},
        ),
        init_runtime=lambda *args: ([], {}),
        run_flash=lambda *args: {},
    )

    class FakePlanner:
        def solve(self, **kwargs):
            assert kwargs["system_prompt"] == "compact rules"
            assert kwargs["user_message"] == "selected task and prior"
            return PlannerResult(finish_result=None, messages=[], stats={})

    class FakeToolkit:
        memory = None

        def cancel_active_and_wait(self):
            pass

        def evaluation_metrics(self, **kwargs):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    from rpent.memory import loading

    monkeypatch.setattr(loading, "prepare_run_memory", lambda *a: None)
    monkeypatch.setattr(cli, "build_planner", lambda *a, **k: FakePlanner())
    monkeypatch.setattr(cli, "get_toolkit", lambda *a, **k: FakeToolkit())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "libero",
            "--planner",
            "codex",
            "--model",
            "offline",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert cli.main() == 0
    assert json.loads((tmp_path / "task_input.json").read_text()) == view
    assert (tmp_path / "runner_cleanup.json").is_file()
