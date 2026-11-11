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

"""CPU-only bounded VLA, real Toolkit/MCP serialization and success closure."""

import base64
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from robots.libero import tools
from robots.libero.task_input import build_task_input
from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.codex import _Recorder
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session.runtime import RuntimeJournal
from rpent.tools.toolkit import ToolCancelled


def context(path, **limits):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "output_dir": str(path),
        "profile_ref": "profile",
        "execution_version": "bounded-vla-v1",
        "planner_observation": {"image_detail": "calibrated"},
        "limits": dict(
            tool_calls=60,
            model_requests=20,
            environment_steps=2000,
            wall_seconds=900,
            **limits,
        ),
    }


class FakeEnv:
    return_all_frames = False
    terminated = False
    truncated = False

    def __init__(self, journal=None, terminal=None):
        self.journal, self.terminal = journal, terminal
        self.steps = 0
        self.sent = []

    def obs(self):
        return {
            "states": np.array([self.steps, 0, 0, 0, 0, 0, 0.04, -0.04]),
            "task_descriptions": "native task",
            "main_images": np.zeros((2, 2, 3), np.uint8) + self.steps,
            "wrist_images": np.zeros((2, 2, 3), np.uint8),
        }

    def reset(self):
        return self.obs(), {}

    def chunk_step(self, actions, return_all_frames=False):
        assert not (self.terminated or self.truncated)
        self.sent.append(np.array(actions))
        observations = []
        for _ in actions:
            self.steps += 1
            observations.append(self.obs())
        self.terminated = self.terminal == "success"
        self.truncated = self.terminal == "truncation"
        if self.journal:
            directory = self.journal.output_dir / "environment_service"
            directory.mkdir(exist_ok=True)
            (directory / "runtime_costs.json").write_text(
                json.dumps(
                    {
                        "identity": self.journal.identity(),
                        "environment_steps_reserved": 30 + self.steps,
                    }
                )
            )
        return (
            observations if return_all_frames else observations[-1],
            np.zeros(len(actions)),
            np.full(len(actions), self.terminated),
            np.full(len(actions), self.truncated),
            {},
        )

    def rsi_chunk_step(self, actions, *, evidence, **kwargs):
        assert evidence["proposal_length"] == 5
        return self.chunk_step(actions, **kwargs)


class FakeVLA:
    def __init__(self):
        self.inputs = []

    def predict(self, obs, options):
        self.inputs.append(
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        )
        assert options == {"mode": "eval"}
        return np.zeros((5, 7), np.float32)


def driver(*, journal=None, terminal=None, cancel=lambda: None):
    env = FakeEnv(journal, terminal)
    model = FakeVLA()
    primitive = tools.LiberoPrimitives(
        env=env,
        model=model,
        sam3_client=None,
        check_cancelled=cancel,
        runtime_journal=journal,
    )
    primitive.set_obs(env.obs())
    return primitive, env, model


def test_latest_observation_instruction_scope_and_chunk_allowance():
    p, env, model = driver()
    first = p.vla_execute("carry to basket", max_chunks=3).data
    second = p.vla_execute("place in basket", max_chunks=1).data
    assert [o["states"][0] for o in model.inputs] == [0, 5, 10, 15]
    assert [o["task_descriptions"] for o in model.inputs] == ["carry to basket"] * 3 + [
        "place in basket"
    ]
    assert p._last_obs["task_descriptions"] == "native task"
    assert first["chunks_used"] == 3 and first["executed_steps"] == 15
    assert first["stop_reason"] == "chunk_allowance"
    assert (
        first["segment_status"] == "allowance_ended_task_unfinished"
        and first["task_success"] is False
    )
    assert second["executed_steps"] == 5 and len(env.sent) == 4
    # Existing contact path still uses its own instruction and native-only success.
    legacy = p.pi0_doubled("turn knob", max_chunks=1).data
    assert legacy["contact_skill_executed"] and not legacy["success"]
    assert model.inputs[-1]["task_descriptions"] == "turn knob"


@pytest.mark.parametrize("value", [0, 25, -1, True, 1.5, "2"])
def test_chunk_bounds_reject_before_prediction(value):
    p, env, model = driver()
    with pytest.raises(ValueError):
        p.vla_execute("task", max_chunks=value)
    assert not env.sent and not model.inputs


@pytest.mark.parametrize("cap", [1, 24])
def test_valid_allowance_endpoints(cap):
    p, env, model = driver()
    assert p.vla_execute("task", max_chunks=cap).data["chunks_used"] == cap
    assert len(model.inputs) == cap and env.steps == 5 * cap


@pytest.mark.parametrize(
    ("terminal", "reason", "success"),
    [("success", "native_success", True), ("truncation", "native_truncation", False)],
)
def test_native_terminal_no_second_prediction(terminal, reason, success):
    p, env, model = driver(terminal=terminal)
    result = p.vla_execute("whole task", max_chunks=24).data
    assert result["stop_reason"] == reason and result["task_success"] is success
    assert result["chunks_used"] == 1 and env.steps == 5
    assert p.vla_execute("another", max_chunks=1).data["chunks_used"] == 0


def test_cancel_after_prediction_records_no_action(tmp_path):
    journal = RuntimeJournal(context(tmp_path))
    p, env, model = driver(journal=journal)

    def check():
        if model.inputs:
            raise ToolCancelled("operator cancel")

    p._check_cancelled = check
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "cancelled" and result["interrupted"]
    assert result["predictions_completed"] == 1 and result["executed_steps"] == 0
    assert not env.sent


def test_prediction_error_is_technical_and_not_task_failure(tmp_path):
    journal = RuntimeJournal(context(tmp_path))
    p, env, model = driver(journal=journal)

    def predict(*args, **kwargs):
        raise ValueError("fake inference failed")

    model.predict = predict
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "technical_error" and not result["interrupted"]
    assert result["predictions_started"] == 1 and result["predictions_completed"] == 0
    assert result["executed_steps"] == 0 and not env.sent
    events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    assert next(e for e in events if e["kind"] == "inference_end")["completed"] is False


def test_remaining_budget_includes_owner_setup_and_stops_before_new_prediction(
    tmp_path,
):
    c = context(tmp_path)
    c["limits"]["environment_steps"] = 39
    journal = RuntimeJournal(c)
    directory = tmp_path / "environment_service"
    directory.mkdir()
    (directory / "runtime_costs.json").write_text(
        json.dumps({"identity": journal.identity(), "environment_steps_reserved": 30})
    )
    p, env, model = driver(journal=journal)
    journal.admit_tool("vla_execute")
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "budget" and "error" not in result
    assert result["executed_steps"] == 5 and len(model.inputs) == 1
    assert result["runtime_budget"]["environment_steps_remaining"] == 4
    events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    for kind in (
        "inference_start",
        "inference_end",
        "action_proposal",
        "action_send",
        "action_feedback",
        "chunk_execution_end",
        "vla_segment_end",
    ):
        assert sum(e["kind"] == kind for e in events) == 1


def test_wall_stop_no_prediction(tmp_path):
    c = context(tmp_path)
    c["limits"]["wall_seconds"] = 0
    journal = RuntimeJournal(c)
    p, env, model = driver(journal=journal)
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "budget" and not model.inputs


def make_toolkit(tmp_path, monkeypatch, *, terminal=None):
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    journal = RuntimeJournal(context(tmp_path / "run"))
    monkeypatch.setattr(
        RuntimeJournal, "from_environment", classmethod(lambda cls: journal)
    )
    primitive, env, model = driver(journal=journal, terminal=terminal)

    def init(self, **kwargs):
        self._primitives = primitive
        primitive._check_cancelled = self.raise_if_cancelled
        self._action_frame_cursor = 0

    monkeypatch.setattr(LiberoToolkit, "init_primitives", init)
    tk = LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    return tk, env, model, journal


def historical_state():
    from rpent.session import EnvState
    from rpent.session.base import StepRecord

    directory = (
        Path(__file__).resolve().parents[5]
        / "artifacts/authorized-20261110/5a-finish-reserve-candidate-01"
    )
    state = EnvState.__new__(EnvState)
    state._output_dir = directory
    state._artifact_lock = threading.Lock()
    rows = json.loads((directory / "states.json").read_text())["steps"]
    state._steps = [StepRecord(**row) for row in rows]
    return state


def test_actual_toolkit_mcp_and_historical_image_serialization(tmp_path, monkeypatch):
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    tk._state = historical_state()
    tk.get_env_state = lambda **kwargs: tools.view_env_state(
        0, state=tk._state, image_detail="calibrated"
    )
    with_server = HttpMcpServer(tk)
    with_server.start()
    try:
        headers = {"Accept": "application/json, text/event-stream"}
        with httpx.Client(trust_env=False, timeout=10) as client:

            def rpc(method, params):
                response = client.post(
                    with_server.url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
                assert "error" not in body, body
                return body["result"]

            catalog = rpc("tools/list", {})["tools"]
            declaration = next(t for t in catalog if t["name"] == "vla_execute")
            assert declaration["inputSchema"]["required"] == [
                "instruction",
                "max_chunks",
            ]
            assert {"pi0_pick", "pi0_doubled", "vla_execute"} <= {
                t["name"] for t in catalog
            }
            invalid = rpc(
                "tools/call",
                {"name": "vla_execute", "arguments": {"instruction": "task"}},
            )
            assert invalid["isError"] and not model.inputs
            segment = rpc(
                "tools/call",
                {
                    "name": "vla_execute",
                    "arguments": {"instruction": "carry", "max_chunks": 2},
                },
            )
            assert not segment["isError"] and len(model.inputs) == 2 and env.steps == 10
            compact = rpc(
                "tools/call", {"name": "view_env_state", "arguments": {"step": 0}}
            )
            full = rpc(
                "tools/call",
                {
                    "name": "view_env_state",
                    "arguments": {"step": 0, "image_detail": "full"},
                },
            )
            for view, names in (
                (compact, ["agentview_high.png", "wrist_high.png"]),
                (
                    full,
                    ["agentview_policy.png", "agentview_high.png", "wrist_high.png"],
                ),
            ):
                data = json.loads(view["content"][0]["text"])
                assert [f["artifact"] for f in data["image_frames"]] == names
                encoded = [b for b in view["content"] if b["type"] == "image"]
                assert len(encoded) == len(names)
                for block, name in zip(encoded, names):
                    assert base64.b64decode(block["data"]) == tk._state.load_bytes(
                        name, step=0
                    )
            assert all(
                f["back_project_compatible"] and f["resolution"] == "high"
                for f in json.loads(compact["content"][0]["text"])["image_frames"]
            )
    finally:
        with_server.stop()
    assert tk._state.exists("agentview_policy.png", step=0)


def test_native_toolkit_closure_and_recorder_use_success_cause(tmp_path, monkeypatch):
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch, terminal="success")
    record = SimpleNamespace(step_idx=1, terminated=True, truncated=False)
    monkeypatch.setattr(tools, "dump_state", lambda *a, **k: record)
    monkeypatch.setattr(
        tools,
        "view_env_state",
        lambda *a, **k: tools.ToolResult(data={"terminated": True}),
    )
    tk._primitives.start_recording()
    tk._state = SimpleNamespace(latest_record=lambda: record)
    monkeypatch.setattr(tk, "_publish_step", lambda record: None)
    result = tk.execute_tool(
        "vla_execute", {"instruction": "whole task", "max_chunks": 24}
    )
    assert result.data["execution_segment"]["stop_reason"] == "native_success"
    assert journal.native_completion()["reason"] == "native_success"
    assert journal.budget_stop() is None
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    assert recorder.request_native_stop()["reason"] == "native_success"
    recorder.observe(
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "interrupted"}},
        }
    )
    assert (
        recorder.stats()["attempt_closed"]
        and recorder.stats()["termination_reason"] == "native_success"
    )
    assert recorder.stats()["planner_budget_stop"] is None and recorder.error is None


def test_codex_stream_native_success_stops_without_another_request(
    tmp_path, monkeypatch
):
    import rpent.planner.codex as codex

    journal = RuntimeJournal(context(tmp_path))
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)

    class Turn:
        interrupts = 0
        requests = 0

        def interrupt(self):
            self.interrupts += 1

        def stream(self):
            self.requests += 1
            yield {"method": "turn/started"}
            journal.complete_native_success(step=1)
            yield {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "type": "mcpToolCall",
                        "tool": "vla_execute",
                        "status": "completed",
                        "result": {"terminated": True},
                    }
                },
            }
            if not self.interrupts:
                self.requests += 1
            yield {
                "method": "turn/completed",
                "payload": {
                    "turn": {
                        "status": "interrupted" if self.interrupts else "completed"
                    }
                },
            }

    turn = Turn()

    class Native:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def thread_start(self, **kwargs):
            return SimpleNamespace(turn=lambda *a, **k: turn)

    monkeypatch.setattr(codex.openai_codex, "Codex", Native)
    planner = codex.CodexPlanner(
        output_dir=str(tmp_path),
        dashboard_events=NullDashboardEventSink(),
        model="offline",
    )
    monkeypatch.setattr(planner, "_build_config", lambda url: None)
    result = {}
    planner._run_session(
        "offline",
        tmp_path / "out",
        tmp_path / "stream",
        tmp_path / "last",
        recorder,
        result,
        "unused",
        SimpleNamespace(runtime_journal=journal),
    )
    assert "error" not in result and turn.interrupts == turn.requests == 1
    assert (
        recorder.stats()["termination_reason"] == "native_success"
        and recorder.stats()["attempt_closed"]
    )


def test_versioned_input_excerpts_preserve_conditions_without_foreign_coordinates():
    root = Path(__file__).resolve().parents[5]
    profile = json.loads((root / "config/run_profile_v9.json").read_text())
    memory = profile["memory"]
    memory_root = next(
        (root / "RPent/memory/libero/.versions").glob(
            "*/snapshots/" + memory["revision"] + "/" + memory["version"]
        )
    )
    c = context(root / "unused")
    c["profile_ref"] = str(root / "config/run_profile_v9.json")
    c.update(
        task={"suite": "libero_object_swap", "task": 2, "seed": 1},
        task_input=profile["task_input"],
    )
    view = build_task_input(c, memory_root, c["task"])
    text = view["system_prompt"] + view["user_message"]
    assert "vla_execute" in text and "not stage" in text
    assert "z>=0.72" not in text and "z<=0.62" not in text and "z0.52" not in text
    assert "task-family/task-family_libero_object_swap_t2.md" in view["missing_sources"]
    for source in view["sources"]:
        assert all(
            line in Path(source["ref"]).read_text()
            for line in source["text"].splitlines()
        )
        assert (
            "applies_when:" in source["text"]
            and "confidence: single-shot" in source["text"]
        )
    assert "while Planner owns placement" in text and view["project_learning"] is False


def test_transport_error_keeps_unknown_execution_and_no_next_prediction(tmp_path):
    journal = RuntimeJournal(context(tmp_path))
    p, env, model = driver(journal=journal)

    def fail(*args, **kwargs):
        raise OSError("fake RPC lost")

    env.rsi_chunk_step = fail
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "technical_error" and result["unknown_steps"] == 5
    assert result["executed_steps"] == 0 and len(model.inputs) == 1


def test_cancel_between_chunks_preserves_completed_count():
    p, env, model = driver()

    def check():
        if env.steps >= 5:
            raise ToolCancelled("cancel between chunks")

    p._check_cancelled = check
    result = p.vla_execute("task", max_chunks=24).data
    assert result["stop_reason"] == "cancelled" and result["chunks_used"] == 1
    assert result["executed_steps"] == 5 and len(model.inputs) == 1


def test_wam_registration_does_not_expose_vla_tool(tmp_path, monkeypatch):
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    tk._tools.clear()
    tk._policy_kind = "wam"
    tk._register_libero_tools()
    names = {tool.name for tool in tk.list_tools()}
    assert "wam_act" in names and not (
        {"vla_execute", "pi0_pick", "pi0_doubled"} & names
    )


def test_compact_pixel_mapping_matches_historical_calibration():
    state = historical_state()
    compact = tools.view_env_state(0, state=state, image_detail="calibrated").data
    for frame in compact["image_frames"]:
        camera = frame["camera"]
        world = state.load(f"{camera}_world_high.npz", step=0)
        valid = np.argwhere(np.isfinite(world).all(axis=-1))
        row, col = map(int, valid[len(valid) // 2])
        projected = tools.back_project(
            row=row, col=col, step=0, camera=camera, resolution="high", state=state
        ).data
        assert projected["world_xyz"] == [round(float(v), 4) for v in world[row, col]]
        import imageio.v2 as imageio

        image = imageio.imread(state.artifact_path(frame["artifact"], step=0))
        assert image.shape[:2] == world.shape[:2] == (1024, 1024)


def test_native_gate_rejection_is_normal_closure_but_other_errors_remain(tmp_path):
    journal = RuntimeJournal(context(tmp_path))
    journal.complete_native_success(step=1)
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    recorder.observe(
        {
            "method": "turn/completed",
            "payload": {
                "turn": {
                    "status": "failed",
                    "error": {
                        "message": "native task completed; no further Planner request forwarded"
                    },
                }
            },
        }
    )
    assert (
        recorder.error is None
        and recorder.stats()["termination_reason"] == "native_success"
    )
    assert recorder.stats()["attempt_closed"]
    other = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    other.observe(
        {
            "method": "turn/completed",
            "payload": {
                "turn": {
                    "status": "failed",
                    "error": {"message": "unrelated transport error"},
                }
            },
        }
    )
    assert (
        other.error == "unrelated transport error"
        and not other.stats()["attempt_closed"]
    )
