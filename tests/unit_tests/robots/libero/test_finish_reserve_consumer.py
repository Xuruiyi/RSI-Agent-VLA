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

"""Frozen finish hints reach the actual Planner without changing admission."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import rpent.planner.codex as codex_module
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.codex import CodexPlanner
from rpent.session.runtime import RuntimeJournal


@pytest.mark.parametrize("reserve", [0, 1])
def test_bound_v3_rule_reaches_prompt_and_preserves_real_tool_cap(
    tmp_path, monkeypatch, reserve
):
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v3",
                "chunk_steps": 5,
                "baseline_chunk_steps": 5,
                "finish_requires_native_success": False,
                "planner_finish_reserve_calls": reserve,
            }
        )
    )
    context = {
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile",
        "output_dir": str(tmp_path),
        "runtime_rules_ref": str(rules),
        "limits": {"tool_calls": 17, "environment_steps": 100},
    }
    journal = RuntimeJournal(context)
    captured = {}

    class Mcp:
        def __init__(self, _toolkit):
            pass

        def start(self):
            return "http://fake.invalid/mcp"

        def stop(self):
            pass

    def fake_session(prompt, output, _raw, _last, _recorder, state, *_args):
        captured["prompt"] = prompt
        output.write_text("offline")
        state["text"] = "offline"

    monkeypatch.setattr(codex_module, "HttpMcpServer", Mcp)
    planner = CodexPlanner(
        output_dir=str(tmp_path),
        output_path=tmp_path / "planner.out",
        dashboard_events=NullDashboardEventSink(),
        model="gpt-6.1-sol",
        reasoning_effort="medium",
    )
    monkeypatch.setattr(planner, "_run_session", fake_session)
    toolkit = SimpleNamespace(
        runtime_journal=journal, cancel_active_and_wait=lambda: None
    )
    planner.solve(
        system_prompt="system", user_message="task", toolkit=toolkit, max_turns=20
    )
    assert ("at most 17 admitted RPent tool calls" in captured["prompt"]) is bool(
        reserve
    )
    events = [json.loads(s) for s in journal.events_path.read_text().splitlines()]
    assert events[0]["effective_rules"]["planner_finish_reserve_calls"] == reserve
    assert sum(e["kind"] == "planner_rule_instruction" for e in events) == reserve
    assert journal.tool_calls == 0 and journal.chunk_steps == 5
    for _ in range(17):
        journal.admit_tool("read")
        journal.emit("tool_end", tool="read", status="completed")
    assert journal.tool_calls == 17
    assert journal.budget_stop()["limit"] == 17


def test_v3_rejects_boolean_reserve_value(tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v3",
                "chunk_steps": 5,
                "baseline_chunk_steps": 5,
                "finish_requires_native_success": False,
                "planner_finish_reserve_calls": True,
            }
        )
    )
    with pytest.raises(ValueError, match="finish reserve"):
        RuntimeJournal({"output_dir": str(tmp_path), "runtime_rules_ref": str(rules)})
