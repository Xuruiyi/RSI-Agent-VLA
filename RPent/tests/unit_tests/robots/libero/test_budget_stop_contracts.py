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

"""Offline budget causes, admitted-tool draining and independent closure."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rpent.planner.codex as codex_module
from robots.libero.result import finalize_run
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.planner.codex import CodexPlanner, _Recorder
from rpent.session.runtime import RuntimeBudgetExceeded, RuntimeJournal


def _context(path, **limits):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile.json",
        "parent_decision_id": None,
        "output_dir": str(path),
        "budget_stop_ref": str(path / "budget_stop.json"),
        "limits": {"tool_calls": 40, "environment_steps": 100, **limits},
    }


def _events(journal):
    return [json.loads(line) for line in journal.events_path.read_text().splitlines()]


def _complete_tool(journal, name="read"):
    journal.admit_tool(name)
    journal.emit("tool_end", tool=name, status="completed", elapsed_seconds=0.0)


def test_tool_40_publishes_before_completion_and_allows_its_admitted_action(tmp_path):
    journal = RuntimeJournal(_context(tmp_path))
    for _ in range(39):
        _complete_tool(journal)
    assert journal.budget_stop() is None
    journal.admit_tool("move")
    marker = journal.budget_stop()
    assert (
        marker["reason"] == "tool_calls" and marker["observed"] == marker["limit"] == 40
    )
    assert marker["stop_after_inflight"] is True and journal.active_tools == 1
    calls = []
    journal.action(
        lambda: (calls.append(1) or {"states": []}, 0, False, False, {}),
        [0] * 7,
        source="move",
    )
    assert calls == [1]
    journal.emit("tool_end", tool="move", status="completed")
    assert journal.active_tools == 0
    with pytest.raises(RuntimeBudgetExceeded):
        journal.admit_tool("finish")
    assert journal.tool_calls == 40


def test_client_service_share_one_complete_first_cause_without_overwrite(tmp_path):
    context = _context(tmp_path / "client")
    context["budget_stop_ref"] = str(tmp_path / "budget_stop.json")
    client = RuntimeJournal(context)
    server = RuntimeJournal({**context, "output_dir": str(tmp_path / "service")})
    first = client.mark_budget_stop("tool_calls", limit=40, observed=40)
    before = Path(context["budget_stop_ref"]).read_bytes()
    second = server.mark_budget_stop("environment_steps", limit=100, observed=100)
    assert first == second
    assert Path(context["budget_stop_ref"]).read_bytes() == before
    assert first["identity"] == client.identity()
    assert _events(client)[first["sequence"]]["kind"] == "budget_stop"
    assert not list(tmp_path.glob(".budget-stop-*"))


def test_action_rejection_publishes_remaining_budget_before_dispatch(tmp_path):
    journal = RuntimeJournal(_context(tmp_path, environment_steps=2))
    with pytest.raises(RuntimeBudgetExceeded):
        journal.action(
            lambda: pytest.fail("must not dispatch"),
            [[0] * 7] * 3,
            source="vla",
            chunk=True,
        )
    marker = journal.budget_stop()
    assert marker["reason"] == "environment_steps" and marker["requested_steps"] == 3
    assert marker["observed"] == journal.reserved_steps == 0


def _run_fake_turn(tmp_path, monkeypatch, *, finish=False):
    journal = RuntimeJournal(_context(tmp_path))
    for _ in range(39):
        _complete_tool(journal)
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    executions = []

    class Turn:
        interrupts = 0
        model_requests = 0

        def interrupt(self):
            self.interrupts += 1

        def stream(self):
            self.model_requests += 1
            yield {"method": "turn/started"}
            journal.admit_tool("finish" if finish else "move")
            yield {"method": "item/started"}
            assert self.interrupts == 0, "admitted last tool must finish"
            executions.append("completed")
            journal.emit(
                "tool_end", tool="finish" if finish else "move", status="completed"
            )
            yield {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "type": "mcpToolCall",
                        "tool": "finish" if finish else "move",
                        "status": "completed",
                        "result": {"_finish": True, "status": "failure"}
                        if finish
                        else {"ok": True},
                    }
                },
            }
            if not finish and not self.interrupts:
                self.model_requests += 1
            yield {
                "method": "turn/completed",
                "payload": {
                    "turn": {
                        "status": "interrupted" if self.interrupts else "completed"
                    }
                },
            }

    turn = Turn()

    class NativeCodex:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def thread_start(self, **_kwargs):
            return SimpleNamespace(turn=lambda *_args, **_options: turn)

    monkeypatch.setattr(codex_module.openai_codex, "Codex", NativeCodex)
    planner = CodexPlanner(
        output_dir=str(tmp_path),
        dashboard_events=NullDashboardEventSink(),
        model="offline",
    )
    monkeypatch.setattr(planner, "_build_config", lambda _url: None)
    state = {}
    planner._run_session(
        "offline",
        tmp_path / "output.txt",
        tmp_path / "stream.jsonl",
        tmp_path / "last.txt",
        recorder,
        state,
        "unused",
        SimpleNamespace(runtime_journal=journal),
    )
    assert "error" not in state
    assert executions == ["completed"]
    return journal, recorder, turn


def test_stream_records_local_stop_after_inflight_and_does_not_request_again(
    tmp_path, monkeypatch
):
    journal, recorder, turn = _run_fake_turn(tmp_path, monkeypatch)
    assert turn.interrupts == turn.model_requests == 1
    stats = recorder.stats()
    assert stats["planner_turn_status"] == "interrupted"
    assert stats["attempt_closed"] is True
    assert stats["termination_reason"] == "verified_budget_exhaustion"
    stop = stats["planner_budget_stop"]
    assert (
        _events(journal)[stop["planner_event_sequence"]]["kind"]
        == "planner_budget_stop"
    )
    raw = [
        json.loads(line)
        for line in (tmp_path / "stream.jsonl").read_text().splitlines()
    ]
    assert any(row.get("type") == "planner_budget_stop" for row in raw)


def test_legal_finish_as_tool_40_keeps_its_result_and_normal_sdk_completion(
    tmp_path, monkeypatch
):
    journal, recorder, turn = _run_fake_turn(tmp_path, monkeypatch, finish=True)
    assert journal.budget_stop()["reason"] == "tool_calls"
    assert turn.interrupts == 0
    assert recorder.finish_result["status"] == "failure"
    assert recorder.stats()["planner_turn_status"] == "completed"
    assert recorder.stats()["planner_budget_stop"] is None


def test_maxturn_controller_has_explicit_cause_but_external_cancel_does_not(tmp_path):
    journal = RuntimeJournal(_context(tmp_path))
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    recorder.turns = 20
    stop = recorder.request_budget_stop()
    assert stop["reason"] == "max_turns"
    recorder.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "interrupted"}}}
    )
    assert recorder.stats()["attempt_closed"]
    external = _Recorder(20, NullDashboardEventSink())
    external.turns = 20
    external.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "interrupted"}}}
    )
    assert external.request_budget_stop() is None
    assert external.stats()["planner_budget_stop"] is None
    assert external.stats()["attempt_closed"] is False


def _result_context(
    tmp_path, journal, recorder, *, pending=False, error=None, aborted=False
):
    service = {
        "identity": journal.identity(),
        "service_closed": True,
        "pending_action": pending,
        "environment_steps_unknown": 0,
        "last_native_success": False,
        "last_feedback_sequence": 5,
        "native_success_source": "native _check_success()",
    }
    service_path = tmp_path / "service.json"
    service_path.write_text(json.dumps(service))
    return RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={},
        environment_success=False,
        agent_error=error,
        elapsed_s=1.0,
        planner="codex",
        model="offline",
        reasoning_effort="medium",
        max_turns=20,
        planner_timeout_s=600,
        finish_result={"operator_aborted": True} if aborted else None,
        stats=recorder.stats(),
        environment_metrics={
            "terminated": False,
            "truncated": False,
            "runtime": {
                "identity": journal.identity(),
                "action_events_ref": str(journal.events_path),
                "environment_service": {"costs_ref": str(service_path)},
            },
        },
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("budget", False),
        ("pending", None),
        ("error", None),
        ("cancel", None),
        ("aborted", None),
        ("identity", None),
    ],
)
def test_independent_budget_failure_requires_explicit_cause_and_complete_native_owner(
    tmp_path, case, expected
):
    journal = RuntimeJournal(_context(tmp_path, tool_calls=1))
    _complete_tool(journal)
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    if case != "cancel":
        recorder.request_budget_stop()
    recorder.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "interrupted"}}}
    )
    if case == "identity":
        marker = journal.budget_stop()
        marker["identity"]["bundle_id"] = "other"
        journal.budget_stop_path.write_text(json.dumps(marker))
    context = _result_context(
        tmp_path,
        journal,
        recorder,
        pending=case == "pending",
        error="provider failed" if case == "error" else None,
        aborted=case == "aborted",
    )
    result = json.loads(finalize_run(context).read_text())
    assert result["benchmark_success"] is expected
    if expected is False:
        assert result["attempt_closed"] and result["verified_budget_exhaustion"]
        assert result["termination_reason"] == "verified_budget_exhaustion"
        assert result["attempt_outcome_source"]["runner_completion"] is None
        assert result["attempt_outcome_source"]["budget_completion"]
    else:
        assert not result["verified_budget_exhaustion"]


def test_provider_error_after_budget_request_does_not_become_budget_success(tmp_path):
    journal = RuntimeJournal(_context(tmp_path, tool_calls=1))
    _complete_tool(journal)
    recorder = _Recorder(20, NullDashboardEventSink(), runtime_journal=journal)
    recorder.request_budget_stop()
    recorder.observe(
        {
            "method": "error",
            "payload": {"message": "provider failed", "will_retry": False},
        }
    )
    recorder.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "failed"}}}
    )
    assert recorder.stats()["attempt_closed"] is False
    assert recorder.stats()["termination_reason"] == "failed"
