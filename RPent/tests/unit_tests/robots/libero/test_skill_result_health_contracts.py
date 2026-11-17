# Copyright 2026 The RPent Authors.

"""Legacy skill terminal facts reach health without replacing native outcomes."""

import json

import pytest

from robots.libero.result import execution_health, finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.session.runtime import RuntimeJournal


@pytest.mark.parametrize(
    "status,unknown,reason,expected",
    [
        ("failed", False, "technical_error", "unresolved"),
        ("interrupted", True, "cancelled", "unresolved"),
        ("interrupted", False, "cancelled", "clear"),
        ("completed", False, "heuristic_stop", "clear"),
    ],
)
def test_journal_skill_result_health_keeps_interruption_and_unknown_separate(
    tmp_path, status, unknown, reason, expected
):
    journal = RuntimeJournal(
        {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
            "output_dir": str(tmp_path),
        }
    )
    journal.emit(
        "skill_execution_result",
        tool="pi0_pick",
        tool_status=status,
        action_feedback_unknown=unknown,
        stop_reason=reason,
        subgoal_outcome="unknown",
        native_outcome="success",
    )
    health = execution_health(
        {"action_events_ref": str(journal.events_path), "identity": journal.identity()}
    )
    assert health["status"] == expected
    assert bool(health["stops"]) == (status == "interrupted")
    if health["issues"]:
        assert health["issues"][0]["native_outcome"] == "success"


@pytest.mark.parametrize(
    "result",
    [
        {"tool_status": "failed"},
        {"tool_status": "interrupted", "action_feedback_unknown": True},
    ],
)
def test_legacy_skill_fault_without_vla_end_stays_technical_and_preserves_native_fact(
    tmp_path, result
):
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={},
        environment_success=True,
        agent_error=None,
        elapsed_s=1,
        planner="flash",
        model="offline",
        reasoning_effort="medium",
        max_turns=1,
        planner_timeout_s=1,
        finish_result={"_finish": True, "status": "success"},
        stats={"planner_turn_status": "completed"},
        environment_metrics={
            "terminated": True,
            "truncated": False,
            "execution_segments": [{**result, "action": "pi0_doubled", "step_idx": 1}],
        },
    )
    final = json.loads(finalize_run(context).read_text())
    assert final["benchmark_success"] is True
    assert final["terminated"] is True
    assert final["validity"] == "technical_invalid"
    assert final["execution_health"]["issues"][0]["action"] == "pi0_doubled"
