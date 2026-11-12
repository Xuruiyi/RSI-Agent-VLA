# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Actual Toolkit -> native finalizer -> aggregate closure and evaluator, CPU fakes.

The aggregate consumer is optional in the standalone RPent checkout. Run with
PYTHONPATH=<aggregate>/core/src:. to exercise this cross-project contract.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit

pytest.importorskip("rsi_vla")

from rsi_vla.adapters.rpent import read_trajectory
from rsi_vla.adapters.runtime_acceptance import inspect_attempt
from rsi_vla.adapters.runtime_closure import finalize_runtime
from rsi_vla.evaluator import evaluate_matrix

from robots.libero import tools
from robots.libero.result import finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.session.base import StepRecord
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize(
    "case,outcome,status",
    [
        ("inference", "unknown", "technical_error"),
        ("inference_then_budget", "unknown", "technical_error"),
        ("partial_rpc", "unknown", "technical_error"),
        ("retry_success", "unknown", "technical_error"),
        ("legacy_inference", "unknown", "technical_error"),
        ("failure", "failure", "completed"),
        ("success", "success", "completed"),
        ("bad_arguments_then_success", "success", "completed"),
        ("unfinished", "unknown", "unknown"),
        ("budget", "failure", "budget_exhausted"),
        ("cancel", "unknown", "cancelled"),
    ],
)
def test_tool_events_reach_final_result_and_consumers(
    tmp_path, monkeypatch, case, outcome, status
):
    protocol = json.loads((ROOT / "config/pilot_v9.json").read_text())
    profile = json.loads((ROOT / "config/run_profile_v9.json").read_text())
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    journal.context.update(
        protocol_id=protocol["protocol_id"],
        bundle_id=protocol["anchor_bundle_id"],
    )
    # The constructor's run_start must belong to the same fixture identity.
    initial = json.loads(journal.events_path.read_text())
    initial.update(journal.identity())
    journal.events_path.write_text(json.dumps(initial) + "\n")
    root = journal.output_dir
    identity = journal.identity()
    task = {"suite": "libero_object_swap", "task": 2, "seed": 1}
    context = {**journal.context, "task": task}
    records = []

    def capture(*args, log, **kwargs):
        record = StepRecord(
            step_idx=len(records),
            state={},
            terminated=env.terminated,
            truncated=env.truncated,
            **log,
        )
        records.append(record)
        return record

    monkeypatch.setattr(tools, "dump_state", capture)
    monkeypatch.setattr(tools, "view_env_state", lambda *a, **k: ToolResult(data={}))
    monkeypatch.setattr(tk, "_publish_step", lambda record: None)
    tk._state = SimpleNamespace(
        records=lambda: records, latest_record=lambda: records[-1]
    )
    original_predict, original_rpc = model.predict, env.rsi_chunk_step

    def fail_predict(*args, **kwargs):
        raise RuntimeError("fake VLA inference unavailable")

    def partial_rpc(actions, **kwargs):
        if env.steps:
            raise ConnectionError("fake RPC lost after two owner-confirmed actions")
        return original_rpc(actions, **kwargs)

    if case in {
        "inference",
        "inference_then_budget",
        "retry_success",
        "legacy_inference",
    }:
        model.predict = fail_predict
    if case == "partial_rpc":
        env.rsi_chunk_step = partial_rpc
    if case in {"success", "bad_arguments_then_success"}:
        env.terminal = "success"
    if case == "cancel":

        def cancel_after_prediction():
            if model.inputs:
                raise ToolCancelled("operator cancelled")

        tk._primitives._check_cancelled = cancel_after_prediction
    if case == "bad_arguments_then_success":
        assert tk.execute_tool("vla_execute", {"instruction": "task"}).is_error
    if case == "legacy_inference":
        tool, arguments = "pi0_doubled", {"prompt": "task"}
    else:
        tool, arguments = "vla_execute", {"instruction": "task", "max_chunks": 2}
    response = tk.execute_tool(tool, arguments)
    if case == "retry_success":
        model.predict = original_predict
        env.terminal = "success"
        tk.execute_tool("vla_execute", {"instruction": "task", "max_chunks": 1})
    if case == "inference":
        assert not env.sent and response.is_error
    if case == "partial_rpc":
        segment = response.data["execution_segment"]
        assert segment["executed_steps"] == 5 and segment["unknown_steps"] == 5

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"identity": identity, **value}))

    # A local file fixture is the native worker; no simulator/model is loaded.
    events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    native = []
    for event in events:
        if event["kind"] == "action_send":
            native.append(
                dict(
                    **identity,
                    sequence=len(native),
                    kind="action_send",
                    source="rlinf.policy",
                )
            )
            native.append(
                dict(
                    **identity,
                    sequence=len(native),
                    kind="rpc_dispatch_end",
                    rpc_evidence={"client_send_sequence": event["sequence"]},
                    execution_mask=[True] * 5,
                    unknown_proposal_indices=[],
                )
            )
    (root / "environment_service").mkdir(exist_ok=True)
    (root / "environment_service/action_events.jsonl").write_text(
        "\n".join(map(json.dumps, native))
    )
    write(
        "environment_service/runtime_costs.json",
        {
            "service_closed": True,
            "pending_action": False,
            "environment_steps_unknown": 3 if case == "partial_rpc" else 0,
            "environment_steps_confirmed": 30 + env.steps,
            "environment_steps_reserved": 30 + env.steps,
            "setup_steps_confirmed": 30,
            "policy_steps_confirmed": env.steps,
            "construction_complete": True,
            "reset_calls": 1,
            "reset_completed": 1,
            "reset_wall_seconds": 0.1,
            "last_native_success": env.terminated,
            "last_feedback_sequence": 7,
            "native_success_source": "fixture native worker _check_success()",
        },
    )
    if case in {"budget", "inference_then_budget"}:
        journal.mark_budget_stop("model_requests", limit=20, observed=20)
        marker = journal.budget_stop()
        planner_sequence = journal.emit("planner_budget_stop", budget_stop=marker)
        stats = {
            "attempt_closed": True,
            "planner_turn_status": "interrupted",
            "planner_budget_stop": {
                **marker,
                "budget_stop_ref": str(journal.budget_stop_path),
                "planner_event_sequence": planner_sequence,
                "planner_events_ref": str(journal.events_path),
            },
        }
    else:
        stats = (
            {}
            if case == "unfinished"
            else {"attempt_closed": True, "planner_turn_status": "completed"}
        )
    finish = (
        None
        if case in {"unfinished", "budget", "success"}
        else {"_finish": True, "status": "failure", "summary": "Planner normal finish"}
    )
    metrics = tk.evaluation_metrics(success=None)
    raw = json.loads(
        finalize_run(
            RunFinalizationContext(
                output_dir=root,
                robot_name="libero",
                task_desc=task,
                environment_success=None,
                agent_error=None,
                elapsed_s=1,
                planner="codex",
                model="gpt-6.1-sol",
                reasoning_effort="medium",
                max_turns=20,
                planner_timeout_s=600,
                finish_result=finish,
                stats=stats,
                environment_metrics=metrics,
            )
        ).read_text()
    )
    original_bytes = (root / "result.json").read_bytes()
    fault = status == "technical_error"
    assert raw["execution_health"]["status"] == ("unresolved" if fault else "clear")
    assert raw["terminated"] is env.terminated
    if fault:
        assert raw["validity"] == "technical_invalid"
        assert raw["error"] is None  # Planner itself completed normally.
    if case == "retry_success":
        assert raw["benchmark_success"] is True  # Environment fact survives.
    if case == "budget":
        assert raw["verified_budget_exhaustion"] is True
    journal.finalize()
    write("runtime_context.json", context)
    write(
        "supervisor.json",
        {
            "status": "completed",
            "cleanup_confirmed": True,
            "exit_code": 0,
            "stop_reason": None,
        },
    )
    write(
        "launcher_costs.json",
        {
            "whole_job_wall_seconds": 1,
            "gpu_device_seconds_envelope": 0,
            "training_seconds": 0,
            "manual_seconds": 0,
        },
    )
    write(
        "request_gate.json",
        {
            "hard_request_cap_enforced": True,
            "requests_forwarded": 1,
            "active_requests_at_close": 0,
        },
    )
    write(
        "model_load.json",
        {
            "loaded": True,
            "embodiment": "libero",
            "action_chunk": 5,
            "prediction_count": 1,
            "last_actions_finite": True,
            "last_action_shape": [5, 7],
            "model_path": "/fixture/model",
            "norm_stats_path": "/fixture/norm",
            "model_load_seconds": 0,
            "effective_config": {
                "pi05": True,
                "openpi": {"config_name": "pi05_libero"},
            },
        },
    )
    write(
        "launch_plan.json",
        {
            "environment": {
                "PI05_CHECKPOINT_PATH": "/fixture/model",
                "PI05_NORM_STATS_PATH": "/fixture/norm",
            }
        },
    )
    (root / "model_requests.jsonl").write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {
                        "kind": "request_admitted",
                        "request_number": 1,
                        "model": "gpt-6.1-sol",
                        "reasoning_effort": "medium",
                    },
                    {
                        "kind": "response_terminal",
                        "request_number": 1,
                        "event_type": "response.completed",
                        "usage": {
                            "input_tokens": 10,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                        },
                    },
                ],
            )
        )
    )
    result = finalize_runtime(context)
    assert (root / "result.json").read_bytes() == original_bytes
    assert result["outcome"] == outcome and result["runtime_status"] == status
    assert result["native_task_result"]["last_native_success"] is env.terminated
    if fault:
        assert result["validity"] == "technical_invalid"
        assert (
            result["execution_health"]["recovery_criterion"]
            == "not_defined_for_execution_faults"
        )
        issues = result["execution_health"]["issues"]
        assert all(e["episode_id"] == context["episode_id"] for e in issues)
        if case != "legacy_inference":
            segment = next(e for e in issues if e["kind"] == "vla_segment_end")
            start = events[segment["segment_sequence"]]
            assert start["kind"] == "vla_segment_start"
            if case == "partial_rpc":
                feedback = next(e for e in issues if e["kind"] == "action_feedback")
                proposal = events[feedback["proposal_sequence"]]
                assert proposal["segment_sequence"] == start["sequence"]
                assert (
                    events[proposal["inference_sequence"]]["kind"] == "inference_start"
                )
    audit = inspect_attempt(root, require_vla=False)
    assert audit["checks"]["execution_health"] is (not fault)
    if fault:
        assert not audit["passed"] and audit["outcome"] == "unknown"
        # Even an explicit raw-path caller cannot bypass the event audit.
        assert not inspect_attempt(root, result_path=root / "result.json")["checks"][
            "execution_health"
        ]
    manifest = [
        {
            "task_key": "position_swap_salad",
            "seed": 1,
            "run_id": context["run_id"],
            "episode_id": context["episode_id"],
            "result_path": str(root / "result.json"),
        }
    ]
    evaluated = evaluate_matrix(
        protocol,
        profile,
        manifest,
        result_id="fixture",
        bundle_id=context["bundle_id"],
        role="development",
    )
    row = next(
        r
        for r in evaluated.assignments
        if r["task_key"] == "position_swap_salad" and r["seed"] == 1
    )
    assert row["outcome"] == outcome
    if fault:
        assert row["validity"] == "invalid"
    if outcome != "unknown":
        assert row["evidence_ref"] == str(root / "independent_result.json")
    (root / "states.json").write_text('{"steps": []}')
    write("transcript.json", {"runtime": {"identity": identity}})
    imported = read_trajectory(
        root / "states.json",
        root / "transcript.json",
        run_id=context["run_id"],
        episode_id=context["episode_id"],
        bundle_id=context["bundle_id"],
        protocol_id=context["protocol_id"],
        independent_result_path=root / "independent_result.json",
    )[-1]
    assert imported.outcome == outcome
    if fault:
        assert imported.flags["technical_invalid"] is True
        raw_import = read_trajectory(
            root / "states.json",
            root / "transcript.json",
            run_id=context["run_id"],
            episode_id=context["episode_id"],
            bundle_id=context["bundle_id"],
            protocol_id=context["protocol_id"],
            independent_result_path=root / "result.json",
        )[-1]
        assert raw_import.outcome == "unknown"
