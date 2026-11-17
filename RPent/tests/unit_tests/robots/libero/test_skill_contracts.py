# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Skill admission and handoffs use the production Toolkit and public inputs."""

import json
from copy import deepcopy

import pytest

from robots.libero.skill_contracts import SCHEMA, planner_contracts, validate_contracts
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled
from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit
from tests.unit_tests.robots.libero.test_skill_routing import toolkit as routed_toolkit

SIGNATURE = {
    key: key + "-v1" for key in ("theta_ref", "processor_ref", "phi_ref", "code_ref")
}


def contract(identity="lift", **changes):
    result = {
        "skill_id": identity,
        "authorized": True,
        "version_signature": deepcopy(SIGNATURE),
        "action_prefix_steps": 5,
        "validation": {
            "status": "verified",
            "evidence_refs": ["independent-validation"],
            "task_scope": [{"suite": "public", "task": 1}],
        },
        "require_verified": True,
        "preconditions": [],
        "completion": [{"signal": "eef_z", "op": "ge", "value": 0.04}],
        "failure_conditions": [],
        "requires": [],
    }
    result.update(changes)
    return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tk, env, model, journal = make_toolkit(tmp_path, monkeypatch)
    # Preserve the actual call, prediction, action, observation and budget path;
    # unrelated image encoding stays at the existing state-capture boundary.
    monkeypatch.setattr(
        tk,
        "_capture_env_state",
        lambda *, result, **kwargs: ToolResult(data={"execution_segment": result}),
    )
    journal.context.update(
        SIGNATURE,
        task={"suite": "public", "task": 1, "seed": 0},
        skill_contracts={"schema": SCHEMA, "skills": {"vla_execute": contract()}},
    )
    original = env.obs

    def obs():
        value = original()
        value["states"][2] = env.steps / 100
        return value

    env.obs = obs
    return tk, env, model, journal


def call(tk, tool="vla_execute", **arguments):
    argument = "instruction" if tool == "vla_execute" else "prompt"
    return tk.execute_tool(
        tool, {argument: "public subgoal", "max_chunks": 1, **arguments}
    ).data


@pytest.mark.parametrize(
    ("change", "status", "reason"),
    [
        ({"authorized": False}, "rejected", "skill_not_authorized"),
        (
            {"version_signature": {**SIGNATURE, "theta_ref": "old-theta"}},
            "deferred",
            "skill_version_mismatch",
        ),
        (
            {
                "validation": {
                    "status": "verified",
                    "evidence_refs": ["old-task-result"],
                    "task_scope": [{"suite": "public", "task": 2}],
                }
            },
            "deferred",
            "outside_verified_task_scope",
        ),
        (
            {
                "validation": {
                    "status": "unverified",
                    "evidence_refs": [],
                    "task_scope": [],
                }
            },
            "deferred",
            "skill_not_verified",
        ),
        (
            {
                "preconditions": [
                    {"signal": "gripper_opening", "op": "ge", "value": 0.09}
                ]
            },
            "rejected",
            "precondition_unsatisfied",
        ),
    ],
)
def test_inapplicable_skills_return_actionable_zero_action_results(
    setup, change, status, reason
):
    tk, env, model, journal = setup
    journal.context["skill_contracts"]["skills"]["vla_execute"].update(change)
    result = call(tk)
    assert result["tool_status"] == status and result["reason"] == reason
    assert result["subgoal_outcome"] == result["native_outcome"] == "unknown"
    assert result["next_action"] and not env.sent and not model.inputs
    events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    assert any(
        row["kind"] == "skill_contract_admission" and row["reason"] == reason
        for row in events
    )
    assert any(
        row["kind"] == "skill_execution_result" and row["tool_status"] == status
        for row in events
    )
    assert events[-1]["kind"] == "tool_end" and events[-1]["status"] == status


def test_unknown_public_start_does_not_consume_private_evaluation_or_predict(setup):
    tk, env, model, _ = setup
    tk.primitives._last_obs.pop("wrist_images")
    result = call(tk)
    assert (
        result["tool_status"] == "deferred"
        and result["reason"] == "public_observation_missing"
    )
    assert not env.sent and not model.inputs


def test_rhob_selected_worker_contract_is_consumed_before_private_evaluation_rpc():
    tk, calls, private, _ = routed_toolkit()
    tk.runtime_journal.context.update(
        SIGNATURE,
        task={"suite": "public", "task": 1},
        skill_contracts={
            "schema": SCHEMA,
            "skills": {"pi0_pick": contract(authorized=False)},
        },
    )
    result = tk._execute_primitive(
        "vla_execute",
        tk.primitives.vla_execute,
        instruction="pick up the salad dressing",
        max_chunks=2,
    ).data
    assert (
        result["reason"] == "skill_not_authorized"
        and result["skill_result"]["skill_id"] == "lift"
    )
    assert not calls and not private.events


def test_tool_subgoal_and_native_results_are_separate_and_failure_is_explicit(setup):
    tk, _, _, journal = setup
    result = call(tk)
    assert (
        result["tool_status"] == "completed" and result["subgoal_outcome"] == "success"
    )
    assert result["native_outcome"] == "unknown"
    journal.context["skill_contracts"]["skills"]["vla_execute"][
        "failure_conditions"
    ] = [{"signal": "eef_z", "op": "ge", "value": 0.09}]
    result = call(tk)
    assert (
        result["tool_status"] == "completed" and result["subgoal_outcome"] == "failure"
    )
    assert (
        result["native_outcome"] == "unknown"
        and tk._skill_relations["lift"]["valid"] is False
    )


def test_handoff_consumes_current_relation_and_contradiction_does_not_resurrect_it(
    setup,
):
    tk, env, _, journal = setup
    journal.context["skill_contracts"]["skills"]["pi0_doubled"] = contract(
        "contact", requires=["lift"], completion=[]
    )
    assert call(tk, "pi0_doubled")["reason"] == "handoff_not_established"
    first = call(tk)
    source = first["skill_result"]["handoff"]["lift"]["source_sequence"]
    assert call(tk, "pi0_doubled")["tool_status"] == "completed"
    observed = env.obs()
    observed["states"][2] = 0.0
    tk.primitives.set_obs(observed)
    assert call(tk, "pi0_doubled")["reason"] == "handoff_not_established"
    assert tk._skill_relations["lift"]["source_sequence"] == source
    observed["states"][2] = 0.2
    tk.primitives.set_obs(observed)
    assert call(tk, "pi0_doubled")["reason"] == "handoff_not_established"


def test_unchanged_observation_cannot_recommit_completion(setup):
    tk, _, _, journal = setup
    first = call(tk)
    source = first["skill_result"]["handoff"]["lift"]["source_sequence"]
    # A handler's claim with no new legal observation cannot advance the goal.
    result = tk._execute_primitive(
        "vla_execute",
        lambda **kwargs: ToolResult(data={"success": True}),
        instruction="public subgoal",
        max_chunks=1,
    ).data
    assert result["subgoal_outcome"] == "unknown"
    assert tk._skill_relations["lift"]["source_sequence"] == source
    assert journal.context["task"]["task"] == 1


@pytest.mark.parametrize("tool", ["vla_execute", "pi0_doubled", "pi0_pick"])
def test_partial_rpc_reports_prediction_and_unknown_send_and_blocks_all_motion(
    setup, tool
):
    tk, env, model, journal = setup
    journal.context.pop("skill_contracts")

    def partial(actions, **kwargs):
        env.steps += 2
        raise TimeoutError("reply lost after two owner steps")

    env.rsi_chunk_step = partial
    result = call(tk, tool)["execution_segment"]
    assert result["predictions_started"] == result["predictions_completed"] == 1
    assert result["executed_steps"] == 0 and result["unknown_steps"] == 5
    assert result["action_feedback_unknown"] and result["tool_status"] == "failed"
    assert result["subgoal_outcome"] == result["native_outcome"] == "unknown"
    before = len(model.inputs)
    assert call(tk)["reason"] == "action_feedback_unknown"
    moved = tk._execute_primitive(
        "release", lambda: pytest.fail("unknown feedback allowed motion")
    )
    assert (
        moved.data["next_action"] == "reconcile_execution"
        and len(model.inputs) == before
    )
    direct = tk.primitives.vla_execute("direct retry", max_chunks=1).data
    assert (
        direct["stop_reason"] == "technical_error"
        and direct["predictions_started"] == 0
    )
    assert len(model.inputs) == before


@pytest.mark.parametrize(
    ("stop", "reason"), [("cancel", "cancelled"), ("budget", "budget")]
)
def test_legacy_contact_stop_preserves_semantics_and_zero_send(setup, stop, reason):
    tk, env, model, journal = setup
    journal.context.pop("skill_contracts")
    if stop == "cancel":
        predict = model.predict

        def cancel_after_prediction(*args, **kwargs):
            result = predict(*args, **kwargs)
            tk.primitives._check_cancelled = lambda: (_ for _ in ()).throw(
                ToolCancelled("cancel after prediction")
            )
            return result

        model.predict = cancel_after_prediction
    else:
        journal.context["limits"]["environment_steps"] = 3
    result = call(tk, "pi0_doubled")["execution_segment"]
    assert result["stop_reason"] == reason and result["tool_status"] == "interrupted"
    assert result["unknown_steps"] == result["executed_steps"] == 0 and not env.sent
    assert result["predictions_completed"] == 1


@pytest.mark.parametrize(
    "signal", ["object_pos", "drawer_qpos", "holding", "native_success"]
)
def test_private_or_unfounded_predicates_are_rejected_before_execution(signal):
    spec = {
        "schema": SCHEMA,
        "skills": {
            "vla_execute": contract(
                completion=[{"signal": signal, "op": "ge", "value": 1}]
            )
        },
    }
    with pytest.raises(ValueError, match="public observation signal"):
        validate_contracts(spec)


def test_planner_consumes_scope_and_conditions_without_private_validation_paths():
    configured = contract(preconditions=[{"signal": "eef_z", "op": "ge", "value": 0.1}])
    configured["validation"]["evidence_refs"] = [
        "environment_service/private-joint-evaluation.json"
    ]
    view = planner_contracts(
        {"skill_contracts": {"schema": SCHEMA, "skills": {"vla_execute": configured}}}
    )
    assert view["skills"]["vla_execute"]["preconditions"] == configured["preconditions"]
    assert (
        view["skills"]["vla_execute"]["validation"]["task_scope"]
        == configured["validation"]["task_scope"]
    )
    assert "private-joint" not in json.dumps(view)


def test_common_planner_contracts_only_expose_the_callable_motion_tool():
    view = planner_contracts(
        {"common_execution": {"allowed_motion_tools": ["vla_execute"]}}
    )
    assert set(view["skills"]) == {"vla_execute"}
