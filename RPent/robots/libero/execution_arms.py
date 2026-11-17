# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Frozen whole-episode consumers using the ordinary LIBERO Toolkit/VLA chain."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np

from rpent.session.runtime import RuntimeBudgetExceeded
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled

SCHEMA = "libero-execution-arm-v1"
SIGNALS = {"eef_z", "eef_lift", "gripper_opening"}
COMMON_SCHEMA = "libero-common-execution-v1"


def validate_common_execution(
    spec: dict[str, Any],
    *,
    action_prefix_steps: int = 5,
    environment_steps: int | None = None,
    tool_calls: int | None = None,
) -> dict[str, Any]:
    """Validate the common action and recording contract before an episode.

    Every prediction consumes one tool allowance, including predictions inside
    an Agent's longer logical segment. Recording follows these same boundaries.
    """
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {
            "schema",
            "observation_cadence",
            "allowed_motion_tools",
            "action_prefix_steps",
        }
        or spec.get("schema") != COMMON_SCHEMA
        or spec.get("observation_cadence") != "prediction"
        or spec.get("allowed_motion_tools") != ["vla_execute"]
        or type(spec.get("action_prefix_steps")) is not int
        or spec["action_prefix_steps"] != 5
        or type(action_prefix_steps) is not int
        or action_prefix_steps != 5
    ):
        raise ValueError("unsupported common LIBERO execution contract")
    if environment_steps is not None or tool_calls is not None:
        if (
            type(environment_steps) is not int
            or environment_steps < 1
            or type(tool_calls) is not int
            or tool_calls < (environment_steps + 4) // 5
        ):
            raise ValueError("tool allowance cannot cover the common physical bound")
    return deepcopy(spec)


def validate_execution_arm(spec: dict[str, Any]) -> dict[str, Any]:
    """Freeze a bounded D/S/R program or an A metadata declaration.

    S advances in list order when a stage completes or its allowance ends.
    R uses the registered complete/timeout/unknown edges, including ``stop``.
    Conditions read only public EEF/gripper observations, never evaluator state.
    """
    spec = deepcopy(spec)
    if spec.get("schema") != SCHEMA or spec.get("arm") not in {"D", "S", "R", "A"}:
        raise ValueError("unsupported LIBERO execution arm")
    if spec["arm"] == "A":
        return spec
    if type(spec.get("max_chunks")) is not int or spec["max_chunks"] < 1:
        raise ValueError("execution arm requires a positive max_chunks")
    if spec["arm"] == "D":
        if spec.get("stages"):
            raise ValueError("D cannot replace the native task with frozen stages")
        return spec
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("S/R requires frozen stages")
    ids = set()
    for stage in stages:
        if (
            not isinstance(stage.get("id"), str)
            or not stage["id"]
            or stage["id"] in ids
        ):
            raise ValueError("stage ids must be nonempty and unique")
        ids.add(stage["id"])
        if (
            not isinstance(stage.get("instruction"), str)
            or not stage["instruction"].strip()
        ):
            raise ValueError("stage requires a frozen instruction")
        if type(stage.get("max_chunks")) is not int or stage["max_chunks"] < 1:
            raise ValueError("stage requires a positive chunk allowance")
        condition = stage.get("completion")
        if condition is not None and (
            not isinstance(condition, dict)
            or set(condition) != {"signal", "op", "value"}
            or condition["signal"] not in SIGNALS
            or condition["op"] not in {"ge", "le"}
            or type(condition["value"]) not in (float, int)
            or not isfinite(condition["value"])
        ):
            raise ValueError(
                "stage condition must use a finite public observation signal"
            )
        if spec["arm"] == "S" and any(key.startswith("on_") for key in stage):
            raise ValueError("S follows list order and cannot contain branch edges")
    if spec["arm"] == "R":
        for stage in stages:
            if any(
                stage.get(key) not in ids | {"stop"}
                for key in ("on_complete", "on_timeout", "on_unknown")
            ):
                raise ValueError("R requires frozen complete/timeout/unknown edges")
    return spec


def select_execution_arm(
    profile: dict[str, Any], task_key: str
) -> dict[str, Any] | None:
    """Select a registered task program while freezing one arm for the profile.

    A task mapping is complete for every task on which the profile is used;
    missing tasks never fall back to another task's instruction. Profiles with
    no mapping retain their original single-program behavior.
    """
    fallback = profile.get("execution_arm")
    if "execution_arm_by_task" not in profile:
        return validate_execution_arm(fallback) if fallback is not None else None
    programs = profile["execution_arm_by_task"]
    if not isinstance(programs, dict) or not programs:
        raise ValueError("execution_arm_by_task requires registered task programs")
    selected = {}
    expected_arm = (
        validate_execution_arm(fallback)["arm"] if fallback is not None else None
    )
    for key, program in programs.items():
        if not isinstance(key, str) or not key or not isinstance(program, dict):
            raise ValueError(
                "task execution programs require nonempty task keys and specs"
            )
        frozen = validate_execution_arm(program)
        if expected_arm is None:
            expected_arm = frozen["arm"]
        if frozen["arm"] != expected_arm:
            raise ValueError("all task execution programs must use the same arm")
        if frozen["arm"] == "D" and "stages" in frozen:
            raise ValueError(
                "D task programs use the native instruction without stages"
            )
        selected[key] = frozen
    if task_key not in selected:
        raise ValueError("no frozen execution program for task: " + task_key)
    return selected[task_key]


def _observe(primitives, initial_z: float | None) -> dict[str, Any]:
    obs = primitives._last_obs
    values = np.asarray(obs.get("states", []), dtype=float).reshape(-1)
    valid = len(values) >= 8 and np.isfinite(values[:8]).all()
    z = float(values[2]) if valid else None
    return {
        "source": "latest_environment_observation",
        "observation_sequence": getattr(primitives, "_observation_sequence", None),
        "eef_z": z,
        "eef_lift": z - initial_z if z is not None and initial_z is not None else None,
        "gripper_opening": float(abs(values[6]) + abs(values[7])) if valid else None,
        "image_shapes": {
            key: list(np.shape(obs[key]))
            for key in ("main_images", "wrist_images")
            if key in obs
        },
    }


def run_execution_arm(
    toolkit, spec: dict[str, Any], *, dynamic_runner=None
) -> ToolResult:
    """Run a whole bounded episode through existing tools, journal and budgets.

    A delegates to the existing dynamic runner supplied by its caller. The CLI
    uses its ordinary Codex path for A and the Flash robot hook for D/S/R.
    Program exhaustion without native terminal feedback remains unknown.
    """
    spec = validate_execution_arm(spec)
    arm = spec["arm"]
    if arm == "A":
        if dynamic_runner is None:
            raise ValueError("A requires the existing dynamic Agent runner")
        return dynamic_runner(toolkit)
    journal = toolkit.runtime_journal
    if (
        getattr(toolkit, "_relation_control", None) is not None
        or journal
        and journal.context.get("relation_control") is not None
    ):
        raise ValueError(
            "frozen execution arms require relation_control absent (no rhoB routing)"
        )
    primitives = toolkit.primitives
    if primitives.policy_kind != "vla":
        raise ValueError("frozen execution arms require the VLA backend")
    instruction = primitives._last_obs["task_descriptions"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("native task instruction is missing")
    stages = spec.get("stages") or [
        {
            "id": "native_task",
            "instruction": instruction,
            "max_chunks": spec["max_chunks"],
        }
    ]
    stage_map = {row["id"]: row for row in stages}
    stage_id, stage_chunks = stages[0]["id"], 0
    first = _observe(primitives, None)
    initial_z = first["eef_z"]
    totals = dict.fromkeys(
        (
            "predictions_started",
            "predictions_completed",
            "executed_steps",
            "unknown_steps",
        ),
        0,
    )
    decisions, segments = [], []
    reason, diagnostic = "program_exhausted", None
    if journal:
        journal.emit(
            "episode_execution_start",
            execution_arm=spec,
            native_instruction=instruction,
            information_scope="public latest image/state observations; no independent evaluation truth",
        )
    try:
        for _ in range(spec["max_chunks"]):
            if primitives.env.terminated or primitives.env.truncated:
                break
            stage = stage_map[stage_id]
            observed = _observe(primitives, initial_z)
            result = toolkit.execute_tool(
                "vla_execute",
                {
                    "instruction": instruction if arm == "D" else stage["instruction"],
                    "max_chunks": 1,
                },
            )
            segment = result.data.get("execution_segment", result.data)
            segments.append(segment)
            for key in totals:
                totals[key] += segment.get(key, 0)
            stage_chunks += segment.get("predictions_completed", 0)
            after = _observe(primitives, initial_z)
            decision = {
                "stage_id": stage_id,
                "stage_chunks": stage_chunks,
                "before": observed,
                "after": after,
                "stop_reason": segment.get("stop_reason"),
            }
            decisions.append(decision)
            if journal:
                journal.emit("episode_execution_observation", arm=arm, **decision)
            if segment.get("tool_status") in {"rejected", "deferred"}:
                reason, diagnostic = (
                    "skill_" + segment["tool_status"],
                    segment.get("reason"),
                )
                break
            if segment.get("stop_reason") in {"technical_error", "budget", "cancelled"}:
                reason, diagnostic = segment["stop_reason"], segment.get("diagnostic")
                break
            if result.data.get("error"):
                reason, diagnostic = "technical_error", result.data["error"]
                break
            if primitives.env.terminated or primitives.env.truncated:
                break
            if segment.get("predictions_completed", 0) != 1:
                reason = "unconfirmed_execution"
                break
            condition = stage.get("completion")
            value = after.get(condition["signal"]) if condition else None
            if condition and value is None:
                edge = "on_unknown"
            elif condition and (
                value >= condition["value"]
                if condition["op"] == "ge"
                else value <= condition["value"]
            ):
                edge = "on_complete"
            elif stage_chunks >= stage["max_chunks"]:
                edge = "on_timeout" if condition else "on_complete"
            else:
                continue
            decision["edge"] = edge
            decision["subgoal_outcome"] = (
                "success" if condition and edge == "on_complete" else "unknown"
            )
            decision["advance_reason"] = (
                "public_predicate"
                if condition and edge == "on_complete"
                else "observation_unknown"
                if edge == "on_unknown"
                else "fixed_allowance"
            )
            if arm == "R":
                next_stage = stage[edge]
            elif edge == "on_unknown":
                reason = "observation_unknown"
                break
            else:
                index = stages.index(stage) + 1
                next_stage = stages[index]["id"] if index < len(stages) else "stop"
            decision["next_stage"] = next_stage
            if journal:
                journal.emit(
                    "episode_execution_transition",
                    arm=arm,
                    stage_id=stage_id,
                    edge=edge,
                    next_stage=next_stage,
                    observation=after,
                )
            if next_stage == "stop":
                reason = "frozen_stop" if arm == "R" else "program_exhausted"
                break
            stage_id, stage_chunks = next_stage, 0
    except RuntimeBudgetExceeded as exc:
        reason, diagnostic = "budget", str(exc)
    except ToolCancelled as exc:
        reason, diagnostic = "cancelled", str(exc)
    except Exception as exc:
        reason, diagnostic = "technical_error", f"{type(exc).__name__}: {exc}"
    terminated, truncated = primitives.env.terminated, primitives.env.truncated
    if reason not in {"technical_error", "cancelled"}:
        if terminated:
            reason = "native_success"
        elif truncated:
            reason = "native_truncation"
    outcome = True if terminated else False if truncated else None
    status = (
        reason.removeprefix("skill_")
        if reason in {"skill_rejected", "skill_deferred"}
        else "technical_error"
        if reason == "technical_error"
        else "cancelled"
        if reason == "cancelled"
        else "budget_exhausted"
        if reason == "budget"
        else "completed"
        if outcome is not None
        else "unknown"
    )
    record = {
        "schema": SCHEMA,
        "arm": arm,
        "spec": spec,
        "native_instruction": instruction,
        "status": status,
        "stop_reason": reason,
        "native_success": outcome,
        "terminated": terminated,
        "truncated": truncated,
        "decisions": decisions,
        "segments": segments,
        **totals,
        "diagnostic": diagnostic,
        "runtime_budget": journal.budget_snapshot() if journal else None,
        "budget_stop_ref": str(journal.budget_stop_path) if journal else None,
    }
    toolkit._episode_execution = record
    if journal:
        journal.emit("episode_execution_end", **record)
    return ToolResult(data=record)
