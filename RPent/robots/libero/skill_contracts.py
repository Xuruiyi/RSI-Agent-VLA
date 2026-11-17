# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Public-observation contracts consumed by the existing LIBERO skill entry."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np

SCHEMA = "libero-skill-contracts-v1"
SKILLS = {"vla_execute", "pi0_pick", "pi0_doubled"}
VERSION_FIELDS = ("theta_ref", "processor_ref", "phi_ref", "code_ref")
SIGNALS = {"eef_z", "gripper_opening", "state_valid", "main_image", "wrist_image"}


def validate_contracts(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate robot-owned conditions without loading a simulator or evaluator."""
    if (
        not isinstance(spec, dict)
        or set(spec) != {"schema", "skills"}
        or spec.get("schema") != SCHEMA
    ):
        raise ValueError("unsupported LIBERO skill contracts")
    skills = spec["skills"]
    if not isinstance(skills, dict) or not skills or not set(skills) <= SKILLS:
        raise ValueError("skill contracts require existing LIBERO motion tools")
    identities = set()
    for tool, contract in skills.items():
        expected = {
            "skill_id",
            "authorized",
            "version_signature",
            "action_prefix_steps",
            "validation",
            "require_verified",
            "preconditions",
            "completion",
            "failure_conditions",
            "requires",
        }
        if not isinstance(contract, dict) or set(contract) != expected:
            raise ValueError(f"incomplete skill contract for {tool}")
        identity = contract["skill_id"]
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity in identities
        ):
            raise ValueError("skill ids must be nonempty and unique")
        identities.add(identity)
        if (
            type(contract["authorized"]) is not bool
            or type(contract["require_verified"]) is not bool
            or type(contract["action_prefix_steps"]) is not int
            or contract["action_prefix_steps"] != 5
            or not isinstance(contract["version_signature"], dict)
            or set(contract["version_signature"]) != set(VERSION_FIELDS)
            or any(
                not isinstance(v, str) or not v
                for v in contract["version_signature"].values()
            )
        ):
            raise ValueError(
                "skill contract requires authorization and executable version dependencies"
            )
        validation = contract["validation"]
        if (
            not isinstance(validation, dict)
            or set(validation) != {"status", "evidence_refs", "task_scope"}
            or validation["status"] not in {"verified", "unverified"}
            or not isinstance(validation["evidence_refs"], list)
            or any(not isinstance(v, str) or not v for v in validation["evidence_refs"])
            or not isinstance(validation["task_scope"], list)
            or any(
                not isinstance(v, dict)
                or set(v) != {"suite", "task"}
                or not isinstance(v["suite"], str)
                or not v["suite"]
                or type(v["task"]) is not int
                or v["task"] < 0
                for v in validation["task_scope"]
            )
            or validation["status"] == "verified"
            and (not validation["evidence_refs"] or not validation["task_scope"])
        ):
            raise ValueError(
                "verified skill scope requires task identities and evidence references"
            )
        for field in ("preconditions", "completion", "failure_conditions"):
            if not isinstance(contract[field], list):
                raise ValueError("skill conditions must be lists")
            for condition in contract[field]:
                if (
                    not isinstance(condition, dict)
                    or set(condition) != {"signal", "op", "value"}
                    or condition["signal"] not in SIGNALS
                    or condition["op"] not in {"ge", "le", "eq"}
                    or type(condition["value"]) not in (int, float, bool)
                    or not isfinite(condition["value"])
                    or condition["signal"]
                    in {"state_valid", "main_image", "wrist_image"}
                    and (
                        condition["op"] != "eq" or type(condition["value"]) is not bool
                    )
                ):
                    raise ValueError(
                        "skill condition must use a finite public observation signal"
                    )
        if not isinstance(contract["requires"], list) or any(
            not isinstance(v, str) or not v for v in contract["requires"]
        ):
            raise ValueError("skill handoff requirements must name skill identities")
    if any(set(contract["requires"]) - identities for contract in skills.values()):
        raise ValueError("skill predecessor must have a registered contract")
    return deepcopy(spec)


def public_observation(primitives: Any) -> dict[str, Any]:
    """Read cached policy inputs only; never request raw object or joint state."""
    obs = getattr(primitives, "_last_obs", {})
    try:
        states = np.asarray(obs.get("states", []), dtype=float).reshape(-1)
        valid = len(states) >= 8 and bool(np.isfinite(states[:8]).all())
    except (TypeError, ValueError):
        states, valid = [], False
    result = {
        "observation_sequence": getattr(primitives, "_observation_sequence", None),
        "observation_epoch": getattr(primitives, "_observation_epoch", None),
        "state_valid": valid,
        "eef_z": float(states[2]) if valid else None,
        "gripper_opening": float(abs(states[6]) + abs(states[7])) if valid else None,
    }
    for signal, key in (("main_image", "main_images"), ("wrist_image", "wrist_images")):
        value = obs.get(key)
        result[signal] = bool(value is not None and np.size(value) > 0)
    if getattr(primitives, "_execution_feedback_unknown", False):
        result.update(
            state_valid=False,
            eef_z=None,
            gripper_opening=None,
            main_image=False,
            wrist_image=False,
        )
    return result


def condition_value(conditions: list[dict], observation: dict) -> bool | None:
    """Return a three-state conjunction; missing measurements stay unknown."""
    values = []
    for row in conditions:
        value = observation.get(row["signal"])
        values.append(
            None
            if value is None
            else value >= row["value"]
            if row["op"] == "ge"
            else value <= row["value"]
            if row["op"] == "le"
            else value == row["value"]
        )
    return False if False in values else None if None in values else True


def planner_contracts(context: dict) -> dict:
    """Expose callable conditions and scope, keeping evaluator paths out of prompts."""
    spec = context.get("skill_contracts")
    configured = validate_contracts(spec)["skills"] if spec is not None else {}
    tools = {"vla_execute"} if context.get("common_execution") is not None else SKILLS
    return {
        "schema": SCHEMA,
        "skills": {
            tool: {
                "skill_id": configured.get(tool, {}).get(
                    "skill_id", tool + "-public-v1"
                ),
                "input": {
                    "instruction_argument": "instruction"
                    if tool == "vla_execute"
                    else "prompt",
                    "max_chunks": "1..24",
                },
                "public_inputs": ["latest policy images", "finite EEF/gripper state"],
                "action_prefix_steps": 5,
                "validation": {
                    key: value
                    for key, value in configured.get(tool, {})
                    .get("validation", {"status": "unverified", "task_scope": []})
                    .items()
                    if key != "evidence_refs"
                },
                **{
                    key: deepcopy(configured[tool][key])
                    for key in (
                        "authorized",
                        "version_signature",
                        "require_verified",
                        "preconditions",
                        "completion",
                        "failure_conditions",
                        "requires",
                    )
                    if tool in configured
                },
            }
            for tool in sorted(tools)
        },
        "result_semantics": "tool completion, public subgoal predicate, and native task outcome are separate; pick heuristics do not prove holding",
    }


def admit_skill(toolkit: Any, tool: str, arguments: dict) -> tuple[dict, dict]:
    """Consume authorization, validation scope and current public preconditions."""
    journal = toolkit.runtime_journal
    context = journal.context if journal is not None else {}
    spec = context.get("skill_contracts")
    configured = validate_contracts(spec)["skills"] if spec is not None else {}
    contract = configured.get(
        tool,
        {
            "skill_id": tool + "-public-v1",
            "preconditions": [],
            "completion": [],
            "failure_conditions": [],
            "requires": [],
        },
    )
    observation = public_observation(toolkit.primitives)
    relations = refresh_relations(toolkit, observation)
    status, reason = "admitted", None
    if getattr(toolkit.primitives, "_execution_feedback_unknown", False):
        status, reason = "deferred", "action_feedback_unknown"
    elif toolkit.primitives.env.terminated or toolkit.primitives.env.truncated:
        status, reason = "rejected", "episode_already_ended"
    elif tool in configured:
        if not contract["authorized"]:
            status, reason = "rejected", "skill_not_authorized"
        elif contract["version_signature"] != {
            name: context.get(name) for name in VERSION_FIELDS
        }:
            status, reason = "deferred", "skill_version_mismatch"
        elif getattr(journal, "chunk_steps", 5) not in {None, 5}:
            status, reason = "rejected", "skill_action_contract_mismatch"
        elif (
            contract["require_verified"]
            and contract["validation"]["status"] != "verified"
        ):
            status, reason = "deferred", "skill_not_verified"
        elif (
            contract["validation"]["status"] == "verified"
            and {key: context.get("task", {}).get(key) for key in ("suite", "task")}
            not in contract["validation"]["task_scope"]
        ):
            status, reason = "deferred", "outside_verified_task_scope"
    instruction = arguments.get("instruction" if tool == "vla_execute" else "prompt")
    allowance = arguments.get(
        "max_chunks",
        24 if tool == "pi0_pick" else 20 if tool == "pi0_doubled" else None,
    )
    if status == "admitted" and (
        not isinstance(instruction, str)
        or not instruction.strip()
        or type(allowance) is not int
        or not 1 <= allowance <= 24
    ):
        status, reason = "rejected", "invalid_skill_input"
    if status == "admitted" and not (
        observation["state_valid"]
        and observation["main_image"]
        and observation["wrist_image"]
    ):
        status, reason = "deferred", "public_observation_missing"
    if status == "admitted":
        value = condition_value(contract["preconditions"], observation)
        if value is not True:
            status, reason = (
                "deferred" if value is None else "rejected",
                "precondition_unknown" if value is None else "precondition_unsatisfied",
            )
        elif any(
            relations.get(identity, {}).get("valid") is not True
            for identity in contract["requires"]
        ):
            status, reason = "deferred", "handoff_not_established"
    admission = {
        "schema": SCHEMA,
        "skill_id": contract["skill_id"],
        "tool": tool,
        "status": status,
        "reason": reason,
        "observation": observation,
        "validation_status": contract.get("validation", {}).get("status", "unverified"),
        "version_signature": contract.get("version_signature"),
        "handoff": deepcopy(relations),
        "next_action": "reconcile_execution"
        if reason == "action_feedback_unknown"
        else "inspect_public_observation_or_select_compatible_skill"
        if reason
        else None,
    }
    return contract, admission


def refresh_relations(toolkit: Any, observation: dict) -> dict:
    """Recheck current predicate relations while retaining their original source."""
    journal = toolkit.runtime_journal
    relations = getattr(toolkit, "_skill_relations", {})
    for relation in relations.values():
        current = condition_value(relation["conditions"], observation)
        failure = (
            condition_value(relation.get("failure_conditions", []), observation)
            if relation.get("failure_conditions")
            else False
        )
        if failure is True:
            current = False
        elif failure is None and current is True:
            current = None
        previous = relation["valid"]
        if previous is True:
            relation["valid"] = current
        relation["checked_observation"] = observation
        if current is not True and previous is True and journal is not None:
            journal.emit(
                "skill_relation_invalidated",
                skill_id=relation["skill_id"],
                source_sequence=relation["source_sequence"],
                observation=observation,
                valid=current,
            )
    toolkit._skill_relations = relations
    return relations


def skill_result(toolkit: Any, contract: dict, admission: dict, data: dict) -> dict:
    """Separate tool, subgoal and native outcomes and preserve handoff provenance."""
    env = toolkit.primitives.env
    reason = data.get("stop_reason")
    tool_status = (
        admission["status"]
        if admission["status"] != "admitted"
        else (
            "interrupted"
            if reason in {"cancelled", "budget"}
            else "failed"
            if data.get("error")
            else "completed"
        )
    )
    observation = public_observation(toolkit.primitives)
    refresh_relations(toolkit, observation)
    changed = (
        observation["observation_epoch"]
        != admission["observation"]["observation_epoch"]
    )
    subgoal = "unknown"
    if tool_status == "completed" and changed and not data.get("unknown_steps", 0):
        if (
            contract["failure_conditions"]
            and condition_value(contract["failure_conditions"], observation) is True
        ):
            subgoal = "failure"
        elif (
            contract["completion"]
            and condition_value(contract["completion"], observation) is True
        ):
            subgoal = "success"
    result = {
        "skill_id": contract["skill_id"],
        "tool_status": tool_status,
        "stop_reason": reason,
        "subgoal_outcome": subgoal,
        "native_outcome": "success"
        if env.terminated
        else "failure"
        if env.truncated
        else "unknown",
        "action_feedback_unknown": bool(data.get("unknown_steps", 0)),
        "observation": observation,
        "completion_meaning": "registered public predicate only; no independent object/holding truth",
        "admission": admission,
    }
    if result["action_feedback_unknown"]:
        toolkit.primitives._execution_feedback_unknown = True
        result["next_action"] = "reconcile_execution"
    journal = toolkit.runtime_journal
    sequence = (
        journal.emit("skill_execution_result", **result)
        if journal is not None
        else None
    )
    if subgoal == "success":
        relations = toolkit._skill_relations
        relation = {
            "skill_id": contract["skill_id"],
            "valid": True,
            "conditions": deepcopy(contract["completion"]),
            "failure_conditions": deepcopy(contract["failure_conditions"]),
            "source_sequence": sequence,
            "established_observation": observation,
            "checked_observation": observation,
        }
        # A repeated result for the same observation cannot establish another completion.
        previous = relations.get(contract["skill_id"])
        if previous is None or previous["established_observation"] != observation:
            relations[contract["skill_id"]] = relation
    elif subgoal == "failure" and contract["skill_id"] in toolkit._skill_relations:
        toolkit._skill_relations[contract["skill_id"]]["valid"] = False
        toolkit._skill_relations[contract["skill_id"]]["failure_source_sequence"] = (
            sequence
        )
    result["handoff"] = deepcopy(getattr(toolkit, "_skill_relations", {}))
    if journal is not None:
        journal.emit(
            "skill_handoff",
            skill_id=contract["skill_id"],
            source_sequence=sequence,
            observation=observation,
            relations=result["handoff"],
        )
    return result
