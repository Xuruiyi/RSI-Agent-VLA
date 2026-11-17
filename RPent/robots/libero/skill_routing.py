# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Finite same-goal skill routing and environment-owned handoff observations."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

PICK_WORKERS = {"pi0_pick", "vla_execute"}
DRAWER_WORKERS = {"vla_execute", "pi0_doubled"}
DRAWER_CRITERION = "rsi-skill-drawer-open-criterion-v1"
VERSION_FIELDS = ("theta_ref", "processor_ref", "phi_ref", "code_ref")


def _route_workers(criterion: dict) -> set[str]:
    if criterion.get("schema") == "rsi-skill-lift-criterion-v1":
        return PICK_WORKERS
    if criterion.get("schema") == DRAWER_CRITERION:
        expected = {
            "region": "wooden_cabinet_1_middle_region",
            "fixture": "wooden_cabinet_1",
            "joint": "wooden_cabinet_1_middle_level",
            "joint_type": "slide",
            "joint_range": [-0.16, 0.01],
            "open_threshold": -0.14,
        }
        if any(criterion.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "drawer criterion differs from the registered middle-joint Open condition"
            )
        return DRAWER_WORKERS
    raise ValueError("unsupported registered skill criterion")


def load_control(journal: Any, tool_names: set[str]) -> dict | None:
    """Validate an optional launcher projection against the current executable."""
    control = journal.context.get("relation_control") if journal is not None else None
    if control is None:
        return None
    if (
        control.get("schema") != "rsi-skill-routing-v1"
        or not control.get("rho_ref")
        or control.get("version_signature")
        != {name: journal.context.get(name) for name in VERSION_FIELDS}
        or any(not journal.context.get(name) for name in VERSION_FIELDS)
    ):
        raise ValueError("skill routing projection has an incompatible version binding")
    seen = set()
    for route in control["routes"]:
        workers = _route_workers(route["criterion"])
        actions = workers | {"handoff", "defer"}
        if route["criterion"]["schema"] == DRAWER_CRITERION and (
            route.get("task_key") != "old_drawer"
            or route["state"] != "old_drawer"
            or type(route.get("max_chunks")) is not int
            or route["max_chunks"] != 24
        ):
            raise ValueError(
                "drawer routing must keep its registered old_drawer goal and 24 chunks"
            )
        identities = {
            (worker, route["instruction"]) for worker in route["trigger_workers"]
        }
        if (
            seen.intersection(identities)
            or not route["trigger_workers"]
            or not set(route["trigger_workers"]) <= workers.intersection(tool_names)
            or route["requested_worker"] not in workers
            or route["selected_worker"] not in actions
            or route["selected_worker"] not in route["allowed_workers"]
            and route["selected_worker"] != "defer"
            or not set(route["allowed_workers"]) <= actions
            or not set(route["allowed_workers"]).intersection(workers) <= tool_names
            or not isinstance(route["instruction"], str)
            or not route["instruction"].strip()
        ):
            raise ValueError(
                "skill routing projection exceeds compatible registered tools"
            )
        seen.update(identities)
    return control


def route_for_call(control: dict | None, name: str, arguments: dict) -> dict | None:
    """Match only a registered tool and exact allowed goal, using no private state."""
    if control is None or name not in PICK_WORKERS | DRAWER_WORKERS:
        return None
    instruction = arguments.get(
        "prompt" if name in {"pi0_pick", "pi0_doubled"} else "instruction"
    )
    return next(
        (
            route
            for route in control["routes"]
            if name in route["trigger_workers"] and route["instruction"] == instruction
        ),
        None,
    )


def observation_route(journal: Any, evidence: dict) -> dict:
    """Admit one private observation before the environment reads native state."""
    if any(evidence.get(key) != value for key, value in journal.identity().items()):
        raise ValueError("skill observation identity mismatch")
    control = journal.context.get("relation_control", {})
    route = next(
        (
            row
            for row in control.get("routes", [])
            if row["route_id"] == evidence.get("route_id")
        ),
        None,
    )
    if (
        route is None
        or evidence.get("worker") != route["selected_worker"]
        or evidence.get("phase") not in {"start", "handoff"}
        or type(evidence.get("client_route_sequence")) is not int
        or evidence["client_route_sequence"] < 0
    ):
        raise ValueError("skill observation is outside the registered route")
    if (
        control.get("schema") != "rsi-skill-routing-v1"
        or not control.get("rho_ref")
        or control.get("version_signature")
        != {name: journal.context.get(name) for name in VERSION_FIELDS}
    ):
        raise ValueError("private skill observation executable version mismatch")
    _route_workers(route["criterion"])
    return route


def record_skill_observation(
    journal: Any, raw: dict, evidence: dict, *, joint_observation: dict | None = None
) -> dict:
    """Persist private ground truth in the environment stream; return no geometry."""
    route = observation_route(journal, evidence)
    control = journal.context["relation_control"]
    criterion = route["criterion"]
    missing = []
    if criterion.get("schema") == DRAWER_CRITERION:
        from copy import deepcopy

        joint = (
            deepcopy(joint_observation) if isinstance(joint_observation, dict) else {}
        )
        for field in ("region", "fixture", "joint", "joint_type", "joint_range"):
            if joint.get(field) != criterion[field]:
                missing.append(field)
        if (
            joint.get("schema") != "rsi-private-drawer-joint-observation-v1"
            or joint.get("status") != "recorded"
        ):
            missing.append("drawer_joint_observation")
        qpos = joint.get("qpos")
        if type(qpos) not in (float, int) or not isfinite(qpos):
            missing.append("qpos")
        missing.extend(str(value) for value in joint.get("missing", []))
        sequence = journal.emit(
            "skill_route_observation",
            route_id=route["route_id"],
            state=route["state"],
            worker=evidence["worker"],
            phase=evidence["phase"],
            client_route_sequence=evidence["client_route_sequence"],
            criterion_ref=criterion["criterion_ref"],
            version_signature=control["version_signature"],
            measurements={"joint": joint},
            missing=missing,
            source="environment_private_joint_observation",
        )
        return {
            "source_sequence": sequence,
            "status": "unknown" if missing else "recorded",
        }

    def xyz(key: str) -> list[float] | None:
        value = raw.get(key)
        try:
            result = [float(v) for v in value]
        except (TypeError, ValueError):
            result = []
        if len(result) != 3 or not all(isfinite(v) for v in result):
            missing.append(key)
            return None
        return result

    gripper = raw.get("robot0_gripper_qpos")
    try:
        gap = sum(abs(float(v)) for v in gripper)
        if len(gripper) != 2 or not isfinite(gap):
            gap = None
    except (TypeError, ValueError):
        gap = None
    if gap is None:
        missing.append("robot0_gripper_qpos")
    measurements = {
        "object_pos": xyz(criterion["object_position_key"]),
        "eef_pos": xyz("robot0_eef_pos"),
        "gripper_opening": gap,
    }
    sequence = journal.emit(
        "skill_route_observation",
        route_id=route["route_id"],
        state=route["state"],
        worker=evidence["worker"],
        phase=evidence["phase"],
        client_route_sequence=evidence["client_route_sequence"],
        criterion_ref=criterion["criterion_ref"],
        object_position_key=criterion["object_position_key"],
        version_signature=control["version_signature"],
        measurements=measurements,
        missing=missing,
        source="environment_raw_observation",
    )
    return {"source_sequence": sequence, "status": "unknown" if missing else "recorded"}


def private_evidence_access(journal: Any, name: str, arguments: dict) -> bool:
    """Keep registered evaluation-only geometry outside Planner file tools."""
    if name not in {"read_text_file", "write_text_file", "list_dir"}:
        return False
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return False
    from rpent.utils.config import get_repo_root

    resolved = (get_repo_root() / path).resolve()
    private = (Path(journal.context["output_dir"]) / "environment_service").resolve()
    return resolved == private or resolved.is_relative_to(private)
