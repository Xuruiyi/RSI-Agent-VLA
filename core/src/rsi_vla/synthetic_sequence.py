"""Deterministic offline tasks for real rule-artifact engineering checks."""

import json
import time
from pathlib import Path

from .contracts import ExperimentResult
from .rule_updates import read_phi_chunk


def evaluate_sequence_matrix(protocol, bundle, *, result_id, candidate_id, problem_ref):
    """Execute every registered sequence with the submitted actual chunk rule."""
    if (
        protocol.get("scope") != "offline"
        or protocol.get("synthetic_engineering") is not True
    ):
        raise PermissionError(
            "synthetic sequence evidence requires an explicit offline protocol"
        )
    started = time.monotonic()
    fixture = json.loads(Path(problem_ref).read_text())
    if fixture.get("schema") != "rsi-synthetic-sequence-problem-v1":
        raise ValueError("unsupported synthetic sequence task")
    tasks = fixture["tasks"]
    expected = {
        (key, seed)
        for key in protocol["task_sets"]["release"]["task_keys"]
        for seed in protocol["task_sets"]["release"]["seeds"]
    }
    if (
        len(tasks) != len(expected)
        or {(row["task_key"], row["seed"]) for row in tasks} != expected
    ):
        raise ValueError("synthetic task matrix does not match the frozen denominator")
    rules = read_phi_chunk(bundle["phi_ref"])
    assignments = []
    for task in tasks:
        if len(task["sequence"]) < rules["chunk_steps"]:
            raise ValueError("synthetic sequence cannot execute candidate chunk")
        observed = task["sequence"][: rules["chunk_steps"]][-1]
        assignments.append(
            {
                "task_key": task["task_key"],
                "seed": task["seed"],
                "outcome": "success"
                if observed == task["expected_observation"]
                else "failure",
                "validity": "valid",
                "evidence_ref": str(problem_ref),
                "outcome_source": "synthetic_task",
            }
        )
    groups = {row["task_key"]: row["group"] for row in protocol["tasks"]}
    metrics = {}
    for group, key in (
        ("base", "success_base"),
        ("controlled", "success_controlled"),
        ("old", "success_old"),
    ):
        rows = [row for row in assignments if groups[row["task_key"]] == group]
        if rows:
            metrics[key] = sum(row["outcome"] == "success" for row in rows) / len(rows)
    constraints = dict.fromkeys(protocol["hard_constraints"], True)
    if set(constraints) - {"safety", "compatibility"}:
        raise ValueError(
            "synthetic sequence consumer supports only no-device safety and parsed rule compatibility"
        )
    return ExperimentResult(
        result_id,
        bundle["bundle_id"],
        protocol["protocol_id"],
        protocol["benchmark_id"],
        "release",
        "offline",
        {
            "information_scope": "meta_exam",
            "synthetic_engineering": True,
            "actual_rules": rules,
        },
        [str(problem_ref), bundle["phi_ref"]],
        protocol["protocol_id"],
        metrics,
        constraints,
        {
            "wall_seconds": time.monotonic() - started,
            "model_requests": 0,
            "gpu_seconds": 0,
        },
        candidate_id=candidate_id,
        assigned_count=len(assignments),
        assignments=assignments,
        outcome="success"
        if all(row["outcome"] == "success" for row in assignments)
        else "failure",
    )
