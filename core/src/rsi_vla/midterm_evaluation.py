# Copyright 2026 RSI-Agent-VLA contributors.

"""Independent finite-matrix evaluation for the registered midterm comparison.

The existing native evaluator supplies outcomes, validity and resource evidence.
This consumer preserves its denominator and enforces the registered old-skill
fraction when the number of initial states differs from the historical pilot.
"""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from math import ceil, comb, sqrt
from pathlib import Path
from statistics import NormalDist

from .evaluation import Gate, assignment_metrics, score
from .evaluator import evaluate_matrix

BEHAVIOR_FIELDS = (
    "control",
    "planner",
    "memory",
    "task_inputs",
    "task_input",
    "relation_routes",
    "relation_skill_starts",
    "planner_observation",
    "common_execution",
    "execution_arm",
    "execution_arm_by_task",
    "vla_segment",
    "collection",
    "execution_version",
)


def _b0_arm(binding):
    bundle, profile = binding["bundle"], binding["profile"]
    return {
        "references": {
            key: bundle[key] for key in ("phi_ref", "rho_ref", "psi_ref", "memory_ref")
        },
        "program": binding["references"]["phi_ref"],
        "relations": binding["records"]["rho_ref"],
        "strategy": binding["records"]["psi_ref"],
        "retrieval": binding["records"]["memory_ref"],
        "behavior": {
            key: deepcopy(profile[key]) for key in BEHAVIOR_FIELDS if key in profile
        },
    }


def freeze_midterm_b0(arm_bindings):
    """Freeze actual four-arm behavior, keeping theta assets and B1 separate."""
    from robots.libero.execution_arms import select_execution_arm

    if set(arm_bindings) != {"D", "S", "R", "A"}:
        raise ValueError("B0 requires all four actual execution bindings")
    for arm, binding in arm_bindings.items():
        profile = binding["profile"]
        mapping = profile.get("execution_arm_by_task")
        task_key = next(iter(mapping)) if isinstance(mapping, dict) and mapping else ""
        selected = select_execution_arm(profile, task_key)
        if selected is None or selected["arm"] != arm:
            raise ValueError("B0 execution arm identity mismatch")
    return {
        "schema": "rsi-midterm-B0-v1",
        "arms": {
            arm: deepcopy(_b0_arm(binding)) for arm, binding in arm_bindings.items()
        },
        "B1": None,
        "B1_status": "not_developed",
    }


def inspect_midterm_bindings(profile, manifest):
    """Cross-check frozen cell/B0 against actual context and native loader IDs."""
    from robots.libero.execution_arms import select_execution_arm

    from .runtime import resolve_reference

    cell = profile.get("midterm_cell")
    fields = {"cell_id", "theta_id", "theta_ref", "processor_ref", "b0_ref", "arm"}
    if (
        not isinstance(cell, dict)
        or set(cell) != fields
        or cell["theta_id"] not in {"theta0", "theta1"}
        or cell["arm"] not in {"D", "S", "R", "A"}
        or any(not isinstance(cell[key], str) or not cell[key] for key in fields)
    ):
        raise ValueError("formal evaluation requires a frozen midterm cell")
    b0 = resolve_reference(cell["b0_ref"])
    if (
        b0.get("schema") != "rsi-midterm-B0-v1"
        or b0.get("B1") is not None
        or b0.get("B1_status") != "not_developed"
    ):
        raise ValueError("formal midterm B0 must retain the undeveloped B1 boundary")
    reports = []
    for row in manifest:
        root = Path(row["result_path"]).parent
        try:
            context = json.loads((root / "runtime_context.json").read_text())
            loaded = json.loads((root / "model_load.json").read_text())
            binding = context["execution_binding"]
            checks = {
                "registered_profile": binding["profile"] == profile,
                "cell_model": all(
                    context.get(key)
                    == cell[key]
                    == binding["bundle"][key]
                    == loaded.get("identity", {}).get(key)
                    for key in ("theta_ref", "processor_ref")
                ),
                "arm": context.get("execution_arm", {}).get("arm") == cell["arm"],
                "task_program": context.get("execution_arm")
                == select_execution_arm(profile, row["task_key"]),
                "common_execution": context.get("common_execution")
                == profile.get("common_execution")
                and context.get("common_execution") is not None,
                "B0": _b0_arm(binding) == b0["arms"][cell["arm"]],
                "model_snapshot": binding["references"]["theta_ref"]
                == resolve_reference(cell["theta_ref"]),
                "native_loaded": loaded.get("loaded") is True
                and loaded.get("prediction_count", 0) > 0
                and loaded.get("last_actions_finite") is True
                and loaded.get("action_chunk") == 5,
            }
            reports.append(
                {
                    "run_id": row["run_id"],
                    "value": all(checks.values()),
                    "checks": checks,
                    "evidence_refs": [
                        str(root / "runtime_context.json"),
                        str(root / "model_load.json"),
                    ],
                }
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reports.append({"run_id": row["run_id"], "value": None, "error": str(exc)})
    values = [row["value"] for row in reports]
    return {
        "cell": deepcopy(cell),
        "value": False
        if False in values
        else True
        if values and all(v is True for v in values)
        else None,
        "runs": reports,
    }


def _policy(protocol):
    policy = protocol.get("midterm_policy")
    if protocol.get("status") != "frozen" or policy != {
        "schema": "rsi-midterm-evaluation-v1",
        "critical_old_min_fraction": 0.5,
    }:
        raise ValueError("midterm evaluation requires its frozen half-success policy")
    return policy


def critical_old_report(protocol, result, *, baseline=None, anchor=None):
    """Check each old task at ceil(n/2), retaining unknown assignments."""
    _policy(protocol)
    assignment_metrics(protocol, result)
    split = protocol["task_sets"]["release"]
    old_keys = [
        task["task_key"]
        for task in protocol["tasks"]
        if task["group"] == "old" and task["task_key"] in split["task_keys"]
    ]
    reports = {}
    for key in old_keys:
        rows = [row for row in result.assignments if row["task_key"] == key]
        successes = sum(
            row["outcome"] == "success" and row["validity"] == "valid" for row in rows
        )
        missing = sum(
            row["outcome"] == "unknown" or row["validity"] != "valid" for row in rows
        )
        minimum = ceil(len(split["seeds"]) / 2)
        comparison_missing = False
        counts = {}
        for label, comparator in (("baseline", baseline), ("anchor", anchor)):
            if comparator is None:
                continue
            assignment_metrics(protocol, comparator)
            other = [row for row in comparator.assignments if row["task_key"] == key]
            comparison_missing |= any(
                row["outcome"] == "unknown" or row["validity"] != "valid"
                for row in other
            )
            counts[label] = sum(
                row["outcome"] == "success" and row["validity"] == "valid"
                for row in other
            )
            minimum = max(minimum, counts[label])
        reports[key] = {
            "successes": successes,
            "assigned": len(rows),
            "unknown_or_invalid": missing,
            "minimum_successes": minimum,
            "comparator_successes": counts,
            "value": None if missing or comparison_missing else successes >= minimum,
        }
    values = [row["value"] for row in reports.values()]
    value = (
        False
        if False in values
        else True
        if values and all(v is True for v in values)
        else None
    )
    return {"value": value, "tasks": reports}


def evaluate_midterm_matrix(protocol, profile, manifest, **arguments):
    """Consume actual native files, then apply the frozen fractional threshold."""
    _policy(protocol)
    if arguments.get("role", "release") != "release":
        raise ValueError("formal midterm evaluator requires independent release role")
    result = evaluate_matrix(protocol, profile, manifest, **arguments)
    value = critical_old_report(protocol, result)["value"]
    bindings = inspect_midterm_bindings(profile, manifest)
    constraints = {**result.hard_constraints, "critical_old_preserved": value}
    if bindings["value"] is not True:
        constraints["compatibility"] = bindings["value"]
    return replace(
        result,
        hard_constraints=constraints,
        intervention={**result.intervention, "midterm_binding_evidence": bindings},
    )


def _wilson(successes, assigned, confidence=0.95):
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / assigned
    denominator = 1 + z * z / assigned
    center = (p + z * z / (2 * assigned)) / denominator
    radius = (
        z
        * sqrt(p * (1 - p) / assigned + z * z / (4 * assigned * assigned))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_comparison(protocol, candidate, baseline):
    """Report all assigned pairs; incomplete pairs never become a smaller test."""
    _policy(protocol)
    for result in (candidate, baseline):
        assignment_metrics(protocol, result)
        if (
            result.protocol_id != protocol["protocol_id"]
            or result.benchmark_id != protocol["benchmark_id"]
            or result.role != "release"
            or result.evidence_level != protocol["scope"]
        ):
            raise ValueError("midterm comparison result provenance mismatch")
    split = protocol["task_sets"]["release"]
    maps = [
        {(row["task_key"], row["seed"]): row for row in result.assignments}
        for result in (candidate, baseline)
    ]
    reports = {}
    for key in split["task_keys"]:
        wins = losses = ties = unknown = 0
        successes = [0, 0]
        for seed in split["seeds"]:
            pair = [rows[(key, seed)] for rows in maps]
            binary = [
                int(row["outcome"] == "success" and row["validity"] == "valid")
                for row in pair
            ]
            successes = [total + value for total, value in zip(successes, binary)]
            if any(
                row["outcome"] == "unknown" or row["validity"] != "valid"
                for row in pair
            ):
                unknown += 1
            elif binary[0] > binary[1]:
                wins += 1
            elif binary[0] < binary[1]:
                losses += 1
            else:
                ties += 1
        n = len(split["seeds"])
        discordant = wins + losses
        exact_p = (
            min(
                1.0,
                2
                * sum(comb(discordant, i) for i in range(min(wins, losses) + 1))
                / 2**discordant,
            )
            if discordant
            else 1.0
        )
        reports[key] = {
            "assigned_pairs": n,
            "candidate_successes": successes[0],
            "baseline_successes": successes[1],
            "success_difference": (successes[0] - successes[1]) / n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "unknown_pairs": unknown,
            "candidate_wilson95": _wilson(successes[0], n),
            "baseline_wilson95": _wilson(successes[1], n),
            "exact_mcnemar_two_sided_p": None if unknown else exact_p,
        }
    return {
        "candidate_result_id": candidate.result_id,
        "baseline_result_id": baseline.result_id,
        "assigned_pairs": candidate.assigned_count,
        "tasks": reports,
        "utility_difference": score(protocol, candidate) - score(protocol, baseline),
        "statistical_scope": "per-task paired initial states; no pooled weighted-J p value; missing/invalid prevent a complete-pair test",
        "equivalence_established": False,
    }


def compare_midterm_theta(protocol, theta1_result, theta0_result):
    """Require two actual bare-VLA cells before reporting a theta comparison."""
    bindings = [
        result.intervention.get("midterm_binding_evidence", {})
        for result in (theta1_result, theta0_result)
    ]
    if any(binding.get("value") is not True for binding in bindings):
        raise ValueError(
            "theta comparison is blocked without both actual cell bindings"
        )
    changed, initial = [binding["cell"] for binding in bindings]
    if (
        changed["theta_id"] != "theta1"
        or initial["theta_id"] != "theta0"
        or changed["arm"] != "D"
        or initial["arm"] != "D"
        or changed["b0_ref"] != initial["b0_ref"]
        or changed["theta_ref"] == initial["theta_ref"]
    ):
        raise ValueError(
            "theta comparison requires distinct theta0/theta1 with frozen D/B0"
        )
    return {
        **paired_comparison(protocol, theta1_result, theta0_result),
        "comparison": "bare_theta1_vs_theta0",
        "training_update_evidence": "independently required from P3",
    }


def evaluate_midterm_gate(
    protocol,
    candidate,
    *,
    candidate_result,
    baseline_result,
    anchor_result,
    current_bundle_id,
    decision_id,
):
    """Supply derived old-skill evidence to the existing three-state Gate.

    The returned candidate result is the independent evaluation record to save
    before saving its decision. This function does not activate a deployment.
    """
    report = critical_old_report(
        protocol, candidate_result, baseline=baseline_result, anchor=anchor_result
    )
    derived = replace(
        candidate_result,
        hard_constraints={
            **candidate_result.hard_constraints,
            "critical_old_preserved": report["value"],
        },
    )
    decision = Gate(protocol).evaluate(
        decision_id,
        candidate,
        current_bundle_id=current_bundle_id,
        candidate_result=derived,
        baseline_result=baseline_result,
        anchor_result=anchor_result,
    )
    return {
        "decision": asdict(decision),
        "candidate_result": asdict(derived),
        "critical_old_report": report,
        "comparison": paired_comparison(protocol, derived, baseline_result),
        "qualification_scope": "comparison old fraction is distinct from initial qualification: success_base=1 and success_old=1",
        "deployment_changed": False,
    }
