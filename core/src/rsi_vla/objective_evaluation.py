"""Independent consumers for registered finite task objectives and paired tests."""

from collections import Counter
from fractions import Fraction
from math import isfinite

from .contracts import Candidate, ExperimentResult, GateDecision


def exact_paired(differences: list[float]) -> dict:
    """Compute a one-sided paired sign-randomization test, retaining zero pairs."""
    values = [Fraction(str(value)) for value in differences]
    if not values:
        raise ValueError("paired test requires assignments")
    distribution = Counter({Fraction(0): 1})
    for value in values:
        updated = Counter()
        for total, count in distribution.items():
            updated[total + abs(value)] += count
            updated[total - abs(value)] += count
        distribution = updated
    statistic = sum(values)
    numerator = sum(
        count for total, count in distribution.items() if total >= statistic
    )
    denominator = 2 ** len(values)
    return {
        "method": "exact_paired",
        "pairs": len(values),
        "mean": float(statistic / len(values)),
        "p_numerator": numerator,
        "p_denominator": denominator,
        "p_value": numerator / denominator,
    }


def objective_policy(protocol: dict) -> dict:
    """Validate the evaluator-owned comparison registration."""
    policy = protocol["objective_comparison"]
    if policy.get("schema") != "rsi-objective-comparison-v1" or policy.get(
        "objective"
    ) not in {"success_gain", "efficiency", "robustness", "new_capability"}:
        raise ValueError("unsupported objective comparison")
    keys = policy.get("task_keys")
    split = protocol["task_sets"]["release"]
    if (
        not isinstance(keys, list)
        or not keys
        or len(set(keys)) != len(keys)
        or not set(keys) <= set(split["task_keys"])
    ):
        raise ValueError("objective tasks must belong to the full registered matrix")
    for key in ("minimum_gain", "anchor_max_drop"):
        value = policy[key]
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("objective threshold must be finite and nonnegative")
    if policy["objective"] == "robustness":
        tasks = {row["task_key"]: row for row in protocol["tasks"]}
        if any(not tasks[key].get("variation_condition") for key in keys):
            raise ValueError("robustness requires registered variation conditions")
    if policy["objective"] == "new_capability":
        tasks = {row["task_key"]: row for row in protocol["tasks"]}
        if any(
            not tasks[key].get("materialization_ref")
            or tasks[key].get("native_success") != "bddl_goal"
            for key in keys
        ):
            raise ValueError(
                "new capability requires materialized tasks and independent native criterion"
            )
    if policy["objective"] == "efficiency" and not policy.get("cost_dimension"):
        raise ValueError("efficiency requires an all-attempt cost dimension")
    statistics = policy.get("statistics")
    if statistics is not None and (
        statistics.get("method") != "exact_paired"
        or type(statistics.get("alpha")) not in (int, float)
        or not 0 < statistics["alpha"] <= 1
    ):
        raise ValueError("invalid paired statistics registration")
    return policy


def _costs(protocol, result, dimension):
    expected = {
        (key, seed)
        for key in protocol["task_sets"]["release"]["task_keys"]
        for seed in protocol["task_sets"]["release"]["seeds"]
    }
    rows = result.intervention.get("attempt_costs_by_slot", [])
    costs = {}
    for row in rows:
        pair = (row["task_key"], row["seed"])
        attempts = row["attempts"]
        if pair in costs or pair not in expected or not attempts:
            raise ValueError("all-attempt cost assignment missing/duplicate/unexpected")
        values = [attempt.get(dimension) for attempt in attempts]
        if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in values):
            raise ValueError("all-attempt cost unknown")
        costs[pair] = sum(values)
    if set(costs) != expected or sum(costs.values()) != result.costs.get(dimension):
        raise ValueError("all-attempt cost denominator or total mismatch")
    return costs


def evaluate_objective(
    protocol: dict,
    decision_id: str,
    candidate: Candidate,
    *,
    current_bundle_id: str,
    candidate_result: ExperimentResult | None = None,
    baseline_result: ExperimentResult | None = None,
    anchor_result: ExperimentResult | None = None,
) -> GateDecision:
    """Return the same three-state Gate contract used by the release consumer."""
    from .evaluation import (
        assignment_metrics,
        critical_old_evidence_matches,
        critical_preserved,
    )

    policy = objective_policy(protocol)
    expected = [
        ("candidate", candidate_result, candidate.bundle_id),
        ("baseline", baseline_result, current_bundle_id),
        ("anchor", anchor_result, protocol["anchor_bundle_id"]),
    ]
    reasons, scores = [], {}
    refs = [r.result_id for _, r, _ in expected if r is not None]
    verdict = "inconclusive"
    identity = (
        candidate.protocol_id == protocol["protocol_id"]
        and candidate.benchmark_id == protocol["benchmark_id"]
    )
    if (
        not identity
        or candidate.base_bundle_id != current_bundle_id
        or candidate.approval_scope != protocol["scope"]
    ):
        reasons.append("candidate registration, scope or baseline mismatch")
    if (
        protocol["scope"] != "offline"
        and protocol.get("dynamic_authorized") is not True
    ):
        reasons.append("dynamic authorization missing")
    candidate_valid = False
    for label, result, bundle in expected:
        valid = (
            result is not None
            and result.bundle_id == bundle
            and result.protocol_id == protocol["protocol_id"]
            and result.benchmark_id == protocol["benchmark_id"]
            and result.evidence_level == protocol["scope"]
            and result.role == "release"
            and result.criteria_ref == protocol["protocol_id"]
        )
        if label == "candidate":
            valid = valid and result.candidate_id == candidate.candidate_id
            candidate_valid = valid and identity
        elif valid and result.candidate_id is not None:
            valid = False
        if not valid:
            reasons.append(label + " independent result identity mismatch/missing")
            continue
        try:
            assignment_metrics(protocol, result)
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append(label + ": " + str(exc))
        if result.unknown_count or result.invalid_count or not result.assigned_count:
            reasons.append(label + " assignments unknown/invalid")
        if not result.data_refs:
            reasons.append(label + " independent sources missing")
        if any(
            not critical_old_evidence_matches(protocol, result)
            if label != "candidate" and key == "critical_old_preserved"
            else result.hard_constraints.get(key) is not True
            for key in protocol["hard_constraints"]
        ):
            reasons.append(label + " hard constraint evidence missing/violated")
    if candidate_valid and any(
        candidate_result.hard_constraints.get(k) is False
        for k in protocol["hard_constraints"]
    ):
        verdict = "reject"
    test = None
    if not reasons:
        preserved = critical_preserved(
            protocol, candidate_result, baseline_result, anchor_result
        )
        if preserved is not True:
            reasons.append("critical old capability lost or unknown")
            verdict = "reject" if preserved is False else "inconclusive"
        else:
            pairs = [
                (key, seed)
                for key in policy["task_keys"]
                for seed in protocol["task_sets"]["release"]["seeds"]
            ]
            maps = [
                {
                    (r["task_key"], r["seed"]): int(r["outcome"] == "success")
                    for r in result.assignments
                }
                for _, result, _ in expected
            ]
            if policy["objective"] == "efficiency":
                if any(
                    maps[0][pair] < max(maps[1][pair], maps[2][pair])
                    for pair in maps[0]
                ):
                    verdict = "reject"
                    reasons.append("efficiency task outcome regression")
                try:
                    values = [
                        _costs(protocol, result, policy["cost_dimension"])
                        for _, result, _ in expected
                    ]
                    values = [
                        {pair: -value for pair, value in costs.items()}
                        for costs in values
                    ]
                except (ValueError, KeyError, TypeError) as exc:
                    reasons.append(str(exc))
                    values = None
            else:
                values = maps
            if values is not None:
                scores = {
                    label: sum(values[i][pair] for pair in pairs) / len(pairs)
                    for i, (label, _, _) in enumerate(expected)
                }
                test = exact_paired(
                    [values[0][pair] - values[1][pair] for pair in pairs]
                )
                if not reasons:
                    if (
                        scores["candidate"] - scores["baseline"]
                        < policy["minimum_gain"]
                        or scores["candidate"]
                        < scores["anchor"] - policy["anchor_max_drop"]
                    ):
                        verdict = "reject"
                        reasons.append("registered engineering threshold failed")
                    elif (
                        policy.get("statistics")
                        and test["p_value"] > policy["statistics"]["alpha"]
                    ):
                        reasons.append("paired statistical precision insufficient")
                    else:
                        verdict = "accept"
                        reasons.append(
                            "registered objective, old capability and hard constraints passed"
                        )
    if test is not None:
        scores["paired_p_value"] = test["p_value"]
        scores["paired_count"] = test["pairs"]
    eligible = verdict == "accept" and protocol.get("comparison_only") is not True
    return GateDecision(
        decision_id=decision_id,
        candidate_id=candidate.candidate_id,
        base_bundle_id=candidate.base_bundle_id,
        bundle_id=candidate.bundle_id,
        protocol_id=protocol["protocol_id"],
        benchmark_id=protocol["benchmark_id"],
        verdict=verdict,
        reasons=reasons,
        release_eligible=eligible,
        scope=protocol["scope"],
        scores=scores,
        result_refs=refs,
    )


def feedback_projection(result: ExperimentResult, *, information_scope: str) -> dict:
    """Expose fixed public feedback; sealed scores remain in the evaluation domain."""
    if information_scope not in {"development", "release", "sealed", "meta_exam"}:
        raise ValueError("unknown feedback scope")
    if information_scope in {"sealed", "meta_exam"}:
        return {
            "schema": "rsi-evaluation-feedback-v1",
            "information_scope": information_scope,
            "status": "withheld",
        }
    return {
        "schema": "rsi-evaluation-feedback-v1",
        "information_scope": information_scope,
        "result_id": result.result_id,
        "metrics": dict(result.metrics),
        "unknown_count": result.unknown_count,
        "invalid_count": result.invalid_count,
    }
