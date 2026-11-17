"""Independent scoring of preregistered research-policy examinations."""

import json
from copy import deepcopy
from math import isfinite
from pathlib import Path

from .contracts import Candidate, ExperimentResult
from .evaluation import EvaluationStore, assignment_metrics, score
from .objective_evaluation import exact_paired


def _read(reference):
    return json.loads(Path(reference).read_text(encoding="utf-8"))


def _number(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("meta examination requires finite nonnegative costs/budgets")
    return value


def _key(row):
    return (row["problem_id"], row["repeat"], row["arm"], row["budget_point"])


def _validate_registration(registration):
    if registration.get("schema") != "rsi-meta-exam-v1":
        raise ValueError("unsupported meta examination registration")
    arms = registration["arms"]
    if set(arms) != {"psi0", "psi1", "psi0_late"}:
        raise ValueError(
            "examination requires baseline, candidate and late-fact control"
        )
    points = registration["budget_points"]
    if (
        not points
        or any(_number(point) == 0 for point in points)
        or points != sorted(set(points))
    ):
        raise ValueError("meta budget points must be strictly increasing and positive")
    if (
        registration["mode"] not in {"endpoint", "milestones"}
        or (registration["mode"] == "endpoint" and len(points) != 1)
        or (registration["mode"] == "milestones" and len(points) < 2)
    ):
        raise ValueError("endpoint/curve registration does not match budget points")
    if registration.get("missing_score") != 0:
        raise ValueError("supported missing/technical-failure policy is fixed zero")
    repetitions = registration["repetitions"]
    problems = registration["problems"]
    if (
        not repetitions
        or len(set(repetitions)) != len(repetitions)
        or not problems
        or len({p["problem_id"] for p in problems}) != len(problems)
    ):
        raise ValueError("duplicate/empty registered problem or process repetition")
    if (
        arms["psi0"]["facts_ref"] != registration["shared_facts_ref"]
        or arms["psi1"]["facts_ref"] != registration["shared_facts_ref"]
    ):
        raise ValueError("core policy comparison must start from the same legal facts")
    snapshots = {}
    for name, arm in arms.items():
        snapshot = _read(arm["policy_snapshot_ref"])
        facts = _read(arm["facts_ref"])
        if (
            snapshot["snapshot_id"] != arm["policy_snapshot_id"]
            or facts.get("legal_scope") != "development_only"
            or facts.get("role") != "facts_not_policy"
        ):
            raise ValueError("policy identity or legal fact scope mismatch")
        if facts.get("information_scope") in {"sealed", "meta_exam"}:
            raise ValueError("sealed facts contaminate the examination start")
        partition = snapshot["problem_partition"]
        for problem in problems:
            if (
                problem["family"] in partition["development_families"]
                or problem["mechanism"] in partition["development_mechanisms"]
                or problem["family"] not in partition["reserved_families"]
                or problem["mechanism"] not in partition["reserved_mechanisms"]
            ):
                raise ValueError(
                    "examination family/mechanism was not held out from policy development"
                )
        snapshots[name] = snapshot
    if snapshots["psi0"]["components"] == snapshots["psi1"]["components"]:
        raise ValueError(
            "no policy update; additional facts cannot create a policy candidate"
        )
    if snapshots["psi0"]["components"] != snapshots["psi0_late"]["components"]:
        raise ValueError("late-fact control must retain fixed baseline policy")
    if not all(
        registration["common"].get(key)
        for key in ("theta_ref", "phi_ref", "non_test_policy_ref")
    ):
        raise ValueError("common execution and non-tested policy must be frozen")
    generation = registration["policy_generation"]
    if generation.get("independent_generations") != 1:
        raise ValueError(
            "one frozen policy examination supports a snapshot-level conclusion"
        )
    for value in generation["costs"].values():
        _number(value)
    for key in ("minimum_gain", "retention_max_drop"):
        _number(registration[key])
    return snapshots


def score_exam(
    registration: dict,
    submissions: list[dict],
    evaluation_store: EvaluationStore | str | Path,
) -> dict:
    """Score one frozen policy over fixed problems, repetitions and budget points.

    Missing and technical submissions receive their registered zero. Results are
    read from the independent evaluation store; caller-provided scores are unused.
    Every point consumes its submitted candidate, never an exam-selected best.
    """
    snapshots = _validate_registration(registration)
    store = (
        evaluation_store
        if isinstance(evaluation_store, EvaluationStore)
        else EvaluationStore(evaluation_store)
    )
    arms = registration["arms"]
    points = registration["budget_points"]
    problems = {row["problem_id"]: row for row in registration["problems"]}
    expected = {
        (problem, repeat, arm, point)
        for problem in problems
        for repeat in registration["repetitions"]
        for arm in arms
        for point in points
    }
    submitted = {}
    for row in submissions:
        pair = _key(row)
        if pair not in expected or pair in submitted:
            raise ValueError("duplicate or unexpected meta examination submission")
        submitted[pair] = row
    values, retention, expenses = {}, {}, {}
    result_snapshots, sources, cost_snapshots, submission_snapshots = {}, [], {}, {}
    notes, known_violation, costs_complete = [], False, True
    for pair in sorted(expected):
        problem, repeat, arm, point = pair
        row = submitted.get(pair)
        values[pair], retention[pair], expenses[pair] = 0.0, None, {}
        if row is None:
            notes.append("missing submission penalized zero: " + str(pair))
            costs_complete = False
            continue
        identity = {
            "exam_id": registration["exam_id"],
            "problem_id": problem,
            "repeat": repeat,
            "arm": arm,
            "budget_point": point,
        }
        receipt = _read(row["submission_receipt_ref"])
        expected_receipt = {
            **identity,
            "schema": "rsi-meta-submission-v1",
            "candidate_ref": row.get("candidate_ref"),
            "result_id": row.get("result_id"),
            "status": row["status"],
            "common": registration["common"],
            "policy_snapshot_id": arms[arm]["policy_snapshot_id"],
            "facts_ref": arms[arm]["facts_ref"],
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ValueError(
                "meta submission changed candidate, policy, facts or common start"
            )
        costs = _read(row["cost_receipt_ref"])
        if (
            costs.get("schema") != "rsi-meta-attempt-cost-v1"
            or any(costs.get(key) != value for key, value in identity.items())
            or costs.get("attempt_costs") != row["attempt_costs"]
        ):
            raise ValueError("meta all-attempt cost receipt identity mismatch")
        attempts = row["attempt_costs"]
        if not isinstance(attempts, list) or not attempts:
            costs_complete = False
        for attempt in attempts:
            for dimension, amount in attempt.items():
                expenses[pair][dimension] = expenses[pair].get(dimension, 0) + _number(
                    amount
                )
        if registration["budget_dimension"] not in expenses[pair]:
            costs_complete = False
        source_key = "|".join(map(str, pair))
        cost_snapshots[source_key] = costs
        submission_snapshots[source_key] = receipt
        sources.extend([row["cost_receipt_ref"], row["submission_receipt_ref"]])
        if row["status"] in {"missing", "technical_failure"}:
            notes.append(row["status"] + " penalized zero: " + str(pair))
            continue
        if row["status"] != "submitted":
            raise ValueError("unsupported meta submission status")
        protocol = _read(problems[problem]["protocol_ref"])
        if registration.get("protocol_snapshots", {}).get(problem) != protocol:
            raise ValueError("meta problem protocol differs from its frozen scoring snapshot")
        candidate = Candidate(**_read(row["candidate_ref"]))
        result = ExperimentResult(**store.get("results", row["result_id"]))
        if (
            result.role != "release"
            or result.bundle_id != candidate.bundle_id
            or result.candidate_id != candidate.candidate_id
            or result.protocol_id != protocol["protocol_id"]
            or result.benchmark_id != protocol["benchmark_id"]
            or result.criteria_ref != protocol["protocol_id"]
            or result.evidence_level != registration["scope"]
        ):
            raise ValueError(
                "meta result is not independent evaluation for its submitted candidate"
            )
        assignment_metrics(protocol, result)
        sources.extend(
            [
                str(store.path("results", result.result_id)),
                row["candidate_ref"],
                problems[problem]["protocol_ref"],
            ]
        )
        result_snapshots[source_key] = deepcopy(store.get("results", result.result_id))
        checks = [
            result.hard_constraints.get(key) for key in protocol["hard_constraints"]
        ]
        if False in checks:
            known_violation = True
        if (
            result.unknown_count
            or result.invalid_count
            or any(v is not True for v in checks)
            or not result.data_refs
        ):
            notes.append(
                "unknown/invalid independent evaluation penalized zero: " + str(pair)
            )
            continue
        measured = score(protocol, result)
        if measured is None:
            notes.append("unknown independent score penalized zero: " + str(pair))
            continue
        values[pair] = measured
        retention[pair] = result.metrics.get("success_old")
    budget_violations = []
    cumulative = {}
    for problem in problems:
        for repeat in registration["repetitions"]:
            for arm in arms:
                running = {}
                for point in points:
                    pair = (problem, repeat, arm, point)
                    for dimension, amount in expenses[pair].items():
                        running[dimension] = running.get(dimension, 0) + amount
                    cumulative[pair] = dict(running)
                    if running.get(registration["budget_dimension"], 0) > point:
                        budget_violations.append(list(pair))
    if budget_violations:
        known_violation = True
        notes.append("registered cumulative research budget exceeded")
    curves = {}
    for arm in arms:
        curves[arm] = [
            sum(
                values[(problem, repeat, arm, point)]
                for problem in problems
                for repeat in registration["repetitions"]
            )
            / (len(problems) * len(registration["repetitions"]))
            for point in points
        ]
    endpoint = {arm: curve[-1] for arm, curve in curves.items()}
    gain = endpoint["psi1"] - endpoint["psi0"]
    old_known = True
    old_preserved = True
    for problem in problems:
        for repeat in registration["repetitions"]:
            for point in points:
                current = retention[(problem, repeat, "psi1", point)]
                baseline = retention[(problem, repeat, "psi0", point)]
                if current is None or baseline is None:
                    old_known = False
                elif current < baseline - registration["retention_max_drop"]:
                    old_preserved = False
    verdict = "reject" if known_violation or not old_preserved else "inconclusive"
    if not known_violation and old_preserved and old_known and costs_complete:
        verdict = "accept" if gain >= registration["minimum_gain"] else "reject"
    test = exact_paired(
        [
            values[(problem, repeat, "psi1", points[-1])]
            - values[(problem, repeat, "psi0", points[-1])]
            for problem in problems
            for repeat in registration["repetitions"]
        ]
    )
    statistics = registration.get("statistics")
    if statistics:
        if (
            statistics.get("method") != "exact_paired"
            or not 0 < statistics["alpha"] <= 1
        ):
            raise ValueError("unsupported meta paired statistics")
        if verdict == "accept" and test["p_value"] > statistics["alpha"]:
            verdict = "inconclusive"
    metrics = {
        "endpoint": endpoint,
        "endpoint_gain": gain,
        "old_preserved": old_preserved if old_known else None,
        "paired_endpoint": test,
        "auc": None,
        "cost_to_target": None,
    }
    if registration["mode"] == "milestones":
        metrics["curves"] = {
            arm: [
                {"budget": point, "score": value} for point, value in zip(points, curve)
            ]
            for arm, curve in curves.items()
        }
        metrics["auc"] = {
            arm: sum(
                (curve[i] + curve[i - 1]) * (points[i] - points[i - 1]) / 2
                for i in range(1, len(points))
            )
            / (points[-1] - points[0])
            for arm, curve in curves.items()
        }
        if registration.get("target_score") is not None:
            metrics["cost_to_target"] = {
                arm: next(
                    (
                        point
                        for point, value in zip(points, curve)
                        if value >= registration["target_score"]
                    ),
                    None,
                )
                for arm, curve in curves.items()
            }
    totals = {}
    for cost in expenses.values():
        for dimension, amount in cost.items():
            totals[dimension] = totals.get(dimension, 0) + amount
    return {
        "schema": "rsi-meta-exam-score-v1",
        "exam_id": registration["exam_id"],
        "verdict": verdict,
        "release_eligible": verdict == "accept",
        "policy_snapshot_id": arms["psi1"]["policy_snapshot_id"],
        "policy_snapshot": snapshots["psi1"],
        "base_policy_snapshot_id": arms["psi0"]["policy_snapshot_id"],
        "protocol_snapshot": deepcopy(registration),
        "registration_snapshot": deepcopy(registration),
        "submissions_snapshot": deepcopy(submissions),
        "result_snapshots": result_snapshots,
        "submission_snapshots": submission_snapshots,
        "cost_snapshots": cost_snapshots,
        "scope": registration["scope"],
        "assignment_refs": sorted(set(sources)),
        "metrics": metrics,
        "costs": {
            "examination_all_attempts": totals,
            "policy_generation": deepcopy(registration["policy_generation"]["costs"]),
            "complete": costs_complete,
        },
        "budget_violations": budget_violations,
        "feedback": {"information_scope": "meta_exam", "status": "withheld"},
        "limitations": [
            "single_policy_snapshot",
            "robot_episodes_are_not_policy_generations",
        ],
        "notes": notes,
    }
