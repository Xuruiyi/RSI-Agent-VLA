"""Fixed protocol consumer and deterministic three-way Gate.

Formal results are supplied by a separate trusted evaluator, never promoted
from development results. Filesystem/process isolation is the deployment ACL.
"""
import json
from math import isfinite
from pathlib import Path
from .contracts import GateDecision
from .storage import Records


def load_protocol(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"protocol_id", "status", "scope", "benchmark_id", "anchor_bundle_id",
                "metrics", "hard_constraints", "min_gain", "anchor_max_drop"}
    if not required <= value.keys():
        raise ValueError("protocol missing fields: " + ", ".join(sorted(required - value.keys())))
    if value["status"] != "frozen" or value["scope"] not in {"offline", "simulation", "real_robot"}:
        raise ValueError("Gate requires a frozen protocol with explicit scope")
    if not value["metrics"] or not value["hard_constraints"]:
        raise ValueError("metrics and hard constraints are required")
    for spec in value["metrics"].values():
        if spec["direction"] not in {"max", "min"} or spec["max"] <= spec["min"] or spec["weight"] < 0:
            raise ValueError("invalid metric normalization")
    return value


def score(protocol, result):
    total = 0.0
    for key, spec in protocol["metrics"].items():
        value = result.metrics.get(key)
        if value is None or not isfinite(value):
            return None
        if not spec["min"] <= value <= spec["max"]:
            raise ValueError("metric outside protocol normalization range: " + key)
        normalized = (value - spec["min"]) / (spec["max"] - spec["min"])
        if spec["direction"] == "min":
            normalized = 1 - normalized
        total += spec["weight"] * normalized
    for key, weight in protocol.get("cost_weights", {}).items():
        if key not in result.costs or not isfinite(result.costs[key]) or result.costs[key] < 0:
            return None
        total -= weight * result.costs[key]
    return total


def assignment_metrics(protocol, result):
    """Recompute scores from every preassigned task/seed, never trust scalars.

    Each row contains task_key, seed, outcome, validity, evidence_ref and
    outcome_source. A declared failure still requires independent terminal
    evidence. Unrun slots remain present with outcome=unknown.
    """
    split = protocol.get("task_sets", {}).get("release")
    if split is None:
        # Small offline state-machine protocols can intentionally omit a matrix.
        if protocol["scope"] != "offline":
            raise ValueError("runtime protocol requires explicit release assignments")
        return result.metrics
    expected = {(key, seed) for key in split["task_keys"] for seed in split["seeds"]}
    rows = {(r["task_key"], r["seed"]): r for r in result.assignments}
    if len(rows) != len(result.assignments) or set(rows) != expected:
        raise ValueError("release assignment matrix missing/duplicate/unexpected slots")
    if result.assigned_count != len(expected):
        raise ValueError("assigned_count does not match fixed denominator")
    unknown = invalid = 0
    for row in rows.values():
        if row.get("outcome") not in {"success", "failure", "unknown"} or row.get("validity") not in {"valid", "invalid"}:
            raise ValueError("invalid assignment label")
        unknown += row["outcome"] == "unknown"
        invalid += row["validity"] == "invalid"
        if row["outcome"] != "unknown" and (not row.get("evidence_ref") or row.get("outcome_source") not in {"native_benchmark", "verified_budget_exhaustion"}):
            raise ValueError("assignment lacks independent native/terminal evidence")
        if row.get("outcome_source") == "verified_budget_exhaustion" and row["outcome"] != "failure":
            raise ValueError("budget exhaustion can establish failure only")
    if (unknown, invalid) != (result.unknown_count, result.invalid_count):
        raise ValueError("unknown/invalid counts disagree with assignment matrix")
    groups = {t["task_key"]: t["group"] for t in protocol["tasks"]}
    result_metrics = {}
    for group in {groups[key] for key, _ in expected}:
        task_scores = []
        for key in split["task_keys"]:
            if groups[key] == group:
                task_rows = [rows[(key, seed)] for seed in split["seeds"]]
                task_scores.append(sum(r["outcome"] == "success" and r["validity"] == "valid" for r in task_rows) / len(task_rows))
        result_metrics["success_" + group] = sum(task_scores) / len(task_scores)
    if any(result.metrics.get(k) != value for k, value in result_metrics.items()):
        raise ValueError("reported metrics disagree with assignment-derived metrics")
    return result_metrics


def critical_preserved(protocol, candidate, baseline, anchor):
    if not protocol.get("task_sets", {}).get("release"):
        return True
    for task in protocol["tasks"]:
        if task["group"] != "old":
            continue
        def successes(result):
            return sum(r["task_key"] == task["task_key"] and r["outcome"] == "success"
                       and r["validity"] == "valid" for r in result.assignments)
        if successes(candidate) < max(1, successes(baseline), successes(anchor)):
            return False
    return True


class EvaluationStore(Records):
    """Place outside the research worker's writable root in real deployments."""
    def save_result(self, result):
        if result.role != "release":
            raise ValueError("independent evaluation store requires role=release")
        return self.put("results", result.result_id, result)

    def freeze_protocol(self, path):
        protocol = load_protocol(path)
        self.put("protocols", protocol["protocol_id"], protocol)
        return protocol


class Gate:
    def __init__(self, protocol):
        self.protocol = protocol

    def evaluate(self, decision_id, candidate, *, current_bundle_id, candidate_result=None,
                 baseline_result=None, anchor_result=None):
        p = self.protocol
        reasons, scores = [], {}
        refs = [r.result_id for r in (candidate_result, baseline_result, anchor_result) if r]
        verdict = "inconclusive"
        if candidate.base_bundle_id != current_bundle_id:
            reasons.append("candidate baseline is stale")
        if candidate.protocol_id != p["protocol_id"] or candidate.benchmark_id != p["benchmark_id"]:
            reasons.append("candidate protocol/benchmark mismatch")
        if p["scope"] != "offline" and p.get("dynamic_authorized") is not True:
            reasons.append("dynamic evaluation/deployment authorization missing")
        if candidate.approval_scope != p["scope"]:
            reasons.append("candidate approval scope mismatch")
        expected = (("candidate", candidate_result, candidate.bundle_id),
                    ("baseline", baseline_result, current_bundle_id),
                    ("anchor", anchor_result, p["anchor_bundle_id"]))
        valid_results = True
        for label, result, bundle in expected:
            if result is None:
                reasons.append(label + " independent result missing")
                valid_results = False
                continue
            if (result.bundle_id != bundle or result.protocol_id != p["protocol_id"]
                    or result.benchmark_id != p["benchmark_id"] or result.role != "release"
                    or result.evidence_level != p["scope"]):
                reasons.append(label + " result provenance mismatch")
                valid_results = False
            if label == "candidate" and result.candidate_id != candidate.candidate_id:
                reasons.append("candidate result identity mismatch")
                valid_results = False
            if result.assigned_count <= 0 or result.unknown_count or result.invalid_count:
                reasons.append(label + " result incomplete/unknown/invalid")
            if min(result.assigned_count, result.unknown_count, result.invalid_count) < 0:
                reasons.append(label + " invalid counts")
            try:
                assignment_metrics(p, result)
            except (ValueError, KeyError, TypeError) as exc:
                reasons.append(label + ": " + str(exc))
                valid_results = False
        candidate_identity_matches = (candidate_result is not None
            and candidate.protocol_id == p["protocol_id"]
            and candidate.benchmark_id == p["benchmark_id"]
            and candidate_result.bundle_id == candidate.bundle_id
            and candidate_result.protocol_id == p["protocol_id"]
            and candidate_result.benchmark_id == p["benchmark_id"]
            and candidate_result.candidate_id == candidate.candidate_id
            and candidate_result.role == "release"
            and candidate_result.evidence_level == p["scope"])
        if candidate_identity_matches and any(candidate_result.hard_constraints.get(k) is False for k in p["hard_constraints"]):
            verdict = "reject"
            reasons.append("hard constraint violated")
        # Known hard violations reject even when other evidence is incomplete.
        # Mismatched result identities are never used as evidence for this candidate.
        if valid_results:
            constraints = [candidate_result.hard_constraints.get(k) for k in p["hard_constraints"]]
            if any(value is False for value in constraints) or not critical_preserved(p, candidate_result, baseline_result, anchor_result):
                verdict = "reject"
                reasons.append("hard constraint violated")
            elif any(value is not True for value in constraints):
                reasons.append("hard constraint evidence missing")
            for label, result, _ in expected:
                try:
                    scores[label] = score(p, result)
                except ValueError as exc:
                    scores[label] = None
                    reasons.append(str(exc))
                if scores[label] is None:
                    reasons.append(label + " score unavailable")
            if not reasons:
                if scores["candidate"] - scores["baseline"] < p["min_gain"]:
                    verdict = "reject"
                    reasons.append("gain below fixed minimum")
                elif scores["candidate"] < scores["anchor"] - p["anchor_max_drop"]:
                    verdict = "reject"
                    reasons.append("long-term anchor regression")
                else:
                    verdict = "accept"
                    reasons.append("hard constraints and fixed score criteria satisfied")
        eligible = (verdict == "accept" and candidate.approval_scope == p["scope"]
                    and (p["scope"] == "offline" or p.get("dynamic_authorized") is True))
        if verdict == "accept" and not eligible:
            reasons.append("deployment scope/authorization not satisfied")
        return GateDecision(decision_id, candidate.candidate_id, candidate.base_bundle_id,
                            candidate.bundle_id, p["protocol_id"], p["benchmark_id"], verdict,
                            reasons, eligible, p["scope"], scores, refs)
