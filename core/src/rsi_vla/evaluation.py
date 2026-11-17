"""Fixed protocol consumer and deterministic three-way Gate.

Formal results are supplied by a separate trusted evaluator, never promoted
from development results. Filesystem/process isolation is the deployment ACL.
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from .contracts import GateDecision
from .storage import Records, atomic_json


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
    if value.get('objective_comparison') is not None:
        from .objective_evaluation import objective_policy
        objective_policy(value)
    policy = value.get("conditional_objective")
    if policy is not None:
        fields = {"minimum_relative_request_reduction", "minimum_absolute_request_reduction", "minimum_improved_tasks", "maximum_task_request_increase", "request_score_weight"}
        if (policy.get("schema") != "rsi-success-ceiling-efficiency-v1" or set(policy) != fields | {"schema"} or
            any(type(policy[key]) not in (int, float) or not isfinite(policy[key]) or policy[key] < 0 for key in fields) or
            policy["minimum_relative_request_reduction"] > 1):
            raise ValueError("invalid conditional efficiency objective")
    return value


def apply_comparison_attempt_evidence(plan, side, result):
    """Preserve all attempt expenses and identity-bound violations in Gate input."""
    attempts = [a for s in plan['slots'] if s['side'] == side for a in s.get('attempts', []) if 'costs' in a]
    if attempts:
        result.costs.clear()
        for field in set().union(*(a['costs'] for a in attempts)):
            values = [a['costs'].get(field) for a in attempts]
            if all(type(value) in (int, float) for value in values):
                result.costs[field] = sum(values)
        if any(field not in result.costs for field in ("whole_job_wall_seconds", "gpu_device_seconds_envelope",
                "model_requests", "tool_calls", "environment_steps", "artifact_bytes")):
            result.hard_constraints['resource_limits'] = None
    if result.intervention.get('objective_comparison_schema') == 'rsi-objective-comparison-v1':
        result.intervention['attempt_costs_by_slot'] = [
            dict(task_key=slot.get('task_key'), seed=slot.get('seed'),
                 attempts=[dict(attempt.get('costs', {})) for attempt in slot.get('attempts', [])])
            for slot in plan['slots'] if slot['side'] == side]
    by_task = {}
    task_costs_complete = True
    for slot in plan['slots']:
        if slot['side'] != side:
            continue
        task = slot.get('task_key')
        values = [attempt.get('costs', {}).get('model_requests') for attempt in slot.get('attempts', [])]
        if task is None or not values or any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in values):
            task_costs_complete = False
            continue
        by_task[task] = by_task.get(task, 0) + sum(values)
    result.intervention['attempt_model_requests_by_task'] = by_task if task_costs_complete else None
    for violation in plan.get('final_attempt_violations', []):
        if violation['side'] == side and violation['constraint'] in result.hard_constraints:
            result.hard_constraints[violation['constraint']] = False


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
        sources = {"native_benchmark", "verified_budget_exhaustion"}
        if (protocol["scope"] == "offline" and protocol.get("synthetic_engineering") is True
                and result.intervention.get("synthetic_engineering") is True):
            sources.add("synthetic_task")
        if row["outcome"] != "unknown" and (not row.get("evidence_ref") or row.get("outcome_source") not in sources):
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


def critical_old_evidence_matches(protocol, result):
    """Require the old-capability label to match complete valid native outcomes."""
    value = result.hard_constraints.get("critical_old_preserved")
    if type(value) is not bool:
        return False
    split = protocol.get("task_sets", {}).get("release", {})
    keys = [task["task_key"] for task in protocol.get("tasks", [])
            if task["group"] == "old" and task["task_key"] in split.get("task_keys", [])]
    rows = [row for row in result.assignments if row["task_key"] in keys]
    if (not keys or not rows or any(row["validity"] != "valid"
            or row["outcome"] not in {"success", "failure"} for row in rows)):
        return False
    established = all(any(row["task_key"] == key and row["outcome"] == "success"
                         for row in rows) for key in keys)
    return value is established


def critical_preserved(protocol, candidate, baseline, anchor):
    if not protocol.get("task_sets", {}).get("release"):
        return True
    # Unknown or technically invalid slots cannot prove loss of old ability.
    if any(result.unknown_count or result.invalid_count or result.assigned_count <= 0
           for result in (candidate, baseline, anchor)):
        return None
    for task in protocol["tasks"]:
        if task["group"] != "old":
            continue
        def successes(result):
            return sum(r["task_key"] == task["task_key"] and r["outcome"] == "success"
                       and r["validity"] == "valid" for r in result.assignments)
        if successes(candidate) < max(1, successes(baseline), successes(anchor)):
            return False
    return True


def conditional_efficiency(protocol, candidate, baseline, anchor):
    """Evaluate the predeclared finite efficiency branch only at raw success J=1."""
    policy = protocol.get("conditional_objective")
    if not policy or score({**protocol, "cost_weights": {}}, baseline) != 1:
        return None
    if (baseline.unknown_count or baseline.invalid_count or baseline.assigned_count <= 0):
        return None
    if policy.get("schema") != "rsi-success-ceiling-efficiency-v1":
        return "inconclusive", "conditional efficiency policy unsupported"
    keys = protocol["task_sets"]["release"]["task_keys"]
    for result in (candidate, baseline, anchor):
        constraints = [result.hard_constraints.get(key) for key in protocol["hard_constraints"]]
        if any(value is False for value in constraints):
            return "reject", "conditional efficiency hard constraint violated"
        if any(value is not True for value in constraints):
            return "inconclusive", "conditional efficiency hard constraint evidence missing"
    for key in keys:
        counts = [sum(row["task_key"] == key and row["outcome"] == "success" and row["validity"] == "valid"
                      for row in result.assignments) for result in (candidate, baseline, anchor)]
        if counts[0] < max(counts[1:]):
            return "reject", "conditional efficiency task success regression: " + key
    maps = [result.intervention.get("attempt_model_requests_by_task") for result in (candidate, baseline)]
    for result, costs in zip((candidate, baseline), maps):
        if (not isinstance(costs, dict) or set(costs) != set(keys) or
            any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in costs.values()) or
            sum(costs.values()) != result.costs.get("model_requests")):
            return "inconclusive", "conditional efficiency all-attempt task request costs unknown"
    candidate_costs, baseline_costs = maps
    reduction = baseline.costs["model_requests"] - candidate.costs["model_requests"]
    if (reduction < policy["minimum_absolute_request_reduction"] or
        reduction + 1e-12 < baseline.costs["model_requests"] * policy["minimum_relative_request_reduction"]):
        return "reject", "conditional efficiency absolute/relative request reduction below minimum"
    if sum(candidate_costs[key] < baseline_costs[key] for key in keys) < policy["minimum_improved_tasks"]:
        return "reject", "conditional efficiency improvement concentrated in too few tasks"
    if any(candidate_costs[key] - baseline_costs[key] > policy["maximum_task_request_increase"] for key in keys):
        return "reject", "conditional efficiency per-task request increase above maximum"
    return "accept", "conditional efficiency success ceiling and all-attempt request criteria satisfied"


class EvaluationStore(Records):
    def candidate_archive(self, research_store, candidate_id, **filters):
        """Trace the exact candidate combination used by independent evaluation."""
        return research_store.candidate_archive(candidate_id, evaluations=self, **filters)

    """Place outside the research worker's writable root in real deployments."""

    def path(self, kind, key):
        if kind == "decisions":
            from .gate_versions import decision_path
            return decision_path(self, key)
        raw = super().path(kind, key)
        pointer = super().path("result_evaluation_current", key)
        if kind != "results" or not pointer.exists():
            return raw
        current = json.loads(pointer.read_text())
        version = super().path("result_evaluation_versions", current["revision_id"])
        receipt = super().path("result_evaluation_revisions", current["revision_id"])
        if (
            current.get("schema") != "result-evaluation-current-v1"
            or current.get("result_id") != key
            or current.get("version_ref") != str(version.resolve())
            or current.get("receipt_ref") != str(receipt.resolve())
        ):
            raise ValueError("result evaluation pointer identity mismatch")
        record = json.loads(receipt.read_text())
        if (
            record.get("schema") != "result-evaluation-revision-v1"
            or record.get("result_id") != key
            or record.get("revision_id") != current["revision_id"]
            or record.get("version_ref") != str(version.resolve())
            or record.get("raw_result_ref") != str(raw.resolve())
            or not raw.is_file()
            or json.loads(version.read_text()).get("result_id") != key
        ):
            raise ValueError("result evaluation version/receipt identity mismatch")
        return version

    def get(self, kind, key):
        value = super().get(kind, key)
        if kind in {"results", "decisions"} and value.get("result_id" if kind == "results" else "decision_id") != key:
            raise ValueError("evaluation canonical identity mismatch")
        return value

    def all(self, kind):
        if kind in {"results", "decisions"}:
            return [
                self.get(kind, path.stem)
                for path in sorted((self.root / kind).glob("*.json"))
            ]
        return super().all(kind)

    def put(self, kind, key, value):
        if kind not in {"results", "decisions"}:
            return super().put(kind, key, value)
        record = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        if record.get("result_id" if kind == "results" else "decision_id") != key:
            raise ValueError("evaluation canonical identity mismatch")
        # Ordinary imports and save_result always retain the immutable raw file.
        path = super().path(kind, key)
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("conflicting " + kind + " identity: " + key)
            return False
        atomic_json(path, record)
        return True

    def public_feedback(self, result_id: str, *, protocol: dict, information_scope: str | None = None) -> dict:
        """Persist the fixed public projection permitted by its frozen protocol."""
        from .contracts import ExperimentResult
        from .objective_evaluation import feedback_projection
        result = ExperimentResult(**self.get('results', result_id))
        if protocol.get('status') != 'frozen' or result.protocol_id != protocol['protocol_id']:
            raise ValueError('feedback requires its frozen evaluation protocol')
        scope = result.intervention.get('information_scope')
        if scope not in {'development', 'release', 'sealed', 'meta_exam'} or information_scope is not None and information_scope != scope:
            raise ValueError('feedback source needs its matching explicit information scope')
        if scope in {'sealed', 'meta_exam'} or protocol.get('feedback_policy', {}).get(scope) != 'aggregate':
            projected = {'schema': 'rsi-evaluation-feedback-v1',
                         'information_scope': scope, 'status': 'withheld'}
        else:
            projected = feedback_projection(result, information_scope=scope)
        envelope = {'schema': 'rsi-public-evaluation-feedback-v1', 'result_id': result_id,
                    'protocol_id': result.protocol_id, 'bundle_id': result.bundle_id,
                    'evaluator_source': {'role': 'evaluation', 'result_id': result_id,
                                         'protocol_id': result.protocol_id}, 'feedback': projected}
        self.put('public_feedback', result_id, envelope)
        return {'feedback': envelope, 'feedback_ref': str(self.path('public_feedback', result_id).resolve())}

    def save_result(self, result):
        if result.role != "release":
            raise ValueError("independent evaluation store requires role=release")
        return self.put("results", result.result_id, result)

    def reparse_result(
        self,
        result,
        *,
        revision_id,
        expected_previous_ref,
        reason,
        source_ref,
        protocol,
        authorization_ref,
        scope_authorization_ref,
        derived_ref_repairs=(),
    ):
        """Version new-task evaluation evidence while preserving all native facts."""
        from .task_policy import task_policy

        scope = json.loads(Path(scope_authorization_ref).read_text())
        authorization = json.loads(Path(authorization_ref).read_text())
        now = datetime.now(timezone.utc)
        if (
            not task_policy(protocol)
            or protocol.get("status") != "frozen"
            or not protocol.get("scope_id")
            or result.role != "release"
            or result.protocol_id != protocol["protocol_id"]
            or result.evidence_level != protocol["scope"]
            or scope.get("scope_id") != protocol.get("scope_id")
            or authorization.get("scope") != scope.get("scope_id")
            or authorization.get("protocol_id") != result.protocol_id
            or Path(authorization["approval_reference"]).resolve()
            != Path(scope_authorization_ref).resolve()
        ):
            raise PermissionError("result reparse requires matching new-task authority")
        for value in (scope, authorization):
            expiry = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
            if (
                value.get("authorized") is not True
                or expiry.tzinfo is None
                or now >= expiry
            ):
                raise PermissionError("result reparse authorization missing or expired")
        if result.benchmark_id != protocol["benchmark_id"]:
            raise ValueError("result reparse changes native facts: benchmark_id")
        source = Path(source_ref).resolve(strict=True)
        if not source.is_file() or not isinstance(reason, str) or not reason.strip():
            raise ValueError("result reparse requires source snapshot and reason")
        raw = super().path("results", result.result_id).resolve(strict=True)
        previous = Path(expected_previous_ref).resolve(strict=True)
        old = json.loads(previous.read_text())
        new = asdict(result)

        def same(left, right):
            return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
                right, sort_keys=True, allow_nan=False
            )

        mutable = {"hard_constraints", "intervention", "data_refs", "notes"}
        if not same(
            {k: v for k, v in old.items() if k not in mutable},
            {k: v for k, v in new.items() if k not in mutable},
        ):
            raise ValueError(
                "result reparse changes native facts, identity, criterion or costs"
            )
        if set(old["hard_constraints"]) != set(new["hard_constraints"]):
            raise ValueError("result reparse changes hard constraint definitions")
        for key, known in old["hard_constraints"].items():
            updated = new["hard_constraints"][key]
            if not same(known, updated) and not (
                known is None and type(updated) is bool
            ):
                raise ValueError(
                    "result reparse changes an established hard constraint"
                )
        if not same(
            {
                k: v
                for k, v in old["intervention"].items()
                if k != "critical_progress_evidence"
            },
            {
                k: v
                for k, v in new["intervention"].items()
                if k != "critical_progress_evidence"
            },
        ):
            raise ValueError("result reparse changes the runtime intervention")
        repairs = []
        substitutions = {}
        native_refs = {row.get("evidence_ref") for row in old["assignments"]}
        native_roots = {Path(ref).parent.resolve() for ref in native_refs if ref}
        for repair in derived_ref_repairs:
            if set(repair) != {"missing_ref", "replacement_ref", "runtime_context_ref", "identity"}:
                raise ValueError("invalid missing independent audit repair record")
            missing_ref = repair["missing_ref"]
            replacement = Path(repair["replacement_ref"]).resolve(strict=True)
            missing = Path(missing_ref)
            context_ref = missing.parent / "runtime_context.json"
            if (missing_ref not in old["data_refs"] or missing_ref in native_refs
                    or missing.name != "independent-audit.json" or missing.exists()
                    or missing.parent.resolve() not in native_roots
                    or missing_ref in substitutions
                    or not replacement.is_relative_to(self.root.resolve())
                    or replacement.name != "independent-audit.json" or not replacement.is_file()
                    or str(context_ref.resolve(strict=True)) != repair["runtime_context_ref"]):
                raise ValueError("only a missing derived independent audit reference can be repaired")
            context = json.loads(context_ref.read_text())
            identity = {key: context[key] for key in ("run_id", "episode_id", "bundle_id", "protocol_id")}
            audit = json.loads(replacement.read_text())
            if (audit.get("identity") != identity or repair.get("identity") != identity
                    or identity["bundle_id"] != result.bundle_id or identity["protocol_id"] != result.protocol_id):
                raise ValueError("repaired independent audit runtime identity mismatch")
            substitutions[missing_ref] = str(replacement)
            repairs.append({**repair, "replacement_ref": str(replacement)})
        expected_refs = [substitutions.get(ref, ref) for ref in old["data_refs"]]
        if new["data_refs"][: len(expected_refs)] != expected_refs:
            raise ValueError(
                "result reparse must preserve every original evidence reference"
            )
        if repairs and any(not Path(ref).is_file() for ref in new["data_refs"]):
            raise ValueError("result reparse cannot repair or hide missing raw/native evidence")
        version = super().path("result_evaluation_versions", revision_id).resolve()
        receipt_path = (
            super().path("result_evaluation_revisions", revision_id).resolve()
        )
        fixed = dict(
            schema="result-evaluation-revision-v1",
            revision_id=revision_id,
            result_id=result.result_id,
            protocol_id=result.protocol_id,
            scope=scope["scope_id"],
            raw_result_ref=str(raw),
            previous_ref=str(previous),
            version_ref=str(version),
            source_ref=str(source),
            reason=reason,
            authorization_ref=str(Path(authorization_ref).resolve()),
            scope_authorization_ref=str(Path(scope_authorization_ref).resolve()),
        )
        if repairs:
            fixed["derived_ref_repairs"] = repairs
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if {
                k: v for k, v in receipt.items() if k != "evaluated_at"
            } != fixed or not same(json.loads(version.read_text()), new):
                raise ValueError("conflicting result evaluation revision identity")
            # A replay never moves a later current pointer back to this revision.
            if self.path("results", result.result_id).resolve() != previous:
                return receipt
        else:
            receipt = {**fixed, "evaluated_at": now.isoformat()}
        current = self.path("results", result.result_id).resolve()
        if current != previous:
            raise ValueError("result evaluation previous reference is stale")
        if previous != raw:
            prior_receipt = super().path("result_evaluation_revisions", previous.stem)
            prior = json.loads(prior_receipt.read_text())
            if (
                previous.parent != version.parent
                or prior.get("result_id") != result.result_id
                or prior.get("version_ref") != str(previous)
            ):
                raise ValueError("result evaluation previous version identity mismatch")
        if version.exists() and not same(json.loads(version.read_text()), new):
            raise ValueError("conflicting immutable result evaluation version")
        self.put("result_evaluation_versions", revision_id, new)
        self.put("result_evaluation_revisions", revision_id, receipt)
        atomic_json(
            super().path("result_evaluation_current", result.result_id),
            dict(
                schema="result-evaluation-current-v1",
                result_id=result.result_id,
                revision_id=revision_id,
                version_ref=str(version),
                receipt_ref=str(receipt_path),
            ),
        )
        return receipt

    def reconsider_gate(self, **arguments):
        """Independently version an inconclusive Gate after same-ID result reparsing."""
        from .gate_versions import reconsider_gate
        return reconsider_gate(self, **arguments)

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
        if p.get('objective_comparison') is not None:
            from .objective_evaluation import evaluate_objective
            return evaluate_objective(p, decision_id, candidate, current_bundle_id=current_bundle_id,
                candidate_result=candidate_result, baseline_result=baseline_result, anchor_result=anchor_result)
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
            if (p.get("comparison_only") is True and label != "candidate"
                    and result.candidate_id is not None):
                reasons.append(label + " result identity mismatch")
                valid_results = False
            if label == "candidate" and result.candidate_id != candidate.candidate_id:
                reasons.append("candidate result identity mismatch")
                valid_results = False
            if result.assigned_count <= 0 or result.unknown_count or result.invalid_count:
                reasons.append(label + " result incomplete/unknown/invalid")
            if p.get("comparison_only") is True:
                if not result.data_refs:
                    reasons.append(label + " independent evidence references missing")
                for key in p["hard_constraints"]:
                    if label != "candidate" and key == "critical_old_preserved":
                        if not critical_old_evidence_matches(p, result):
                            reasons.append(label + " critical old capability evidence unknown/conflicting")
                        continue
                    if result.hard_constraints.get(key) is not True:
                        reasons.append(label + " hard constraint evidence unestablished: " + key)
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
            preserved = critical_preserved(p, candidate_result, baseline_result, anchor_result)
            if p.get("comparison_only") is True:
                minimum = p.get("initial_requirements", {}).get("metric_minima", {}).get("success_base")
                base = candidate_result.metrics.get("success_base")
                if (minimum is not None and base is not None and base < minimum
                        and candidate_result.assigned_count > 0
                        and not candidate_result.unknown_count and not candidate_result.invalid_count):
                    verdict = "reject"
                    reasons.append("candidate base capability below predeclared minimum")
            if any(value is False for value in constraints) or preserved is False:
                verdict = "reject"
                reasons.append("hard constraint violated")
            elif preserved is None:
                reasons.append("critical old capability evidence incomplete")
            elif any(value is not True for value in constraints):
                reasons.append("hard constraint evidence missing")
            efficiency_branch = (p.get("conditional_objective") and baseline_result.assigned_count > 0
                and not (baseline_result.unknown_count or baseline_result.invalid_count)
                and score({**p, "cost_weights": {}}, baseline_result) == 1)
            score_protocol = ({**p, "cost_weights": {"model_requests": p["conditional_objective"]["request_score_weight"]}}
                              if efficiency_branch else p)
            for label, result, _ in expected:
                try:
                    scores[label] = score(score_protocol, result)
                except ValueError as exc:
                    scores[label] = None
                    reasons.append(str(exc))
                if scores[label] is None:
                    reasons.append(label + " score unavailable")
            if not reasons:
                efficiency = conditional_efficiency(p, candidate_result, baseline_result, anchor_result)
                if efficiency is not None and efficiency[0] != "accept":
                    verdict, reason = efficiency
                    reasons.append(reason)
                elif scores["candidate"] - scores["baseline"] + (1e-12 if efficiency else 0) < p["min_gain"]:
                    verdict = "reject"
                    reasons.append("gain below fixed minimum")
                elif scores["candidate"] < scores["anchor"] - p["anchor_max_drop"]:
                    verdict = "reject"
                    reasons.append("long-term anchor regression")
                else:
                    verdict = "accept"
                    reasons.append(efficiency[1] if efficiency else "hard constraints and fixed score criteria satisfied")
        if p.get("conditional_objective"):
            branch = "pending_complete_baseline"
            if valid_results and baseline_result.assigned_count > 0 and not (baseline_result.unknown_count or baseline_result.invalid_count):
                branch = "efficiency" if score({**p, "cost_weights": {}}, baseline_result) == 1 else "success_gain"
            reasons.append("objective branch: " + branch)
        eligible = (verdict == "accept" and p.get("comparison_only") is not True
                    and candidate.approval_scope == p["scope"]
                    and (p["scope"] == "offline" or p.get("dynamic_authorized") is True))
        if verdict == "accept" and not eligible:
            reasons.append("comparison-only acceptance grants no deployment authorization"
                if p.get("comparison_only") is True else "deployment scope/authorization not satisfied")
        return GateDecision(decision_id, candidate.candidate_id, candidate.base_bundle_id,
                            candidate.bundle_id, p["protocol_id"], p["benchmark_id"], verdict,
                            reasons, eligible, p["scope"], scores, refs)
