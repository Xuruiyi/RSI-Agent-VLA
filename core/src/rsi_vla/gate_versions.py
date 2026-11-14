"""Finite independent reconsideration of an inconclusive canonical Gate."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .contracts import ExperimentResult
from .storage import Records, atomic_json

SIDES = ("candidate", "baseline", "anchor")
STABLE = ("decision_id", "candidate_id", "base_bundle_id", "bundle_id", "protocol_id",
          "benchmark_id", "scope", "result_refs", "scores")


def _read(ref):
    return json.loads(Path(ref).read_text())


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _raw(store, kind, key):
    return Records.path(store, kind, key).resolve()


def _result_input(store, result_id, reference, seen=()):
    """Validate an immutable result revision chain without advancing a pointer."""
    path = Path(reference).resolve(strict=True)
    raw = _raw(store, "results", result_id)
    value = _read(path)
    if value.get("result_id") != result_id or path in seen:
        raise ValueError("Gate result revision identity/cycle mismatch")
    if path == raw:
        return value, {path}
    if path.parent != _raw(store, "result_evaluation_versions", path.stem).parent:
        raise ValueError("Gate input is not a canonical result revision")
    receipt = _read(_raw(store, "result_evaluation_revisions", path.stem))
    if (receipt.get("schema") != "result-evaluation-revision-v1" or receipt.get("revision_id") != path.stem
            or receipt.get("result_id") != result_id or receipt.get("version_ref") != str(path)
            or receipt.get("raw_result_ref") != str(raw) or receipt.get("protocol_id") != value["protocol_id"]):
        raise ValueError("Gate result revision receipt mismatch")
    previous, ancestry = _result_input(store, result_id, receipt["previous_ref"], (*seen, path))
    mutable = {"hard_constraints", "intervention", "data_refs", "notes"}
    if not _same({k: v for k, v in previous.items() if k not in mutable},
                 {k: v for k, v in value.items() if k not in mutable}):
        raise ValueError("Gate result revision changes native facts or costs")
    if set(previous["hard_constraints"]) != set(value["hard_constraints"]):
        raise ValueError("Gate result revision changes hard constraint definitions")
    for key, known in previous["hard_constraints"].items():
        updated = value["hard_constraints"][key]
        if not _same(known, updated) and not (known is None and type(updated) is bool):
            raise ValueError("Gate result revision changes established evidence")
    if not _same({k: v for k, v in previous["intervention"].items() if k != "critical_progress_evidence"},
                 {k: v for k, v in value["intervention"].items() if k != "critical_progress_evidence"}):
        raise ValueError("Gate result revision changes Runtime intervention")
    repairs = {row["missing_ref"]: row["replacement_ref"] for row in receipt.get("derived_ref_repairs", [])}
    expected = [repairs.get(ref, ref) for ref in previous["data_refs"]]
    if value["data_refs"][:len(expected)] != expected:
        raise ValueError("Gate result revision drops original evidence references")
    return value, ancestry | {path}


def decision_path(store, decision_id):
    """Resolve the logical Gate while checking its immutable revision receipt."""
    raw = _raw(store, "decisions", decision_id)
    pointer = _raw(store, "gate_evaluation_current", decision_id)
    if not pointer.exists():
        return raw
    current = _read(pointer)
    revision_id = current["revision_id"]
    version = _raw(store, "gate_evaluation_versions", revision_id)
    receipt_path = _raw(store, "gate_evaluation_revisions", revision_id)
    if (current.get("schema") != "gate-evaluation-current-v1" or current.get("decision_id") != decision_id
            or current.get("version_ref") != str(version) or current.get("receipt_ref") != str(receipt_path)):
        raise ValueError("Gate evaluation pointer identity mismatch")
    receipt = _read(receipt_path)
    original, value = _read(raw), _read(version)
    if (receipt.get("schema") != "gate-evaluation-revision-v1" or receipt.get("revision_id") != revision_id
            or receipt.get("decision_id") != decision_id or receipt.get("version_ref") != str(version)
            or receipt.get("raw_decision_ref") != str(raw) or receipt.get("raw_decision_snapshot") != original
            or original.get("verdict") != "inconclusive"
            or not _same(value, receipt.get("decision_snapshot"))
            or any(not _same(original.get(key), value.get(key)) for key in STABLE)):
        raise ValueError("Gate evaluation version/receipt identity mismatch")
    previous = _read(receipt["previous_ref"])
    if previous.get("verdict") != "inconclusive" or previous != receipt.get("previous_decision_snapshot"):
        raise ValueError("Gate reconsideration previous decision is no longer inconclusive")
    if set(receipt.get("result_version_refs", {})) != set(SIDES):
        raise ValueError("Gate evaluation result revision bindings are missing")
    from .contracts import Candidate
    from .evaluation import Gate
    protocol, candidate = receipt["protocol_snapshot"], Candidate(**receipt["candidate_snapshot"])
    old_results, new_results, changed = {}, {}, False
    if set(receipt.get("previous_result_refs", {})) != set(SIDES):
        raise ValueError("Gate previous result revision bindings are missing")
    for side, result_id in zip(SIDES, value["result_refs"]):
        old_ref = Path(receipt["previous_result_refs"][side]).resolve()
        old, _ = _result_input(store, result_id, old_ref)
        new, ancestry = _result_input(store, result_id, receipt["result_version_refs"][side])
        if old_ref not in ancestry:
            raise ValueError("Gate result revision is not a legal successor")
        changed = changed or Path(receipt["result_version_refs"][side]).resolve() != old_ref
        old_results[side], new_results[side] = ExperimentResult(**old), ExperimentResult(**new)
    if not changed:
        raise ValueError("Gate revision has no new same-ID result evidence")
    for inputs, expected in ((old_results, previous), (new_results, value)):
        reproduced = Gate(protocol).evaluate(decision_id, candidate,
            current_bundle_id=receipt["current_bundle_id"], candidate_result=inputs["candidate"],
            baseline_result=inputs["baseline"], anchor_result=inputs["anchor"])
        if not _same(asdict(reproduced), expected):
            raise ValueError("Gate revision receipt does not reproduce its independent decisions")
    return version


def reconsider_gate(store, *, decision_id, candidate, current_bundle_id, result_ids,
                     previous_result_refs, revision_id, expected_previous_ref, reason,
                     source_ref, protocol, frozen_protocol_ref, authorization_ref,
                     scope_authorization_ref):
    """Recompute only an inconclusive Gate after same-ID evidence reparsing.

    All native facts, scores, result IDs, criterion definitions and logical Gate
    identities stay fixed. The previous result paths must reproduce the previous
    complete Gate object; at least one current result must be a legal successor.
    """
    from .evaluation import Gate
    from .task_policy import task_policy

    authorization = _read(authorization_ref)
    scope = _read(scope_authorization_ref)
    now = datetime.now(timezone.utc)
    if (not task_policy(protocol) or protocol.get("status") != "frozen"
            or _read(frozen_protocol_ref) != protocol or scope.get("scope_id") != protocol.get("scope_id")
            or authorization.get("scope") != scope.get("scope_id")
            or authorization.get("protocol_id") != protocol.get("protocol_id")
            or Path(authorization["approval_reference"]).resolve() != Path(scope_authorization_ref).resolve()):
        raise PermissionError("Gate reconsideration requires the unchanged frozen protocol and existing authority")
    for authority in (scope, authorization):
        expiry = datetime.fromisoformat(authority["expires_at"].replace("Z", "+00:00"))
        if authority.get("authorized") is not True or expiry.tzinfo is None or now >= expiry:
            raise PermissionError("Gate reconsideration authority missing or expired")
    source = Path(source_ref).resolve(strict=True)
    if not source.is_file() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Gate reconsideration needs an evaluation source and reason")
    raw = _raw(store, "decisions", decision_id)
    original = _read(raw)
    previous_path = Path(expected_previous_ref).resolve(strict=True)
    previous = _read(previous_path)
    if original.get("verdict") != "inconclusive" or previous.get("verdict") != "inconclusive":
        raise ValueError("only an inconclusive raw and previous Gate can be reconsidered")
    if (set(result_ids) != set(SIDES) or set(previous_result_refs) != set(SIDES)
            or [result_ids[side] for side in SIDES] != original["result_refs"]):
        raise ValueError("Gate reconsideration cannot change result IDs or denominator")
    if previous_path != raw:
        prior = _read(_raw(store, "gate_evaluation_revisions", previous_path.stem))
        if (prior.get("version_ref") != str(previous_path)
                or prior.get("raw_decision_ref") != str(raw)
                or prior.get("result_version_refs") != {
                    side: str(Path(previous_result_refs[side]).resolve()) for side in SIDES}):
            raise ValueError("previous Gate result versions must retain its immutable receipt inputs")
    old_results, new_results, result_refs, changed = {}, {}, {}, False
    for side in SIDES:
        result_id = result_ids[side]
        old, _ = _result_input(store, result_id, previous_result_refs[side])
        current_ref = store.path("results", result_id).resolve()
        value, ancestry = _result_input(store, result_id, current_ref)
        old_ref = Path(previous_result_refs[side]).resolve()
        if old_ref not in ancestry:
            raise ValueError("Gate result revision is not a successor of the previous input")
        changed = changed or current_ref != old_ref
        old_results[side], new_results[side] = ExperimentResult(**old), ExperimentResult(**value)
        result_refs[side] = str(current_ref)
    gate = Gate(protocol)
    def evaluate(results):
        return asdict(gate.evaluate(decision_id, candidate, current_bundle_id=current_bundle_id,
            candidate_result=results["candidate"], baseline_result=results["baseline"], anchor_result=results["anchor"]))
    if not _same(evaluate(old_results), previous):
        raise ValueError("previous result versions do not reproduce the previous Gate")
    decision = evaluate(new_results)
    if not changed:
        raise ValueError("Gate reconsideration needs a new same-ID legal result reparse")
    if any(not _same(previous.get(key), decision.get(key)) for key in STABLE):
        raise ValueError("Gate reconsideration changes identities, scores, result IDs or criteria")
    version = _raw(store, "gate_evaluation_versions", revision_id)
    receipt_path = _raw(store, "gate_evaluation_revisions", revision_id)
    fixed = dict(schema="gate-evaluation-revision-v1", revision_id=revision_id, decision_id=decision_id,
        raw_decision_ref=str(raw), raw_decision_snapshot=original, previous_ref=str(previous_path),
        previous_decision_snapshot=previous, version_ref=str(version), decision_snapshot=decision,
        protocol_snapshot=protocol, frozen_protocol_ref=str(Path(frozen_protocol_ref).resolve()),
        candidate_snapshot=asdict(candidate), current_bundle_id=current_bundle_id,
        result_ids=dict(result_ids), previous_result_refs={side: str(Path(previous_result_refs[side]).resolve()) for side in SIDES},
        result_version_refs=result_refs, source_ref=str(source), reason=reason,
        authorization_ref=str(Path(authorization_ref).resolve()),
        scope_authorization_ref=str(Path(scope_authorization_ref).resolve()))
    current = store.path("decisions", decision_id).resolve()
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if {key: value for key, value in receipt.items() if key != "evaluated_at"} != fixed or _read(version) != decision:
            raise ValueError("conflicting Gate reconsideration revision identity")
        if current != previous_path:
            return receipt
    else:
        receipt = {**fixed, "evaluated_at": now.isoformat()}
    if current != previous_path:
        raise ValueError("Gate reconsideration previous reference is stale")
    if _read(current).get("verdict") != "inconclusive":
        raise ValueError("current Gate is already terminal")
    if version.exists() and _read(version) != decision:
        raise ValueError("conflicting immutable Gate revision")
    store.put("gate_evaluation_versions", revision_id, decision)
    store.put("gate_evaluation_revisions", revision_id, receipt)
    atomic_json(_raw(store, "gate_evaluation_current", decision_id),
        dict(schema="gate-evaluation-current-v1", decision_id=decision_id, revision_id=revision_id,
             version_ref=str(version), receipt_ref=str(receipt_path)))
    return receipt


def decision_inputs(store, decision_id):
    """Collect immutable old/current Gate and result inputs for existing role mounts."""
    resolved = store.path("decisions", decision_id).resolve()
    raw = _raw(store, "decisions", decision_id)
    inputs = {raw, resolved}
    if resolved == raw:
        return inputs
    inputs.add(_raw(store, "gate_evaluation_current", decision_id))
    def result_chain(result_id, ref):
        path = Path(ref).resolve()
        if path in inputs:
            return
        inputs.add(path)
        if path != _raw(store, "results", result_id):
            receipt_path = _raw(store, "result_evaluation_revisions", path.stem)
            inputs.add(receipt_path)
            result_chain(result_id, _read(receipt_path)["previous_ref"])
    path = resolved
    seen = set()
    while path != raw:
        if path in seen:
            raise ValueError("Gate revision ancestry cycle")
        seen.add(path)
        inputs.add(path)
        receipt_path = _raw(store, "gate_evaluation_revisions", path.stem)
        inputs.add(receipt_path)
        receipt = _read(receipt_path)
        for side in SIDES:
            for refs in ("previous_result_refs", "result_version_refs"):
                result_chain(receipt["result_ids"][side], receipt[refs][side])
        path = Path(receipt["previous_ref"]).resolve()
    return inputs
