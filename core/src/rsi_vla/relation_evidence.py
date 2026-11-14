"""Independent skill feedback from environment-owned development observations."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from .contracts import ExecutionEvent, RunBundle, record
from .cooperation import review_skill_handoff
from .relation_learning import version_signature

IDENTITY = ("run_id", "episode_id", "bundle_id", "protocol_id")


def _stream(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sequences = [row["sequence"] for row in rows]
    if any(type(value) is not int for value in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("skill feedback stream has invalid or duplicate sequences")
    return rows


def review_skill_feedback(output):
    """Review registered routes without using tool success or terminal outcome.

    Missing client/environment events and unknown measurements retain null
    feedback. Conflicting identities, versions or registered criteria are errors.
    The caller owns the independent evaluation role and report persistence.
    """
    root = Path(output).resolve()
    context_path = root / "runtime_context.json"
    context = json.loads(context_path.read_text())
    if context.get("information_scope") != "development":
        raise ValueError("skill feedback accepts development evidence only")
    identity = {name: context[name] for name in IDENTITY}
    control = context.get("relation_control", {})
    signature = {name: context.get(name) for name in ("theta_ref", "processor_ref", "phi_ref", "code_ref")}
    if (control.get("schema") != "rsi-skill-routing-v1" or not control.get("rho_ref")
            or control["rho_ref"] != context.get("rho_ref") or control.get("version_signature") != signature):
        raise ValueError("skill feedback routing/version binding mismatch")
    routes = {row["route_id"]: row for row in control["routes"]}
    if len(routes) != len(control["routes"]):
        raise ValueError("duplicate registered skill route")
    client_path = root / "action_events.jsonl"
    environment_path = root / "environment_service/action_events.jsonl"
    client, environment = _stream(client_path), _stream(environment_path)
    client_kinds = {"relation_policy_loaded", "relation_route_decision", "skill_route_start", "skill_route_handoff"}
    for row in [r for r in client if r["kind"] in client_kinds] + [r for r in environment if r["kind"] == "skill_route_observation"]:
        if any(row.get(name) != value for name, value in identity.items()):
            raise ValueError("skill feedback crosses execution identity")
    loaded = [row for row in client if row["kind"] == "relation_policy_loaded" and row.get("relation") == "rhoB"]
    for row in loaded:
        if row.get("rho_ref") != control["rho_ref"] or row.get("version_signature") != signature:
            raise ValueError("loaded skill policy differs from the bound version")
    raw = {row["sequence"]: row for row in environment if row["kind"] == "skill_route_observation"}
    references = [str(context_path), str(client_path), str(environment_path)]
    references.extend(str(Path(route["criterion"]["criterion_ref"].partition("#")[0]).resolve())
                      for route in routes.values())
    reviews, observed_routes = [], set()
    for decision in [row for row in client if row["kind"] == "relation_route_decision"]:
        route = routes.get(decision.get("route_id"))
        if route is None:
            raise ValueError("decision refers to an unregistered skill route")
        fields = {"rho_ref": control["rho_ref"], "route_id": route["route_id"], "state": route["state"],
                  "requested_worker": route["requested_worker"], "selected_worker": route["selected_worker"],
                  "criterion_ref": route["criterion"]["criterion_ref"]}
        if (any(decision.get(key) != value for key, value in fields.items())
                or decision.get("trigger_worker") not in route["trigger_workers"]):
            raise ValueError("skill decision differs from its registered route")
        observed_routes.add(route["route_id"])
        sequence = decision["sequence"]
        entry = {"route_id": route["route_id"], "state": route["state"], "worker": route["selected_worker"],
                 "client_route_sequence": sequence, "criterion_ref": route["criterion"]["criterion_ref"],
                 "version_signature": signature, "success": None, "checks": {}, "missing": [],
                 "environment_source_sequences": {}, "client_source_sequences": {"decision": sequence}}
        if not loaded or any(route["route_id"] not in row.get("route_ids", []) for row in loaded):
            entry["missing"].append("actual rhoB policy load receipt")
        criterion_ref = route["criterion"]["criterion_ref"]
        criterion_path, _, pointer = criterion_ref.partition("#")
        references.append(str(Path(criterion_path).resolve()))
        criterion = None
        try:
            from .runtime import resolve_reference
            criterion = resolve_reference(criterion_ref)
        except FileNotFoundError:
            entry["missing"].append("registered criterion file")
        if criterion is not None and {**criterion, "criterion_ref": criterion_ref} != route["criterion"]:
            raise ValueError("registered criterion differs from its independent source file")
        observations = []
        for phase in ("start", "handoff"):
            rows = [row for row in client if row["kind"] == "skill_route_" + phase
                    and row.get("client_route_sequence") == sequence]
            if len(rows) > 1:
                raise ValueError("duplicate client skill observation link")
            if not rows:
                entry["missing"].append("client " + phase)
                observations.append(None)
                continue
            row = rows[0]
            if any(row.get(key) != value for key, value in fields.items()) or row.get("decision_sequence") != sequence:
                raise ValueError("client skill observation has conflicting route binding")
            entry["client_source_sequences"][phase] = row["sequence"]
            source_sequence = row.get("environment_observation", {}).get("source_sequence")
            observation = raw.get(source_sequence)
            if observation is None:
                entry["missing"].append("environment " + phase)
            else:
                expected = {**identity, "route_id": route["route_id"], "state": route["state"],
                            "worker": route["selected_worker"], "client_route_sequence": sequence,
                            "criterion_ref": criterion_ref, "version_signature": signature, "phase": phase}
                if any(observation.get(key) != value for key, value in expected.items()):
                    raise ValueError("client link refers to a different environment observation")
                entry["environment_source_sequences"][phase] = source_sequence
            observations.append(observation)
        positions = entry["client_source_sequences"]
        if "start" in positions and "handoff" in positions and not sequence < positions["start"] < positions["handoff"]:
            raise ValueError("client skill observation ordering is invalid")
        if not entry["missing"] and criterion is not None and all(observations):
            review = review_skill_handoff(observations[0], observations[1], route["criterion"])
            entry.update(success=review["success"], checks=review["checks"], missing=review["missing"])
            entry.update({key: review[key] for key in ("eligibility", "joint_progress", "measurement_scope") if key in review})
        reviews.append(entry)
    for route_id in routes.keys() - observed_routes:
        route = routes[route_id]
        reviews.append({"route_id": route_id, "state": route["state"], "worker": route["selected_worker"],
                        "criterion_ref": route["criterion"]["criterion_ref"], "version_signature": signature,
                        "success": None, "checks": {}, "missing": ["client route decision"],
                        "environment_source_sequences": {}, "client_source_sequences": {}})
    return {"schema": "rsi-skill-feedback-review-v1", "identity": identity, "rho_ref": control["rho_ref"],
            "version_signature": signature, "information_scope": "development", "role": "evaluation",
            "reviews": reviews, "source_refs": list(dict.fromkeys(references)),
            "feedback_scope": "independent_observational_handoff_only"}


def attach_skill_feedback(store, result, report, report_ref):
    """Save a new development view/result; original L0 remains an exact prefix."""
    if (result.role != "development" or result.intervention.get("information_scope") != "development"
            or report.get("schema") != "rsi-skill-feedback-review-v1" or report.get("role") != "evaluation"
            or report.get("information_scope") != "development"
            or report.get("identity") != result.intervention.get("identity") or not report_ref):
        raise ValueError("independent skill feedback does not match the development result")
    bundle = RunBundle(**store.get("bundles", result.bundle_id))
    signature = version_signature(bundle)
    if report.get("rho_ref") != bundle.rho_ref or report.get("version_signature") != signature:
        raise ValueError("independent skill feedback has an incompatible version")
    view_id = result.intervention.get("evidence_view_id")
    view = store.get("evidence_views", view_id)
    if view["bundle_id"] != result.bundle_id:
        raise ValueError("skill feedback result/view mismatch")
    facts = deepcopy(view["l0"])
    sequence = max((event["sequence"] for event in facts), default=-1) + 1
    identity = report["identity"]
    for index, review in enumerate(report["reviews"]):
        if review.get("version_signature") != signature or (review.get("success") is not None and type(review.get("success")) is not bool):
            raise ValueError("invalid independent skill review")
        ids = {}
        decision_sequence = review.get("client_route_sequence")
        decision_id = str(decision_sequence) if type(decision_sequence) is int else "unknown-" + str(index)
        for phase in ("start", "handoff", "review"):
            event_id = identity["run_id"] + "-relation-" + decision_id + "-" + phase
            ids[phase] = event_id
            flags = {"review_ref": str(report_ref), "route_id": review["route_id"],
                     "environment_source_sequences": review["environment_source_sequences"],
                     "client_source_sequences": review["client_source_sequences"]}
            complete_sources = all(phase in review["environment_source_sequences"] for phase in ("start", "handoff"))
            if phase == "review" and type(review.get("success")) is bool:
                if not complete_sources or not review.get("criterion_ref") or type(decision_sequence) is not int:
                    raise ValueError("boolean skill feedback lacks original decision/start/handoff sources")
                flags["route_feedback"] = {"evidence_kind": "independent_criterion", "criterion_ref": review["criterion_ref"],
                    "state": review["state"], "worker": review["worker"], "success": review["success"],
                    "version_signature": signature, "start_event_id": ids["start"], "handoff_event_id": ids["handoff"]}
            event = ExecutionEvent(event_id, identity["run_id"], identity["episode_id"], sequence,
                result.bundle_id, "independent_skill_" + phase, "independent_skill_feedback_review",
                flags=flags, missing=list(review.get("missing", [])))
            facts.append(record(event))
            sequence += 1
    derived_view_id = view_id + "-relations"
    store.put("evidence_views", derived_view_id, {**view, "view_id": derived_view_id,
                                               "l0": facts, "policy_ref": "independent-skill-feedback-v1"})
    enriched = replace(result, result_id=result.result_id + "-relations",
        intervention={**result.intervention, "evidence_view_id": derived_view_id,
                      "skill_feedback_review_ref": str(report_ref)},
        data_refs=[*result.data_refs, str(report_ref), str(store.path("evidence_views", derived_view_id).resolve())])
    store.save_development_result(enriched)
    return enriched
