"""Bounded observational rho updates from admitted development metadata.

A optimizes recorded-event media coverage against reference count, not visual
quality or native success. B needs an explicit independent handoff criterion;
terminal task success is never synthesized into worker supervision. C estimates
observed development screening yield, without attributing causality to research.
Program-frozen snapshots support offline replay and finite development trials.
Candidates additionally require new screened results and independent consumer receipts.
"""

from copy import deepcopy
from dataclasses import replace
from math import isfinite
import tempfile

from .contracts import RunBundle
from .storage import atomic_json

SCHEMA = "rsi-relation-feedback-v1"
CONTEXT_SCHEMA = "rsi-research-context-policy-v2"


def research_context(*, task_key, operator, evidence_category, signature, data_context):
    """Keep operator, evidence, executable and data applicability in the decision key."""
    import json
    if (not all(isinstance(v, str) and v for v in (task_key, operator, evidence_category))
            or set(signature) != set(VERSION_FIELDS) or not all(signature.values())
            or not isinstance(data_context, dict) or not data_context):
        raise ValueError("research context needs task/operator/evidence/version/data applicability")
    value = dict(task_key=task_key, operator=operator, evidence_category=evidence_category,
                 version_signature=dict(signature), data_context=deepcopy(data_context))
    return {"key": json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), **value}


def contextual_research_policy(legacy, *, max_explorations, data_context):
    """Create an explicit new policy without reinterpreting legacy screening samples."""
    if (type(max_explorations) is not int or not 1 <= max_explorations <= 32
            or not isinstance(data_context, dict) or not data_context):
        raise ValueError("contextual rhoC needs finite exploration bound and data context")
    return {"type": CONTEXT_SCHEMA, "values": {}, "feedback": [], "exploration_reservations": [],
            "max_explorations": max_explorations, "data_context": deepcopy(data_context),
            "legacy_baseline": deepcopy(legacy), "legacy_feedback_semantics": "unchanged_not_contextual"}


def contextual_research_decision(policy, job, *, context, evidence_available, inspected,
                                 experiment_permitted, legal_repairs, technical_admitted,
                                 data_admitted, exploration_used=0):
    """Finite exploration changes preference only after all ordinary prerequisites pass."""
    if policy.get("type") != CONTEXT_SCHEMA:
        raise ValueError("contextual research requires its explicit policy version")
    if (not context or context.get("data_context") != policy["data_context"]
            or context.get("operator") not in legal_repairs):
        raise ValueError("research context differs from the bound operator/data applicability")
    if type(exploration_used) is not int or exploration_used < 0:
        raise ValueError("exploration usage must be a nonnegative count")
    row = {"context": deepcopy(context), "policy_type": CONTEXT_SCHEMA,
           "action": "defer", "reason": "prerequisites_missing", "exploration": False,
           "unexecuted_task_gain": None, "exploration_used_before": exploration_used}
    if job.state["status"] != "running" or not evidence_available:
        return row
    if not inspected:
        return {**row, "action": "inspect", "reason": "inspect_legal_evidence"}
    if not experiment_permitted or not technical_admitted or not data_admitted:
        return row
    feedback = policy.get("values", {}).get(context["key"], {})
    known = feedback.get("screening_pass", {})
    if known.get("count", 0) and known["total"] / known["count"] > .5:
        return {**row, "action": "bounded_experiment", "reason": "context_screening_yield"}
    if exploration_used < policy["max_explorations"]:
        return {**row, "action": "bounded_experiment", "reason": "bounded_context_reevaluation",
                "exploration": True}
    return {**row, "reason": "registered_exploration_exhausted"}

VERSION_FIELDS = ("theta_ref", "processor_ref", "phi_ref", "code_ref")


def version_signature(bundle):
    """Bind skill applicability to its executable and perception versions."""
    return {name: getattr(bundle, name) for name in VERSION_FIELDS}


def select_media(policy, facts, selected_media=None):
    """Select recorded L1 references while retaining every L0 fact elsewhere."""
    available = {ref for event in facts for ref in event["media"]}
    if selected_media is not None:
        chosen = set(selected_media)
    elif policy.get("type") == "fixed":
        if policy.get("retain_l0") != "all" or policy.get("media") != "all_available":
            raise ValueError("unsupported rhoA policy")
        chosen = available
    elif policy.get("type") == SCHEMA:
        action = best_action(policy, "media", ("all_available", "one_per_event"))
        chosen = ({min(event["media"]) for event in facts if event["media"]}
                  if action == "one_per_event" else available)
    else:
        raise ValueError("unsupported rhoA policy")
    if not chosen <= available:
        raise ValueError("media outside legal recorded facts")
    return sorted(chosen)


def best_action(policy, state, allowed, *, minimum_actions=1):
    """Choose only measured actions; ties follow the allowed action order."""
    values = policy.get("values", {}).get(state, {})
    measured = [a for a in allowed if values.get(a, {}).get("count", 0)]
    if len(measured) < minimum_actions:
        return None
    return max(measured, key=lambda a: values[a]["total"] / values[a]["count"])


def _sample(policy, state, action, reward):
    if type(reward) not in (float, int) or not isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("feedback target must be finite and within [0, 1]")
    value = policy["values"].setdefault(state, {}).setdefault(action, {"count": 0, "total": 0.0})
    value["count"] += 1
    value["total"] += reward


def _view(tools, result):
    view_id = result["intervention"].get("evidence_view_id")
    if not view_id:
        return None
    view = tools.store.get("evidence_views", view_id)
    if view["bundle_id"] != result["bundle_id"]:
        raise ValueError("relation feedback result/view bundle mismatch")
    identity = result["intervention"].get("identity", {})
    for event in view["l0"]:
        if event["active_bundle"] != result["bundle_id"] or any(
            event[key] != identity[key] for key in ("run_id", "episode_id") if key in identity
        ):
            raise ValueError("relation feedback crosses development execution")
    return view


def learn_relations(tools, destination, learning_id, result_ids, *, max_feedback=2048, relations=None):
    """Update empirical means in a separate store, with an explicit sample bound.

    Replay is idempotent by result/event identity. Existing learned statistics
    and source identities are inherited, so another round cannot silently reset
    its denominator. No model, Runtime, payload or formal namespace is read.
    """
    if destination.root.resolve() == tools.store.root.resolve():
        raise ValueError("relation learning requires a separate derived store")
    if type(max_feedback) is not int or not 1 <= max_feedback <= 10000:
        raise ValueError("max_feedback must be an integer within 1..10000")
    selected = list(relations if relations is not None else ("rhoA", "rhoB", "rhoC"))
    if (not selected or len(set(selected)) != len(selected) or
            not set(selected) <= {"rhoA", "rhoB", "rhoC"}):
        raise ValueError("select a nonempty unique subset of rhoA/B/C")
    results = tools.query_development(result_ids)
    path = destination.path("relation_learning", learning_id)
    if path.exists():
        state = destination.get("relation_learning", learning_id)
        if (state["base_bundle_id"] != tools.base.bundle_id or
                state["max_feedback"] != max_feedback or state["frozen"] or
                state.get("relations", ["rhoA", "rhoB", "rhoC"]) != selected):
            raise ValueError("learning identity is frozen or belongs to another base/bound")
    else:
        before = tools.store.get("rho", tools.base.rho_ref)
        state = {"schema": SCHEMA, "learning_id": learning_id,
                 "base_bundle_id": tools.base.bundle_id, "parent_rho_ref": tools.base.rho_ref,
                 "protocol_id": tools.base.protocol_id, "max_feedback": max_feedback, "relations": selected,
                 "frozen": False, "feedback": [], "missing": deepcopy(before.get("missing", [])),
                 "snapshot": deepcopy(before)}
        for name in selected:
            if name == "rhoC" and before[name].get("type") == CONTEXT_SCHEMA:
                continue
            if before[name].get("type") != SCHEMA:
                state["snapshot"][name] = {"type": SCHEMA, "values": {}}
            elif before.get("schema") != SCHEMA:
                raise ValueError("learned relation snapshot schema mismatch")
        state["feedback"] = deepcopy(before.get("feedback", []))
    seen = {row["feedback_id"] for row in state["feedback"]}

    def add(key, relation, context, action, reward, refs, **extra):
        if key in seen:
            return
        if len(state["feedback"]) >= max_feedback:
            raise ValueError("relation feedback bound exhausted")
        _sample(state["snapshot"][relation], context, action, reward)
        state["feedback"].append({"feedback_id": key, "relation": relation,
            "state": context, "action": action, "target": reward, "source_refs": refs, **extra})
        seen.add(key)

    for result in results:
        view = _view(tools, result)
        if view is None:
            state["missing"].append({"result_id": result["result_id"], "reason": "no development L0 view"})
            continue
        result_key = result["result_id"]
        source_key = result["intervention"].get("identity", {}).get("run_id", result_key)
        media_events = [e for e in view["l0"] if e["media"]]
        available = {m for e in media_events for m in e["media"]}
        for action in (("all_available", "one_per_event") if "rhoA" in selected else ()):
            if not available:
                break
            chosen = (available if action == "all_available" else {min(e["media"]) for e in media_events})
            coverage = sum(bool(chosen.intersection(e["media"])) for e in media_events) / len(media_events)
            reward = coverage - .1 * len(chosen) / len(available)
            add(source_key + ":A:" + action, "rhoA", "media", action, reward,
                [view["view_id"]], criterion="recorded_event_coverage_minus_reference_cost",
                coverage=coverage, retained_references=len(chosen), available_references=len(available))
        events = {event["event_id"]: event for event in view["l0"]}
        signature = version_signature(RunBundle(**tools.store.get("bundles", result["bundle_id"])))
        count_b = 0
        for event in (view["l0"] if "rhoB" in selected else []):
            feedback = event["flags"].get("route_feedback")
            if feedback is None:
                continue
            if (feedback.get("evidence_kind") != "independent_criterion" or
                    not feedback.get("criterion_ref") or type(feedback.get("success")) is not bool or
                    feedback.get("version_signature") != signature or
                    not isinstance(feedback.get("worker"), str) or not feedback["worker"] or
                    not isinstance(feedback.get("state"), str) or not feedback["state"]):
                raise ValueError("route feedback needs an independent criterion and matching version")
            refs = [feedback.get("start_event_id"), feedback.get("handoff_event_id")]
            if any(ref not in events for ref in refs) or refs[0] == refs[1]:
                raise ValueError("route feedback needs recorded skill start and handoff")
            if not events[refs[0]]["sequence"] < events[refs[1]]["sequence"] <= event["sequence"]:
                raise ValueError("route feedback has invalid start/handoff ordering")
            context = feedback["state"]
            bindings = state["snapshot"]["rhoB"].setdefault("version_signatures", {})
            if context in bindings and bindings[context] != signature:
                raise ValueError("route state mixes incompatible skill versions")
            bindings[context] = signature
            add(source_key + ":B:" + event["event_id"], "rhoB", context, feedback["worker"],
                float(feedback["success"]), refs + [event["event_id"]],
                criterion=feedback["criterion_ref"], version_signature=signature)
            count_b += 1
        if "rhoB" in selected and not count_b:
            state["missing"].append({"result_id": result_key, "relation": "rhoB",
                "reason": "no independent worker/start/handoff feedback; terminal outcome is not route supervision"})
        screening = result["intervention"].get("screening_checks")
        if "rhoC" not in selected or state["snapshot"]["rhoC"].get("type") == CONTEXT_SCHEMA:
            continue
        if result["intervention"].get("runtime_loaded") is True and isinstance(screening, dict) and screening:
            known = all(type(value) is bool for value in screening.values())
            if known:
                for assignment in result["assignments"]:
                    if assignment["validity"] != "valid":
                        continue
                    add(source_key + ":C:" + assignment["task_key"], "rhoC", assignment["task_key"],
                        "bounded_experiment", float(all(screening.values())), [result_key],
                        criterion="observed_development_screening_yield", attribution="observational_not_causal")
        else:
            state["missing"].append({"result_id": result_key, "relation": "rhoC",
                "reason": "no loaded trial with explicit development screening feedback"})
    if "rhoC" in selected and state["snapshot"]["rhoC"].get("type") == CONTEXT_SCHEMA:
        policy = state["snapshot"]["rhoC"]
        reservations = {row["selection_id"]: row for row in policy.get("exploration_reservations", [])}
        for row in tools.store.all("research_relation_selections"):
            if (row.get("rho_ref") == tools.base.rho_ref and row.get("protocol_id") == tools.protocol["protocol_id"]
                    and row.get("exploration_count", 0)):
                reservations[row["selection_id"]] = {"selection_id": row["selection_id"],
                    "exploration_count": row["exploration_count"], "protocol_id": row["protocol_id"]}
        policy["exploration_reservations"] = list(reservations.values())
        for row in tools.store.all("contextual_research_feedback"):
            if row.get("base_bundle_id") != tools.base.bundle_id or row.get("protocol_id") != tools.protocol["protocol_id"]:
                continue
            key = row["feedback_id"]
            if key in seen:
                continue
            if len(state["feedback"]) >= max_feedback:
                raise ValueError("relation feedback bound exhausted")
            policy = state["snapshot"]["rhoC"]
            if row["context"]["data_context"] != policy["data_context"]:
                continue
            values = policy["values"].setdefault(row["context"]["key"], {})
            for dimension in ("technical_valid", "screening_pass", "old_capability"):
                value = row["feedback"][dimension]
                if dimension == "screening_pass" and row["feedback"]["technical_valid"] is not True:
                    continue
                if value is not None:
                    item = values.setdefault(dimension, {"count": 0, "total": 0.0})
                    item["count"] += 1
                    item["total"] += float(value)
            policy.setdefault("feedback", []).append(deepcopy(row))
            state["feedback"].append({"feedback_id": key, "relation": "rhoC", "state": row["context"]["key"],
                "context": row["context"], "feedback": row["feedback"], "source_refs": row["source_refs"],
                "criterion": "contextual_development_feedback_v2", "attribution": "observational_not_causal"})
            seen.add(key)
    state["missing"] = [dict(items) for items in dict.fromkeys(tuple(sorted(row.items())) for row in state["missing"])]
    atomic_json(path, state)
    return state


def freeze_relations(tools, destination, learning_id, *, rho_ref, bundle_id, development_trial=False):
    """Freeze a new rho and child bundle in the derived store, without activation.

    The existing bundle, psi, phi and active version remain owned by their normal
    consumers. The caller marks an offline replacement or a development trial;
    neither operation submits a Candidate or activates a version.
    """
    if destination.root.resolve() == tools.store.root.resolve():
        raise ValueError("relation freezing requires a separate derived store")
    state = destination.get("relation_learning", learning_id)
    if state["base_bundle_id"] != tools.base.bundle_id or not state["feedback"]:
        raise ValueError("freezing needs feedback at the immutable learning base")
    if state["frozen"] and state["replacement"] != {"rho_ref": rho_ref, "bundle_id": bundle_id}:
        raise ValueError("frozen relation replacement identity cannot change")
    if rho_ref == tools.base.rho_ref or bundle_id == tools.base.bundle_id:
        raise ValueError("relation replacement requires new snapshot and bundle identities")
    snapshot = {**state["snapshot"], "schema": SCHEMA, "learned": True,
                "parent_rho_ref": state["parent_rho_ref"], "feedback": state["feedback"],
                "missing": state["missing"],
                "status": "frozen_development_trial" if development_trial else "frozen_offline_policy",
                "protocol_id": state["protocol_id"]}
    diff = {"base_bundle_id": tools.base.bundle_id, "bundle_id": bundle_id,
            "changes": {"rho_ref": rho_ref}, "parent_rho_ref": tools.base.rho_ref,
            "learning_id": learning_id,
            "scope": "development_policy_trial" if development_trial else "offline_policy_replacement",
            "consumers": ["EventLedger.view", "ResearchTools.relation_evidence_views",
                          "ResearchTools.prepare_relation_step"]}
    target = replace(tools.base, bundle_id=bundle_id, rho_ref=rho_ref,
                     parent_bundle_id=tools.base.bundle_id,
                     diff_ref=str(destination.path("relation_diffs", learning_id).resolve()))
    destination.put("relation_diffs", learning_id, diff)
    destination.put("rho", rho_ref, snapshot)
    for kind, ref in (("rho", tools.base.rho_ref), ("psi", tools.base.psi_ref), ("memory", tools.base.memory_ref)):
        destination.put(kind, ref, tools.store.get(kind, ref))
    destination.save_bundle(tools.base)
    destination.save_bundle(target)
    state["frozen"] = True
    state["replacement"] = {"rho_ref": rho_ref, "bundle_id": bundle_id}
    atomic_json(destination.path("relation_learning", learning_id), state)
    return target


def relation_policy(tools):
    """Read only an explicitly enabled new-protocol finite relation operator."""
    from .task_policy import task_policy
    policy = task_policy(tools.protocol).get("relation_learning")
    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        raise ValueError("relation learning needs explicit new-protocol opt-in")
    relations = policy.get("relations")
    bound = policy.get("max_feedback")
    if (not isinstance(relations, list) or not relations or any(not isinstance(name, str) for name in relations)
            or len(set(relations)) != len(relations)
            or not set(relations) <= {"rhoA", "rhoB", "rhoC"}
            or type(bound) is not int or not 1 <= bound <= 10000):
        raise ValueError("relation policy needs finite allowed relations and feedback bound")
    if "relation_feedback_update" not in tools.domain_policy.get("rule_operators", []):
        raise ValueError("relation operator outside human domain policy")
    return dict(policy)


def preview_relation_feedback(tools, result_ids):
    """Expose supported new feedback, never model-authored strategy parameters."""
    policy = relation_policy(tools)
    before = tools.store.get("rho", tools.base.rho_ref)
    old = {row["feedback_id"] for row in before.get("feedback", [])}
    from .storage import ResearchStore
    with tempfile.TemporaryDirectory(prefix="rsi-relation-preview-") as tmp:
        state = learn_relations(tools, ResearchStore(tmp), "preview", result_ids,
                                max_feedback=policy["max_feedback"], relations=policy["relations"])
    counts = {name: sum(row["relation"] == name and row["feedback_id"] not in old
                        for row in state["feedback"]) for name in policy["relations"]}
    supported = [name for name in policy["relations"] if counts[name]]
    if "rhoB" in supported and not any(len(actions) >= 2 for actions in state["snapshot"]["rhoB"]["values"].values()):
        supported.remove("rhoB")
    return {"supported_relations": supported, "new_feedback_counts": counts,
            "max_feedback": policy["max_feedback"], "missing": state["missing"],
            "feedback_origin": "program_recomputed_legal_development_only"}


def write_relation_trial(tools, *, experiment_id, bundle_id, result_ids, relations):
    """Materialize a finite model-selected relation update using program feedback."""
    from .storage import ResearchStore, identifier
    policy = relation_policy(tools)
    if (not isinstance(relations, list) or not relations or any(not isinstance(name, str) for name in relations)
            or len(set(relations)) != len(relations)
            or not set(relations) <= set(policy["relations"])):
        raise ValueError("relation selection outside the registered finite operator")
    supported = preview_relation_feedback(tools, result_ids)["supported_relations"]
    if not set(relations) <= set(supported):
        raise ValueError("selected relation has no new supported development feedback")
    derived = ResearchStore(tools.store.root / "relation_trial_artifacts" / identifier(experiment_id))
    state_path = derived.path("relation_learning", experiment_id)
    if not state_path.exists():
        learn_relations(tools, derived, experiment_id, result_ids,
                        max_feedback=policy["max_feedback"], relations=relations)
    target = freeze_relations(tools, derived, experiment_id, rho_ref=experiment_id + "-rho",
                              bundle_id=bundle_id, development_trial=True)
    snapshot = derived.get("rho", target.rho_ref)
    tools.store.put("rho", target.rho_ref, snapshot)
    tools.store.save_bundle(target)
    return target, {"rho_ref": target.rho_ref, "diff_ref": target.diff_ref}
