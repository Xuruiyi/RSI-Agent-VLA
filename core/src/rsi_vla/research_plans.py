# Copyright 2026 RSI-Agent-VLA Authors.
"""Bounded Research proposals over trusted existing operation templates."""

from copy import deepcopy
from math import isfinite

from .storage import identifier

KINDS = {
    "failure_unknown": "failure",
    "inefficient_success": "normal",
    "robustness_retention": "capability_boundary",
    "new_capability": "new_goal",
}


def _costs(value):
    if (
        not isinstance(value, dict)
        or not value
        or any(
            type(v) not in (int, float) or not isfinite(v) or v < 0
            for v in value.values()
        )
    ):
        raise ValueError("plan needs finite nonnegative registered costs")
    return dict(value)


def discover_research_opportunity(
    tools,
    opportunity_id,
    *,
    kind,
    result_ids,
    external_value,
    hypotheses,
    expected_cost,
    target=None,
):
    """Keep four opportunity purposes on the existing legal development boundary."""
    if kind not in KINDS:
        raise ValueError("unknown research opportunity kind")
    value = tools.discover_opportunity(
        opportunity_id,
        source=KINDS[kind],
        result_ids=result_ids,
        external_value=external_value,
        competing_explanations=hypotheses,
        expected_cost=expected_cost,
        target=target,
    )
    return tools._save(
        "research_opportunity_kinds",
        opportunity_id,
        {
            "opportunity_id": opportunity_id,
            "kind": kind,
            "source": value["source"],
            "capability_gain": None,
        },
    )


def consume_research_plan(tools, proposal, *, policy):
    """Validate model choices and register fixed operations without granting authority.

    ``policy`` is host-owned. Its capabilities contain complete existing operation
    templates, legal input references, maximum costs and expected artifact names.
    The model can select those names; it cannot provide commands or permissions.
    """
    fields = {
        "plan_id",
        "opportunity_id",
        "base_bundle_id",
        "parent_plan_id",
        "steps",
        "distinguishing_prediction",
        "stop_condition",
        "requires_visual_evidence",
    }
    if not isinstance(proposal, dict) or set(proposal) != fields:
        raise ValueError("invalid bounded plan fields")
    identity = identifier(proposal["plan_id"])
    if tools.store.path("research_plans", identity).exists():
        saved = tools._owned("research_plans", identity)
        if saved["proposal"] != proposal or saved["policy"] != policy:
            raise ValueError("conflicting plan identity")
        return saved
    opportunity = tools._owned("opportunities", proposal["opportunity_id"])
    if proposal["base_bundle_id"] != tools.base.bundle_id:
        raise ValueError("plan starting bundle differs from research job")
    for name in ("distinguishing_prediction", "stop_condition"):
        if not isinstance(proposal[name], str) or not proposal[name].strip():
            raise ValueError("plan needs prediction and stop condition")
    if type(proposal["requires_visual_evidence"]) is not bool:
        raise ValueError("visual evidence requirement must be explicit")
    steps = proposal["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= policy["max_steps"]:
        raise ValueError("plan exceeds registered finite step bound")
    parent = proposal["parent_plan_id"]
    depth = 1
    if parent is not None:
        old = tools._owned("research_plans", parent)
        if old["proposal"]["opportunity_id"] != proposal["opportunity_id"]:
            raise ValueError("plan revision belongs to another opportunity")
        if not tools.store.path("research_plan_feedback", parent).exists():
            raise ValueError("reconcile original plan before revision")
        feedback = tools._owned("research_plan_feedback", parent)
        if feedback["action"] == "reconcile":
            raise ValueError(
                "unknown original effects must be reconciled before revision"
            )
        depth = old["version"] + 1
        if any(
            p["proposal"]["parent_plan_id"] == parent
            for p in tools.store.all("research_plans")
        ):
            raise ValueError("plan already has a successor")
    if depth > policy["max_versions"]:
        raise ValueError("plan revision bound exhausted")
    operations, total, names = [], {}, set()
    training = False
    for step in steps:
        if not isinstance(step, dict) or set(step) != {
            "step_id",
            "capability",
            "input_refs",
            "cost_limit",
            "expected_artifacts",
        }:
            raise ValueError("invalid plan step fields")
        name = identifier(step["step_id"])
        if name in names:
            raise ValueError("duplicate plan step")
        names.add(name)
        capability = policy["capabilities"].get(step["capability"])
        if capability is None:
            raise ValueError("unsupported research capability")
        if not isinstance(step["input_refs"], list) or not set(
            step["input_refs"]
        ) <= set(capability["input_refs"]):
            raise ValueError("plan dependency outside trusted capability")
        if step["expected_artifacts"] != capability["expected_artifacts"]:
            raise ValueError("plan artifacts differ from trusted consumer")
        costs = _costs(step["cost_limit"])
        if any(v > capability["cost_limit"].get(k, -1) for k, v in costs.items()):
            raise ValueError("plan exceeds capability cost bound")
        for key, value in costs.items():
            total[key] = total.get(key, 0) + value
        training |= capability.get("training", False) is True
        operation = deepcopy(capability["operation"])
        operation["operation_id"] = identity + "-" + name
        operations.append(operation)
    unknown_effects = []
    for evidence in tools.query_development(opportunity["result_ids"]):
        view_id = evidence["intervention"].get("evidence_view_id")
        if view_id:
            view = tools.store.get("evidence_views", view_id)
            if view["bundle_id"] != evidence["bundle_id"]:
                raise ValueError("opportunity evidence view belongs to another bundle")
            unknown_effects.extend(
                e["event_id"]
                for e in view["l0"]
                if e.get("action_feedback_unknown") is True
                or (e.get("feedback") or {}).get("action_feedback_unknown") is True
                or e.get("flags", {}).get("action_feedback_unknown") is True
                or e.get("next_action") == "reconcile_execution"
            )
    action, gaps = "registered", []
    if policy.get("unresolved_operations") or unknown_effects:
        action, gaps = "reconcile", ["original operation effects and costs"]
    elif (
        proposal["requires_visual_evidence"]
        and policy.get("payload_modality", "text_only") == "text_only"
    ):
        action, gaps = (
            "defer",
            ["required visual content absent from actual Research payload"],
        )
    elif training and policy.get("training_dispatch_allowed") is not True:
        action, gaps = "defer", ["training dispatch is outside current authorization"]
    elif tools.job.state["status"] != "running" or any(
        v > tools.job.remaining().get(k, -1) for k, v in total.items()
    ):
        action, gaps = "defer", ["remaining registered job budget"]
    if (
        action != "reconcile"
        and opportunity.get("admission", {}).get("action") == "defer"
    ):
        action, gaps = "defer", opportunity["admission"]["missing_evidence"]
    return tools._save(
        "research_plans",
        identity,
        {
            "plan_id": identity,
            "proposal": deepcopy(proposal),
            "policy": deepcopy(policy),
            "version": depth,
            "action": action,
            "missing_evidence": gaps,
            "operations": operations if action == "registered" else [],
            "cost_limit": total,
            "next_action": "dispatch_registered_operations"
            if action == "registered"
            else action,
            "reconcile_operations": list(policy.get("unresolved_operations", [])),
            "unknown_effect_event_ids": unknown_effects,
            "execution_authorized": False,
            "hypotheses_resolved": False,
        },
    )


def record_plan_feedback(tools, plan_id, *, receipts):
    """Return actual operation receipts to research; unknown effects stay unresolved."""
    plan = tools._owned("research_plans", plan_id)
    if plan["action"] != "registered":
        raise ValueError("feedback requires a registered plan")
    expected = {op["operation_id"] for op in plan["operations"]}
    if (
        not isinstance(receipts, list)
        or {r.get("operation_id") for r in receipts} != expected
        or len(receipts) != len(expected)
    ):
        raise ValueError("feedback requires the original operation denominator")
    import json
    from pathlib import Path

    for receipt in receipts:
        path = Path(receipt["receipt_ref"])
        stored = json.loads(path.read_text())
        operation = next(
            op
            for op in plan["operations"]
            if op["operation_id"] == receipt["operation_id"]
        )
        if (
            stored.get("operation") != operation
            or stored.get("status") != receipt["status"]
        ):
            raise ValueError("feedback differs from original operation receipt")
        if receipt.get("value") != stored.get("value"):
            raise ValueError("feedback value differs from original receipt")
    unknown = any(r["status"] != "completed" for r in receipts)
    record = {
        "plan_id": plan_id,
        "receipts": receipts,
        "action": "reconcile" if unknown else "research_followup",
        "hypotheses_resolved": False,
        "capability_gain": None,
        "next_request_requires_new_identity": True,
    }
    if unknown:
        previous = [
            r
            for r in tools.store.all("research_plan_observations")
            if r.get("plan_id") == plan_id
        ]
        for row in previous:
            if row["receipts"] == receipts:
                return row
        return tools._save(
            "research_plan_observations",
            plan_id + "-unknown-" + str(len(previous)),
            record,
        )
    return tools._save("research_plan_feedback", plan_id, record)
