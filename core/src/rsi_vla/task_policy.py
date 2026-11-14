"""Opt-in policies for new tasks; frozen comparison consumers keep their rules."""

from math import isfinite


def number(value, name):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("finite nonnegative value required: " + name)
    return value


def task_policy(protocol):
    """Read a versioned task policy without reinterpreting protocol10 or its projection."""
    policy = protocol.get("task_policy")
    if policy is None:
        return {}
    if (
        protocol.get("scope_id") == "continuation10-20261113"
        or protocol.get("protocol_id")
        == "rsi-libero-pro-improvement-protocol10-public-seeds01"
    ):
        raise ValueError(
            "new task policy requires a new protocol identity, not protocol10"
        )
    if policy.get("schema") != "rsi-task-policy-v1":
        raise ValueError("unsupported task policy schema")
    return policy


def admit_stage(phase, *, required, cleanup, reserve):
    """Admit the next necessary stage and disjoint mandatory reserves, without charging.

    Vectors use the phase ledger's dimensions. Required includes only the next
    stage; cleanup and reserve must exclude costs already included in that stage.
    Unknown hard costs still need reconciliation by their owning ledger.
    """
    if phase.get("unresolved_costs") != []:
        raise ValueError("phase hard costs unresolved or unreported")
    dimensions = set(phase["limits"])
    if not required or any(
        not set(v) <= dimensions for v in (required, cleanup, reserve)
    ):
        raise ValueError("stage vectors must use registered phase dimensions")
    needed, shortfall = {}, {}
    for key in dimensions:
        cap = number(phase["limits"][key], key)
        used = number(phase["used"][key], key)
        left = number(phase["remaining"][key], key)
        if abs(cap - used - left) > 1e-6:
            raise ValueError("phase arithmetic inconsistent: " + key)
        needed[key] = sum(
            number(v.get(key, 0), key) for v in (required, cleanup, reserve)
        )
        shortfall[key] = max(0, needed[key] - left)
    return {
        "resource_fit": not any(shortfall.values()),
        "additional_required": shortfall,
        "required_now": needed,
        "scope_completion_guaranteed": False,
        "phase_unchanged": True,
    }


def research_branches(protocol, *, j, capability_established):
    """Choose declared research objectives; this never grants execution or release."""
    branches = task_policy(protocol).get("research_branches", [])
    rows = []
    for branch in branches:
        objective = branch["objective"]
        if objective not in {
            "success_gain",
            "efficiency",
            "robustness",
            "new_capability",
            "repair",
        }:
            raise ValueError("unsupported research objective")
        if not branch.get("criterion_ref"):
            raise ValueError("research branch needs a testable criterion")
        allowed = (objective != "success_gain" or j < 1) and (
            capability_established or objective in {"repair", "new_capability"}
        )
        rows.append({**branch, "research_allowed": allowed,
                     "evaluation_ref": branch.get("evaluation_ref")})
    return rows


def exploration_policy(tools, operator=None):
    """Return a registered hypothesis-collection scope, never a dynamic permit."""
    if tools.domain_policy.get("research_mode") != "exploratory":
        return None
    policy = task_policy(tools.protocol).get("exploration")
    if not policy or not all(
        policy.get(k)
        for k in (
            "hypothesis",
            "criterion_ref",
            "stop_condition",
            "assignments",
            "cost_limit",
        )
    ):
        raise ValueError(
            "exploration needs a registered hypothesis, criterion, slots, stop and budget"
        )
    if policy["criterion_ref"] != tools.protocol["protocol_id"]:
        raise ValueError("exploration criterion must belong to this new protocol")
    split = tools.protocol["task_sets"]["development"]
    if any(
        row["task_key"] not in split["task_keys"] or row["seed"] not in split["seeds"]
        for row in policy["assignments"]
    ):
        raise ValueError("exploration cannot insert assignments outside development")
    for key, value in policy["cost_limit"].items():
        if number(value, key) > tools.job.remaining().get(key, -1):
            raise ValueError("exploration exceeds original job budget")
    if operator is not None and operator not in policy.get("operators", []):
        return None
    return policy


def select_gpu(cards, policy):
    """Select from fresh telemetry using an explicit service memory/compatibility policy."""
    required = number(policy["required_memory_mib"], "required_memory_mib")
    margin = number(policy["safety_margin_mib"], "safety_margin_mib")
    minimum = number(policy["minimum_compute_capability"], "minimum_compute_capability")
    names = policy.get("compatible_names")
    eligible = []
    for card in cards:
        free = card.get("free_memory_mib")
        capability = card.get("compute_capability")
        if free is None or capability is None:
            continue
        if names is not None and card.get("name") not in names:
            continue
        try:
            compute = float(capability)
        except (TypeError, ValueError):
            continue
        if not isfinite(compute) or compute < minimum:
            continue
        if number(free, "free_memory_mib") < required + margin:
            continue
        load = card.get("utilization_percent")
        eligible.append(
            (load if load is not None else 101, -free, card["physical_gpu"])
        )
    return min(eligible)[2] if eligible else None


def resource_wait(policy, *, waited_seconds, offline_remaining_seconds):
    """Return a bounded resource checkpoint; waiting never cancels or unpauses a task."""
    maximum = number(policy["max_wait_seconds"], "max_wait_seconds")
    interval = number(policy["check_interval_seconds"], "check_interval_seconds")
    if interval == 0:
        raise ValueError("resource check interval must be positive")
    waited = number(waited_seconds, "waited_seconds")
    available = number(offline_remaining_seconds, "offline_remaining_seconds")
    delay = min(interval, max(0, maximum - waited), available)
    return {
        "state": "waiting_resources",
        "next_check_seconds": delay,
        "wait_budget_exhausted": delay == 0,
        "task_closed": False,
        "resume_requires": "fresh resources, valid authorization and no manual pause/cancel",
    }


def review_goal_value(protocol, opportunity, evidence, *, policy):
    """Recompute domain value from independently supplied development measurements.

    The evaluation role owns the policy and measurement sources. This pure
    consumer checks their bindings; its dictionary API is not a role boundary.
    A value confirmation permits preparation for a future evaluation protocol,
    and establishes neither a capability gain nor release eligibility.
    """
    if not task_policy(protocol) or opportunity.get("source") != "new_goal":
        raise ValueError("goal value review requires a new-task goal opportunity")
    target = opportunity["target"]
    if (
        opportunity.get("protocol_id") != protocol["protocol_id"]
        or target["domain"] != policy["domain"]
        or target["kind"] not in policy["allowed_goal_kinds"]
        or target["value_criterion_ref"] != policy["value_criterion_ref"]
        or target["independent_value_test_ref"] != policy["independent_value_test_ref"]
        or opportunity["goal_review"]["policy_ref"] != policy["policy_ref"]
    ):
        raise ValueError("goal does not match the independent domain value policy")
    if (
        evidence.get("schema") != "rsi-goal-value-evidence-v1"
        or evidence.get("protocol_id") != protocol["protocol_id"]
        or evidence.get("opportunity_id") != opportunity["opportunity_id"]
        or evidence.get("target") != target
        or evidence.get("criterion_ref") != policy["value_criterion_ref"]
        or evidence.get("test_ref") != policy["independent_value_test_ref"]
        or evidence.get("information_scope") != "development"
    ):
        raise ValueError("independent goal value evidence binding mismatch")
    refs = evidence.get("evidence_refs")
    if not isinstance(refs, list) or not refs or any(
        not isinstance(ref, str) or not ref.strip() for ref in refs
    ):
        raise ValueError("independent goal value review needs measurement sources")
    minimum = number(policy["minimum"], "goal value minimum")
    metric = policy["metric"]
    measured = evidence.get("metrics", {}).get(metric)
    validity = evidence.get("validity")
    if validity not in {"valid", "invalid", "unknown"}:
        raise ValueError("invalid goal value measurement validity")
    if measured is not None:
        measured = number(measured, metric)
    status = "inconclusive"
    if validity == "valid" and measured is not None:
        status = "confirmed" if measured >= minimum else "rejected"
    return {
        "schema": "rsi-goal-value-review-v1",
        "role": "evaluation",
        "protocol_id": protocol["protocol_id"],
        "opportunity_id": opportunity["opportunity_id"],
        "target": dict(target),
        "policy_ref": policy["policy_ref"],
        "value_criterion_ref": policy["value_criterion_ref"],
        "independent_value_test_ref": policy["independent_value_test_ref"],
        "value_status": status,
        "metric": metric,
        "observed": measured,
        "minimum": minimum,
        "evidence_refs": list(refs),
        "capability_gain": None,
        "requires_new_evaluation_protocol": True,
        "release_eligible": False,
    }


def opportunity_admission(protocol, opportunity, remaining_budget):
    """Retain an opportunity and decide preparation against its original job budget.

    This checks the proposed next action only. It does not charge, issue a
    permit, change a formal matrix, or treat a domain-value claim as verified.
    New goals are handed to independent value review before experiment design.
    """
    if not task_policy(protocol):
        raise ValueError("opportunity admission requires an explicit new task policy")
    expected = opportunity["expected_cost"]
    shortfall, unregistered = {}, []
    for dimension, value in expected.items():
        required = number(value, dimension)
        if dimension not in remaining_budget:
            unregistered.append(dimension)
        left = number(remaining_budget.get(dimension, 0), dimension)
        shortfall[dimension] = max(0, required - left)
    missing = []
    if any(shortfall.values()) or unregistered:
        missing.append("remaining job budget cannot admit the proposed next action")
    if opportunity["source"] == "new_goal":
        missing.append("independent domain value review and a new experiment protocol")
    return {
        "action": "defer" if missing else "prepare",
        "resource_fit": not any(shortfall.values()) and not unregistered,
        "additional_required": shortfall,
        "unregistered_cost_dimensions": unregistered,
        "missing_evidence": missing,
        "remaining_budget": dict(remaining_budget),
        "value_status": "unverified" if opportunity["source"] == "new_goal" else None,
        "execution_authorized": False,
        "release_eligible": False,
    }
