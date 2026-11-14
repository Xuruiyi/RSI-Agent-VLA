"""Permission-preserving consumers for fixed and feedback-learned relations."""


def fixed_route(rho_snapshot, requested_worker, allowed_workers, *, state=None,
                version_signature=None):
    """Route measured compatible skills within the explicitly authorized workers."""
    from .relation_learning import SCHEMA, best_action
    policy = rho_snapshot["rhoB"]
    if policy.get("type") == SCHEMA:
        if (state is not None and version_signature is not None and
                policy.get("version_signatures", {}).get(state) == version_signature):
            ordered = ([requested_worker] if requested_worker in allowed_workers else [])
            ordered += sorted(set(allowed_workers) - set(ordered))
            action = best_action(policy, state, ordered, minimum_actions=2)
            if action is not None:
                value = policy["values"][state][action]
                if value["total"] / value["count"] > .5:
                    return action
    elif policy.get("type") != "fixed" or policy.get("route") != "existing_authorized_worker":
        raise ValueError("unsupported rhoB policy")
    return requested_worker if requested_worker in allowed_workers else "defer"


def fixed_research_step(rho_snapshot, job, *, evidence_available, legal_repairs=(),
                        inspected=False, experiment_permitted=False, state=None):
    """Inspect then use observed screening yield, under the unchanged job budget."""
    from .relation_learning import SCHEMA
    policy = rho_snapshot["rhoC"]
    if policy.get("type") != SCHEMA and (policy.get("type") != "fixed" or
            policy.get("order") != ["inspect", "bounded_experiment", "defer"]):
        raise ValueError("unsupported rhoC policy")
    if job.state["status"] != "running" or not evidence_available:
        return "defer"
    if not inspected:
        return "inspect"
    if not experiment_permitted or not legal_repairs:
        return "defer"
    if policy.get("type") == SCHEMA:
        value = policy.get("values", {}).get(state, {}).get("bounded_experiment")
        if value and value["count"] and value["total"] / value["count"] <= .5:
            return "defer"
    return "bounded_experiment"



DRAWER_CRITERION = "rsi-skill-drawer-open-criterion-v1"


def skill_route_workers(criterion):
    """Validate the registered goal criterion and return its compatible skills."""
    from math import isfinite
    if criterion.get("schema") == DRAWER_CRITERION:
        expected = {"region": "wooden_cabinet_1_middle_region", "fixture": "wooden_cabinet_1",
                    "joint": "wooden_cabinet_1_middle_level", "joint_type": "slide",
                    "joint_range": [-0.16, 0.01], "open_threshold": -0.14}
        if not criterion.get("criterion_ref") or any(criterion.get(key) != value for key, value in expected.items()):
            raise ValueError("drawer criterion must bind the registered middle slide joint and strict Open threshold")
        return {"vla_execute", "pi0_doubled"}
    if (criterion.get("schema") != "rsi-skill-lift-criterion-v1" or not criterion.get("criterion_ref")
            or not isinstance(criterion.get("object_position_key"), str)
            or not criterion["object_position_key"].endswith("_pos")):
        raise ValueError("skill route needs a registered independent lift or drawer criterion")
    for field in ("min_lift_m", "max_eef_object_distance_m", "min_gripper_opening", "max_gripper_opening"):
        value = criterion[field]
        if type(value) not in (float, int) or not isfinite(value) or value < 0:
            raise ValueError("skill lift criterion needs finite nonnegative thresholds")
    if (criterion["min_lift_m"] <= 0 or criterion["max_eef_object_distance_m"] <= 0 or
            criterion["min_gripper_opening"] > criterion["max_gripper_opening"]):
        raise ValueError("skill lift criterion thresholds are inconsistent")
    return {"pi0_pick", "vla_execute"}


def _review_drawer_handoff(start, handoff, criterion):
    """Use two private scalar joint readings; an already-open start is ineligible."""
    from math import isfinite
    values, missing = [], []
    fields = ("region", "fixture", "joint", "joint_type", "joint_range")
    for phase, event in (("start", start), ("handoff", handoff)):
        observed = event.get("measurements", {}).get("joint") or {}
        for field in fields:
            if observed.get(field) != criterion[field]:
                missing.append(phase + ":" + field)
        if observed.get("schema") != "rsi-private-drawer-joint-observation-v1" or observed.get("status") != "recorded":
            missing.append(phase + ":drawer_joint_observation")
        qpos = observed.get("qpos")
        if type(qpos) not in (int, float) or not isfinite(qpos):
            missing.append(phase + ":qpos")
        if observed.get("missing"):
            missing.extend(phase + ":" + str(item) for item in observed["missing"])
        values.append(qpos)
    checks = {"start_not_open": None, "handoff_open": None}
    eligibility, success, progress = "unknown", None, None
    if not missing:
        before, after = values
        checks = {"start_not_open": before >= criterion["open_threshold"],
                  "handoff_open": after < criterion["open_threshold"]}
        progress = before - after
        if not checks["start_not_open"]:
            eligibility = "not_eligible_start_already_open"
        else:
            eligibility, success = "eligible", checks["handoff_open"]
    return {"success": success, "checks": checks, "missing": missing, "eligibility": eligibility,
            "joint_progress": progress, "criterion_ref": criterion["criterion_ref"],
            "source_sequences": [start["sequence"], handoff["sequence"]],
            "measurement_scope": "observational_drawer_open_only"}


def prepare_skill_routes(rho_snapshot, *, rho_ref, version_signature, registered_routes):
    """Project registered same-goal LIBERO skill choices into the existing Runtime.

    Only registered lift/drawer compatibility groups participate. Goal text, observation
    criterion and permitted skills remain registered inputs; rho cannot invent
    tools, change an instruction, or increase an action allowance.
    """
    from copy import deepcopy
    from .relation_learning import VERSION_FIELDS

    if set(version_signature) != set(VERSION_FIELDS) or any(
        not isinstance(value, str) or not value for value in version_signature.values()
    ):
        raise ValueError("skill routing needs the bound executable version signature")
    routes, seen = [], set()
    for registered in registered_routes:
        route = deepcopy(registered)
        for field in ("route_id", "state", "instruction"):
            if not isinstance(route.get(field), str) or not route[field].strip():
                raise ValueError("skill route needs a registered identity, state and exact goal")
        if route["route_id"] in seen:
            raise ValueError("duplicate skill route identity")
        seen.add(route["route_id"])
        workers = skill_route_workers(route["criterion"])
        if route["criterion"]["schema"] == DRAWER_CRITERION and (
                route.get("task_key") != "old_drawer" or route["state"] != "old_drawer"
                or route.get("max_chunks") != 24 or type(route.get("max_chunks")) is not int):
            raise ValueError("drawer route must keep its registered old_drawer goal and 24 chunks")
        if route["requested_worker"] not in workers:
            raise ValueError("only registered compatible goal skills can be routed")
        triggers = route.setdefault("trigger_workers", [route["requested_worker"]])
        if (not isinstance(triggers, list) or not triggers or len(triggers) != len(set(triggers))
                or not set(triggers) <= workers):
            raise ValueError("skill route triggers must match its registered goal compatibility group")
        allowed = route["allowed_workers"]
        if (not isinstance(allowed, list) or len(allowed) != len(set(allowed))
                or not set(allowed) <= workers | {"handoff", "defer"}):
            raise ValueError("skill route exceeds its registered goal compatibility group")
        route["selected_worker"] = fixed_route(
            rho_snapshot, route["requested_worker"], allowed,
            state=route["state"], version_signature=version_signature,
        )
        routes.append(route)
    return {"schema": "rsi-skill-routing-v1", "rho_ref": rho_ref,
            "version_signature": dict(version_signature), "routes": routes}


def review_skill_handoff(start, handoff, criterion):
    """Evaluate private environment observations independently of skill labels.

    The caller must read the original environment-owned stream and the registered
    criterion. Lift geometry and drawer joint opening are separate observational
    conditions, not claims of causal grasping or native task completion.
    """
    from math import isfinite, sqrt

    identity = ("run_id", "episode_id", "bundle_id", "protocol_id", "route_id", "state",
                "worker", "client_route_sequence", "criterion_ref", "version_signature")
    if (start.get("kind") != "skill_route_observation" or
            handoff.get("kind") != "skill_route_observation" or
            start.get("phase") != "start" or handoff.get("phase") != "handoff" or
            any(start.get(key) is None or start.get(key) != handoff.get(key) for key in identity) or
            start.get("clock_domain") != "environment_service" or
            handoff.get("clock_domain") != "environment_service" or
            not start["sequence"] < handoff["sequence"] or
            start["criterion_ref"] != criterion["criterion_ref"] or
            (criterion.get("schema") == "rsi-skill-lift-criterion-v1" and any(
                event.get("object_position_key") != criterion["object_position_key"] for event in (start, handoff)))):
        raise ValueError("independent skill observations have conflicting identity or criterion")
    skill_route_workers(criterion)
    if criterion["schema"] == DRAWER_CRITERION:
        return _review_drawer_handoff(start, handoff, criterion)
    measurements = [event.get("measurements", {}) for event in (start, handoff)]
    missing = []
    for phase, value in zip(("start", "handoff"), measurements):
        for key in ("object_pos", "eef_pos"):
            xyz = value.get(key)
            if (not isinstance(xyz, list) or len(xyz) != 3 or
                    any(type(v) not in (int, float) or not isfinite(v) for v in xyz)):
                missing.append(phase + ":" + key)
        gap = value.get("gripper_opening")
        if type(gap) not in (int, float) or not isfinite(gap) or gap < 0:
            missing.append(phase + ":gripper_opening")
    checks = {}
    if not missing:
        before, after = measurements
        lift = after["object_pos"][2] - before["object_pos"][2]
        distance = sqrt(sum((x-y)**2 for x, y in zip(after["eef_pos"], after["object_pos"])))
        checks = {"object_lift": lift >= criterion["min_lift_m"],
                  "eef_object_proximity": distance <= criterion["max_eef_object_distance_m"],
                  "gripper_opening": criterion["min_gripper_opening"] <= after["gripper_opening"] <= criterion["max_gripper_opening"]}
    return {"success": all(checks.values()) if not missing else None,
            "checks": checks, "missing": missing, "criterion_ref": criterion["criterion_ref"],
            "source_sequences": [start["sequence"], handoff["sequence"]],
            "measurement_scope": "observational_handoff_only"}
