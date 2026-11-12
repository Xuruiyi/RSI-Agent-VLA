"""Finite phi updates consumed by RPent's runtime_rules_ref adapter."""
import json
from pathlib import Path


RULE_SCHEMA = "rsi-runtime-rules-v1"
GUARD_RULE_SCHEMA = "rsi-runtime-rules-v2"
RESERVE_RULE_SCHEMA = "rsi-runtime-rules-v3"


def validate_runtime_rules(value):
    if not isinstance(value, dict):
        raise ValueError("runtime rules must be a JSON object")
    schema = value.get("schema")
    fields = {"schema", "chunk_steps", "baseline_chunk_steps"}
    if schema in {GUARD_RULE_SCHEMA, RESERVE_RULE_SCHEMA}:
        fields.add("finish_requires_native_success")
        if schema == RESERVE_RULE_SCHEMA:
            fields.add("planner_finish_reserve_calls")
    elif schema != RULE_SCHEMA:
        raise ValueError("unsupported runtime rule schema")
    if set(value) != fields:
        raise ValueError("runtime rule fields must match the finite schema")
    cap, steps = value["baseline_chunk_steps"], value["chunk_steps"]
    if type(cap) is not int or type(steps) is not int or not 1 <= steps <= cap:
        raise ValueError("chunk_steps must be an integer within the baseline cap")
    if schema in {GUARD_RULE_SCHEMA, RESERVE_RULE_SCHEMA} and type(value["finish_requires_native_success"]) is not bool:
        raise ValueError("finish_requires_native_success must be a boolean")
    if schema == RESERVE_RULE_SCHEMA and (type(value["planner_finish_reserve_calls"]) is not int or
                                         value["planner_finish_reserve_calls"] not in {0, 1}):
        raise ValueError("planner_finish_reserve_calls must be 0 or 1")
    return dict(value)


def load_runtime_rules(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("runtime_rules_ref must be an absolute JSON path")
    return validate_runtime_rules(json.loads(path.read_text(encoding="utf-8")))


def read_phi_chunk(phi_ref):
    """Read the existing authoritative control pointer or a rule artifact."""
    filename, _, pointer = phi_ref.partition("#")
    path = Path(filename)
    if not path.is_absolute():
        raise ValueError("phi_ref must reference a local absolute artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("invalid control JSON pointer")
        for part in pointer[1:].split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
    if value.get("schema") in {RULE_SCHEMA, GUARD_RULE_SCHEMA, RESERVE_RULE_SCHEMA}:
        return validate_runtime_rules(value)
    cap = value.get("action_chunk")
    if type(cap) is not int or cap < 1:
        raise ValueError("base phi lacks a verified positive action_chunk")
    return {"schema": RULE_SCHEMA, "chunk_steps": cap, "baseline_chunk_steps": cap}


def rule_update(before, *, chunk_steps=None, finish_requires_native_success=None,
                planner_finish_reserve_calls=None):
    """Choose exactly one finite behavior update, preserving the other behavior."""
    before = validate_runtime_rules(before)
    if sum(value is not None for value in (chunk_steps, finish_requires_native_success,
                                           planner_finish_reserve_calls)) != 1:
        raise ValueError("choose exactly one rule intervention")
    if chunk_steps is not None:
        after = validate_runtime_rules({**before, "chunk_steps": chunk_steps})
        name, old, new = "chunk_steps", before["chunk_steps"], chunk_steps
    elif finish_requires_native_success is not None:
        if finish_requires_native_success is not True:
            raise ValueError("this repair scope only enables the native finish guard")
        after = validate_runtime_rules({**before, "schema": (RESERVE_RULE_SCHEMA if
            before["schema"] == RESERVE_RULE_SCHEMA else GUARD_RULE_SCHEMA),
                                       "finish_requires_native_success": finish_requires_native_success})
        name = "finish_requires_native_success"
        old, new = before.get(name, False), finish_requires_native_success
    else:
        if type(planner_finish_reserve_calls) is not int or planner_finish_reserve_calls != 1:
            raise ValueError("this repair scope only enables one reserved finish call hint")
        after = validate_runtime_rules({**before, "schema": RESERVE_RULE_SCHEMA,
            "finish_requires_native_success": before.get("finish_requires_native_success", False),
            "planner_finish_reserve_calls": 1})
        name, old, new = "planner_finish_reserve_calls", before.get("planner_finish_reserve_calls", 0), 1
    if old == new:
        raise ValueError("a rule trial requires an actual behavior change")
    return after, {name: {"before": old, "after": new}}


def write_rule_trial(store, base, *, bundle_id, artifact_id, chunk_steps=None,
                     finish_requires_native_success=None, planner_finish_reserve_calls=None):
    """Persist actual bounded config changes; theta and all other slots freeze."""
    from dataclasses import replace
    before = read_phi_chunk(base.phi_ref)
    after, changes = rule_update(before, chunk_steps=chunk_steps,
                                finish_requires_native_success=finish_requires_native_success,
                                planner_finish_reserve_calls=planner_finish_reserve_calls)
    timing_changed = "chunk_steps" in changes
    rule_ref = str(store.path("runtime_rules", artifact_id).resolve())
    diff_ref = str(store.path("rule_diffs", artifact_id).resolve())
    difference = {"base_bundle_id": base.bundle_id, "bundle_id": bundle_id,
                  "base_phi_ref": base.phi_ref, "target_phi_ref": rule_ref,
                  "changes": changes,
                  "frozen": {name: getattr(base, name) for name in (
                      "theta_ref", "processor_ref", "rho_ref", "psi_ref", "memory_ref",
                      "backend_ref", "code_ref", "protocol_id")},
                  "consumer": ("RSI_RUNTIME_CONTEXT.runtime_rules_ref -> LIBERO _vlm_chunk" if timing_changed
                               else "RSI_RUNTIME_CONTEXT.runtime_rules_ref -> Planner counted-budget finish hint"
                               if "planner_finish_reserve_calls" in changes
                               else "RSI_RUNTIME_CONTEXT.runtime_rules_ref -> LIBERO Toolkit native finish guard"),
                  "requires_new_timing_baseline": timing_changed,
                  "comparison_requirement": ("new matched control-timing baseline required" if timing_changed
                      else "same frozen theta/action chunk/budgets/scoring; independent task comparison required")}
    if not timing_changed:
        difference["frozen_control"] = {key: before[key] for key in ("chunk_steps", "baseline_chunk_steps")}
    store.put("runtime_rules", artifact_id, after)
    store.put("rule_diffs", artifact_id, difference)
    target = replace(base, bundle_id=bundle_id, parent_bundle_id=base.bundle_id,
                     phi_ref=rule_ref, diff_ref=diff_ref)
    store.save_bundle(target)
    return target, difference
