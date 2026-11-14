"""Read-only protocol10 checks before any future improvement dispatch.

This consumer reuses the existing assignment/score contracts. It neither issues
permits nor mutates phase ledgers, active releases, candidates or Runtime state.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from .contracts import ExperimentResult, RunBundle
from .evaluation import assignment_metrics, critical_preserved, load_protocol, score
from .storage import Records, ResearchStore

DIMENSIONS = (
    "simulation_episodes",
    "model_requests",
    "tool_calls",
    "environment_steps",
    "wall_seconds",
    "gpu_seconds",
    "storage_bytes",
)
TASKS = ("base_salad", "position_swap_salad", "old_milk", "old_drawer")
ACTIVE = "initial-medium-release08-deployment09"
BASELINE = "initial-deployment09-protocol10"
PER_START = dict(zip(DIMENSIONS, (1, 20, 60, 2000, 900, 900, 1073741824)))
STARTS = {
    "baseline": 8,
    "candidate": 8,
    "development": 2,
    "technical_recovery": 2,
    "inheritance": 1,
    "inheritance_recovery": 1,
}


def load_frozen_protocol(path: str | Path, frozen_path: str | Path) -> dict[str, Any]:
    """Load an exact frozen protocol and reject changed task/resource semantics."""
    p = load_protocol(path)
    if p != json.loads(Path(frozen_path).read_text(encoding="utf-8")):
        raise ValueError("protocol differs from frozen copy")
    release = p["task_sets"]["release"]
    dev = p["task_sets"]["development"]
    expected = {
        "status": "frozen",
        "scope": "simulation",
        "dynamic_authorized": False,
        "scope_id": "continuation10-20261113",
        "comparison_only": True,
        "anchor_bundle_id": BASELINE,
        "current_bundle_id": BASELINE,
        "active_bundle_id": ACTIVE,
        "current_active_hint": False,
        "min_gain": 0.02,
        "anchor_max_drop": 0.02,
    }
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("protocol identity, authorization or fixed Gate changed")
    if (
        release["task_keys"] != list(TASKS)
        or release["seeds"] != [0, 1]
        or release["assignments_per_bundle"] != 8
        or dev["seeds"] != [2]
        or dev["maximum_starts"] != 2
        or dev["task_keys"] != list(TASKS)
        or any(v.get("authorized") is not False for v in p["task_sets"].values())
        or p.get("allowed_runtime_splits") != []
    ):
        raise ValueError("fixed public denominator/split changed")
    if [(t["task_key"], t["group"]) for t in p["tasks"]] != list(
        zip(TASKS, ("base", "controlled", "old", "old"))
    ):
        raise ValueError("task group mapping changed")
    weights = {k: v["weight"] for k, v in p["metrics"].items()}
    if (
        weights
        != {"success_base": 0.4, "success_controlled": 0.35, "success_old": 0.25}
        or p["cost_weights"]
    ):
        raise ValueError("fixed J changed")
    b = p["budget"]
    if (
        b["runtime_per_start"] != PER_START
        or b["runtime_starts"] != STARTS
        or b["max_runtime_starts"] != 22
        or b["max_recoveries_per_slot"] != 1
        or b["automatic_retries"] != 0
        or b["research_requests"] != 1
    ):
        raise ValueError("Runtime/recovery envelope changed")
    if b["research_proposal"] != dict(zip(DIMENSIONS, (0, 1, 0, 0, 180, 0, 67108864))):
        raise ValueError("Research envelope changed")
    if b["offline_pool"] != {
        "wall_seconds": 1200,
        "storage_bytes": 1073741824,
        "gpu_wait_seconds_included": 300,
    }:
        raise ValueError("shared offline pool changed")
    if p["models"] != {
        "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
        "research": {"model": "gpt-6.1-sol", "reasoning_effort": "high"},
    }:
        raise ValueError("Planner/Research model settings changed")
    if (
        p["ceiling_policy"]["complete_baseline_J_equals_one"]
        != "freeze_stop_before_research_or_candidate"
    ):
        raise ValueError("ceiling stop policy changed")
    return p


def derive_baseline(
    protocol: dict[str, Any],
    *,
    source_research: str | Path,
    source_release: str | Path,
    destination: str | Path,
    profile_path: str | Path,
) -> dict[str, Any]:
    """Save a new protocol-bound baseline using the actual active deployment inputs.

    Snapshot IDs retain their immutable values. The destination has no imported
    results, qualification, release history or active marker.
    """
    source = ResearchStore(source_research)
    active = json.loads((Path(source_release) / "active.json").read_text())
    original = RunBundle(**source.get("bundles", ACTIVE))
    if (
        active.get("bundle_id") != ACTIVE
        or active.get("activation") != "qualified_initial"
        or original.protocol_id != "rsi-libero-pro-initial-release08-deployment09"
    ):
        raise ValueError("source is not current qualified deployment09")
    old_profile = json.loads(Path(original.backend_ref).read_text())
    new_profile = json.loads(Path(profile_path).read_text())
    metadata = {
        "profile_id",
        "dynamic_authorized",
        "deployment_authorized",
        "release_authorized",
        "authorization_reference",
        "profile_status",
        "changed_configuration",
        "protocol10_derivation",
    }
    if {k: v for k, v in old_profile.items() if k not in metadata} != {
        k: v for k, v in new_profile.items() if k not in metadata
    }:
        raise ValueError("baseline derivation changed actual execution inputs")
    binding = new_profile.get("protocol10_derivation", {})
    if (
        new_profile.get("profile_id") != BASELINE
        or binding.get("source_bundle_id") != ACTIVE
        or binding.get("protocol_id") != protocol["protocol_id"]
        or any(
            new_profile.get(k) is not False
            for k in (
                "dynamic_authorized",
                "deployment_authorized",
                "release_authorized",
            )
        )
        or old_profile["control"].get("planner_segment_handoff_hint", False)
        is not False
    ):
        raise ValueError("baseline profile provenance/authorization/hint mismatch")
    target = ResearchStore(destination)
    profile_ref = str(target.path("profiles", BASELINE).resolve())
    bundle = replace(
        original,
        bundle_id=BASELINE,
        protocol_id=protocol["protocol_id"],
        parent_bundle_id=ACTIVE,
        backend_ref=profile_ref,
        theta_ref=profile_ref + "#/model",
        processor_ref=profile_ref + "#/model/normalization",
        phi_ref=profile_ref + "#/control",
        diff_ref=None,
    )
    # Read every source first; a missing snapshot cannot produce a partial baseline.
    snapshots = [
        (kind, ref, source.get(kind, ref))
        for kind, ref in (
            ("psi", original.psi_ref),
            ("memory", original.memory_ref),
            ("rho", original.rho_ref),
        )
    ]
    target.put("profiles", BASELINE, new_profile)
    for kind, ref, value in snapshots:
        target.put(kind, ref, value)
    target.save_bundle(bundle)
    result = {
        "bundle_id": BASELINE,
        "parent_bundle_id": ACTIVE,
        "protocol_id": protocol["protocol_id"],
        "profile_ref": profile_ref,
        "execution_code_ref": bundle.code_ref,
        "execution_inputs_equal": True,
        "qualifications_imported": False,
        "results_imported": False,
        "source_active_unchanged": True,
    }
    Records(destination).put("protocol_derivations", BASELINE, result)
    return result


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError("finite nonnegative value required: " + label)
    return value


def resource_envelope(
    protocol: dict[str, Any],
    *,
    offline_used_seconds: float = 0,
    offline_used_bytes: int = 0,
    runtime_starts_used: int = 0,
    research_completed: bool = False,
) -> dict[str, Any]:
    """Compute gross and still-required vectors without reserving spent work twice."""
    b = protocol["budget"]
    seconds = _number(offline_used_seconds, "offline used seconds")
    size = _number(offline_used_bytes, "offline used bytes")
    pool = b["offline_pool"]
    if seconds > pool["wall_seconds"] or size > pool["storage_bytes"]:
        raise ValueError("shared offline pool exhausted")
    if type(runtime_starts_used) is not int or not 0 <= runtime_starts_used <= b["max_runtime_starts"]:
        raise ValueError("invalid already-consumed Runtime starts")
    runtime = {
        k: v * b["max_runtime_starts"] for k, v in b["runtime_per_start"].items()
    }
    gross = {k: runtime[k] + b["research_proposal"][k] for k in DIMENSIONS}
    gross["wall_seconds"] += pool["wall_seconds"]
    gross["storage_bytes"] += pool["storage_bytes"]
    remaining = dict(gross)
    remaining["wall_seconds"] -= seconds
    remaining["storage_bytes"] -= size
    for k in DIMENSIONS:
        remaining[k] -= runtime_starts_used * b["runtime_per_start"][k]
        if research_completed:
            remaining[k] -= b["research_proposal"][k]
    return {
        "runtime": runtime,
        "research": b["research_proposal"],
        "gross": gross,
        "remaining": remaining,
        "offline_consumed": {"wall_seconds": seconds, "storage_bytes": size},
        "gpu_wait_included_in_offline_seconds": pool["gpu_wait_seconds_included"],
    }


def admit_resources(
    protocol: dict[str, Any],
    phase: dict[str, Any],
    *,
    offline_used_seconds: float,
    offline_used_bytes: int,
    runtime_starts_used: int = 0,
    research_completed: bool = False,
    next_required: dict[str, float] | None = None,
    cleanup: dict[str, float] | None = None,
    reserve: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Use the task's admission policy; protocol10 retains its full envelope."""
    from .task_policy import admit_stage, task_policy

    mode = task_policy(protocol).get("resource_admission", "whole_comparison")
    if mode == "staged":
        if next_required is None or cleanup is None or reserve is None:
            raise ValueError("staged admission requires explicit next stage, cleanup and reserve vectors")
        return admit_stage(phase, required=next_required, cleanup=cleanup, reserve=reserve)
    if mode != "whole_comparison":
        raise ValueError("unsupported resource admission policy")
    envelope = resource_envelope(
        protocol,
        offline_used_seconds=offline_used_seconds,
        offline_used_bytes=offline_used_bytes,
        runtime_starts_used=runtime_starts_used,
        research_completed=research_completed,
    )
    if "unresolved_costs" not in phase or phase["unresolved_costs"] != []:
        raise ValueError("phase hard costs unresolved or unreported")
    shortfall = {}
    for k in DIMENSIONS:
        cap = _number(phase["limits"][k], "phase limit " + k)
        used = _number(phase["used"][k], "phase used " + k)
        left = _number(phase["remaining"][k], "phase remaining " + k)
        if abs(cap - used - left) > 1e-6:
            raise ValueError("phase arithmetic inconsistent: " + k)
        shortfall[k] = max(0, envelope["remaining"][k] - left)
    return {
        "resource_fit": not any(shortfall.values()),
        "additional_required": shortfall,
        "envelope": envelope,
        "phase_unchanged": True,
    }


def baseline_readiness(
    protocol: dict[str, Any], result: ExperimentResult
) -> dict[str, Any]:
    """Require matched facts and hard constraints; new tasks may declare other objectives."""
    from .task_policy import research_branches, task_policy

    policy = task_policy(protocol)
    if (
        result.bundle_id != protocol.get("current_bundle_id", BASELINE)
        or result.protocol_id != protocol["protocol_id"]
        or result.benchmark_id != protocol["benchmark_id"]
        or result.role != "release"
        or result.evidence_level != protocol["scope"]
        or result.candidate_id is not None
    ):
        raise ValueError("baseline identity/provenance mismatch")
    assignment_metrics(protocol, result)
    if result.unknown_count or result.invalid_count or not result.data_refs:
        return {
            "action": "stop_incomplete_baseline",
            "J": None,
            "candidate_generation_allowed": False,
        }
    required = [result.hard_constraints.get(k) for k in protocol["hard_constraints"]]
    if any(v is not True for v in required):
        return {
            "action": "stop_baseline_constraints",
            "J": score(protocol, result),
            "candidate_generation_allowed": False,
        }
    capability = (
        result.metrics["success_base"] >= 1
        and result.metrics["success_old"] >= 0.5
        and critical_preserved(protocol, result, result, result) is True
    )
    if policy.get("research_branches"):
        j = score(protocol, result)
        branches = research_branches(protocol, j=j, capability_established=capability)
        allowed = any(b["research_allowed"] for b in branches)
        return {"action": "ready_for_declared_research" if allowed else "stop_no_research_branch",
                "J": j, "candidate_generation_allowed": allowed, "research_branches": branches,
                "release_qualification": "requires independent objective-specific evaluation"}
    if not capability:
        return {
            "action": "stop_baseline_capability_unestablished",
            "J": score(protocol, result),
            "candidate_generation_allowed": False,
        }
    j = score(protocol, result)
    ceiling = j == 1
    return {
        "action": "freeze_stop_ceiling" if ceiling else "ready_for_bounded_research",
        "J": j,
        "candidate_generation_allowed": not ceiling,
        "reason": "fixed J has no positive-gain headroom"
        if ceiling
        else "same fixed min_gain=.02 and old-task protection apply",
    }


def validate_attempts(
    protocol: dict[str, Any], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check frozen slots and bounded recovery, preserving first valid failure."""
    runs, outputs, previous, selected = set(), set(), {}, {}
    base_counts = {"baseline": 0, "candidate": 0, "development": 0, "inheritance": 0}
    recoveries = {"technical_recovery": 0, "inheritance_recovery": 0}
    cumulative = dict.fromkeys(DIMENSIONS, 0)
    side_bundles = {}
    last_order = -1
    for row in attempts:
        if row["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("attempt protocol mismatch")
        if row["versions"] != protocol["versions"]:
            raise ValueError("execution/evaluation/scheduling version mismatch")
        if row["run_id"] in runs or row["output_ref"] in outputs:
            raise ValueError("run/output identity reused")
        runs.add(row["run_id"])
        outputs.add(row["output_ref"])
        order = _number(row["started_order"], "chronological start order")
        if order <= last_order:
            raise ValueError("attempts not strictly chronological")
        last_order = order
        side, task, seed = row["side"], row["task_key"], row["seed"]
        if side not in base_counts or task not in TASKS:
            raise ValueError("unregistered side/task")
        if side in ("baseline", "candidate") and seed not in (0, 1):
            raise ValueError("comparison seed outside frozen denominator")
        if side == "development" and seed != 2:
            raise ValueError("development cannot reuse comparison initial state")
        if side == "inheritance" and (task, seed) != ("position_swap_salad", 2):
            raise ValueError("unregistered inheritance task/initial state")
        if side == "baseline" and row["bundle_id"] != BASELINE:
            raise ValueError("baseline bundle identity changed")
        if side in side_bundles and side_bundles[side] != row["bundle_id"]:
            raise ValueError("side bundle changed within frozen matrix")
        side_bundles[side] = row["bundle_id"]
        slot = (side, task, seed)
        prior = previous.get(slot)
        if prior:
            if (
                slot in selected
                or prior.get("recovery_of_run_id") is not None
                or row.get("recovery_of_run_id") != prior["run_id"]
                or prior["validity"] != "invalid"
                or prior["failure_class"] != "technical"
                or prior["cost_closed"] is not True
                or prior["cleanup_complete"] is not True
                or row["bundle_id"] != prior["bundle_id"]
                or row.get("recovery_cause_resolved") is not True
            ):
                raise ValueError(
                    "recovery requires closed technical fault and cannot replace valid outcome"
                )
            group = (
                "inheritance_recovery"
                if side == "inheritance"
                else "technical_recovery"
            )
            recoveries[group] += 1
            if recoveries[group] > STARTS[group]:
                raise ValueError("round recovery cap exhausted")
        else:
            if row.get("recovery_of_run_id") is not None:
                raise ValueError("recovery source absent")
            base_counts[side] += 1
            if base_counts[side] > STARTS[side]:
                raise ValueError("side start cap exhausted")
        if row["validity"] not in ("valid", "invalid") or row["outcome"] not in (
            "success",
            "failure",
            "unknown",
        ):
            raise ValueError("invalid attempt outcome/technical label")
        if row["cost_closed"] is not True or row["cleanup_complete"] is not True:
            raise ValueError("attempt has unresolved cost/cleanup")
        for k in DIMENSIONS:
            cost = _number(row["costs"][k], "attempt cost " + k)
            if cost > PER_START[k] or (k == "simulation_episodes" and cost != 1):
                raise ValueError("per-start resource cap exceeded: " + k)
            cumulative[k] += cost
        previous[slot] = row
        if row["validity"] == "valid":
            selected[slot] = row["run_id"]
    if len(attempts) > 22:
        raise ValueError("maximum Runtime starts exhausted")
    matrix = [
        {
            "side": side,
            "task_key": task,
            "seed": seed,
            "selected_run_id": selected.get((side, task, seed)),
            "outcome": next(
                (
                    r["outcome"]
                    for r in attempts
                    if r["run_id"] == selected.get((side, task, seed))
                ),
                "unknown",
            ),
        }
        for side in ("baseline", "candidate")
        for task in TASKS
        for seed in (0, 1)
    ]
    return {
        "formal_denominator_per_side": 8,
        "matrix": matrix,
        "total_starts": len(attempts),
        "base_starts": base_counts,
        "recoveries": recoveries,
        "all_attempt_costs": cumulative,
    }


def check_stage(
    protocol: dict[str, Any],
    phase: dict[str, Any],
    *,
    stage: str,
    offline_used_seconds: float,
    offline_used_bytes: int,
    baseline: ExperimentResult | None = None,
    attempts: list[dict[str, Any]] | None = None,
    research_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only next-stage preflight consumed before future preparation."""
    if stage not in protocol["stage_order"]:
        raise ValueError("unregistered stage")
    attempt_report = validate_attempts(protocol, attempts or [])
    research_completed = False
    if research_receipt is not None:
        if (
            research_receipt.get("model") != protocol["models"]["research"]
            or research_receipt.get("protocol_id") != protocol["protocol_id"]
            or research_receipt.get("complete") is not True
            or not research_receipt.get("request_id")
        ):
            raise ValueError("Research receipt identity/model/closure mismatch")
        for k in DIMENSIONS:
            value = _number(research_receipt["costs"][k], "Research cost " + k)
            if value > protocol["budget"]["research_proposal"][k]:
                raise ValueError("Research resource cap exceeded: " + k)
        if research_receipt["costs"]["model_requests"] != 1:
            raise ValueError("Research receipt must retain exactly one request")
        research_completed = True
    for k, value in attempt_report["all_attempt_costs"].items():
        research_cost = research_receipt["costs"][k] if research_completed else 0
        if phase["used"][k] < value + research_cost:
            raise ValueError(
                "phase used does not retain all attempt/Research costs: " + k
            )
    report = admit_resources(
        protocol,
        phase,
        offline_used_seconds=offline_used_seconds,
        offline_used_bytes=offline_used_bytes,
        runtime_starts_used=len(attempts or []),
        research_completed=research_completed,
    )
    report.update(
        stage=stage,
        dispatch=False,
        dynamic_admitted=False,
        authorization="no new dynamic authority; exact centralized amendment and new permits required",
        versions=protocol["versions"],
    )
    if attempts is not None:
        report["attempts"] = attempt_report
    if stage != "baseline":
        if baseline is None:
            report["baseline"] = {
                "action": "stop_baseline_missing",
                "candidate_generation_allowed": False,
            }
        else:
            report["baseline"] = baseline_readiness(protocol, baseline)
    report["preparation_ready"] = report["resource_fit"] and (
        stage == "baseline" or report["baseline"]["candidate_generation_allowed"]
    )
    return report


def runtime_projection(
    protocol: dict[str, Any], *, frozen_ref: str | Path, authorization_ref: str | Path
) -> dict[str, Any]:
    """Map frozen protocol10 names to the existing Runtime schema without drift.

    The versioned field map is deterministic; only affirmative administrative
    authorization and its explicit release/development split admission are added afterward.
    """
    frozen = json.loads(Path(frozen_ref).read_text())
    if frozen != protocol:
        raise ValueError("runtime projection source differs from frozen protocol")
    authority = json.loads(Path(authorization_ref).read_text())
    if (
        authority.get("authorized") is not True
        or authority.get("scope") != protocol["scope_id"]
        or authority.get("protocol_id") != protocol["protocol_id"]
        or authority.get("frozen_protocol_ref") != str(Path(frozen_ref).resolve())
        or authority.get("allowed_runtime_splits") not in (["release"], ["release", "development"])
    ):
        raise ValueError(
            "explicit matching protocol10 administrative authority required"
        )
    projected = deepcopy(protocol)
    projected["runtime_config"] = dict(protocol["models"]["planner"])
    projected["runtime_config"]["research_llm"] = dict(protocol["models"]["research"])
    projected["budget"]["token_policy"] = protocol["budget"]["tokens"].split(";", 1)[0].strip()
    cap = protocol["budget"]["runtime_per_start"]
    projected["budget"]["per_attempt_proposal"] = {
        "full_job_wall_seconds": cap["wall_seconds"],
        "planner_timeout_seconds": 600,
        "max_turns": 20,
        "max_episode_steps": cap["environment_steps"],
        "tool_calls": cap["tool_calls"],
        "llm_input_tokens_including_cached": 1000000,
        "llm_output_tokens": 8000,
        "artifact_bytes": cap["storage_bytes"],
        "local_recoveries": 1,
        "reset_retries": 0,
        "model_requests": cap["model_requests"],
        "token_limits_enforced": False,
    }
    projected["runtime_projection"] = {
        "version": "protocol10-to-existing-runtime-v1",
        "source_frozen_protocol_ref": str(Path(frozen_ref).resolve()),
        "source_protocol_id": protocol["protocol_id"],
        "authorization_ref": str(Path(authorization_ref).resolve()),
        "field_map": {
            "runtime_config": "models.planner + models.research",
            "budget.token_policy": "budget.tokens policy identifier",
            "budget.per_attempt_proposal": "budget.runtime_per_start + frozen Planner600/20, token-reference, local1/reset0 controls",
        },
    }
    projected["dynamic_authorized"] = True
    projected["allowed_runtime_splits"] = list(authority["allowed_runtime_splits"])
    return projected


def prepare_baseline_attempt(
    *,
    root: str | Path,
    source_research: str | Path,
    preparation_root: str | Path,
    protocol_path: str | Path,
    frozen_path: str | Path,
    phase: dict[str, Any],
    administrative_authorization: str | Path,
    permit_path: str | Path,
    run_id: str,
    task_key: str,
    seed: int,
    output_dir: str | Path,
    scratch_dir: str | Path,
    offline_used_seconds: float,
    offline_used_bytes: int,
    attempts: list[dict[str, Any]] | None = None,
    cuda_device: int = 0,
) -> dict[str, Any]:
    """Prepare one bound baseline through the actual Runtime consumer; never execute.

    Uses a new administrative shadow store to leave the frozen research baseline
    and the formally active deployment unchanged. All checks precede those writes.
    """
    from .runtime import prepare_runtime
    from .storage import identifier

    protocol = load_frozen_protocol(protocol_path, frozen_path)
    preflight = check_stage(
        protocol,
        phase,
        stage="baseline",
        attempts=attempts,
        offline_used_seconds=offline_used_seconds,
        offline_used_bytes=offline_used_bytes,
    )
    if not preflight["preparation_ready"]:
        raise ValueError("full remaining protocol10 envelope is not admitted")
    projection = runtime_projection(
        protocol, frozen_ref=frozen_path, authorization_ref=administrative_authorization
    )
    identifier(run_id)
    if task_key not in TASKS or type(seed) is not int or seed not in (0, 1):
        raise ValueError("baseline attempt outside frozen task/seed denominator")
    for row in attempts or []:
        if row["run_id"] == run_id:
            raise ValueError("new run identity required")
        if row["side"] == "baseline" and (row["task_key"], row["seed"]) == (
            task_key,
            seed,
        ):
            raise ValueError(
                "this entry prepares first attempts only; recovery needs its registered closure consumer"
            )
    source = ResearchStore(source_research)
    bundle = RunBundle(**source.get("bundles", BASELINE))
    if (
        bundle.protocol_id != protocol["protocol_id"]
        or bundle.parent_bundle_id != ACTIVE
    ):
        raise ValueError("derived baseline identity/source mismatch")
    profile = json.loads(Path(bundle.backend_ref).read_text())
    if profile.get("dynamic_authorized") is not False:
        raise ValueError(
            "source baseline profile is not the frozen unauthorized profile"
        )
    binding = source.get("protocol_derivations", BASELINE)
    if (
        binding.get("parent_bundle_id") != ACTIVE
        or binding.get("protocol_id") != protocol["protocol_id"]
    ):
        raise ValueError("baseline derivation receipt mismatch")
    output, scratch, prep = (
        Path(x).resolve() for x in (output_dir, scratch_dir, preparation_root)
    )
    if (
        output == scratch
        or output.is_relative_to(scratch)
        or scratch.is_relative_to(output)
        or prep == output
        or prep == scratch
        or prep.is_relative_to(output)
        or prep.is_relative_to(scratch)
        or output.is_relative_to(prep)
        or scratch.is_relative_to(prep)
        or output.exists()
        or scratch.exists()
        or prep.exists()
    ):
        raise ValueError("new separate output/scratch/preparation roots required")
    permit = json.loads(Path(permit_path).read_text())
    expected = {
        "authorized": True,
        "scope": protocol["scope_id"],
        "run_id": run_id,
        "bundle_id": BASELINE,
        "protocol_id": protocol["protocol_id"],
        "task_key": task_key,
        "seed": seed,
        "simulation_episodes": 1,
        "gpu_devices": [cuda_device],
        "output_dir": str(output),
        "storage_write_roots": [str(output), str(scratch)],
        "administrative_authorization_ref": str(
            Path(administrative_authorization).resolve()
        ),
    }
    if any(permit.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "concrete single-attempt permit identity/domain/authority mismatch"
        )
    limits = {
        "wall_seconds": 900,
        "planner_seconds": 600,
        "tool_calls": 60,
        "environment_steps": 2000,
        "storage_bytes": 1073741824,
        "gpu_seconds": 900,
        "model_requests": 20,
    }
    if permit.get("limits") != limits:
        raise ValueError("permit must bind the exact frozen per-start limits")
    for row in attempts or []:
        if row["output_ref"] in (str(output), str(scratch)):
            raise ValueError("output identity already consumed")
    snapshots = [
        (kind, ref, source.get(kind, ref))
        for kind, ref in (
            ("psi", bundle.psi_ref),
            ("memory", bundle.memory_ref),
            ("rho", bundle.rho_ref),
        )
    ]
    shadow = ResearchStore(prep / "state/research")
    authorized_profile = deepcopy(profile)
    authorized_profile["dynamic_authorized"] = True
    authorized_profile_ref = str(shadow.path("profiles", BASELINE))
    projected_bundle = replace(
        bundle,
        backend_ref=authorized_profile_ref,
        theta_ref=authorized_profile_ref + "#/model",
        processor_ref=authorized_profile_ref + "#/model/normalization",
        phi_ref=authorized_profile_ref + "#/control",
    )
    shadow.put("profiles", BASELINE, authorized_profile)
    for kind, ref, value in snapshots:
        shadow.put(kind, ref, value)
    shadow.save_bundle(projected_bundle)
    Records(prep).put("protocols", protocol["protocol_id"], projection)
    projected_path = Records(prep).path("protocols", protocol["protocol_id"])
    plan = prepare_runtime(
        root,
        prep / "state",
        protocol_path=projected_path,
        run_id=run_id,
        bundle_id=BASELINE,
        task_key=task_key,
        seed=seed,
        split="release",
        output_dir=output,
        authorization_path=permit_path,
        cuda_device=cuda_device,
        storage_write_roots=[scratch],
    )
    receipt = {
        "stage": "baseline",
        "prepared_via": "rsi_vla.runtime.prepare_runtime",
        "execute": False,
        "dispatch": False,
        "preflight": preflight,
        "runtime_projection_ref": str(projected_path),
        "plan": plan,
        "source_baseline_unchanged": True,
        "formally_active_unchanged": True,
        "ready_for_separately_authorized_dispatch": not plan["blockers"],
    }
    Records(prep).put("plans", run_id, receipt)
    return receipt


def main() -> int:
    """Print deterministic JSON; exit 2 when the requested stage lacks admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("baseline", "research", "development", "candidate", "inheritance"),
    )
    parser.add_argument("--offline-used-seconds", required=True, type=float)
    parser.add_argument("--offline-used-bytes", required=True, type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--attempts", type=Path)
    parser.add_argument("--research-receipt", type=Path)
    parser.add_argument("--derive-to", type=Path)
    parser.add_argument("--source-research", type=Path)
    parser.add_argument("--source-release", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    try:
        protocol = load_frozen_protocol(args.protocol, args.frozen)
        derivation = None
        if args.derive_to:
            if not all((args.source_research, args.source_release, args.profile)):
                raise ValueError(
                    "derivation requires explicit source research/release and profile"
                )
            derivation = derive_baseline(
                protocol,
                source_research=args.source_research,
                source_release=args.source_release,
                destination=args.derive_to,
                profile_path=args.profile,
            )
        result = (
            ExperimentResult(**json.loads(args.baseline.read_text()))
            if args.baseline
            else None
        )
        report = check_stage(
            protocol,
            json.loads(args.phase.read_text()),
            stage=args.stage,
            offline_used_seconds=args.offline_used_seconds,
            offline_used_bytes=args.offline_used_bytes,
            baseline=result,
            attempts=json.loads(args.attempts.read_text()) if args.attempts else None,
            research_receipt=json.loads(args.research_receipt.read_text())
            if args.research_receipt
            else None,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(
            json.dumps(
                {"dispatch": False, "preparation_ready": False, "error": str(exc)}
            )
        )
        return 2
    if derivation is not None:
        report["derivation"] = derivation
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["preparation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
