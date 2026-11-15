"""Frozen theta x B preparation consumed by the existing Runtime and evaluator.

Only plans and research bundles are produced. Runtime permits, independent
assessment and production Gate/activation remain separate consumers.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from .contracts import RunBundle
from .runtime import prepare_runtime, resolve_reference
from .storage import ResearchStore


B_FIELDS = ("phi_ref", "rho_ref", "memory_ref", "backend_ref")


def _b_snapshot(store, bundle):
    """Keep shared references unique; snapshots detect changed program or prompts."""
    return {"references": {key: bundle[key] for key in B_FIELDS},
            "program": resolve_reference(bundle["phi_ref"]),
            "relations": store.get("rho", bundle["rho_ref"]),
            "retrieval": store.get("memory", bundle["memory_ref"]),
            "profile": resolve_reference(bundle["backend_ref"])}


def freeze_theta_b(artifacts, *, experiment_id, base_bundle_id, b_bundle_ids, theta_versions):
    """Freeze B0/B1 contents and explicit existing or unavailable theta versions."""
    from .storage import identifier
    identifier(experiment_id)
    if set(b_bundle_ids) != {"B0", "B1"} or not theta_versions:
        raise ValueError("B0/B1 and explicit theta versions required")
    store = ResearchStore(Path(artifacts) / "research")
    base = store.get("bundles", base_bundle_id)
    frozen = {}
    for label, bundle_id in b_bundle_ids.items():
        bundle = store.get("bundles", bundle_id)
        # The experiment varies only theta and the explicitly frozen B package.
        if any(bundle[key] != base[key] for key in ("protocol_id", "psi_ref", "code_ref")):
            raise ValueError("B packages must share protocol, research strategy and code")
        frozen[label] = {"source_bundle_id": bundle_id, **_b_snapshot(store, bundle)}
    versions = deepcopy(theta_versions)
    for theta in versions.values():
        if (theta.get("theta_ref") is None) != (theta.get("processor_ref") is None):
            raise ValueError("theta and processor must both exist or both be unknown")
        theta["snapshot"] = (resolve_reference(theta["theta_ref"])
                             if theta.get("theta_ref") else None)
        theta["processor_snapshot"] = (_processor_snapshot(theta["processor_ref"], theta["snapshot"])
                                       if theta.get("processor_ref") else None)
        theta["b_applicability"] = {label: "unknown" for label in frozen}
    return {"schema": "rsi-theta-b-experiment-v1", "experiment_id": experiment_id,
            "base_bundle": base, "B": frozen, "theta": versions,
            "production_eligible": False, "capability_gain": "unknown"}


def _processor_snapshot(reference, model):
    binding = resolve_reference(reference) if "#" in reference else reference
    path = Path(binding)
    if not path.is_absolute():
        checkpoint = Path(model.get("model_path", model.get("checkpoint", "")))
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[3] / checkpoint
        path = checkpoint / path
    if path.is_dir():
        path = path / "norm_stats.json"
    return {"binding": binding, "stats": path.read_text() if path.is_file() else None}


def prepare_theta_b_cell(root, artifacts, experiment, *, theta_id, b_id, bundle_id,
                         fallback_b_id=None, **runtime_args):
    """Prepare a real Runtime plan, exposing unknown theta/B and any fallback.

    Unknown applicability is retained in the plan. A caller may explicitly choose
    a different frozen B package; that run is labeled as its effective cell.
    This function never executes Runtime or bypasses its production admission.
    """
    if experiment.get("schema") != "rsi-theta-b-experiment-v1":
        raise ValueError("unsupported theta x B experiment")
    theta = experiment["theta"][theta_id]
    requested = experiment["B"][b_id]
    effective_id = fallback_b_id or b_id
    effective = experiment["B"][effective_id]
    metadata = {"experiment_id": experiment["experiment_id"], "theta_id": theta_id,
        "requested_b": b_id, "effective_b": effective_id,
        "fallback": effective_id != b_id,
        "fallback_reason": "explicit_unknown_applicability" if effective_id != b_id else None,
        "theta_b_applicability": theta["b_applicability"][b_id],
        "production_eligible": False, "capability_gain": "unknown"}
    if not theta.get("theta_ref"):
        return {"status": "blocked_theta_unavailable", "binding": metadata, "plan": None}
    if theta["snapshot"] != resolve_reference(theta["theta_ref"]) or theta["processor_snapshot"] != _processor_snapshot(theta["processor_ref"], theta["snapshot"]):
        raise ValueError("frozen theta or processor content changed")
    store = ResearchStore(Path(artifacts) / "research")
    for label, b in ((b_id, requested), (effective_id, effective)):
        if _b_snapshot(store, store.get("bundles", b["source_bundle_id"])) != {k: v for k, v in b.items() if k != "source_bundle_id"}:
            raise ValueError("frozen B content changed: " + label)
    base = RunBundle(**experiment["base_bundle"])
    if runtime_args.get("split") not in {"development", "release", "smoke"}:
        raise ValueError("theta x B prepares evaluation runs only")
    proposal = replace(base, bundle_id=bundle_id, theta_ref=theta["theta_ref"],
        processor_ref=theta["processor_ref"], **effective["references"],
        parent_bundle_id=base.bundle_id, diff_ref=experiment["experiment_id"])
    store.save_bundle(proposal)
    plan = prepare_runtime(root, artifacts, bundle_id=bundle_id, **runtime_args)
    plan["context"]["theta_b_experiment"] = metadata
    return {"status": "prepared_pending_runtime_authorization", "binding": metadata, "plan": plan}


def evaluate_theta_b_cell(prepared, result):
    """Associate independent result identity without promoting it to Gate evidence."""
    plan = prepared.get("plan")
    if plan is None:
        raise ValueError("no prepared theta x B cell")
    context = plan["context"]
    for field in ("run_id", "episode_id", "bundle_id", "protocol_id"):
        if result.get(field) != context[field]:
            raise ValueError("evaluation result identity differs: " + field)
    return {"binding": prepared["binding"], "result": deepcopy(result),
            "production_eligible": False, "capability_gain": "unknown",
            "next_consumer": "independent protocol evaluator and existing Gate"}
