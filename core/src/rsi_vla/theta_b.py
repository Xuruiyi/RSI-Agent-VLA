"""Frozen theta x B preparation consumed by the existing Runtime and evaluator.

Only plans and research bundles are produced. Runtime permits, independent
assessment and production Gate/activation remain separate consumers.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

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
        if theta.get("load_receipt_ref"):
            from .training import training_handoff
            loaded, _handoff = training_handoff(theta["load_receipt_ref"], require_load=True)
            if (loaded["theta_ref"] != theta["theta_ref"] or loaded["processor_ref"] != theta["processor_ref"]):
                raise ValueError("frozen theta does not match actual loader receipt")
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
        "fallback_reason": ("explicit_" + theta["b_applicability"][b_id] + "_applicability") if effective_id != b_id else None,
        "theta_b_applicability": theta["b_applicability"][b_id],
        "theta_ref": theta.get("theta_ref"), "processor_ref": theta.get("processor_ref"),
        "load_receipt_ref": theta.get("load_receipt_ref"),
        "applicability": theta["b_applicability"][b_id],
        "production_eligible": False, "capability_gain": "unknown"}
    import json
    for label in {b_id, effective_id}:
        review = theta.get("applicability_evidence", {}).get(label)
        if review and json.loads(Path(review["evidence_ref"]).read_text()) != review["evidence"]:
            raise ValueError("frozen theta/B applicability evidence changed")
    if theta["b_applicability"][effective_id] == "inapplicable":
        return {"status": "blocked_theta_b_inapplicable", "binding": metadata, "plan": None}
    if theta.get("load_receipt_ref"):
        from .training import training_handoff
        training_handoff(theta["load_receipt_ref"], require_load=True)
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
    return {"binding": prepared["binding"], "intervention": deepcopy(prepared["binding"]), "result": deepcopy(result),
            "production_eligible": False, "capability_gain": "unknown",
            "next_consumer": "independent protocol evaluator and existing Gate"}


def training_theta_version(
    *, training_result_ref: str | Path, load_result_ref: str | Path, scope_id: str
) -> dict[str, Any]:
    """Export theta only after the saved model passes the actual RPent loader."""
    from .training import training_handoff

    trained, before = training_handoff(training_result_ref, scope_id=scope_id)
    loaded, after = training_handoff(
        load_result_ref, scope_id=scope_id, require_load=True
    )
    if (
        loaded.get("training_result_ref") != str(Path(training_result_ref).resolve())
        or before["weights_identity"] != after["weights_identity"]
        or before["processor_snapshot"] != after["processor_snapshot"]
        or trained["run_id"] != loaded["run_id"]
    ):
        raise PermissionError("loaded theta is not the completed training output")
    return {
        "theta_ref": loaded["theta_ref"],
        "processor_ref": loaded["processor_ref"],
        "training_result_ref": str(Path(training_result_ref).resolve()),
        "load_receipt_ref": str(Path(load_result_ref).resolve()),
        "actual_model_load": "passed",
        "capability_gain": "unknown",
    }


def register_training_bundle(
    artifacts: str | Path,
    *,
    base_bundle_id: str,
    bundle_id: str,
    training_result_ref: str | Path,
    load_result_ref: str | Path,
    scope_id: str,
    candidate_id: str | None = None,
    benchmark_id: str | None = None,
    cost_receipt_refs: tuple[str | Path, ...] | list[str | Path] = (),
    development_result_ids: tuple[str, ...] | list[str] = (),
    accounting_ref: str | Path | None = None,
    approval_scope: str = "offline",
    current_operation_id: str | None = None,
) -> dict[str, Any]:
    """Save an unqualified theta candidate for independent matrix evaluation."""
    from dataclasses import asdict

    from .adapters.pi05_training import training_bundle_proposal

    version = training_theta_version(
        training_result_ref=training_result_ref,
        load_result_ref=load_result_ref,
        scope_id=scope_id,
    )
    store = ResearchStore(Path(artifacts) / "research")
    base = RunBundle(**store.get("bundles", base_bundle_id))
    bundle = training_bundle_proposal(
        base,
        bundle_id=bundle_id,
        theta_ref=version["theta_ref"],
        processor_ref=version["processor_ref"],
        diff_ref=version["training_result_ref"],
    )
    store.save_bundle(bundle)
    result = {
        "bundle_id": bundle_id,
        "bundle": asdict(bundle),
        **version,
        "qualification": "pending_independent_evaluation",
        "production_eligible": False,
    }
    if candidate_id is not None:
        from .contracts import Candidate
        from .training import training_handoff

        if not benchmark_id:
            raise ValueError(
                "training Candidate requires its independent benchmark identity"
            )
        trained, _ = training_handoff(training_result_ref, scope_id=scope_id)
        loaded, _ = training_handoff(
            load_result_ref, scope_id=scope_id, require_load=True
        )
        references = list(cost_receipt_refs)
        accounting_snapshot = None
        pending_cost_operations = []
        if accounting_ref:
            import json

            accounting_snapshot = json.loads(Path(accounting_ref).read_text())
            if accounting_snapshot.get("scope_id") != scope_id:
                raise PermissionError(
                    "training Candidate accounting belongs to another task"
                )
            for operation, reservation in accounting_snapshot["reservations"].items():
                if reservation.get("status") != "settled":
                    if operation != current_operation_id:
                        raise PermissionError("training Candidate has an unsettled predecessor cost")
                    pending_cost_operations.append(operation)
                    continue
                if reservation.get("status") == "settled":
                    reference = reservation["receipt_ref"]
                    receipt = json.loads(Path(reference).read_text())
                    if (
                        receipt.get("actual") != reservation.get("actual")
                        or receipt.get("operation_id") != operation
                    ):
                        raise PermissionError(
                            "Host cost receipt differs from settled accounting"
                        )
                    if str(Path(reference).resolve()) not in {
                        str(Path(ref).resolve()) for ref in references
                    }:
                        references.append(reference)
        costs, receipts = _training_candidate_costs(
            references,
            scope_id=scope_id,
            required_operations={trained["operation_id"], loaded["operation_id"]},
        )
        fields = ("theta_ref", "processor_ref", "diff_ref")
        changes = {
            key: getattr(bundle, key)
            for key in fields
            if getattr(base, key) != getattr(bundle, key)
        }
        for key in development_result_ids:
            finding = store.get("development_results", key)
            if (
                finding.get("bundle_id") != bundle_id
                or finding.get("protocol_id") != bundle.protocol_id
            ):
                raise ValueError(
                    "training Candidate development finding belongs to another bundle/protocol"
                )
        provenance_ref = store.path("training_candidate_sources", candidate_id)
        store.put(
            "training_candidate_sources",
            candidate_id,
            {
                "schema": "rsi-training-candidate-sources-v1",
                "candidate_id": candidate_id,
                "training_result_ref": str(Path(training_result_ref).resolve()),
                "load_result_ref": str(Path(load_result_ref).resolve()),
                "host_cost_receipts": receipts,
                "cost_scope": ("all settled task operations before current Candidate serialization; current disk is peak occupancy"
                               if accounting_ref else "all supplied completed and failed Host operations; current disk is peak occupancy"),
                "pending_cost_operations": pending_cost_operations,
                "accounting_ref": str(accounting_ref) if accounting_ref else None,
                "accounting_snapshot": accounting_snapshot,
                "task_gain": "unknown",
            },
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            base_bundle_id=base_bundle_id,
            bundle_id=bundle_id,
            protocol_id=bundle.protocol_id,
            benchmark_id=benchmark_id,
            changes=changes,
            artifact_refs=[
                str(training_result_ref),
                str(load_result_ref),
                str(provenance_ref),
            ],
            development_results=list(development_result_ids),
            costs=costs,
            compatibility={
                "actual_model_load": "passed",
                "task_gain": "unknown",
                "qualification": "pending_independent_evaluation",
                "source_receipt_ref": str(provenance_ref),
                "pending_serialization_cost_operations": pending_cost_operations,
            },
            approval_scope=approval_scope,
            novelty_type="parameter_update",
        )
        store.save_candidate(candidate)
        result.update(
            candidate_id=candidate_id,
            candidate_ref=str(store.path("candidates", candidate_id)),
            candidate=asdict(candidate),
        )
    return result


def record_theta_b_applicability(
    experiment: dict[str, Any],
    *,
    theta_id: str,
    b_id: str,
    decision: str,
    evidence_ref: str | Path,
    fallback_b_id: str | None = None,
) -> dict[str, Any]:
    """Bind compatibility evidence without silently relabeling a fallback cell."""
    import json

    if decision not in {"applicable", "inapplicable", "unknown"}:
        raise ValueError("unsupported theta/B applicability decision")
    theta = experiment["theta"][theta_id]
    if b_id not in experiment["B"] or fallback_b_id not in {None, *experiment["B"]}:
        raise ValueError("applicability references an unfrozen B package")
    evidence = json.loads(Path(evidence_ref).read_text())
    if (
        evidence.get("schema") != "rsi-theta-b-applicability-v1"
        or evidence.get("reviewer_role") != "evaluation"
        or evidence.get("experiment_id") != experiment["experiment_id"]
        or evidence.get("theta_ref") != theta.get("theta_ref")
        or evidence.get("processor_ref") != theta.get("processor_ref")
        or evidence.get("b_id") != b_id
        or evidence.get("decision") != decision
    ):
        raise ValueError("theta/B applicability evidence identity differs")
    if decision == "applicable" and any(evidence.get(key) is not True for key in (
            "action_contract_compatible", "processor_compatible", "frozen_b_compatible")):
        raise ValueError("applicable theta/B requires all concrete compatibility checks")
    result = deepcopy(experiment)
    result["theta"][theta_id]["b_applicability"][b_id] = decision
    result["theta"][theta_id].setdefault("applicability_evidence", {})[b_id] = {
        "evidence_ref": str(Path(evidence_ref).resolve()),
        "evidence": evidence,
        "fallback_b_id": fallback_b_id,
    }
    return result


def prepare_theta_b_matrix(
    root: str | Path,
    artifacts: str | Path,
    experiment: dict[str, Any],
    *,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prepare every explicitly assigned cell, retaining blocked theta and fallback."""
    identities = [(row["theta_id"], row["b_id"]) for row in cells]
    expected = {(theta, b) for theta in experiment["theta"] for b in experiment["B"]}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("theta x B matrix requires all unique frozen cells")
    prepared = [prepare_theta_b_cell(root, artifacts, experiment, **row) for row in cells]
    effective = {(row["binding"]["theta_id"], row["binding"]["effective_b"])
                 for row in prepared if row["plan"] is not None}
    return {
        "requested_cells": [list(cell) for cell in sorted(expected)],
        "effective_cells": [list(cell) for cell in sorted(effective)],
        "missing_effective_cells": [list(cell) for cell in sorted(expected - effective)],
        "schema": "rsi-theta-b-matrix-preparation-v1",
        "experiment_id": experiment["experiment_id"],
        "cells": prepared,
        "production_eligible": False,
        "capability_gain": "unknown",
    }


def _training_candidate_costs(
    cost_receipt_refs: list[str | Path], *, scope_id: str, required_operations: set[str]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Consume settled Host costs, retaining failed attempts and peak occupancy."""
    import json
    import math

    totals, operations, sources = {}, set(), []
    names = {
        "wall_seconds": "whole_job_wall_seconds",
        "gpu_seconds": "gpu_device_seconds_envelope",
        "storage_bytes": "artifact_bytes",
        "current_disk_bytes": "peak_current_disk_bytes",
    }
    for reference in cost_receipt_refs:
        receipt = json.loads(Path(reference).read_text())
        operation = receipt.get("operation_id")
        if (
            receipt.get("scope_id") != scope_id
            or not operation
            or operation in operations
            or receipt.get("cleanup_confirmed") is not True
            or receipt.get("unknown_dimensions")
            or not receipt.get("actual")
        ):
            raise PermissionError(
                "training candidate cost receipt is duplicate, unresolved or differently scoped"
            )
        report = json.loads(Path(receipt["supervisor_ref"]).read_text())
        if (
            report.get("identity", {}).get("run_id") != operation
            or report.get("cleanup_confirmed") is not True
            or report.get("remaining_process_ids") != []
        ):
            raise PermissionError(
                "training candidate cost receipt lacks original Host process closure"
            )
        if operation in required_operations and (
            receipt.get("status") != "completed"
            or report.get("status") != "completed"
            or report.get("exit_code") != 0
        ):
            raise PermissionError("training or loader Host job was not completed")
        for key, value in receipt["actual"].items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    "training candidate cost contains an unknown or invalid value"
                )
            name = names.get(key, key)
            totals[name] = (
                max(totals.get(name, 0), value)
                if key == "current_disk_bytes"
                else totals.get(name, 0) + value
            )
        operations.add(operation)
        sources.append(
            {
                "receipt_ref": str(Path(reference).resolve()),
                "receipt": receipt,
                "supervisor": report,
            }
        )
    if not required_operations <= operations:
        raise PermissionError(
            "training candidate costs omit its completed training or loading operation"
        )
    return totals, sources
