"""Launcher-owned data provenance and admission; never converts traces to demonstrations."""

import json
from pathlib import Path
from typing import Any

from .storage import atomic_json


def register_collection(context):
    root = Path(context["output_dir"])
    identity = {
        k: context[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    independent_path = root / "independent_result.json"
    independent = json.loads(independent_path.read_text())
    if any(independent.get(k) != v for k, v in identity.items()):
        raise ValueError("collection outcome identity mismatch")
    data_root = Path(context["collection"]["root"])
    data_root.relative_to(root)
    import numpy as np

    scope = context["information_scope"]
    source_kind = "independent_evaluation" if scope in {"release", "evaluation", "sealed", "final", "test"} else "autonomous_execution"
    entries = []
    for path in sorted(data_root.rglob("episode_*")):
        if not path.is_dir():
            continue
        complete, errors, persisted = False, [], 0
        metadata = {}
        record_path = (
            path / "episode.json"
            if (path / "episode.json").exists()
            else path / "collection.json"
        )
        try:
            metadata = json.loads(record_path.read_text())
            if (
                metadata.get(
                    "runtime_identity",
                    metadata.get("collection_config", {}).get("identity"),
                )
                != identity
            ):
                raise ValueError("collection identity mismatch")
            # Flywheel owns a local episode_id; the runtime episode is separately bound below.
            for batch in metadata["batches"]:
                if batch["start"] != persisted:
                    raise ValueError("noncontiguous collection batch")
                with np.load(path / batch["file"], allow_pickle=False) as data:
                    if len(data["actions"]) != batch["count"]:
                        raise ValueError("persisted transition count mismatch")
                persisted += batch["count"]
            if persisted != metadata["persisted_steps"]:
                raise ValueError("collection persisted count mismatch")
            if path.name.endswith(".partial"):
                errors.append("interrupted_or_not_finalized")
            else:
                complete = (
                    metadata.get("alignment_complete") is True
                    and metadata.get("pending_dispatch") is None
                )
            if metadata.get("unflushed_observed_steps", 0):
                errors.append("unflushed_tail")
        except (OSError, ValueError, KeyError) as exc:
            complete = False
            errors.append(str(exc))
        entries.append(
            {
                "runtime_identity": identity,
                "collection_episode_id": metadata.get("episode_id"),
                "path": str(path),
                "metadata_ref": str(record_path),
                "persisted_steps": persisted,
                "alignment_complete": complete,
                "pending_dispatch": metadata.get("pending_dispatch"),
                "unflushed_observed_steps": metadata.get("unflushed_observed_steps"),
                "errors": errors,
                "source_kind": source_kind,
                "information_scope": context["information_scope"],
                "task": context["task"],
                "action_contract": metadata.get("action_contract"),
                "native_task_fact": independent.get("native_task_result"),
                "technical_validity": independent.get("validity"),
                "outcome": independent.get("outcome"),
                "stop_reason": independent.get("runtime_stop_reason"),
                "independent_result_ref": str(independent_path),
                "allowed_uses": ["preservation", "development_diagnosis"]
                if context["information_scope"] == "development"
                else ["preservation", "independent_evaluation_only"],
                "training_admission": "not_admitted_" + source_kind,
                "missing": [
                    "controller_applied_action",
                    "reset_setup_trajectory",
                    "nonreturned_rpc_transitions",
                ],
                "clock": "nominal_20Hz_separate_from_control_index_and_host_monotonic",
            }
        )
    manifest = {
        "schema": "rsi-execution-dataset-manifest-v1",
        "identity": identity,
        "collection_config": context["collection"],
        "episodes": entries,
        "alignment_complete": len(entries) == 1
        and all(e["alignment_complete"] for e in entries),
        "source_kind": source_kind,
        "training_admission": "not_admitted",
        "required_release_evidence": context["collection"].get(
            "required_for_release", True
        ),
        "cost_scope": "all raw and derived files under supervised output; included in task and phase storage",
        "missing": [] if entries else ["collection_not_available"],
        "evidence_refs": [
            str(independent_path),
            str(root / "environment_service/action_events.jsonl"),
            str(root / "action_events.jsonl"),
        ],
        "selection_mechanisms": {
            "media_retention": {"source_ref": str(root / "action_events.jsonl"), "probability": None,
                                "basis": "per-artifact outlet receipts when available; historical coverage unknown"},
            "outcome_label": {"source_ref": str(independent_path), "probability": None},
            "repair_selection": {"probability": None, "basis": "unknown"},
        },
        "loss_bound": "up to flush_steps-1 returned transitions plus in-flight RPC and incomplete write; no zero-loss guarantee",
    }
    from .process_accounting import accounted_storage_bytes, logical_snapshot

    files = logical_snapshot([root])
    temporary = {value["inode"]: value["size"] for path, value in files.items()
                 if ".tmp" in Path(path).name}
    persistent = {value["inode"]: value["size"] for path, value in files.items()
                  if ".tmp" not in Path(path).name and value["inode"] not in temporary}
    manifest["artifact_storage"] = {
        "persistent_bytes": sum(persistent.values()), "temporary_bytes": sum(temporary.values()),
        "cumulative_new_bytes_lower_bound": accounted_storage_bytes(root),
        "exact_cumulative_bytes": None, "sampled_before_manifest_write": True,
        "deletion_refunds_cumulative_cost": False,
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def dataset_file_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    """Record local payload identities so later consumers detect changed inputs."""
    root = Path(manifest["dataset_root"]).resolve()
    paths = [
        p
        for folder in (root / "meta", root / "data", root / "images", root / "videos")
        if folder.exists()
        for p in folder.rglob("*")
        if p.is_file()
    ]
    paths.append(Path(manifest["provenance_ref"]).resolve())
    return {
        str(path.resolve()): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(set(paths))
    }


def review_training_dataset(
    *,
    manifest_ref: str | Path,
    output_root: str | Path,
    scope_id: str,
    protocol_id: str,
    reviewer_id: str,
    quality_review_ref: str | Path | None = None,
    evaluation_isolation_review_ref: str | Path | None = None,
    payload_audit_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one local dataset and publish scoped admission, rejection or unknown.

    Independent quality/isolation reviews remain evaluator-owned inputs. Missing
    review evidence is unknown; invalid payloads and unsupported sources are
    rejected. This operation neither acquires data nor asserts model capability.
    """
    from .adapters.pi05_training import (
        audit_demonstration_payload,
        dataset_identity,
        require_training_data_admission,
        validate_dataset_manifest,
    )

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest_ref = Path(manifest_ref).resolve()
    manifest = json.loads(manifest_ref.read_text())
    result = {
        "schema": "rsi-training-data-admission-v1",
        "scope_id": scope_id,
        "protocol_id": protocol_id,
        "reviewer_id": reviewer_id,
        "original_manifest_ref": str(manifest_ref),
        "decision": "unknown",
        "allowed_uses": ["preservation"],
        "errors": [],
        "model_capability": "unknown",
        "training_steps": 0,
    }
    receipt_ref = root / "data-admission.json"
    try:
        validate_dataset_manifest(manifest)
        result["dataset"] = dataset_identity(manifest)
        scope = manifest.get("information_scope", "training")
        if scope in {"evaluation", "release", "sealed", "final", "test"}:
            raise PermissionError("independent evaluation data cannot enter training")
        if (
            manifest.get("scope_id", scope_id) != scope_id
            or manifest.get("protocol_id", protocol_id) != protocol_id
        ):
            raise PermissionError("dataset belongs to a different task/protocol")
        if any(
            row.get("information_scope", scope)
            in {"evaluation", "release", "sealed", "final", "test"}
            for row in manifest["episodes"]
        ):
            raise PermissionError("evaluation episode mixed into training data")
        before = dataset_file_snapshot(manifest)
        # A caller-owned cache is evidence to check, never independent admission.
        # Evaluation reads the actual payload even when the supplied snapshot matches.
        audit = audit_demonstration_payload(manifest)
        if payload_audit_ref:
            cached = json.loads(Path(payload_audit_ref).read_text())
            if cached.get("payload_snapshot") != before:
                raise ValueError(
                    "provided payload audit is stale or lacks payload identity"
                )
            if cached != audit:
                raise ValueError(
                    "provided payload audit does not match independent payload review"
                )
        if (
            audit.get("schema") != "rsi-demonstration-payload-audit-v1"
            or audit.get("dataset") != result["dataset"]
            or audit.get("checks_passed") is not True
        ):
            raise ValueError("payload audit rejected the dataset or its identity")
        if dataset_file_snapshot(manifest) != before:
            raise ValueError("dataset changed while its payload was being audited")
        atomic_json(root / "payload-audit.json", audit)
        result.update(
            payload_audit_ref=str(root / "payload-audit.json"),
            payload_snapshot=before,
            approved_training_tasks=sorted(
                {row["task"] for row in manifest["episodes"]}
            ),
        )
        missing = [
            name
            for name, ref in (
                ("quality_review", quality_review_ref),
                ("evaluation_isolation_review", evaluation_isolation_review_ref),
            )
            if not ref or not Path(ref).is_file()
        ]
        if missing:
            result["errors"] = [
                "missing_independent_evidence:" + name for name in missing
            ]
        else:
            result.update(
                decision="admitted",
                quality_review_ref=str(Path(quality_review_ref).resolve()),
                evaluation_isolation_review_ref=str(
                    Path(evaluation_isolation_review_ref).resolve()
                ),
                allowed_uses=["training", "preservation"],
            )
            atomic_json(receipt_ref, result)
            require_training_data_admission(
                manifest,
                admission_ref=receipt_ref,
                scope_id=scope_id,
                protocol_id=protocol_id,
                authority=reviewer_id,
            )
    except (OSError, ValueError, KeyError, TypeError, PermissionError) as exc:
        result.update(
            decision="rejected", allowed_uses=["preservation"], errors=[str(exc)]
        )
    atomic_json(receipt_ref, result)
    return {
        "decision": result["decision"],
        "admission_ref": str(receipt_ref),
        "artifact_ref": str(receipt_ref),
        "manifest_ref": str(manifest_ref),
        "dataset_root": manifest.get("dataset_root"),
        "errors": result["errors"],
        "model_capability": "unknown",
    }
