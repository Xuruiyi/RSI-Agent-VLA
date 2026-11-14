"""Launcher-owned data provenance and admission; never converts traces to demonstrations."""

import json
from pathlib import Path
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

    entries = []
    for path in sorted(data_root.rglob("episode_*")):
        if not path.is_dir():
            continue
        complete, errors, persisted = False, [], 0
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
            metadata = {}
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
                "source_kind": "autonomous_execution",
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
                "training_admission": "not_admitted_autonomous_execution",
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
        "source_kind": "autonomous_execution",
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
        "loss_bound": "up to flush_steps-1 returned transitions plus in-flight RPC and incomplete write; no zero-loss guarantee",
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest
