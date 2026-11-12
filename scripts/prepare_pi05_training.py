#!/usr/bin/env python3
"""Prepare/check the installed pi05 chain; this command never trains."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.adapters.pi05_training import (build_recipe, checkpoint_tensor_inventory, dataset_acquisition_proposal,
    acquire_dataset_metadata, acquire_demonstration_files, materialize_demonstration_subset,
    inspect_checkpoint, inspect_native_config, read_native_batch,
    static_training_memory, validate_dataset_manifest, verify_installed_sources)
from rsi_vla.training import TrainingBudget, dynamic_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="pi05-resource-pilot")
    parser.add_argument("--read-batch", action="store_true")
    parser.add_argument("--acquire", choices=["metadata", "demonstrations"])
    parser.add_argument("--permit")
    parser.add_argument("--metadata-root")
    parser.add_argument("--subset-output")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    output = Path(args.output).resolve()
    os.environ["HF_DATASETS_CACHE"] = str(output / "datasets-cache")
    if args.acquire:
        if not args.permit:
            raise PermissionError("dataset acquisition requires a separate permit")
        permit = json.loads(Path(args.permit).read_text())
        with dynamic_lock():
            if args.acquire == "metadata":
                report = acquire_dataset_metadata(output, permit)
            else:
                if not args.metadata_root or not args.subset_output:
                    raise ValueError("demonstration acquisition requires --metadata-root and --subset-output")
                acquire_demonstration_files(args.metadata_root, output, permit)
                report = materialize_demonstration_subset(args.metadata_root, output, args.subset_output)
        print(json.dumps(report, indent=2))
        return
    if not args.model_path or not args.norm_stats_path:
        raise ValueError("offline preparation requires --model-path and --norm-stats-path")
    if output.exists():
        raise ValueError("prepare output must be a new directory")
    report = {"installed": verify_installed_sources(),
              "checkpoint": inspect_checkpoint(args.model_path, args.norm_stats_path),
              "data_acquisition_proposal": dataset_acquisition_proposal(),
              "dynamic_authorized": False, "status": "blocked_no_qualified_dataset"}
    safe = Path(args.model_path) / "model.safetensors"
    if safe.is_file():
        report["tensor_inventory"] = checkpoint_tensor_inventory(safe)
        report["static_training_memory"] = static_training_memory(safe)
    # Proposal only. Training and loading need their own explicit authorization.
    budget = TrainingBudget(900, 900, 1, 64 * 1024**3, 3)
    dataset_root = "/UNQUALIFIED_DATASET"
    manifest = None
    if args.dataset_manifest:
        manifest = json.loads(Path(args.dataset_manifest).read_text())
        report["dataset"] = validate_dataset_manifest(manifest)
        dataset_root = report["dataset"]["dataset_root"]
        report["status"] = "prepared_pending_training_authorization"
    output.mkdir(parents=True)
    norm_snapshot = output / "processor/physical-intelligence/libero/norm_stats.json"
    norm_snapshot.parent.mkdir(parents=True)
    shutil.copyfile(report["checkpoint"]["norm_stats"], norm_snapshot)
    report["processor_snapshot"] = str(norm_snapshot)
    recipe = build_recipe(dataset_root=dataset_root, model_path=args.model_path,
                          norm_stats_path=norm_snapshot, output_root=output,
                          run_id=args.run_id, budget=budget)
    report["native_config"] = inspect_native_config(recipe)
    if args.read_batch:
        if manifest is None:
            raise ValueError("--read-batch needs a qualified local dataset manifest")
        report["batch"] = read_native_batch(recipe, manifest)
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    permit = {"kind": "pi05-training-pilot", "authorized": False,
              "cuda_device": None,
              "run_id": args.run_id, "dataset_root": dataset_root,
              "model_path": str(Path(args.model_path).resolve()),
              "budget": budget.__dict__}
    (output / "training-permit-proposal.json").write_text(json.dumps(permit, indent=2) + "\n")
    if manifest is not None:
        (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
