#!/usr/bin/env python3
"""Run a scoped prepared RLinf pi05 resource pilot, finite learning or load check."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.adapters.pi05_training import (
    build_learning_recipe,
    build_recipe,
    inspect_checkpoint,
    inspect_native_config,
    read_native_batch,
    require_training_data_admission,
    validate_dataset_manifest,
    verify_installed_sources,
)
from rsi_vla.training import (
    TrainingBudget,
    dynamic_lock,
    finalize_training_job,
    finalize_native_model_meter,
    require_scoped_training_permit,
    supervise_process,
    training_handoff,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--permit", required=True)
    parser.add_argument("--cuda-device", required=True, type=int)
    parser.add_argument("--verify-load", action="store_true")
    parser.add_argument("--training-result", help="actual completed trainer receipt for the load successor")
    parser.add_argument("--check-only", action="store_true", help="validate live admission and config without GPU launch")
    args = parser.parse_args()
    root = Path(args.prepared).resolve()
    permit = json.loads(Path(args.permit).read_text())
    recipe = json.loads((root / "recipe.json").read_text())
    model = recipe["actor"]["model"]
    run_id = recipe["runner"]["logger"]["experiment_name"]
    budget = TrainingBudget(**permit["budget"])
    checkpoint = None
    training_result_ref = args.training_result or permit.get("training_result_ref")
    if args.verify_load:
        trained, trained_handoff = training_handoff(training_result_ref, scope_id=permit["scope_id"])
        checkpoint = Path(trained_handoff["model_descriptor"]["model_path"])
    if args.cuda_device < 0:
        raise ValueError("one nonnegative CUDA device ID required")
    learning = recipe["rsi_training"].get("mode") == "learning"
    kind = "pi05-load-check" if args.verify_load else "pi05-training-learning" if learning else "pi05-training-pilot"
    artifact_root = root / (run_id + "-load-check" if args.verify_load else run_id)
    admission = require_scoped_training_permit(permit, run_id=run_id, budget=budget,
                            dataset_root=recipe["data"]["train_data_paths"],
                            model_path=checkpoint if args.verify_load else model["model_path"],
                            artifact_root=artifact_root, kind=kind, cuda_device=args.cuda_device)
    recipe_budget = TrainingBudget(**json.loads((root / "training-permit-proposal.json").read_text())["budget"])
    builder = build_learning_recipe if learning else build_recipe
    kwargs = {"learning_plan": recipe["rsi_training"]["learning_plan"]} if learning else {}
    expected = builder(dataset_root=recipe["data"]["train_data_paths"],
                            model_path=model["model_path"],
                            norm_stats_path=model["openpi_data"]["norm_stats_path"],
                            output_root=root, run_id=run_id, budget=recipe_budget if args.verify_load else budget, **kwargs)
    if recipe != expected:
        raise ValueError("recipe differs from registered bounded native training configuration")
    verify_installed_sources()
    inspect_checkpoint(checkpoint if args.verify_load else model["model_path"], model["openpi_data"]["norm_stats_path"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets-cache")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    validation = validate_dataset_manifest(manifest)
    if not args.verify_load:
        if learning and recipe["rsi_training"]["learning_plan"]["data_admission_ref"] != permit.get("data_admission_ref"):
            raise PermissionError("learning plan and permit bind different data admissions")
        if learning and any(recipe["rsi_training"]["learning_plan"][key] != permit.get(key)
                            for key in ("candidate_id", "protocol_id")):
            raise PermissionError("learning plan and permit candidate/protocol identities differ")
        require_training_data_admission(manifest, admission_ref=permit.get("data_admission_ref"),
            scope_id=permit["scope_id"], protocol_id=permit.get("protocol_id"),
            authority=permit.get("data_admission_authority"))
    if validation["dataset_root"] != recipe["data"]["train_data_paths"]:
        raise ValueError("manifest and recipe refer to different datasets")
    inspect_native_config(recipe)
    if args.verify_load and (budget.reserved_devices != 1 or budget.max_steps != 1 or budget.wall_seconds > 900
                            or budget.gpu_device_seconds > 900 or budget.artifact_bytes > 1024**3):
        raise ValueError("load check permits one reserved device/one prediction, at most 900s and 1GiB")
    if artifact_root.exists():
        raise ValueError("training run artifacts already exist; reconcile its receipt")
    if args.check_only:
        print(json.dumps({"status": "preflight_passed_no_launch", "admission": admission,
                          "kind": kind, "training_started": False}, indent=2))
        return 0
    if args.verify_load:
        batch = json.loads((root / run_id / "batch-check.json").read_text())
    else:
        batch = read_native_batch(recipe, manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.cuda_device),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "core/src"),
               PYTHONDONTWRITEBYTECODE="1", RAY_ADDRESS="local",
               HF_DATASETS_CACHE=str(artifact_root / "datasets-cache"),
               RSI_TRAINING_SCOPE_ID=permit["scope_id"],
               RSI_TRAINING_OPERATION_ID=permit["operation_id"], RSI_TRAINING_KIND=kind)
    if args.verify_load:
        command = [sys.executable, "-c",
                   "from rsi_vla.adapters.pi05_training import native_verify_load; import sys; native_verify_load(*sys.argv[1:])",
                   str(root / "recipe.json"), str(checkpoint), str(root / run_id / "probes-after.pt"),
                   str(artifact_root / "load-check.json")]
    else:
        command = [sys.executable, "-c",
                   "from rsi_vla.adapters.pi05_training import native_train; import sys; native_train(sys.argv[1])",
                   str(root / "recipe.json")]
    with dynamic_lock():
        require_scoped_training_permit(permit, run_id=run_id, budget=budget,
            dataset_root=recipe["data"]["train_data_paths"],
            model_path=checkpoint if args.verify_load else model["model_path"],
            artifact_root=artifact_root, kind=kind, cuda_device=args.cuda_device, require_submission=True)
        artifact_root.mkdir()
        (artifact_root / "batch-check.json").write_text(json.dumps(batch, indent=2) + "\n")
        try:
            report = supervise_process(command, budget=budget, artifact_root=artifact_root, env=env)
        finally:
            # The supervisor persists its closure before re-raising an interrupt
            # or monitoring error; retain a failed native receipt for the Host.
            supervision_ref = artifact_root / "supervision.json"
            if supervision_ref.exists():
                report = json.loads(supervision_ref.read_text())
                result = finalize_training_job(prepared_root=root, artifact_root=artifact_root,
                    permit=permit, supervision=report, python_executable=sys.executable,
                    training_result_ref=training_result_ref)
                finalize_native_model_meter(artifact_root=artifact_root, permit=permit,
                    supervision=report, result=result)
        succeeded = result["status"] == "completed"
    print(json.dumps(report, indent=2))
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
