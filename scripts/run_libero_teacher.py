#!/usr/bin/env python3
# Copyright 2026 RSI-Agent-VLA Authors.
"""Supervise one registered scripted-teacher simulation attempt without a VLA."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.adapters.libero_teacher import TEACHER_KIND, require_teacher_permit
from rsi_vla.training import dynamic_lock, supervise_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--permit", required=True)
    parser.add_argument("--cuda-device", required=True, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    permit_ref = str(Path(args.permit).resolve())
    registration_ref = str(Path(args.registration).resolve())
    permit = json.loads(Path(permit_ref).read_text())
    root = Path(permit["output_root"]).resolve()
    registration, assignment, budget = require_teacher_permit(
        registration_ref, permit, output_root=root, cuda_device=args.cuda_device
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed_no_launch",
                    "scope_id": permit["scope_id"],
                    "operation_id": permit["operation_id"],
                    "native_started": False,
                },
                indent=2,
            )
        )
        return 0
    if root.exists():
        raise FileExistsError(
            "teacher attempt identity already has artifacts; reconcile its receipt"
        )
    project = Path(__file__).resolve().parents[1]
    context = {
        "schema_version": "1.0",
        "run_id": permit["run_id"],
        "episode_id": permit["run_id"],
        "bundle_id": "scripted_teacher:" + registration["teacher_plan"]["teacher_id"],
        "protocol_id": permit["protocol_id"],
        "profile_ref": registration_ref,
        "code_ref": str(project / "core/src/rsi_vla/adapters/libero_teacher.py"),
        "source_manifest_ref": registration["source_manifest_ref"],
        "output_dir": str(root),
        "information_scope": "teacher_only_registered_native_geometry",
        "task": {**registration["teacher_plan"]["task"], "seed": assignment["seed"]},
        "limits": {
            "environment_steps": budget.max_steps,
            "tool_calls": permit["tool_call_limit"],
            "wall_seconds": budget.wall_seconds,
            "model_requests": 0,
        },
    }
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONPATH=str(project / "core/src") + os.pathsep + str(project / "RPent"),
        JAX_PLATFORMS="cpu",
        TMPDIR=str(root / "scratch"),
        XDG_CACHE_HOME=str(root / "cache"),
        MPLCONFIGDIR=str(root / "cache/matplotlib"),
        HF_DATASETS_CACHE=str(root / "cache/datasets"),
    )
    with dynamic_lock():
        require_teacher_permit(
            registration_ref,
            permit,
            output_root=root,
            cuda_device=args.cuda_device,
            require_submission=True,
        )
        root.mkdir(parents=True)
        (root / "scratch").mkdir()
        (root / "cache").mkdir()
        context_ref = root / "runtime-context.json"
        context_ref.write_text(json.dumps(context, indent=2) + "\n")
        (root / "teacher-registration-snapshot.json").write_text(
            json.dumps(registration, indent=2) + "\n"
        )
        command = [
            sys.executable,
            "-c",
            "from rsi_vla.adapters.libero_teacher import collect_native_teacher; import sys; collect_native_teacher(*sys.argv[1:])",
            registration_ref,
            permit_ref,
            str(context_ref),
        ]
        report = supervise_process(command, budget=budget, artifact_root=root, env=env)
        succeeded = (
            report["returncode"] == 0
            and report["stop_reason"] is None
            and report["cleanup_confirmed"]
            and report["resource_limits_satisfied"]
        )
        native_ref = root / "native-result.json"
        native = json.loads(native_ref.read_text()) if native_ref.is_file() else None
        result = {
            "schema": "rsi-libero-teacher-job-result-v1",
            "scope_id": permit["scope_id"],
            "operation_id": permit["operation_id"],
            "run_id": permit["run_id"],
            "kind": TEACHER_KIND,
            "status": "completed" if succeeded and native is not None else "failed",
            "native_result_ref": str(native_ref) if native is not None else None,
            "native_start_ref": str(root / "native-start.json"),
            "supervision_ref": str(root / "supervision.json"),
            "model_meter_ref": str(root / "model-meter.json"),
            "native_success": native.get("native_success")
            if native is not None
            else None,
            "collection_quality_pass": native.get("collection_quality_pass")
            if native is not None
            else None,
            "training_admission": "not_admitted",
            "source_manifest_ref": registration["source_manifest_ref"],
        }
        (root / "job-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
