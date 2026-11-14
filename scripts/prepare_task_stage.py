#!/usr/bin/env python3
"""Prepare one new-policy Runtime stage with its actual limits; never dispatch."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.improvement_protocol import admit_resources
from rsi_vla.runtime import prepare_runtime
from rsi_vla.storage import atomic_json
from rsi_vla.task_policy import resource_wait, select_gpu, task_policy


def prepare(protocol_ref, phase_ref, request):
    """Use Runtime's real per-attempt envelope plus disjoint cleanup and reserves."""
    protocol = json.loads(Path(protocol_ref).read_text())
    policy = task_policy(protocol)
    if policy.get("resource_admission") != "staged":
        raise ValueError(
            "stage preparation requires a new protocol with staged admission"
        )
    gpu = request["cuda_device"]
    if "gpu_cards" in request:
        gpu = select_gpu(request["gpu_cards"], policy["gpu_admission"])
        if gpu is None:
            return {
                "dispatch": False,
                "stage_preparation_ready": False,
                "resource_wait": resource_wait(
                    policy["resource_wait"],
                    waited_seconds=request["waited_seconds"],
                    offline_remaining_seconds=request["offline_remaining_seconds"],
                ),
            }
    plan = prepare_runtime(
        request["repository_root"],
        request["artifacts"],
        protocol_path=protocol_ref,
        run_id=request["run_id"],
        bundle_id=request["bundle_id"],
        task_key=request["task_key"],
        seed=request["seed"],
        split=request["split"],
        output_dir=request["output_dir"],
        cuda_device=gpu,
    )
    required = {**plan["context"]["limits"], "simulation_episodes": 1}
    phase = json.loads(Path(phase_ref).read_text())
    # Planner timeout is nested in wall time, not a second ledger resource.
    required.pop("planner_seconds", None)
    report = admit_resources(
        protocol,
        phase,
        offline_used_seconds=0,
        offline_used_bytes=0,
        next_required=required,
        cleanup=request["cleanup"],
        reserve=request["reserve"],
    )
    return {
        "plan": plan,
        "admission": report,
        "dispatch": False,
        "stage_preparation_ready": report["resource_fit"],
        "completion": "next stage only; later branches require fresh admission",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.protocol, args.phase, json.loads(args.request.read_text()))
    atomic_json(args.output, report)
    return 0 if report["stage_preparation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
