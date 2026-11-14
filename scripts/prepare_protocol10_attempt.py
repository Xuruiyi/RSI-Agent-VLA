#!/usr/bin/env python3
"""Prepare one protocol10 baseline Runtime plan; this entry never executes it."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))


def main() -> int:
    from rsi_vla.improvement_protocol import prepare_baseline_attempt

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "source-research",
        "preparation-root",
        "protocol",
        "frozen",
        "phase",
        "administrative-authorization",
        "permit",
        "output",
        "scratch",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--offline-used-seconds", type=float, required=True)
    parser.add_argument("--offline-used-bytes", type=int, required=True)
    parser.add_argument("--attempts", type=Path)
    args = parser.parse_args()
    try:
        report = prepare_baseline_attempt(
            root=args.root,
            source_research=args.source_research,
            preparation_root=args.preparation_root,
            protocol_path=args.protocol,
            frozen_path=args.frozen,
            phase=json.loads(args.phase.read_text()),
            administrative_authorization=args.administrative_authorization,
            permit_path=args.permit,
            run_id=args.run,
            task_key=args.task_key,
            seed=args.seed,
            output_dir=args.output,
            scratch_dir=args.scratch,
            offline_used_seconds=args.offline_used_seconds,
            offline_used_bytes=args.offline_used_bytes,
            cuda_device=args.cuda_device,
            attempts=json.loads(args.attempts.read_text()) if args.attempts else None,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"dispatch": False, "execute": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_separately_authorized_dispatch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
