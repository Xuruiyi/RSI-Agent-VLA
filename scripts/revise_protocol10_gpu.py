#!/usr/bin/env python3
# Copyright 2026 RSI-Agent-VLA Authors.
"""Revise existing unsubmitted protocol10 GPU/source bindings without dispatch."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.protocol10_gpu import revise_registration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument(
        "--devices",
        type=Path,
        help="JSON object mapping registered run ID to physical GPU",
    )
    parser.add_argument("--refresh-source-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        devices = json.loads(args.devices.read_text()) if args.devices else {}
        report = revise_registration(
            args.configuration,
            args.root,
            args.revision_id,
            devices=devices,
            refresh_source_manifest=args.refresh_source_manifest,
        )
        print(
            json.dumps(
                {
                    "revision_id": report["revision_id"],
                    "status": report["status"],
                    "dispatch": False,
                    "binding": report["after_binding"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "dispatch": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
