#!/usr/bin/env python3
"""Freeze or prepare theta x B evaluation cells; never execute or publish."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.storage import atomic_json
from rsi_vla.theta_b import freeze_theta_b, prepare_theta_b_cell, evaluate_theta_b_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "prepare", "evaluate"])
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if Path(args.output).exists():
        raise FileExistsError("output already exists; preserve the original binding")
    if args.action == "freeze":
        result = freeze_theta_b(args.artifacts, **request)
    elif args.action == "prepare":
        result = prepare_theta_b_cell(Path(__file__).resolve().parents[1], args.artifacts, **request)
    else:
        result = evaluate_theta_b_cell(**request)
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
