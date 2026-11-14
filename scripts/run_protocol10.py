#!/usr/bin/env python3
# ruff: noqa: UP017 - Python 3.10 support
# Copyright 2026 RSI-Agent-VLA Authors.
"""Inspect or advance one registered protocol10 operation; never create permits."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.protocol10_backend import Protocol10Backend, read
from rsi_vla.protocol10_coordinator import Coordinator


def main(argv=None):
    """Consume explicit JSON configuration, admission facts and typed events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "advance", "reconcile", "event"))
    parser.add_argument(
        "--cpu-boundary-factory",
        help="explicit offline fixture boundary; rejected for production",
    )
    parser.add_argument(
        "--trigger-id", help="stable trusted dispatch identity; never replays business"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument(
        "--inputs", type=Path, help="time, matching authorization and resource facts"
    )
    parser.add_argument("--event", type=Path, help="typed event JSON; never a command")
    args = parser.parse_args(argv)
    try:
        config = read(args.configuration)
        if args.cpu_boundary_factory and not config.get("offline_fixture"):
            raise ValueError("CPU boundary injection requires offline fixture scope")
        from rsi_vla.protocol10_roles import operator_command

        role_status = operator_command(
            args.configuration,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *(argv if argv is not None else sys.argv[1:]),
            ],
            boundary_factory=args.cpu_boundary_factory,
        )
        if role_status is not None:
            return role_status
        if config.get("offline_fixture") and not args.cpu_boundary_factory:
            raise ValueError(
                "offline fixtures require injected in-process boundaries and are not CLI dispatch configurations"
            )
        backend = Protocol10Backend(args.root, config)
        coordinator = Coordinator(
            args.root / "coordinator", backend.registration, backend
        )
        if args.command == "status":
            result = coordinator.status()
        elif args.command == "event":
            if not args.event:
                raise ValueError("event requires --event")
            result = coordinator.event(read(args.event))
        else:
            if not args.inputs:
                raise ValueError("advance/reconcile requires --inputs")
            inputs = read(args.inputs)
            inputs["now"] = datetime.now(timezone.utc).isoformat()
            result = coordinator.advance(
                inputs,
                reconcile_only=args.command == "reconcile",
                trigger_id=args.trigger_id,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "dispatch": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
