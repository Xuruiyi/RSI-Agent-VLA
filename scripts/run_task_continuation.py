#!/usr/bin/env python3
# Copyright 2026 RSI-Agent-VLA Authors.
"""Advance the registered new-task continuation using existing role isolation."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.protocol10_roles import operator_command
from rsi_vla.task_continuation import TaskContinuationBackend, drive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "event", "advance", "reconcile", "drive")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--event", type=Path)
    parser.add_argument("--trigger-id")
    args = parser.parse_args(argv)
    config = json.loads(args.configuration.read_text())
    if config.get("offline_fixture"):
        raise ValueError("production CLI does not dispatch offline fixtures")
    role_status = operator_command(
        args.configuration,
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ],
    )
    if role_status is not None:
        return role_status
    backend = TaskContinuationBackend(args.root, config)
    coordinator = Coordinator(args.root / "coordinator", backend.registration, backend)

    def inputs():
        value = json.loads(args.inputs.read_text())
        value["now"] = datetime.now(timezone.utc).isoformat()
        state = coordinator.status()
        operation = state["pending"]
        if operation is None and state["status"] not in {
            "completed",
            "cancelled",
            "terminated",
        }:
            operation = backend.observe(state, value).get("next_operation")
        if operation is not None and operation["dynamic"]:
            from rsi_vla.protocol10_roles import broker_request

            fresh = broker_request(
                {"action": "task_admission", "operation_id": operation["operation_id"]}
            )
            value["resources"] = {**value.get("resources", {}), **fresh["resources"]}
            value["wait_remaining_seconds"] = fresh["wait_remaining_seconds"]
        return value

    if args.command == "status":
        result = coordinator.status()
    elif args.command == "event":
        result = coordinator.event(json.loads(args.event.read_text()))
    elif args.command == "drive":
        result = drive(
            coordinator, inputs, policy=config["continuation"]["resource_wait"]
        )
    else:
        result = coordinator.advance(
            inputs(),
            reconcile_only=args.command == "reconcile",
            trigger_id=args.trigger_id,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
