#!/usr/bin/env python3
"""Run the task-scoped Autoloop adapter and detached operation supervision."""

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.midterm_host import (
    MidtermHost,
    build_autoloop_rules,
    build_configuration,
    worker,
    engineering_probe,
    prepare_deployment,
    launch_deployment,
    monitor_deployment,
    prove_unsubmitted_preparation,
    prepare_preparation_repair,
    activate_preparation_repair,
    observe_repair,
    reconcile_ended_runtime,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "serve",
            "worker",
            "supervise",
            "operation",
            "rules",
            "status",
            "prepare",
            "deploy",
            "launch",
            "monitor",
            "probe",
            "prove-preparation",
            "prepare-repair",
            "activate-repair",
            "observe",
            "reconcile-ended",
        ),
    )
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--operation-id")
    parser.add_argument("--continuation-configuration", type=Path)
    parser.add_argument("--hub-configuration", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--autoloop-root", type=Path)
    parser.add_argument("--port", type=int, default=3361)
    parser.add_argument("--probe-directory", type=Path)
    parser.add_argument("--probe-seconds", type=float, default=12)
    parser.add_argument("--repair-id")
    parser.add_argument("--monitor-pid", type=int)
    args = parser.parse_args()
    if args.command == "probe":
        engineering_probe(args.operation_id, args.probe_directory, args.probe_seconds)
        return 0
    if args.configuration is None:
        parser.error("--configuration is required")
    if args.command == "prove-preparation":
        print(
            json.dumps(
                prove_unsubmitted_preparation(args.configuration, args.operation_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "prepare-repair":
        print(
            json.dumps(
                prepare_preparation_repair(
                    args.configuration,
                    args.operation_id,
                    args.repair_id,
                    monitor_pid=args.monitor_pid,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "activate-repair":
        print(
            json.dumps(
                activate_preparation_repair(args.configuration), ensure_ascii=False
            )
        )
        return 0
    if args.command == "observe":
        observe_repair(args.configuration)
        return 0
    if args.command == "reconcile-ended":
        print(
            json.dumps(
                reconcile_ended_runtime(args.configuration, args.operation_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "deploy":
        result = prepare_deployment(
            args.continuation_configuration,
            args.configuration,
            node=args.node,
            autoloop_root=args.autoloop_root,
            port=args.port,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "launch":
        print(json.dumps(launch_deployment(args.configuration), ensure_ascii=False))
        return 0
    if args.command == "monitor":
        monitor_deployment(args.configuration)
        return 0
    if args.command == "prepare":
        from rsi_vla.storage import atomic_json

        config = build_configuration(
            args.continuation_configuration, args.configuration, args.hub_configuration
        )
        atomic_json(args.configuration, config)
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.command == "worker":
        print(worker(args.configuration))
        return 0
    host = MidtermHost(args.configuration)
    if args.command == "rules":
        print(
            json.dumps(build_autoloop_rules(host.config), ensure_ascii=False, indent=2)
        )
    elif args.command == "status":
        print(
            json.dumps(
                {"core": host.coordinator().status(), "accounting": host.ledger.read()},
                ensure_ascii=False,
            )
        )
    elif args.command == "serve":
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda _signal, _frame: stop.set())
        host.serve(stop)
    elif args.command == "supervise":
        host.supervise(args.operation_id)
    else:
        host.operation(args.operation_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
