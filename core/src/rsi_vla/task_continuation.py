# Copyright 2026 RSI-Agent-VLA Authors.
"""Result-driven continuation for new tasks over the existing role consumers."""

import json
import os
import time
from copy import deepcopy
from pathlib import Path

from .storage import atomic_json, identifier
from .task_policy import resource_wait

SCHEMA = "rsi-task-continuation-v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _field(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def probe_admission(configuration, operation_id, *, gpu_probe=None):
    """Host-read fresh phase and GPU facts for a registered operation; grant no authority."""
    from .task_policy import select_gpu, task_policy

    if (
        configuration.get("schema") != "rsi-task-completion-v1"
        or configuration.get("continuation", {}).get("schema") != SCHEMA
    ):
        raise PermissionError(
            "resource probe requires a registered new-task continuation"
        )
    matches = [
        op for op in configuration["operations"] if op["operation_id"] == operation_id
    ]
    if len(matches) != 1:
        raise ValueError("resource probe requires an exact registered operation")
    operation = matches[0]
    phase = _read(configuration["phase_ref"])
    blockers, telemetry = [], None
    if operation["kind"] == "runtime" and operation["dynamic"]:
        row = next(
            s
            for s in configuration["runtime_slots"]
            if s["run_id"] == operation["payload"]["run_id"]
        )
        policy = task_policy(_read(configuration["protocol_ref"]))["gpu_admission"]
        try:
            if gpu_probe is None:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.free,utilization.gpu,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                cards = []
                for line in result.stdout.splitlines():
                    index, name, free, load, capability = [
                        v.strip() for v in line.split(",")
                    ]
                    cards.append(
                        {
                            "physical_gpu": int(index),
                            "name": name,
                            "free_memory_mib": int(free),
                            "utilization_percent": int(load)
                            if load.isdigit()
                            else None,
                            "compute_capability": capability,
                        }
                    )
            else:
                cards = gpu_probe()
            assigned = [
                card for card in cards if card["physical_gpu"] == row["cuda_device"]
            ]
            telemetry = {"assigned": row["cuda_device"], "cards": assigned}
            if select_gpu(assigned, policy) != row["cuda_device"]:
                blockers.append("registered_gpu_unavailable")
        except (OSError, ValueError, RuntimeError) as exc:
            blockers.append("gpu_telemetry_unavailable:" + type(exc).__name__)
        except Exception as exc:
            # subprocess errors remain a resource wait, never assumed free GPU.
            import subprocess

            if not isinstance(exc, subprocess.SubprocessError):
                raise
            blockers.append("gpu_telemetry_unavailable:" + type(exc).__name__)
    return {
        "resources": {
            "available": phase["remaining"],
            "required": operation.get("resource_requirements", {}),
            "blockers": blockers,
            "unresolved_costs": phase.get("unresolved_costs", []),
        },
        "wait_remaining_seconds": phase["remaining"]["wall_seconds"],
        "gpu_telemetry": telemetry,
        "authorization_granted": False,
    }


class TaskContinuationBackend:
    """Select registered successors from receipts without repeating submissions.

    A finite graph references the existing TaskCompletionBackend operations.
    Conditions inspect role receipt values; they cannot supply commands, permits,
    or resource limits. The existing Coordinator owns pending intents and signals.
    """

    def __init__(self, root, configuration, *, invoke=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = deepcopy(configuration)
        graph = self.config["continuation"]
        if (
            self.config.get("schema") != "rsi-task-completion-v1"
            or graph.get("schema") != SCHEMA
        ):
            raise ValueError("continuation requires an explicitly registered new task")
        self.nodes = graph["nodes"]
        self.operations = {op["operation_id"]: op for op in self.config["operations"]}
        if len(self.operations) != len(self.config["operations"]):
            raise ValueError("duplicate operation identity")
        self.start = graph["start"]
        visiting, visited, referenced = set(), set(), set()

        def visit(key):
            identifier(key)
            if key in visiting:
                raise ValueError(
                    "continuation must be finite; use bounded registered attempts"
                )
            if key in visited:
                return
            visiting.add(key)
            node = self.nodes[key]
            if "terminal" in node:
                if set(node) != {"terminal"} or not isinstance(node["terminal"], str):
                    raise ValueError("terminal node requires a reason")
            else:
                op_id = node["operation_id"]
                if op_id not in self.operations or op_id in referenced:
                    raise ValueError(
                        "each continuation operation must have one owning node"
                    )
                referenced.add(op_id)
                edges = node["transitions"]
                if (
                    not edges
                    or "when" in edges[-1]
                    or any("when" not in e for e in edges[:-1])
                ):
                    raise ValueError("ordered branches require one final default")
                for edge in edges:
                    if "when" in edge:
                        condition = edge["when"]
                        if (
                            set(condition) != {"path", "equals"}
                            or not isinstance(condition["path"], list)
                            or not condition["path"]
                            or any(not isinstance(k, str) for k in condition["path"])
                        ):
                            raise ValueError(
                                "branch requires a receipt field path and equality"
                            )
                    visit(edge["next"])
            visiting.remove(key)
            visited.add(key)

        visit(self.start)
        if visited != set(self.nodes):
            raise ValueError("unreachable continuation node")
        registration = self.root / "registration.json"
        if registration.exists():
            if _read(registration) != self.config:
                raise ValueError("conflicting continuation registration")
        else:
            atomic_json(registration, self.config)
        self.registration = {
            "registration_ref": str(registration.resolve()),
            "scope_authorization_ref": self.config["scope_authorization_ref"],
            "protocol_ref": self.config["protocol_ref"],
            "schema": SCHEMA,
        }
        self.invoke = invoke or self._invoke

    def _invoke(self, operation, action, inputs):
        from .protocol10_roles import dispatch

        if os.environ.get("RSI_ISOLATED_ROLE") != "operator":
            raise PermissionError(
                "production continuation requires the existing host role broker"
            )
        return dispatch(
            self.config,
            self.root / "backend",
            action,
            operation=operation,
            inputs=inputs,
        )

    def _path(self, operation):
        return (
            self.root / "receipts" / (identifier(operation["operation_id"]) + ".json")
        )

    def _receipt(self, operation, value):
        return {
            "operation_id": operation["operation_id"],
            "status": "completed",
            "receipt_ref": str(self._path(operation).resolve()),
            "value": value,
        }

    def observe(self, progress, inputs):
        key = self.start
        while True:
            node = self.nodes[key]
            if "terminal" in node:
                return {"terminal": node["terminal"]}
            operation = self.operations[node["operation_id"]]
            receipt = progress["receipts"].get(operation["operation_id"])
            if receipt is None:
                return {"next_operation": deepcopy(operation)}
            stored = _read(receipt["receipt_ref"])
            if (
                stored.get("operation") != operation
                or stored.get("status") != "completed"
                or stored.get("value") != receipt.get("value")
            ):
                raise ValueError("successor requires the original operation receipt")
            for edge in node["transitions"]:
                condition = edge.get("when")
                if (
                    condition is None
                    or _field(receipt["value"], condition["path"])
                    == condition["equals"]
                ):
                    key = edge["next"]
                    break

    def execute(self, operation, inputs):
        path = self._path(operation)
        if path.exists():
            return self.reconcile(operation, inputs)
        atomic_json(path, {"status": "started", "operation": operation})
        value = self.invoke(operation, "execute", inputs)
        atomic_json(
            path, {"status": "completed", "operation": operation, "value": value}
        )
        return self._receipt(operation, value)

    def reconcile(self, operation, inputs):
        path = self._path(operation)
        if not path.exists():
            return {
                "operation_id": operation["operation_id"],
                "status": "not_submitted",
                "submission_absent": True,
                "receipt_ref": str(path),
            }
        saved = _read(path)
        if saved.get("operation") != operation:
            raise ValueError("conflicting native operation identity")
        if saved["status"] == "completed":
            return self._receipt(operation, saved["value"])
        from .protocol10_roles import ROLES
        from .task_completion import ACTION_ROLES

        role = ACTION_ROLES[operation["kind"]]
        role_root = Path(self.config["isolation_task_root"]) / ROLES[role]
        returns = sorted(
            (role_root / "role_receipts").glob(
                operation["operation_id"] + "-execute-revision-*.json"
            )
        )
        if returns:
            original = _read(returns[-1])
            if (
                original["request"]["configuration"] != self.config
                or original["request"]["task"]["operation"] != operation
            ):
                raise ValueError("original role return binding differs")
            if not original.get("error") and "value" in original:
                value = original["value"]
                atomic_json(
                    path,
                    {"status": "completed", "operation": operation, "value": value},
                )
                return self._receipt(operation, value)
        if operation["kind"] != "runtime":
            # Native Research reconcile consumes saved transport output only;
            # other non-dynamic role consumers use immutable store operations.
            value = self.invoke(operation, "reconcile", inputs)
            atomic_json(
                path, {"status": "completed", "operation": operation, "value": value}
            )
            return self._receipt(operation, value)
        return {
            "operation_id": operation["operation_id"],
            "status": "unknown",
            "reason": "submission_unknown",
            "receipt_ref": str(path),
        }


def drive(coordinator, load_inputs, *, policy, sleep=time.sleep, clock=time.monotonic):
    """Advance until a durable stop, with bounded fresh resource rechecks.

    The caller supplies fresh admission inputs on every iteration. Unknown native
    submissions, manual pauses and terminal outcomes return immediately. Only
    resource waits are automatically polled; elapsed waits survive process restart.
    """
    path = coordinator.root / "resource-wait.json"
    record = _read(path) if path.exists() else {"waited_seconds": 0.0, "checks": 0}
    if "active_wait_started" in record:
        elapsed = clock() - record["active_wait_started"]
        if elapsed < 0:
            raise RuntimeError(
                "wait clock changed; reconcile original wait before continuing"
            )
        record["waited_seconds"] += elapsed
        del record["active_wait_started"]
        atomic_json(path, record)
    while True:
        state = coordinator.advance(load_inputs())
        if state["status"] in {"completed", "terminated", "cancelled", "paused"}:
            return state
        reason = (state.get("wait") or {}).get("reason")
        if reason is None:
            continue
        if reason not in {
            "budget_wait",
            "resource_evidence_required",
            "waiting_resources",
        }:
            return state
        wait = resource_wait(
            policy,
            waited_seconds=record["waited_seconds"],
            offline_remaining_seconds=load_inputs().get("wait_remaining_seconds", 0),
        )
        if wait["wait_budget_exhausted"]:
            return state
        began = clock()
        record["active_wait_started"] = began
        atomic_json(path, record)
        sleep(min(30.0, wait["next_check_seconds"]))
        record["waited_seconds"] += max(0.0, clock() - began)
        del record["active_wait_started"]
        record["checks"] += 1
        atomic_json(path, record)
