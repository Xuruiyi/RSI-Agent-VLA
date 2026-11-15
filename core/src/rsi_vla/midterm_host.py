"""Trusted task adapter using Autoloop queues/timers and existing role consumers.

The command worker only wakes this host. Registered operations, permits, resource
reservations and detached whole-command supervisors remain host-owned.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .storage import atomic_json, identifier
from .task_continuation import TaskContinuationBackend

COUNTERS = frozenset(
    {
        "runtime_starts",
        "training_starts",
        "training_pilots",
        "learning_candidates",
        "load_checks",
        "gpu_recoveries",
        "external_worker_rounds",
    }
)
CAPACITIES = frozenset({"concurrent_gpu_devices", "active_coding_agents"})
_children = []


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("UTC authorization timestamp required")
    return result


def numbers(values):
    if not isinstance(values, dict) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in values.values()
    ):
        raise ValueError("finite nonnegative resource vector required")
    return values


def runtime_permit_binding(configuration, slot, *, authorization_path=None):
    """Derive permit bindings through the real no-side-effect Runtime planner."""
    from .runtime import prepare_runtime

    plan = prepare_runtime(
        configuration["repository_root"],
        configuration["artifacts_root"],
        protocol_path=configuration["protocol_ref"],
        run_id=slot["run_id"],
        bundle_id=slot["bundle_id"],
        task_key=slot["task_key"],
        seed=slot["seed"],
        split=slot["split"],
        output_dir=slot["output_ref"],
        authorization_path=authorization_path,
        cuda_device=slot["cuda_device"],
        storage_write_roots=[slot["scratch_ref"]],
    )
    return {
        key: deepcopy(plan["context"][key])
        for key in ("collection", "code_ref", "storage_write_roots")
        if key in plan["context"]
    }


class MidtermContinuationBackend(TaskContinuationBackend):
    """Consume host-owned ended-operation facts without replacing native results."""

    def __init__(self, host):
        self.host = host
        super().__init__(
            host.config["continuation_root"], host.core_config, invoke=host.invoke
        )

    def observe(self, progress, inputs):
        if any(row.get('value',{}).get('research_policy_rejected') is True for row in progress['receipts'].values()):
            return {'terminal':'research_policy_rejected'}
        stopped = [row['value'] for row in progress['receipts'].values() if row.get('value',{}).get('research_technical_stop') is True]
        if stopped:
            return {'terminal': stopped[-1].get('technical_terminal', 'research_pre_forward_technical_stop')}
        return super().observe(progress,inputs)

    def reconcile(self, operation, inputs):
        path = (
            self.host.root
            / "jobs"
            / operation["operation_id"]
            / "host-ended-receipt.json"
        )
        if not path.exists():
            template = self.host.config['operations'].get(operation['operation_id'], {})
            if template.get('command') and template.get('permit_ref'):
                from .fixed_command import reconcile_fixed_backend
                receipt = reconcile_fixed_backend(self, operation)
                if receipt is not None:
                    return receipt
            return super().reconcile(operation, inputs)
        value = read(path)
        if (
            value.get("operation") != operation
            or value.get("status") != "completed"
            or value.get("scope_id") != self.host.config["scope_id"]
            or value.get("owner_uid") != os.getuid()
                or (value.get("value", {}).get("native_outcome") != "unknown" and value.get("value", {}).get("research_technical_stop") is not True and value.get('value',{}).get('research_policy_rejected') is not True)
            or value["value"].get("execution_ended") is not True
        ):
            raise PermissionError(
                "host-ended receipt does not bind the original operation"
            )
        report = read(value["value"]["supervisor_ref"])
        cpu_preparation = value['value'].get('research_preparation_technical_stop') is True
        if cpu_preparation and (operation['kind'] != 'research_prepare' or operation['dynamic'] is not False):
            raise PermissionError('CPU preparation receipt cannot close a native operation')
        policy_rejected = value['value'].get('research_policy_rejected') is True
        if policy_rejected and (operation['kind'] != 'research' or operation['dynamic'] is not True):
            raise PermissionError('policy rejection cannot close another native operation')
        report_closed = (report.get('status') == 'completed' and report.get('exit_code') == 1) if cpu_preparation or policy_rejected else report.get('status') == 'interrupted'
        if (
            report.get("identity", {}).get("run_id") != operation["operation_id"]
            or not report_closed
            or report.get("cleanup_confirmed") is not True
            or report.get("remaining_process_ids") != []
        ):
            raise PermissionError(
                "original supervisor no longer supports ended-operation closure"
            )
        return {
            "operation_id": operation["operation_id"],
            "status": "completed",
            "receipt_ref": str(path),
            "value": value["value"],
        }


class MidtermLedger:
    """Reserve against the live task ledger and retain uncertain costs.

    Args:
        scope_authorization_ref: Current task authorization, never a historical scope.
        accounting_ref: Shared authoritative ledger, defaulting to write_root/accounting.json.
    """

    def __init__(self, scope_authorization_ref, accounting_ref=None):
        self.scope_ref = Path(scope_authorization_ref).resolve()
        scope = read(self.scope_ref)
        self.root = Path(scope["write_root"]).resolve()
        self.path = Path(accounting_ref or self.root / "accounting.json").resolve()
        if not self.path.is_relative_to(self.root):
            raise PermissionError("accounting must remain in the task write domain")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock():
            if not self.path.exists():
                limits = numbers(scope["outer_limits"])
                atomic_json(
                    self.path,
                    {
                        "schema": "rsi-midterm-accounting-v1",
                        "scope_id": scope["scope_id"],
                        "limits": limits,
                        "used": dict.fromkeys(limits, 0),
                        "reservations": {},
                        "unresolved_costs": [],
                    },
                )

    @contextmanager
    def lock(self):
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def authority(self, *, active=True):
        scope = read(self.scope_ref)
        if (
            scope.get("schema") != "rsi-midterm-scope-v1"
            or scope.get("authorized") is not True
            or scope.get("owner_uid") != os.getuid()
            or self.scope_ref.stat().st_uid != os.getuid()
            or Path(scope["write_root"]).resolve() != self.root
            or self.root
            not in [Path(p).resolve() for p in scope["storage_write_roots"]]
        ):
            raise PermissionError("scope owner or write domain mismatch")
        if active and (
            scope.get("status") != "active"
            or scope.get("dynamic_authorized") is not True
            or utcnow() >= timestamp(scope["expires_at"])
        ):
            raise PermissionError("task paused, closed or expired")
        return scope

    def _read(self):
        value = read(self.path)
        scope = self.authority(active=False)
        if (
            value.get("schema") != "rsi-midterm-accounting-v1"
            or value.get("scope_id") != scope["scope_id"]
            or value["limits"] != scope["outer_limits"]
        ):
            raise PermissionError("ledger differs from original task limits")
        numbers(value["used"])
        return value

    def read(self):
        with self.lock():
            return self._read()

    @staticmethod
    def _held(record, dimension):
        if record["status"] not in {"active", "unknown"}:
            return 0
        if dimension in record.get("charged_counters", []):
            return 0
        if (
            record["status"] == "unknown"
            and record.get("cleanup_confirmed") is True
            and dimension in record.get("actual", {})
            and dimension not in record.get("missing", [])
        ):
            return 0
        return max(
            0,
            record["budget"].get(dimension, 0)
            - record.get("actual", {}).get(dimension, 0),
        )

    @staticmethod
    def _phase(record, budget):
        phase = read(record["phase_ref"])
        if (
            phase.get("unresolved_costs") and not record.get("evidence_closure")
        ) or any(
            value > phase["remaining"].get(key, -1)
            for key, value in budget.items()
            if key in phase["limits"]
        ):
            raise PermissionError(
                "live phase remaining resources cannot admit operation"
            )

    def reserve(
        self,
        operation_id,
        *,
        budget,
        phase_ref,
        gpu_devices,
        receipt_ref,
        evidence_closure=False,
    ):
        """Persist one reservation before starting its independent supervisor."""
        identifier(operation_id)
        numbers(budget)
        phase_ref, receipt_ref = Path(phase_ref).resolve(), Path(receipt_ref).resolve()
        if (
            not phase_ref.is_relative_to(self.root)
            or not receipt_ref.is_relative_to(self.root)
            or len(set(gpu_devices)) != len(gpu_devices)
            or any(type(device) is not int or device < 0 for device in gpu_devices)
        ):
            raise PermissionError(
                "reservation paths/devices are outside task registration"
            )
        with self.lock():
            scope = self.authority()
            state = self._read()
            record = {
                "scope_id": scope["scope_id"],
                "operation_id": operation_id,
                "owner_uid": scope["owner_uid"],
                "status": "active",
                "phase_ref": str(phase_ref),
                "budget": budget,
                "gpu_devices": gpu_devices,
                "receipt_ref": str(receipt_ref),
            }
            if evidence_closure:
                if gpu_devices or set(budget) - {"storage_bytes", "current_disk_bytes"}:
                    raise PermissionError(
                        "evidence closure cannot reserve model, GPU or new execution starts"
                    )
                record["evidence_closure"] = True
            old = state["reservations"].get(operation_id)
            if old:
                if any(
                    old.get(key) != value
                    for key, value in record.items()
                    if key != "status"
                ):
                    raise ValueError("conflicting operation reservation")
                return deepcopy(old)
            if state.get("unresolved_costs") and not evidence_closure:
                raise PermissionError("unresolved task costs require reconciliation")
            if set(budget) - (state["limits"].keys() - CAPACITIES):
                raise ValueError("unregistered hard resource dimension")
            for dimension, amount in budget.items():
                held = sum(
                    self._held(item, dimension)
                    for item in state["reservations"].values()
                )
                if dimension == 'storage_bytes':
                    held += sum(row['upper_bound_bytes'] for row in state.get('historical_control_uncertainties', []))
                if (
                    state["used"].get(dimension, 0) + held + amount
                    > state["limits"][dimension]
                ):
                    raise PermissionError("outer task budget exhausted: " + dimension)
            allocated = [
                device
                for item in state["reservations"].values()
                if item["status"] == "active"
                or (
                    item["status"] == "unknown"
                    and item.get("cleanup_confirmed") is not True
                )
                for device in item["gpu_devices"]
            ]
            if (
                set(allocated).intersection(gpu_devices)
                or len(set(allocated + gpu_devices))
                > scope["outer_limits"]["concurrent_gpu_devices"]
            ):
                raise PermissionError("GPU lease already held or concurrency exceeded")
            self._phase(record, budget)
            record["reserved_at"] = utcnow().isoformat()
            state["reservations"][operation_id] = record
            atomic_json(self.path, state)
            return deepcopy(record)

    def validate_reservation(
        self, operation_id, *, budget=None, phase_ref=None, gpu_devices=None
    ):
        """Recheck live authority and the exact reservation at the actual consumer."""
        with self.lock():
            scope = self.authority()
            record = self._read()["reservations"][identifier(operation_id)]
            if (
                record["status"] != "active"
                or record["scope_id"] != scope["scope_id"]
                or record["owner_uid"] != os.getuid()
                or (
                    phase_ref is not None
                    and Path(phase_ref).resolve() != Path(record["phase_ref"])
                )
                or (gpu_devices is not None and gpu_devices != record["gpu_devices"])
            ):
                raise PermissionError("operation lacks its live matching reservation")
            required = numbers(budget or record["budget"])
            if any(
                value > record["budget"].get(key, -1) for key, value in required.items()
            ):
                raise PermissionError("consumer exceeds reserved envelope")
            self._phase(record, required)
            return deepcopy(record)

    def mark_submitted(self, operation_id):
        """Record committed submission; native markers establish actual launch counts."""
        with self.lock():
            self.authority()
            state = self._read()
            record = state["reservations"][operation_id]
            if record.get("submitted_at"):
                return
            if record["status"] != "active":
                raise PermissionError("cannot submit a closed reservation")
            record["submitted_at"] = utcnow().isoformat()
            atomic_json(self.path, state)

    def register_candidate(self, candidate_id, binding):
        """Freeze one learning candidate's data, recipe and selection rules by identity."""
        identifier(candidate_id)
        if not isinstance(binding, dict) or not binding:
            raise ValueError(
                "candidate requires its registered data/recipe/stopping binding"
            )
        with self.lock():
            scope = self.authority()
            state = self._read()
            candidates = state.setdefault("learning_candidate_bindings", {})
            value = {
                "candidate_id": candidate_id,
                "scope_id": scope["scope_id"],
                "binding": binding,
            }
            if candidate_id in candidates and candidates[candidate_id] != value:
                raise ValueError(
                    "candidate identity already binds different data/recipe/stopping rules"
                )
            candidates[candidate_id] = value
            atomic_json(self.path, state)
            return deepcopy(value)

    def settle(self, operation_id, *, actual, cleanup_confirmed, unknown_dimensions=()):
        """Charge measured cumulative use once and hold all unknown remainder."""
        numbers(actual)
        with self.lock():
            state = self._read()
            record = state["reservations"][operation_id]
            if record["status"] == "settled":
                if record["actual"] != actual or not cleanup_confirmed or unknown_dimensions:
                    raise ValueError("conflicting settled operation receipt")
                return
            previous = record.get("actual", {})
            if any(
                actual.get(key, -1) < value
                for key, value in previous.items()
                if key != "current_disk_bytes"
            ):
                raise ValueError("cannot remove observed operation costs")
            for key, value in actual.items():
                if key not in state["limits"] or key in CAPACITIES:
                    raise ValueError("actual costs must use measured task dimensions")
                if key == "current_disk_bytes":
                    # This dimension is an inventory, not cumulative materialization.
                    state["used"][key] = sum(
                        path.stat().st_size
                        for path in self.root.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    )
                else:
                    state["used"][key] = (
                        state["used"].get(key, 0) + value - previous.get(key, 0)
                    )
            record["actual"] = actual
            if set(unknown_dimensions) - record["budget"].keys():
                raise ValueError("unknown costs must remain within registered resource dimensions")
            missing = (set(record["budget"]) - actual.keys()) | set(unknown_dimensions)
            record["status"] = (
                "settled" if cleanup_confirmed and not missing else "unknown"
            )
            record["cleanup_confirmed"] = cleanup_confirmed
            record["missing"] = sorted(missing)
            unresolved = [
                item
                for item in state["unresolved_costs"]
                if item.get("operation_id") != operation_id
            ]
            if record["status"] == "unknown":
                unresolved.append(
                    {
                        "operation_id": operation_id,
                        "missing": sorted(missing),
                        "cleanup_confirmed": cleanup_confirmed,
                    }
                )
            state["unresolved_costs"] = unresolved
            atomic_json(self.path, state)


def worker_receipt(state, *, scope_id, poll_seconds):
    """Use real Autoloop later timers; a timer never resumes a paused task."""
    status = state["status"]
    later = [{"key": scope_id + "-wake", "cancel": True}]
    if status in {"running", "ready", "waiting_resources"}:
        later = [
            {
                "key": scope_id + "-wake",
                "at": (utcnow() + timedelta(seconds=poll_seconds)).isoformat(),
                "about": scope_id,
            }
        ]
    return {"summary": scope_id + "：" + status, "ask": [], "later": later}


def build_autoloop_rules(config):
    """Return a fixed deterministic command plugin, with no external notifications."""
    scope = identifier(config["scope_id"])
    agent = scope + "-driver"
    return {
        "timezone": "UTC",
        "askFirst": [],
        "sources": {scope: {"name": scope}},
        "agents": {
            agent: {
                "name": "受限业务 CLI",
                "worker": scope,
                "use": "command",
                "command": [
                    config["python"],
                    config["script_ref"],
                    "worker",
                    "--configuration",
                    config["configuration_ref"],
                ],
            }
        },
        "channels": {},
        "rules": [
            {
                "name": scope,
                "when": {"source": scope, "types": ["midterm.*"]},
                "to": agent,
                "cwd": config["repository_root"],
                "group": scope,
                "resume": False,
                "tell": "never",
            }
        ],
    }


def build_configuration(
    continuation_configuration_ref,
    configuration_ref,
    hub_configuration_ref,
    *,
    python=None,
    script_ref=None,
    poll_seconds=30,
):
    """Build finite operation templates from the registered current-task consumer."""
    core = read(continuation_configuration_ref)
    scope = read(core["scope_authorization_ref"])
    root = Path(scope["write_root"])
    operations = {}
    protocol = read(core["protocol_ref"])
    for operation in core["operations"]:
        kind = operation["kind"]
        template = {
            "budget": {
                "storage_bytes": 64 * 1024**2,
                "current_disk_bytes": 64 * 1024**2,
            },
            "gpu_devices": [],
            "supervisor_limits": {"wall_seconds": 180, "storage_bytes": 64 * 1024**2},
        }
        if kind == "runtime":
            slot = next(
                row
                for row in core["runtime_slots"]
                if row["run_id"] == operation["payload"]["run_id"]
            )
            limits = {
                key: value
                for key, value in protocol["budget"]["runtime_per_start"].items()
                if key != "simulation_episodes"
            }
            limits["planner_seconds"] = protocol["budget"]["per_attempt_proposal"][
                "planner_timeout_seconds"
            ]
            template.update(
                budget={
                    "runtime_starts": 1,
                    "gpu_seconds": limits["gpu_seconds"],
                    "model_requests": limits["model_requests"],
                    "storage_bytes": limits["storage_bytes"],
                    "current_disk_bytes": limits["storage_bytes"],
                },
                gpu_devices=[slot["cuda_device"]],
                output_ref=slot["output_ref"],
                start_marker_ref=str(Path(slot["output_ref"]) / "runtime_context.json"),
                permit_template={"limits": limits},
                supervisor_limits={
                    key: limits[key]
                    for key in ("wall_seconds", "storage_bytes", "gpu_seconds")
                },
            )
        elif kind == "research":
            request = next(
                row
                for row in core["research_requests"]
                if row["request_id"] == operation["payload"]["request_id"]
            )
            limits = {
                **protocol["task_policy"]["research_per_request"],
                "model_requests": 1,
            }
            template.update(
                budget={
                    "model_requests": 1,
                    "research_requests": 1,
                    "storage_bytes": limits["storage_bytes"],
                    "current_disk_bytes": limits["storage_bytes"],
                },
                output_ref=request["output_ref"],
                permit_template={"limits": limits},
                supervisor_limits={
                    key: limits[key] for key in ("wall_seconds", "storage_bytes")
                },
            )
        if core.get('research_request_envelope') == 'complete_v2' and operation.get('payload', {}).get('request_id'):
            cap = protocol['task_policy']['research_per_request']
            template['budget'].update(storage_bytes=cap['storage_bytes'], current_disk_bytes=cap['storage_bytes'])
            template['supervisor_limits'].update(wall_seconds=cap['wall_seconds'], storage_bytes=cap['storage_bytes'])
        operations[operation["operation_id"]] = template
    return {
        "schema": "rsi-midterm-host-v1",
        "scope_id": scope["scope_id"],
        "scope_authorization_ref": core["scope_authorization_ref"],
        "accounting_ref": str(root / "accounting.json"),
        "configuration_ref": str(configuration_ref),
        "hub_configuration_ref": str(hub_configuration_ref),
        "repository_root": core["repository_root"],
        "python": python or sys.executable,
        "script_ref": script_ref
        or str(Path(core["repository_root"]) / "scripts/run_midterm_host.py"),
        "continuation_configuration_ref": str(continuation_configuration_ref),
        "continuation_root": str(
            Path(continuation_configuration_ref).parent / "continuation"
        ),
        "inputs_ref": str(Path(continuation_configuration_ref).parent / "inputs.json"),
        "transport_root": str(root / "autoloop/transport"),
        "poll_seconds": poll_seconds,
        "operations": operations,
    }


class MidtermHost:
    """Thin business adapter; Autoloop owns the queue, timers and worker lifecycle."""

    def __init__(self, configuration_ref):
        self.configuration_ref = Path(configuration_ref).resolve()
        self.config = read(self.configuration_ref)
        if self.config.get("schema") != "rsi-midterm-host-v1":
            raise ValueError("explicit new midterm host configuration required")
        self.ledger = MidtermLedger(
            self.config["scope_authorization_ref"], self.config.get("accounting_ref")
        )
        scope = self.ledger.authority(active=False)
        if (
            self.config["scope_id"] != scope["scope_id"]
            or self.configuration_ref.stat().st_uid != scope["owner_uid"]
        ):
            raise PermissionError("host configuration owner/scope mismatch")
        self.root = Path(
            self.config.get("host_root", self.ledger.root / "operator/midterm-host")
        ).resolve()
        if not self.root.is_relative_to(self.ledger.root):
            raise PermissionError("host state escapes the task write domain")
        self.root.mkdir(parents=True, exist_ok=True)
        registration = (
            self.root
            / "registrations"
            / (identifier(self.configuration_ref.stem) + ".json")
        )
        if registration.exists() and read(registration) != self.config:
            raise ValueError(
                "host configuration changed; register a new reviewed configuration identity"
            )
        atomic_json(registration, self.config)
        self.core_config = read(self.config["continuation_configuration_ref"])
        if self.core_config["scope_authorization_ref"] != str(self.ledger.scope_ref):
            raise PermissionError("continuation belongs to a different scope")
        self.transport = Path(
            self.config.get("transport_root", self.ledger.root / "autoloop/transport")
        )
        if not self.transport.resolve().is_relative_to(self.ledger.root):
            raise PermissionError("worker transport escapes task write domain")
        for name in ("requests", "responses"):
            (self.transport / name).mkdir(parents=True, exist_ok=True)

    def backend(self):
        return MidtermContinuationBackend(self)

    def coordinator(self):
        from .protocol10_coordinator import Coordinator

        backend = self.backend()
        return Coordinator(
            Path(self.config["continuation_root"]) / "coordinator",
            backend.registration,
            backend,
        )

    def inputs(self):
        value = read(self.config["inputs_ref"])
        value["now"] = utcnow().isoformat()
        state = self.coordinator().status()
        operation = state["pending"] or self.backend().observe(state, value).get(
            "next_operation"
        )
        if (
            operation
            and operation["dynamic"]
            and not (
                self.root
                / "jobs"
                / operation["operation_id"]
                / "host-ended-receipt.json"
            ).exists()
        ):
            from .task_continuation import probe_admission

            fresh = probe_admission(self.core_config, operation["operation_id"])
            value["resources"] = {**value.get("resources", {}), **fresh["resources"]}
            template = self.config["operations"][operation["operation_id"]]
            if "command" in template and template.get("gpu_devices"):
                from .fixed_command import probe_fixed_gpu
                gpu = probe_fixed_gpu(template)
                value["resources"]["blockers"] = [*value["resources"].get("blockers", []), *gpu["blockers"]]
                value["fixed_gpu_admission"] = gpu["gpu_telemetry"]
        return value

    def invoke(self, operation, action, inputs):
        """Dispatch standard operations through their existing OS-isolated roles."""
        from .phase_receipts import commit
        from .protocol10_roles import run_role
        from .protocol10_roles import ROLES

        template = self.config["operations"][operation["operation_id"]]
        if "command" in template:
            if action != "execute":
                if template.get('permit_ref'):
                    from .fixed_command import fixed_result
                    return fixed_result(self,operation,template)
                raise RuntimeError(
                    "registered command submission is unknown; inspect its original receipt"
                )
            returned = subprocess.run(
                template["command"], cwd=self.config["repository_root"], check=False
            )
            if returned.returncode and not Path(template["result_ref"]).exists():
                raise RuntimeError(
                    "registered command failed: " + str(returned.returncode)
                )
            if template.get('permit_ref'):
                from .fixed_command import fixed_result
                return fixed_result(self,operation,template)
            receipt = read(template["result_ref"])
            if receipt.get("operation_id") != operation["operation_id"]:
                raise ValueError("registered command result identity mismatch")
            if (
                receipt.get("scope_id", self.config["scope_id"])
                != self.config["scope_id"]
            ):
                raise ValueError("registered command result scope mismatch")
            return receipt
        for role, relative in ROLES.items():
            (Path(self.core_config["isolation_task_root"]) / relative).mkdir(
                parents=True, exist_ok=True
            )
            (Path(self.core_config["isolation_task_root"]) / "scratch" / role).mkdir(
                parents=True, exist_ok=True
            )
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)

        task = {
            "backend_root": str(Path(self.config["continuation_root"]) / "backend"),
            "action": action,
            "operation": operation,
            "inputs": inputs,
        }
        returned = run_role(self.core_config, task)
        if returned.get("error"):
            raise RuntimeError(returned["error"])
        for proposal in returned.get("phase_proposals", []):
            commit(proposal, self.core_config["phase_ref"])
        return returned["value"]

    def materialize(self, operation):
        """Create only the current operation's exact permit and Research context."""
        kind = operation["kind"]
        template = self.config["operations"][operation["operation_id"]]
        if "command" in template:
            from .fixed_command import fixed_permit, fixed_storage_context
            fixed_storage_context(self, operation, template)
            fixed_permit(self, operation, template)
            return
        from .research_supervision import lifecycle
        request_envelope = lifecycle(self,operation)
        if kind == "research_prepare":
            self.select_research_input(operation["payload"]["request_id"])
        if not operation["dynamic"] or kind not in {"runtime", "research"}:
            return
        from .task_completion import TaskCompletionBackend

        native = TaskCompletionBackend(self.core_config)
        scope, _admin = native.authority()
        cfg = next(
            row
            for row in self.core_config[
                "runtime_slots" if kind == "runtime" else "research_requests"
            ]
            if row["run_id" if kind == "runtime" else "request_id"]
            == operation["payload"]["run_id" if kind == "runtime" else "request_id"]
        )
        target = Path(cfg["permit_ref"]).resolve()
        output, scratch = (
            Path(cfg["output_ref"]).resolve(),
            Path(cfg["scratch_ref"]).resolve(),
        )
        if any(
            not path.is_relative_to(self.ledger.root)
            for path in (target, output, scratch)
        ):
            raise PermissionError("operation write domain escapes scope")
        permit = deepcopy(template["permit_template"])
        permit.update(
            authorized=True,
            scope=scope["scope_id"],
            protocol_id=native.protocol["protocol_id"],
            phase_snapshot_ref=self.core_config["phase_ref"],
            administrative_authorization_ref=self.core_config[
                "administrative_authorization_ref"
            ],
            source_manifest_ref=self.core_config["source_manifest_ref"],
            output_dir=cfg["output_ref"],
            expires_at=scope["expires_at"],
            storage_write_roots=[str(output), str(scratch)],
            operation_id=operation["operation_id"],
            accounting_ref=str(self.ledger.path),
            owner_uid=scope["owner_uid"],
        )
        if kind == "runtime":
            permit.update(runtime_permit_binding(self.core_config, cfg))
            permit.update(
                run_id=cfg["run_id"],
                bundle_id=cfg["bundle_id"],
                task_key=cfg["task_key"],
                seed=cfg["seed"],
                simulation_episodes=1,
                gpu_devices=[cfg["cuda_device"]],
                approval_reference=self.core_config["scope_authorization_ref"],
            )
        else:
            tools, cfg = native.tools(cfg["request_id"])
            transport = tools.store.get("bounded_transports", cfg["request_id"])
            if cfg.get("relation_operator"):
                gates = tools.domain_policy["relation_admission"]
                admission = tools.research_relation_admission(
                    cfg["result_ids"],
                    legal_repairs=[cfg["relation_operator"]],
                    selection_id=cfg["request_id"],
                    technical_admitted=gates["technical_admitted"],
                    data_admitted=gates["data_admitted"],
                )
                atomic_json(
                    tools.store.path("followup_admission_receipts", cfg["request_id"]),
                    {
                        "request_id": cfg["request_id"],
                        "selection": admission,
                        "consumer": "ResearchTools.research_relation_admission",
                        "model_forwarded": False,
                    },
                )
                if admission["action"] != "prepare":
                    raise PermissionError("Research context admission deferred")
            from .research_preflight import operator_preflight

            preflight = operator_preflight(
                tools,
                self.core_config["repository_root"],
                protocol_path=self.core_config["protocol_ref"],
                result_ids=cfg["result_ids"],
            )
            atomic_json(cfg["preflight_ref"], preflight)
            permit.update(
                schema="rsi-bounded-research-permit-v1",
                job_id=cfg["job_id"],
                request_id=cfg["request_id"],
                bundle_id=tools.base.bundle_id,
                job_limits=tools.job.state["limits"],
                **scope["models"]["research"],
            )
            if request_envelope is not None:
                from .research_supervision import compact_input_binding
                permit.update(compact_input_binding(tools.store.path('bounded_transports',cfg['request_id'])))
                left = (datetime.fromisoformat(request_envelope['deadline'])-utcnow()).total_seconds()
                if left <= 0:
                    raise PermissionError('complete Research request deadline expired before permit materialization')
                permit['limits']['wall_seconds'] = min(permit['limits']['wall_seconds'],left)
            else:
                permit['input'] = transport
        if target.exists() and read(target) != permit:
            raise ValueError("existing current-operation permit differs")
        atomic_json(target, permit)
        native.validate_dynamic(kind, operation["payload"])
        if kind == "runtime":
            # Validate the same complete permit in the actual downstream planner
            # before publishing a host submission intent or reserving a start.
            runtime_permit_binding(
                self.core_config, cfg, authorization_path=str(target)
            )

    def select_research_input(self, request_id):
        """Materialize the registered deterministic L0 selection after independent import."""
        from .task_completion import TaskCompletionBackend

        native = TaskCompletionBackend(self.core_config)
        tools, cfg = native.tools(request_id)
        policy = read(cfg["domain_policy_ref"])
        selection = policy.get("evidence_selection")
        if not selection:
            return
        results = tools.query_development(cfg["result_ids"])
        native.preserve_development_views(results)
        ids = {}
        for result in results:
            view = tools.store.get(
                "evidence_views", result["intervention"]["evidence_view_id"]
            )
            events = view["l0"]
            chosen = [event for event in events if event["kind"] in selection["kinds"]][
                : selection["first_matches"]
            ]
            chosen += (
                events[-selection["last_events"] :] if selection["last_events"] else []
            )
            wanted = {event["event_id"] for event in chosen}
            ids[result["result_id"]] = [
                event["event_id"] for event in events if event["event_id"] in wanted
            ]
        policy["evidence_input"] = {
            "mode": "selected_l0",
            "reason": "registered first matching kinds plus final events; full L0 retained",
            "event_ids": ids,
        }
        for field in ("media_refs", "sdk_projection"):
            if field in selection:
                policy["evidence_input"][field] = selection[field]
        atomic_json(cfg["domain_policy_ref"], policy)

    def advance(self):
        """Submit one current operation; repeated wakes reconcile its original job."""
        _children[:] = [child for child in _children if child.poll() is None]
        # Independent supervisor facts remain readable while the native role owns
        # its Coordinator and business locks for a potentially eight-hour job.
        for submission in sorted((self.root / "jobs").glob("*/submission.json")):
            job = submission.parent
            if (job / "host-ended-receipt.json").exists() or (job/'fixed-reconciled.json').exists():
                continue
            if (job / "completion.json").exists():
                if (
                    read(job / "completion.json")["status"] != "completed"
                    and not (job / "host-ended-receipt.json").exists()
                ):
                    template = self.config['operations'].get(job.name,{})
                    if template.get('command') and template.get('permit_ref') and process_matches(job/'process.json'):
                        return {'status':'running','operation_id':job.name,'submission_ref':str(submission)}
                    from .fixed_command import recover_fixed_completion
                    reconciled = recover_fixed_completion(self, job)
                    if reconciled is not None:
                        return reconciled
                    operation = next(row for row in self.core_config['operations'] if row['operation_id'] == job.name)
                    if operation['kind'] == 'research' and (job/'supervisor.json').exists():
                        from .research_supervision import research_policy_rejection_evidence, close_policy_rejected_research
                        report = read(job/'supervisor.json')
                        if report.get('status') == 'completed' and report.get('exit_code') == 1:
                            if process_matches(job/'process.json'):
                                return {'status':'running','operation_id':job.name,'submission_ref':str(submission)}
                            if research_policy_rejection_evidence(self,job.name,report) is not None:
                                close_policy_rejected_research(self,job.name,report)
                                return {'status':'ready','operation_id':job.name,'policy_rejection_closed':True}
                    return {"status": "unknown", "operation_id": job.name}
                continue
            if process_matches(job / "process.json"):
                return {
                    "status": "running",
                    "operation_id": job.name,
                    "submission_ref": str(submission),
                }
            if (job / "supervisor.json").exists():
                self.finish(job.name, read(job / "supervisor.json"))
                return {"status": "ready", "operation_id": job.name}
            return {
                "status": "unknown",
                "operation_id": job.name,
                "submission_ref": str(submission),
            }
        coordinator = self.coordinator()
        state = coordinator.status()
        scope = self.ledger.authority(active=False)
        if (
            state["status"] in {"completed", "terminated", "cancelled"}
            or state["paused"]
        ):
            return {"status": state["status"], "core": state}
        if scope["status"] != "active":
            return {"status": "paused"}
        if utcnow() >= timestamp(scope["expires_at"]):
            return {"status": "expired"}
        fresh_inputs = self.inputs()
        operation = state["pending"] or self.backend().observe(state, fresh_inputs).get(
            "next_operation"
        )
        if operation is None:
            return {"status": coordinator.advance(self.inputs())["status"]}
        identity = operation["operation_id"]
        job = self.root / "jobs" / identifier(identity)
        intent = job / "submission.json"
        receipt = job / "completion.json"
        if intent.exists():
            saved = read(intent)
            if saved["operation"] != operation:
                raise ValueError("conflicting original supervisor submission")
            if (job / "host-ended-receipt.json").exists():
                reconciled = coordinator.advance(self.inputs(), reconcile_only=True)
                return {"status": "ready", "core": reconciled, "operation_id": identity}
            if receipt.exists():
                report = read(receipt)
                if report["status"] != "completed":
                    return {
                        "status": "unknown",
                        "operation_id": identity,
                        "receipt_ref": str(receipt),
                    }
                return {"status": "ready", "operation_id": identity}
            return {
                "status": "running"
                if process_matches(job / "process.json")
                else "unknown",
                "operation_id": identity,
                "submission_ref": str(intent),
            }
        template = self.config["operations"][identity]
        if template.get("candidate_id"):
            self.ledger.register_candidate(
                template["candidate_id"], template["candidate_binding"]
            )
        if operation["dynamic"] and fresh_inputs.get("resources", {}).get("blockers"):
            return self.resource_wait()
        if utcnow() + timedelta(
            seconds=template["supervisor_limits"]["wall_seconds"]
        ) >= timestamp(scope["expires_at"]):
            return {
                "status": "expired",
                "reason": "whole_operation_cannot_fit_authorization",
            }
        if template.get("admission_lease_ref"):
            with Path(template["admission_lease_ref"]).open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return self.resource_wait()
        self.materialize(operation)
        self.ledger.reserve(
            identity,
            budget=template["budget"],
            phase_ref=self.core_config["phase_ref"],
            gpu_devices=template.get("gpu_devices", []),
            receipt_ref=receipt,
            evidence_closure=(
                operation["kind"] in {"audit", "development_base_collect"}
                and not operation["dynamic"]
                and any(
                    item.get("value", {}).get("execution_ended") is True
                    for item in state["receipts"].values()
                )
            ),
        )
        if fresh_inputs.get("fixed_gpu_admission") is not None:
            atomic_json(job / "gpu-admission.json", fresh_inputs["fixed_gpu_admission"])
        wait_ref = self.root / "resource-wait.json"
        if wait_ref.exists():
            wait = read(wait_ref)
            wait["active"] = False
            atomic_json(wait_ref, wait)
        atomic_json(
            intent,
            {
                "operation": operation,
                "owner_uid": os.getuid(),
                "at": utcnow().isoformat(),
            },
        )
        self.ledger.mark_submitted(identity)
        process = subprocess.Popen(
            [
                self.config["python"],
                self.config["script_ref"],
                "supervise",
                "--configuration",
                str(self.configuration_ref),
                "--operation-id",
                identity,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atomic_json(job / "process.json", process_identity(process.pid))
        _children.append(process)
        return {
            "status": "running",
            "operation_id": identity,
            "supervisor_pid": process.pid,
            "submission_ref": str(intent),
        }

    def resource_wait(self):
        """Account actual time between resource rechecks without owning a timer."""
        path = self.root / "resource-wait.json"
        record = read(path) if path.exists() else {"waited_seconds": 0}
        now = utcnow()
        elapsed = (
            max(0, (now - timestamp(record["last_check_at"])).total_seconds())
            if record.get("last_check_at") and record.get("active", True)
            else 0
        )
        with self.ledger.lock():
            state = self.ledger._read()
            remaining = (
                state["limits"]["resource_wait_seconds"]
                - state["used"]["resource_wait_seconds"]
            )
            charged = min(remaining, elapsed)
            state["used"]["resource_wait_seconds"] += charged
            atomic_json(self.ledger.path, state)
        record.update(
            waited_seconds=record["waited_seconds"] + charged,
            last_check_at=now.isoformat(),
            active=True,
        )
        atomic_json(path, record)
        return {
            "status": "waiting_resources" if remaining > elapsed else "waiting",
            "wait": record,
        }

    def operation(self, identity):
        """Execute through the existing Coordinator's durable intent/receipt path."""
        self.ledger.validate_reservation(
            identity, phase_ref=self.core_config["phase_ref"]
        )
        coordinator = self.coordinator()
        state = coordinator.status()
        operation = state["pending"] or self.backend().observe(
            state, self.inputs()
        ).get("next_operation")
        if operation is None or operation["operation_id"] != identity:
            raise PermissionError("operation is no longer the registered successor")
        value = coordinator.advance(
            self.inputs(), reconcile_only=state["pending"] is not None
        )
        if self.core_config.get('research_receipt_mode') == 'reference_v1' and operation.get('payload', {}).get('request_id'):
            saved_value = {'status': value['status'], 'operation_id': identity,
                           'receipt': value['receipts'].get(identity),
                           'coordinator_progress_ref': str(Path(self.config['continuation_root'])/'coordinator/progress.json')}
        else:
            saved_value = value
        atomic_json(self.root / "jobs" / identity / "core-result.json", saved_value)
        if identity not in value["receipts"]:
            raise RuntimeError(
                "original operation did not yield a completed receipt: "
                + str(value.get("wait"))
            )

    def supervise(self, identity):
        """Own a complete job independently of both command worker and host daemon."""
        from .adapters.runtime_supervisor import supervise
        from .process_accounting import StorageMeter

        identity = identifier(identity)
        template = self.config["operations"][identity]
        job = self.root / "jobs" / identity
        record = self.ledger.validate_reservation(
            identity, phase_ref=self.core_config["phase_ref"]
        )
        context = {
            "dynamic_authorized": True,
            "authorization_ref": str(self.ledger.scope_ref),
            "run_id": identity,
            "episode_id": "host-operation",
            "bundle_id": self.core_config["base_bundle_id"],
            "protocol_id": read(self.core_config["protocol_ref"])["protocol_id"],
            "output_dir": str(job),
            "gpu_count": len(record["gpu_devices"]),
            "limits": template["supervisor_limits"],
            "storage_write_roots": [str(self.ledger.root)],
        }
        from .research_supervision import complete_context
        operation = next(item for item in self.core_config['operations'] if item['operation_id']==identity)
        context = complete_context(self,operation,context)
        if "command" in template and template.get("permit_ref"):
            from .fixed_command import fixed_storage_context
            context.update(fixed_storage_context(self, operation, template))
        if not context.get("storage_meter_state_ref"):
            context["storage_baseline"] = StorageMeter(
                context["storage_write_roots"]
            ).baseline
        atomic_json(job / "runtime_context.json", context)
        report = supervise(
            [
                self.config["python"],
                self.config["script_ref"],
                "operation",
                "--configuration",
                str(self.configuration_ref),
                "--operation-id",
                identity,
            ],
            context_path=job / "runtime_context.json",
            terminate_grace_s=10,
        )
        self.finish(identity, report)

    def finish(self, identity, report):
        """Reconcile a saved whole-command report without submitting any operation."""
        if report.get("identity", {}).get("run_id") != identity:
            raise ValueError("original supervisor report identity mismatch")
        operation = next(
            item
            for item in self.core_config["operations"]
            if item["operation_id"] == identity
        )
        if (
            operation["kind"] == "runtime"
            and report.get("status") == "interrupted"
            and report.get("cleanup_confirmed") is True
        ):
            self.close_ended_runtime(identity, report)
            return
        if operation['kind']=='research' and report.get('status')=='interrupted' and report.get('cleanup_confirmed') is True:
            self.close_ended_research(identity,report)
            return
        template = self.config["operations"][identity]
        job = self.root / "jobs" / identity
        self.charge_phase(identity, report)
        actual = {}
        budget = template["budget"]
        fixed = "command" in template and template.get("permit_ref") is not None
        for counter in (() if fixed else COUNTERS.intersection(budget)):
            marker = template.get("start_marker_ref")
            if marker is not None:
                actual[counter] = budget[counter] if Path(marker).exists() else 0
        if "gpu_seconds" in budget:
            actual["gpu_seconds"] = report["gpu_device_seconds_envelope"]
        if "storage_bytes" in budget:
            actual["storage_bytes"] = report.get('business_storage_delta_bytes',report["storage_bytes"])
        # Native consumers own request telemetry. Read actual forward records, never estimates.
        output = Path(template["output_ref"]) if template.get("output_ref") else None
        if (
            "model_requests" in budget
            and output is not None
            and (output / "request_gate.json").exists()
        ):
            from .adapters.runtime_closure import read_model_events

            gate = read(output / "request_gate.json")
            native_context = read(output / "runtime_context.json")
            identity_fields = {
                key: native_context[key]
                for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
            }
            events = read_model_events(output, gate, identity_fields)
            if not gate["cleanup_complete"] or gate["active_requests_at_close"]:
                raise RuntimeError(
                    "native request gate has unresolved in-flight requests"
                )
            actual["model_requests"] = sum(
                event.get("kind") == "request_admitted" for event in events
            )
            if "research_requests" in budget:
                actual["research_requests"] = actual["model_requests"]
        if "current_disk_bytes" in budget:
            actual["current_disk_bytes"] = report["storage_accounting"][
                "current_new_bytes"
            ]
        measurements = None
        if fixed:
            from .fixed_command import charge_fixed_phase, fixed_measurements
            measurements = fixed_measurements(self, identity, template, report)
            actual.update(measurements["actual"])
            charge_fixed_phase(self, identity, measurements)
        completed = (
            report["status"] == "completed"
            and report["exit_code"] == 0
            and report["cleanup_confirmed"]
        )
        self.ledger.settle(
            identity, actual=actual, cleanup_confirmed=report["cleanup_confirmed"],
            unknown_dimensions=measurements["unknown_dimensions"] if measurements else (),
        )
        from .research_supervision import bill_control
        bill_control(self,identity,report)
        atomic_json(
            job / "completion.json",
            {
                "scope_id": self.config["scope_id"],
                "operation_id": identity,
                "status": "completed" if completed else "unknown",
                "supervisor_ref": str(job / "supervisor.json"),
                "actual": actual,
                "cleanup_confirmed": report["cleanup_confirmed"],
                **({"fixed_receipt_evidence": measurements["evidence"], "unknown_dimensions": measurements["unknown_dimensions"]} if measurements else {}),
            },
        )

    def charge_phase(self, identity, report):
        """Reuse native Runtime settlement, then book only whole-command overhead."""
        if any(
            item["operation_id"] == identity
            for item in read(self.core_config["phase_ref"]).get(
                "host_ended_cost_reconciliations", []
            )
        ):
            return
        operation = next(
            item
            for item in self.core_config["operations"]
            if item["operation_id"] == identity
        )
        template = self.config["operations"][identity]
        native_wall, native_storage, native_gpu = 0, 0, 0
        job_native_wall, job_native_storage = 0, 0
        output = Path(template["output_ref"]) if template.get("output_ref") else None
        if (
            operation["kind"] == "runtime"
            and output is not None
            and (output / "runtime_context.json").exists()
        ):
            from .adapters.runtime_acceptance import inspect_attempt
            from .research_cycle import _phase_charge

            audit = inspect_attempt(output, require_vla=False)
            costs = {
                key: value
                for key, value in audit["costs"].items()
                if type(value) in (int, float)
            }
            slot = next(
                item
                for item in self.core_config["runtime_slots"]
                if item["run_id"] == operation["payload"]["run_id"]
            )
            _phase_charge(
                {**read(slot["permit_ref"]), "permit_ref": slot["permit_ref"]},
                slot["run_id"],
                costs,
                output,
                charge_kind="protocol10_trial_charges",
            )
            native_wall = costs.get("whole_job_wall_seconds", 0)
            native_storage = costs.get("artifact_bytes", 0)
            native_gpu = costs.get("gpu_device_seconds_envelope", 0)
        elif (
            operation["kind"] == "research"
            and output is not None
            and (output / "execution-costs.json").exists()
        ):
            costs = read(output / "execution-costs.json")["costs"]
            request_id = operation['payload']['request_id']
            phase_native = read(self.core_config['phase_ref'])
            if any(row['request_id'] == request_id for row in phase_native.get('research_charges', [])):
                native_wall = costs.get('whole_job_wall_seconds', 0)
                native_storage = costs.get('artifact_bytes', 0)
                if 'business_storage_delta_bytes' in report:
                    native_storage += sum(row['storage_bytes'] for row in phase_native.get('derived_storage_charges', []) if row['key'] == request_id+'-derived')
            from .storage import ResearchStore
            request = next(row for row in self.core_config['research_requests'] if row['request_id'] == request_id)
            nested = ResearchStore(self.core_config['research_store_ref']).get('jobs', request['job_id'])
            for charge in nested['charges']:
                if charge['reason'] in {'llm '+request_id, 'failed LLM startup '+request_id} or charge['reason'].startswith('new derived evidence storage '+request_id+'-derived '):
                    job_native_wall += charge['costs'].get('whole_job_wall_seconds', 0)
                    job_native_storage += charge['costs'].get('artifact_bytes', 0)
        phase = read(self.core_config["phase_ref"])
        entry = {
            "operation_id": identity,
            "wall_seconds": max(0, report["wall_seconds"] - native_wall),
            "storage_bytes": max(0, report.get('business_storage_delta_bytes',report["storage_bytes"]) - native_storage),
            "supervisor_ref": str(self.root / "jobs" / identity / "supervisor.json"),
        }
        if "gpu_seconds" in template["budget"]:
            entry["gpu_seconds"] = max(
                0, report["gpu_device_seconds_envelope"] - native_gpu
            )
        from .research_supervision import bill_research_overhead
        job_entry = entry
        if operation['kind'] == 'research':
            job_entry = {**entry,'wall_seconds':max(0,report['wall_seconds']-job_native_wall),
                         'storage_bytes':max(0,report.get('business_storage_delta_bytes',report['storage_bytes'])-job_native_storage)}
        bill_research_overhead(self, operation, job_entry)
        entries = phase.setdefault("host_command_charges", [])
        existing = [item for item in entries if item["operation_id"] == identity]
        if existing:
            if existing != [entry]:
                raise ValueError("conflicting host command overhead receipt")
            return
        entries.append(entry)
        for key in ("wall_seconds", "storage_bytes", "gpu_seconds"):
            if key not in entry:
                continue
            phase["used"][key] += entry[key]
            phase["remaining"][key] = max(0, phase["limits"][key] - phase["used"][key])
        atomic_json(self.core_config["phase_ref"], phase)

    def close_ended_research(self, operation_id, report):
        from .research_supervision import close_ended_research
        return close_ended_research(self,operation_id,report)

    def close_ended_runtime(self, operation_id, report):
        """Record controlled execution end, retaining unknown native task outcome."""
        operation = next(
            item
            for item in self.core_config["operations"]
            if item["operation_id"] == operation_id
        )
        if operation["kind"] != "runtime":
            raise ValueError("ended Runtime closure requires a Runtime operation")
        job = self.root / "jobs" / identifier(operation_id)
        closure_ref = job / "host-ended-receipt.json"
        if closure_ref.exists():
            return read(closure_ref)
        slot = next(
            item
            for item in self.core_config["runtime_slots"]
            if item["run_id"] == operation["payload"]["run_id"]
        )
        context = read(Path(slot["output_ref"]) / "runtime_context.json")
        expected = {
            "run_id": slot["run_id"],
            "episode_id": f"{slot['task_key']}-s{slot['seed']}",
            "bundle_id": slot["bundle_id"],
            "protocol_id": read(self.core_config["protocol_ref"])["protocol_id"],
        }
        if (
            any(context.get(key) != value for key, value in expected.items())
            or read(job / "submission.json")["operation"] != operation
            or report.get("identity", {}).get("run_id") != operation_id
            or report.get("status") != "interrupted"
            or report.get("cleanup_confirmed") is not True
            or report.get("remaining_process_ids") != []
        ):
            raise PermissionError(
                "supervisor/context does not establish this ended Runtime operation"
            )
        process_evidence = inspect_owned_runtime_processes(
            self.root / "host-process.json",
            operation_id,
            slot["output_ref"],
            operation_anchor_ref=job / "process.json",
        )
        if (
            process_evidence["live_processes"]
            or process_evidence["unreadable_owned_processes"]
        ):
            raise PermissionError(
                "Runtime owners/forwarders have not been fully reconciled"
            )
        model_requests, request_evidence = ended_request_count(
            slot["output_ref"], context, process_evidence
        )
        record = self.ledger.read()["reservations"][operation_id]
        prior_reservation_ref = job / "closure-control/previous-reservation.json"
        if not prior_reservation_ref.exists():
            immutable_json(prior_reservation_ref, record)
        actual = dict(record.get("actual", {}))
        actual.update(
            runtime_starts=1,
            gpu_seconds=report["gpu_device_seconds_envelope"],
            storage_bytes=report["storage_bytes"],
        )
        if "current_disk_bytes" in record["budget"]:
            actual["current_disk_bytes"] = report["storage_accounting"][
                "current_new_bytes"
            ]
        if model_requests is not None:
            actual["model_requests"] = model_requests
        self.ledger.settle(operation_id, actual=actual, cleanup_confirmed=True)
        native_costs = ended_native_counts(slot["output_ref"], expected)
        self.reconcile_ended_phase(
            operation_id, slot, report, model_requests, native_costs
        )
        source_ref = capture_closure_sources(self, job / "closure-control/source")
        closure = {
            "schema": "rsi-host-ended-operation-receipt-v1",
            "scope_id": self.config["scope_id"],
            "owner_uid": os.getuid(),
            "operation": operation,
            "status": "completed",
            "value": {
                "operation_id": operation_id,
                "run_id": slot["run_id"],
                "execution_ended": True,
                "execution_status": "interrupted",
                "native_outcome": "unknown",
                "valid_task_result": False,
                "termination_class": "budget_stop"
                if report.get("stop_reason")
                in {"storage_bytes", "wall_seconds", "gpu_seconds_envelope"}
                else "controlled_interruption",
                "stop_reason": report.get("stop_reason"),
                "supervisor_ref": str(job / "supervisor.json"),
                "runtime_context_ref": str(
                    Path(slot["output_ref"]) / "runtime_context.json"
                ),
                "closure_source_manifest_ref": str(source_ref),
                "previous_ledger_reservation_ref": str(prior_reservation_ref),
                "process_evidence": process_evidence,
                "request_evidence": request_evidence,
                "model_requests": model_requests,
                "actual": actual,
                "native_counts": native_costs,
                "technical_recovery": False,
                "resubmitted": False,
                "native_results_synthesized": False,
            },
        }
        immutable_json(closure_ref, closure)
        return closure

    def reconcile_ended_phase(
        self, operation_id, slot, report, model_requests, native_costs
    ):
        """Resolve measured dimensions without charging the same outer envelope twice."""
        path = self.core_config["phase_ref"]
        phase = read(path)
        rows = phase.setdefault("host_ended_cost_reconciliations", [])
        if any(row["operation_id"] == operation_id for row in rows):
            return
        observed = {
            "wall_seconds": report["wall_seconds"],
            "gpu_seconds": report["gpu_device_seconds_envelope"],
            "storage_bytes": report["storage_bytes"],
            **native_costs,
        }
        if model_requests is not None:
            observed["model_requests"] = model_requests
        native_entries = [
            row
            for row in phase.get("protocol10_trial_charges", [])
            if row["run_id"] == slot["run_id"]
        ]
        host_entries = [
            row
            for row in phase.get("host_command_charges", [])
            if row["operation_id"] == operation_id
        ]
        if len(native_entries) != 1 or len(host_entries) != 1:
            # New interruptions first use the existing settlement consumer once.
            self.charge_phase(operation_id, report)
            phase = read(path)
            rows = phase.setdefault("host_ended_cost_reconciliations", [])
            native_entries = [
                row
                for row in phase.get("protocol10_trial_charges", [])
                if row["run_id"] == slot["run_id"]
            ]
            host_entries = [
                row
                for row in phase.get("host_command_charges", [])
                if row["operation_id"] == operation_id
            ]
        if len(native_entries) != 1 or len(host_entries) != 1:
            raise PermissionError(
                "ended-operation costs lack their original phase receipts"
            )
        booked = native_entries[0]["costs"]
        newly_charged = {}
        for key, value in observed.items():
            already = booked.get(key, 0) + host_entries[0].get(key, 0)
            if already > value:
                raise ValueError(
                    "ended-operation observation removes previously charged resources"
                )
            if value > already:
                newly_charged[key] = value - already
                phase["used"][key] += value - already
                phase["remaining"][key] = max(
                    0, phase["limits"][key] - phase["used"][key]
                )
        mapping = {
            "whole_job_wall_seconds": "wall_seconds",
            "gpu_device_seconds_envelope": "gpu_seconds",
            "artifact_bytes": "storage_bytes",
            "environment_steps": "environment_steps",
            "tool_calls": "tool_calls",
            "model_requests": "model_requests",
        }
        previous_unresolved = deepcopy(phase.get("unresolved_costs", []))
        for unresolved in phase.get("unresolved_costs", []):
            if (
                unresolved.get("run_id") == slot["run_id"]
                and unresolved.get("output_ref") == slot["output_ref"]
            ):
                unresolved["missing"] = [
                    key
                    for key in unresolved["missing"]
                    if mapping.get(key, key) not in observed
                ]
        phase["unresolved_costs"] = [
            row for row in phase.get("unresolved_costs", []) if row.get("missing") != []
        ]
        rows.append(
            {
                "operation_id": operation_id,
                "run_id": slot["run_id"],
                "observed": observed,
                "newly_charged": newly_charged,
                "existing_native_and_host_entries_retained": True,
                "previous_unresolved_costs": previous_unresolved,
                "supervisor_ref": str(
                    self.root / "jobs" / operation_id / "supervisor.json"
                ),
            }
        )
        atomic_json(path, phase)

    def publish_completions(self):
        """Publish stable completion events; repeated HTTP sends are source-deduplicated."""
        hub = read(self.config["hub_configuration_ref"])
        if not hub["url"].startswith(("http://127.0.0.1:", "http://localhost:")):
            raise PermissionError("midterm hub must listen locally")
        for receipt in sorted((self.root / "jobs").glob("*/completion.json")):
            published = receipt.parent / "event-published.json"
            if published.exists():
                continue
            value = read(receipt)
            event = {
                "id": "complete-" + value["operation_id"],
                "type": "midterm.completed",
                "subject": self.config["scope_id"],
                "summary": value["status"],
                "detail": str(receipt),
            }
            request = urllib.request.Request(
                hub["url"] + "/v1/events",
                data=json.dumps(event).encode(),
                headers={
                    "Authorization": "Bearer " + hub["source_key"],
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                atomic_json(
                    published, {"event": event, "response": json.load(response)}
                )
        for receipt in sorted((self.root / "jobs").glob("*/host-ended-receipt.json")):
            published = receipt.parent / "ended-event-published.json"
            if published.exists():
                continue
            value = read(receipt)
            event = ended_completion_event(
                self.config["scope_id"], value["operation"]["operation_id"], receipt
            )
            atomic_json(
                published,
                {
                    "event": event,
                    "response": hub_request(self.config, "/v1/events", event=event),
                },
            )

    def serve(self, stop):
        """Serve quick wake requests; all scheduling stays in Autoloop."""
        with (self.root / "host.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(self.root / "host-process.json", process_identity(os.getpid()))
            while not stop.is_set():
                for request in sorted((self.transport / "requests").glob("*.json")):
                    response = self.transport / "responses" / request.name
                    if response.exists():
                        continue
                    value = read(request)
                    try:
                        if (
                            set(value) != {"request_id", "scope_id", "action"}
                            or value["request_id"] != request.stem
                            or value["scope_id"] != self.config["scope_id"]
                            or value["action"] != "wake"
                        ):
                            raise ValueError("unregistered worker request")
                        result = self.advance()
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        RuntimeError,
                        PermissionError,
                    ) as exc:
                        result = {
                            "status": "blocked",
                            "error": type(exc).__name__ + ": " + str(exc),
                        }
                    atomic_json(response, result)
                try:
                    self.publish_completions()
                except (OSError, ValueError) as exc:
                    atomic_json(
                        self.root / "event-error.json",
                        {"error": str(exc), "at": utcnow().isoformat()},
                    )
                stop.wait(0.1)


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "owner_uid": os.getuid(),
        "start_ticks": stat[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_matches(path):
    if not Path(path).exists():
        return False
    saved = read(path)
    try:
        if (
            Path(f"/proc/{saved['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
            == "Z"
        ):
            return False
        current = process_identity(saved["pid"])
        return (
            all(current[key] == saved[key] for key in current)
            and Path(f"/proc/{saved['pid']}").stat().st_uid == saved["owner_uid"]
        )
    except (OSError, ProcessLookupError):
        return False


def fresh_runtime_actor_evidence(output):
    """Establish this D run spawned its actors inside a fresh role namespace."""
    root = Path(output)
    evidence = {"established": False}
    try:
        context = read(root / "runtime_context.json")
        identity = {
            key: context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        plan = read(root / "launch_plan.json")
        snapshot_ref = Path(context["code_snapshot_ref"])
        snapshot = read(snapshot_ref)
        if (
            context.get("execution_arm", {}).get("arm") != "D"
            or context.get("planner", {}).get("backend") != "flash"
            or any(
                snapshot.get("identity", {}).get(key) != value
                or plan.get("context", {}).get(key) != value
                for key, value in identity.items()
            )
            or context.get("output_dir") != str(root.resolve())
        ):
            return evidence
        command = plan["command"]
        endpoint_flags = (
            "--env-endpoint",
            "--am-endpoint",
            "--vla-endpoint",
            "--wam-endpoint",
            "--sam3-endpoint",
        )
        if any(
            any(
                argument == flag or argument.startswith(flag + "=")
                for flag in endpoint_flags
            )
            for argument in command
        ):
            return evidence
        if "--output-dir" not in command or command[
            command.index("--output-dir") + 1
        ] != str(root.resolve()):
            return evidence
        required = {
            "RPent/rpent/utils/daemon.py": ["self._proc = subprocess.Popen("],
            "RPent/robots/libero/robot_spec.py": [
                "def _spawn_env_server(",
                "daemon.start()",
                "def _spawn_vla_server(",
                "am.spawn_worker(",
            ],
            "core/src/rsi_vla/runtime.py": [
                "output.mkdir(parents=True, exist_ok=False)",
                "with RequestGate(plan['context']) as gate:",
            ],
            "core/src/rsi_vla/adapters/request_gate.py": [
                "self.sent = 0",
                'ThreadingHTTPServer(("127.0.0.1", 0), Handler)',
            ],
            "scripts/run_isolated_role.py": ['"--unshare-pid"', '"--die-with-parent"'],
            "core/src/rsi_vla/protocol10_roles.py": [
                "def run_role(",
                "_sandbox_command(",
            ],
        }
        refs = []
        for name, fragments in required.items():
            if name not in snapshot["files"]:
                return evidence
            path = snapshot_ref.parent / name
            source = path.read_text()
            if any(fragment not in source for fragment in fragments):
                return evidence
            refs.append(str(path))
        log = (root / "run.log").read_text()
        spawns = dict(
            re.findall(
                r"\[daemon\] (env_server|vla_server|sam3_server) spawned \(pid=(\d+)\)",
                log,
            )
        )
        if (
            set(spawns) != {"env_server", "vla_server", "sam3_server"}
            or str(root.resolve()) not in log
        ):
            return evidence
        return {
            "established": True,
            "basis": "fresh output; local endpoints; recorded subprocess spawns; fresh PID namespace and request gate",
            "launch_plan_ref": str(root / "launch_plan.json"),
            "run_log_ref": str(root / "run.log"),
            "code_snapshot_ref": str(snapshot_ref),
            "source_refs": refs,
            "spawned_namespace_pids": {name: int(pid) for name, pid in spawns.items()},
        }
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        evidence["missing"] = type(exc).__name__ + ": " + str(exc)
    return evidence


def inspect_owned_runtime_processes(
    host_anchor_ref, operation_id, output, *, operation_anchor_ref=None, actor_evidence=None
):
    """Inspect this owned operation from the same PID namespace as its live host."""
    if not process_matches(host_anchor_ref):
        raise PermissionError(
            "live host birth anchor is not visible; process cleanup remains unknown"
        )
    context_ref = str((Path(output) / "runtime_context.json").resolve())
    matches, unreadable, excluded_preexisting = [], [], []
    operation_anchor = None
    freshness = {"established": False}
    if operation_anchor_ref is not None:
        anchor_ref = Path(operation_anchor_ref)
        operation_anchor = read(anchor_ref)
        if (
            anchor_ref.stat().st_uid != os.getuid()
            or operation_anchor.get("owner_uid") != os.getuid()
            or operation_anchor.get("boot_id")
            != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            or int(operation_anchor["start_ticks"]) <= 0
        ):
            raise PermissionError(
                "operation birth anchor does not bind this owner and boot"
            )
        freshness = actor_evidence if actor_evidence is not None else fresh_runtime_actor_evidence(output)
    control_ancestors = {os.getpid()}
    ancestor = os.getpid()
    while ancestor > 1:
        try:
            ancestor = int(
                Path(f"/proc/{ancestor}/stat").read_text().rsplit(")", 1)[1].split()[1]
            )
        except (OSError, ValueError, IndexError):
            break
        control_ancestors.add(ancestor)
    for directory in Path("/proc").glob("[0-9]*"):
        pid = int(directory.name)
        if pid in control_ancestors:
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            argv = (
                (directory / "cmdline")
                .read_bytes()
                .decode(errors="replace")
                .split("\0")
            )
            in_arguments = any(
                operation_id in part or str(Path(output).resolve()) in part
                for part in argv
            )
            if (
                operation_anchor is not None
                and freshness["established"]
                and int(stat[19]) < int(operation_anchor["start_ticks"])
                and pid != operation_anchor["pid"]
                and not in_arguments
            ):
                excluded_preexisting.append(
                    {
                        "pid": pid,
                        "start_ticks": stat[19],
                        "operation_start_ticks": operation_anchor["start_ticks"],
                        "owner_uid": os.getuid(),
                        "reason": "predates fresh operation and has no operation argument/owner association",
                        "environment_read": False,
                    }
                )
                continue
            if in_arguments or (
                operation_anchor is not None and pid == operation_anchor["pid"]
            ):
                matches.append(
                    {
                        "pid": pid,
                        "via": "arguments"
                        if in_arguments
                        else "operation_owner_marker",
                    }
                )
                continue
            environment = (directory / "environ").read_bytes().split(b"\0")
            in_environment = (
                "RSI_RUNTIME_CONTEXT=" + context_ref
            ).encode() in environment
            if in_arguments or in_environment:
                matches.append(
                    {
                        "pid": pid,
                        "via": "arguments"
                        if in_arguments
                        else "runtime_context_environment",
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable.append(pid)
    return {
        "owner_uid": os.getuid(),
        "host_anchor_ref": str(host_anchor_ref),
        "host_pid_namespace_confirmed": True,
        "live_processes": matches,
        "unreadable_owned_processes": unreadable,
        "excluded_control_ancestor_pids": sorted(control_ancestors),
        "operation_anchor_ref": str(operation_anchor_ref)
        if operation_anchor_ref is not None
        else None,
        "operation_birth_anchor": operation_anchor,
        "fresh_actor_evidence": freshness,
        "excluded_preexisting_processes": excluded_preexisting,
    }


def ended_request_count(output, context, process_evidence):
    """Use the durable pre-forward counter and D consumer to prove exact zero."""
    root = Path(output)
    evidence = {
        "basis": "unknown",
        "live_gate_ref": str(root / "request_gate_live.json"),
    }
    if (
        process_evidence.get("host_pid_namespace_confirmed") is not True
        or process_evidence.get("live_processes")
        or process_evidence.get("unreadable_owned_processes")
    ):
        return None, evidence
    try:
        identity = {
            key: context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        live = read(root / "request_gate_live.json")
        snapshot_ref = Path(context["code_snapshot_ref"])
        snapshot = read(snapshot_ref)
        source = snapshot_ref.parent / "core/src/rsi_vla/adapters/request_gate.py"
        gate_source = source.read_text()
        admit = gate_source.split("    def admit(self, payload):", 1)[1].split(
            "    def observe_event(", 1
        )[0]
        post = gate_source.split("            def do_POST(self):", 1)[1].split(
            "    def __exit__(", 1
        )[0]
        source_order = admit.index("self.sent += 1") < admit.index(
            "self.publish_live()"
        ) and post.index("gate.admit(payload)") < post.index(
            "asyncio.run(self.forward("
        )
        frozen_d = (
            context.get("execution_arm", {}).get("arm") == "D"
            and context.get("planner", {}).get("backend") == "flash"
            and context.get("execution_information", {}).get("online_planner") is False
            and context.get("execution_information", {}).get("model_requests_expected")
            == 0
        )
        events_path = root / "model_requests.jsonl"
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines() if line]
            if events_path.exists()
            else []
        )
        if (
            source_order
            and frozen_d
            and live.get("requests_forwarded") == 0
            and all(
                live.get("identity", {}).get(key) == value
                for key, value in identity.items()
            )
            and all(
                snapshot.get("identity", {}).get(key) == value
                for key, value in identity.items()
            )
            and not any(event.get("kind") == "request_admitted" for event in events)
        ):
            evidence.update(
                basis="durable_counter_written_before_any_forward; frozen_D; all_owners_stopped",
                requests_forwarded=0,
                actual_gate_source_ref=str(source),
                code_snapshot_ref=str(snapshot_ref),
            )
            return 0, evidence
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        evidence["missing"] = type(exc).__name__ + ": " + str(exc)
    return None, evidence


def ended_native_counts(output, identity):
    """Recover only persisted native counters, without inventing missing finalizers."""
    root = Path(output)
    costs = {}
    try:
        events = [
            json.loads(line)
            for line in (root / "action_events.jsonl").read_text().splitlines()
            if line
        ]
        if all(
            all(event.get(key) == value for key, value in identity.items())
            for event in events
        ):
            costs["tool_calls"] = sum(
                event.get("kind") == "tool_call" for event in events
            )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        service = read(root / "environment_service/runtime_costs.json")
        if (
            all(
                service.get("identity", service).get(key) == value
                for key, value in identity.items()
            )
            and service.get("pending_action") is False
            and service.get("environment_steps_unknown") == 0
            and service.get("environment_steps_reserved")
            == service.get("environment_steps_confirmed")
            and type(service.get("environment_steps_confirmed")) is int
        ):
            costs["environment_steps"] = service["environment_steps_confirmed"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return costs


def capture_closure_sources(host, destination):
    """Preserve a new host-control version alongside the original operation snapshot."""
    destination = Path(destination)
    manifest_ref = destination / "manifest.json"
    if manifest_ref.exists():
        return manifest_ref
    original_ref = Path(host.core_config["source_manifest_ref"])
    original = read(original_ref)
    changed = []
    files = list(
        dict.fromkeys(
            [
                *original["files"],
                "core/src/rsi_vla/midterm_host.py",
                "scripts/run_midterm_host.py",
                "core/src/rsi_vla/research_supervision.py",
                "scripts/prepare_research_storage_recovery.py",
            ]
        )
    )
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("reviewed source path escapes repository")
        payload = (Path(host.config["repository_root"]) / relative).read_bytes()
        before = original_ref.parent / "source" / relative
        if not before.exists() or payload != before.read_bytes():
            changed.append(name)
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    immutable_json(
        manifest_ref,
        {
            **original,
            "files": files,
            "parent_manifest_ref": str(original_ref),
            "source_revision": "host-ended-closure-v1",
            "changed_files": changed,
            "scope": "host control closure; original operation source snapshot retained",
        },
    )
    return manifest_ref


def reconcile_ended_runtime(configuration_ref, operation_id):
    """Reconcile one ended operation and emit a single ordinary Autoloop wake."""
    host = MidtermHost(configuration_ref)
    job = host.root / "jobs" / identifier(operation_id)
    operation = next(item for item in host.core_config['operations'] if item['operation_id']==operation_id)
    if operation['kind'] == 'research_prepare':
        from .research_supervision import close_failed_research_preparation
        closure = close_failed_research_preparation(host,operation_id,read(job/'supervisor.json'))
    elif operation['kind'] == 'research':
        report = read(job/'supervisor.json')
        if report.get('status') == 'completed' and report.get('exit_code') == 1:
            from .research_supervision import close_policy_rejected_research
            closure = close_policy_rejected_research(host,operation_id,report)
        else:
            closure = host.close_ended_research(operation_id,report)
    else:
        closure = host.close_ended_runtime(operation_id,read(job/'supervisor.json'))
    control = job / "closure-control"
    previous_ref = control / "previous-host.json"
    if not previous_ref.exists():
        immutable_json(previous_ref, read(host.root / "host-process.json"))
    previous = read(previous_ref)
    restart_ref = control / "replacement-host.json"
    if not restart_ref.exists():
        if process_matches(previous_ref):
            owned = owned_command_identity(previous["pid"], configuration_ref, "serve")
            if any(
                owned[key] != previous[key]
                for key in ("pid", "owner_uid", "start_ticks", "boot_id")
            ):
                raise PermissionError(
                    "original host birth identity changed before closure deployment"
                )
            os.kill(previous["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 5
            while process_matches(previous_ref) and time.monotonic() < deadline:
                time.sleep(0.05)
            if process_matches(previous_ref):
                raise RuntimeError("owned previous host did not stop")
        with (host.root / "host.log").open("ab") as stream:
            process = subprocess.Popen(
                [
                    host.config["python"],
                    host.config["script_ref"],
                    "serve",
                    "--configuration",
                    str(configuration_ref),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _children.append(process)
        immutable_json(restart_ref, process_identity(process.pid))
    elif not process_matches(restart_ref):
        raise RuntimeError(
            "closure replacement host is no longer alive; inspect its original receipt"
        )
    event_ref = control / "event.json"
    if not event_ref.exists():
        event = ended_completion_event(
            host.config["scope_id"], operation_id, job / "host-ended-receipt.json"
        )
        atomic_json(
            event_ref,
            {
                "event": event,
                "response": hub_request(host.config, "/v1/events", event=event),
            },
        )
    return {
        "scope_id": host.config["scope_id"],
        "operation_id": operation_id,
        "closure_ref": str(job / "host-ended-receipt.json"),
        "native_outcome": closure["value"].get("native_outcome",'unknown'),
        "event_ref": str(event_ref),
        "replacement_host_ref": str(restart_ref),
        "runtime_resubmitted": False,
        "business_node_manually_dispatched": False,
    }


def ended_completion_event(scope_id, operation_id, receipt_ref):
    return {
        "id": "ended-" + operation_id,
        "type": "midterm.completed",
        "subject": scope_id,
        "summary": "controlled execution ended; native outcome remains unknown",
        "detail": str(receipt_ref),
    }


def worker(configuration_ref):
    """Send no event prose, executable arguments or permits over the worker boundary."""
    config = read(configuration_ref)
    transport = Path(config["transport_root"])
    context = read(os.environ["AGENT_AUTOLOOP_CONTEXT"])
    scope = config["scope_id"]
    if any(
        event.get("source") != scope
        or (
            event.get("subject") != scope
            and not (event.get("type") == "timer.due" and event.get("subject") == "")
        )
        for event in context.get("events", [])
    ) or context.get("answers"):
        raise PermissionError("worker context is outside registered source/subject")
    if any(
        timer.get("key") != scope + "-wake" or timer.get("about") != scope
        for timer in context.get("timers", [])
    ):
        raise PermissionError("unregistered timer identity")
    identity = uuid4().hex
    atomic_json(
        transport / "requests" / (identity + ".json"),
        {"request_id": identity, "scope_id": scope, "action": "wake"},
    )
    response = transport / "responses" / (identity + ".json")
    deadline = time.monotonic() + 10
    while not response.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    state = (
        read(response)
        if response.exists()
        else {"status": "running", "reason": "host_response_pending"}
    )
    result = worker_receipt(state, scope_id=scope, poll_seconds=config["poll_seconds"])
    return (
        result["summary"]
        + "\n```agent-autoloop\n"
        + json.dumps(
            {k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False
        )
        + "\n```"
    )


def engineering_probe(operation_id, directory, seconds):
    """One real engineering side effect used for controlled host-restart acceptance."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    submission = root / "submitted.json"
    if submission.exists():
        raise FileExistsError(
            "engineering operation already submitted; never execute twice"
        )
    atomic_json(
        submission,
        {
            "operation_id": operation_id,
            "owner_uid": os.getuid(),
            "pid": os.getpid(),
            "submitted_at": utcnow().isoformat(),
        },
    )
    time.sleep(seconds)
    atomic_json(
        root / "result.json",
        {
            "operation_id": operation_id,
            "owner_uid": os.getuid(),
            "status": "completed",
            "completed_at": utcnow().isoformat(),
            "model_requests": 0,
            "gpu_devices": [],
        },
    )


def prepare_deployment(
    continuation_ref, configuration_ref, *, node, autoloop_root, port=3361
):
    """Prepare an isolated real Autoloop instance and the registered engineering prefix."""
    original = read(continuation_ref)
    scope = read(original["scope_authorization_ref"])
    root = Path(scope["write_root"])
    hub_home = root / "autoloop/home"
    hub_home.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "AGENT_AUTOLOOP_HOME": str(hub_home)}
    hub_config_ref = hub_home / "config.json"
    # The supported runtime initializes its projection from rules on import.
    if not (hub_home / "rules.json").exists():
        provisional = build_configuration(
            continuation_ref, configuration_ref, root / "operator/midterm-hub.json"
        )
        atomic_json(hub_home / "rules.json", build_autoloop_rules(provisional))
    if not hub_config_ref.exists():
        runtime = (Path(autoloop_root) / "host/runtime.ts").resolve().as_uri()
        program = (
            f"import {{addKey,db}} from {json.dumps(runtime)};"
            f"const name={json.dumps(scope['scope_id'])};"
            "console.log(JSON.stringify({worker:{name,key:addKey('worker',name)},"
            "source:{name,key:addKey('source',name)},owner:addKey('owner','owner')}));db.close();"
        )
        completed = subprocess.run(
            [str(node), "--input-type=module", "-e", program],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        credentials = json.loads(completed.stdout)
        url = f"http://127.0.0.1:{port}"
        atomic_json(
            hub_config_ref,
            {
                **credentials,
                "url": url,
                "hub": {"listen": f"127.0.0.1:{port}", "publicUrl": url},
                "channels": {},
            },
        )
        hub_config_ref.chmod(0o600)
    hub = read(hub_config_ref)
    private_hub_ref = root / "operator/midterm-hub.json"
    atomic_json(
        private_hub_ref,
        {"url": hub["url"], "source_key": hub["source"]["key"], "owner": hub["owner"]},
    )
    private_hub_ref.chmod(0o600)
    core = deepcopy(original)
    probe_id = scope["scope_id"] + "-engineering-probe"
    probe_dir = root / "operator/engineering-probe"
    probe = {
        "operation_id": probe_id,
        "kind": "host_command",
        "stage": "P1-engineering",
        "dynamic": False,
        "payload": {},
    }
    core["operations"].insert(0, probe)
    core["continuation"]["nodes"]["engineering-probe"] = {
        "operation_id": probe_id,
        "transitions": [{"next": core["continuation"]["start"]}],
    }
    core["continuation"]["start"] = "engineering-probe"
    derived_ref = Path(continuation_ref).parent / "host-continuation-configuration.json"
    atomic_json(derived_ref, core)
    config = build_configuration(
        derived_ref, configuration_ref, private_hub_ref, poll_seconds=2
    )
    config.update(
        node=str(node),
        autoloop_root=str(autoloop_root),
        autoloop_home=str(hub_home),
        engineering_probe_id=probe_id,
        engineering_probe_dir=str(probe_dir),
    )
    config["operations"][probe_id].update(
        command=[
            config["python"],
            config["script_ref"],
            "probe",
            "--operation-id",
            probe_id,
            "--probe-directory",
            str(probe_dir),
            "--probe-seconds",
            "12",
        ],
        result_ref=str(probe_dir / "result.json"),
        admission_lease_ref=str(root / "operator/engineering-admission.lock"),
        supervisor_limits={"wall_seconds": 30, "storage_bytes": 64 * 1024**2},
    )
    atomic_json(configuration_ref, config)
    atomic_json(hub_home / "rules.json", build_autoloop_rules(config))
    return {
        "configuration_ref": str(configuration_ref),
        "hub_home": str(hub_home),
        "scope_id": scope["scope_id"],
    }


def hub_request(config, path, *, event=None):
    hub = read(config["hub_configuration_ref"])
    key = hub["source_key"] if event is not None else hub["owner"]
    request = urllib.request.Request(
        hub["url"] + path,
        data=json.dumps(event).encode() if event is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def prove_unsubmitted_preparation(configuration_ref, operation_id):
    """Prove the observed collection-permit rejection preceded Runtime dispatch.

    This deliberately recognizes the actual preparation failure, rather than
    treating an absent output alone as evidence that a robot operation never ran.
    """
    configuration_ref = Path(configuration_ref).resolve()
    config = read(configuration_ref)
    core_config = read(config["continuation_configuration_ref"])
    scope = read(config["scope_authorization_ref"])
    if (
        scope["owner_uid"] != os.getuid()
        or configuration_ref.stat().st_uid != os.getuid()
        or config["scope_id"] != scope["scope_id"]
    ):
        raise PermissionError("preparation evidence owner/scope mismatch")
    root = Path(
        config.get("host_root", Path(scope["write_root"]) / "operator/midterm-host")
    )
    operation = next(
        item
        for item in core_config["operations"]
        if item["operation_id"] == operation_id
    )
    if operation["kind"] != "runtime":
        raise ValueError("preparation repair requires a registered Runtime operation")
    slot = next(
        item
        for item in core_config["runtime_slots"]
        if item["run_id"] == operation["payload"]["run_id"]
    )
    job = root / "jobs" / identifier(operation_id)
    submission = read(job / "submission.json")
    report = read(job / "supervisor.json")
    if submission["operation"] != operation or submission["owner_uid"] != os.getuid():
        raise PermissionError("original submission owner/operation mismatch")
    if (
        process_matches(job / "process.json")
        or report.get("identity", {}).get("run_id") != operation_id
        or report.get("cleanup_confirmed") is not True
        or report.get("remaining_process_ids")
        or report.get("status") != "completed"
        or report.get("exit_code") != 1
    ):
        raise PermissionError(
            "original supervisor submission/cleanup remains unresolved"
        )
    returns = sorted(
        (Path(core_config["isolation_task_root"]) / "runtime/role_receipts").glob(
            operation_id + "-execute-revision-*.json"
        )
    )
    if len(returns) != 1:
        raise ValueError("one original rejected role execution receipt is required")
    returned = read(returns[0])
    if (
        returned.get("error")
        != "ValueError: authorization collection binding missing or different"
        or returned.get("role") != "runtime"
        or returned["request"]["configuration"] != core_config
        or returned["request"]["task"]["operation"] != operation
    ):
        raise PermissionError(
            "role failure does not establish this pre-dispatch preparation rejection"
        )
    output = Path(slot["output_ref"])
    if output.exists() and any(output.iterdir()):
        raise PermissionError(
            "Runtime output is present; inspect native submission instead of retrying"
        )
    snapshot = Path(core_config["source_manifest_ref"])
    manifest = read(snapshot)
    paths = ("core/src/rsi_vla/runtime.py", "core/src/rsi_vla/task_completion.py")
    if any(path not in manifest["files"] for path in paths):
        raise PermissionError("original reviewed consumer sources are unavailable")
    runtime_source = (snapshot.parent / "source" / paths[0]).read_text()
    completion_source = (snapshot.parent / "source" / paths[1]).read_text()
    runtime_prepare = runtime_source.split("def prepare_runtime(", 1)[1].split(
        "def execute_runtime(", 1
    )[0]
    runtime_branch = completion_source.split('        if kind == "runtime":', 1)[
        1
    ].split('        if kind == "research_feedback":', 1)[0]
    if (
        "raise ValueError('authorization collection binding missing or different')"
        not in runtime_prepare
        or "execute_runtime(" in runtime_prepare
        or runtime_branch.index("plan = prepare_runtime(")
        >= runtime_branch.index("execute_runtime(plan)")
    ):
        raise PermissionError(
            "original source does not prove the rejection preceded execution"
        )
    record = read(config["accounting_ref"])["reservations"][operation_id]
    if (
        record.get("actual", {}).get("runtime_starts") != 0
        or record.get("cleanup_confirmed") is not True
    ):
        raise PermissionError(
            "original operation has unresolved actual starts or cleanup"
        )
    proof = {
        "schema": "rsi-unsubmitted-preparation-proof-v1",
        "scope_id": config["scope_id"],
        "operation_id": operation_id,
        "run_id": slot["run_id"],
        "owner_uid": os.getuid(),
        "failure_stage": "prepare_runtime_authorization_collection_validation",
        "runtime_starts": 0,
        "model_requests": 0,
        "submission_absent": True,
        "cleanup_confirmed": True,
        "role_receipt_ref": str(returns[0]),
        "supervisor_ref": str(job / "supervisor.json"),
        "source_manifest_ref": str(snapshot),
        "source_refs": [str(snapshot.parent / "source" / path) for path in paths],
        "output_ref": str(output),
        "preserved_actual": record.get("actual", {}),
        "not_gpu_technical_recovery": True,
    }
    return proof


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError("conflicting repair artifact: " + str(path))
    atomic_json(path, value)


def prepare_preparation_repair(
    configuration_ref, operation_id, repair_id, *, monitor_pid=None
):
    """Close a proven unsubmitted attempt and register a linked remaining graph.

    Existing progress, role errors, permits, source snapshots and completion
    receipts are retained. This prepares no model request or Runtime launch.
    """
    identifier(repair_id)
    host = MidtermHost(configuration_ref)
    proof = prove_unsubmitted_preparation(configuration_ref, operation_id)
    destination = host.ledger.root / "operator/p1" / repair_id
    original_monitor = None
    if host.config.get("engineering_probe_id"):
        if monitor_pid is None:
            raise ValueError(
                "original engineering monitor PID is required for owned handoff"
            )
        original_monitor = owned_command_identity(
            monitor_pid, configuration_ref, "monitor"
        )
        immutable_json(destination / "original-monitor.json", original_monitor)
    proof_ref = destination / "unsubmitted-proof.json"
    if proof_ref.exists():
        original_proof = read(proof_ref)
        if {k: v for k, v in proof.items() if k != "preserved_actual"} != {
            k: v for k, v in original_proof.items() if k != "preserved_actual"
        }:
            raise ValueError("preparation repair evidence changed")
        proof = original_proof
    immutable_json(proof_ref, proof)
    actual = {**proof["preserved_actual"], "model_requests": 0}
    host.ledger.settle(operation_id, actual=actual, cleanup_confirmed=True)
    with host.ledger.lock():
        ledger = host.ledger._read()
        entry = {
            "operation_id": operation_id,
            "proof_ref": str(proof_ref),
            "actual": actual,
        }
        entries = ledger.setdefault("preparation_reconciliations", [])
        if entry not in entries:
            entries.append(entry)
            atomic_json(host.ledger.path, ledger)
    core = deepcopy(host.core_config)
    old_operation = next(
        item for item in core["operations"] if item["operation_id"] == operation_id
    )
    successor_id = identifier(operation_id + "-" + repair_id)
    old_operation["operation_id"] = successor_id
    start = next(
        key
        for key, node in core["continuation"]["nodes"].items()
        if node.get("operation_id") == operation_id
    )
    core["continuation"]["nodes"][start]["operation_id"] = successor_id
    core["continuation"]["start"] = start
    reachable = set()

    def visit(key):
        if key in reachable:
            return
        reachable.add(key)
        for edge in core["continuation"]["nodes"][key].get("transitions", []):
            visit(edge["next"])

    visit(start)
    core["continuation"]["nodes"] = {
        key: value
        for key, value in core["continuation"]["nodes"].items()
        if key in reachable
    }
    used = {
        node["operation_id"]
        for node in core["continuation"]["nodes"].values()
        if "operation_id" in node
    }
    core["operations"] = [
        item for item in core["operations"] if item["operation_id"] in used
    ]
    slot = next(
        item for item in core["runtime_slots"] if item["run_id"] == proof["run_id"]
    )
    slot["permit_ref"] = str(destination / "permits" / (successor_id + ".json"))
    old_manifest_ref = Path(core["source_manifest_ref"])
    old_manifest = read(old_manifest_ref)
    source_ref = destination / "source/manifest.json"
    changed = []
    for name in old_manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("reviewed source path escapes repository")
        payload = (Path(core["repository_root"]) / relative).read_bytes()
        if payload != (old_manifest_ref.parent / "source" / relative).read_bytes():
            changed.append(name)
        target = source_ref.parent / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("repair reviewed source changed after registration")
        target.write_bytes(payload)
    manifest = {
        **old_manifest,
        "parent_manifest_ref": str(old_manifest_ref),
        "source_revision": repair_id,
        "changed_files": changed,
        "reason": "complete Runtime permit bindings and proven unsubmitted preparation continuation",
    }
    immutable_json(source_ref, manifest)
    core["source_manifest_ref"] = str(source_ref)
    core["preparation_repair"] = {
        "repair_id": repair_id,
        "original_configuration_ref": str(
            host.config["continuation_configuration_ref"]
        ),
        "original_operation_id": operation_id,
        "successor_operation_id": successor_id,
        "proof_ref": str(proof_ref),
        "same_logical_slot": True,
        "scientific_repetition": False,
        "gpu_technical_recovery": False,
    }
    core_ref = destination / "continuation-configuration.json"
    immutable_json(core_ref, core)
    new_host_ref = destination / "host.json"
    new_host = build_configuration(
        core_ref,
        new_host_ref,
        host.config["hub_configuration_ref"],
        python=host.config["python"],
        script_ref=host.config["script_ref"],
        poll_seconds=host.config["poll_seconds"],
    )
    new_host.update(
        host_root=str(destination / "host-state"),
        transport_root=str(destination / "transport"),
        node=host.config["node"],
        autoloop_root=host.config["autoloop_root"],
        autoloop_home=host.config["autoloop_home"],
        preparation_repair=core["preparation_repair"],
        original_host_configuration_ref=str(configuration_ref),
    )
    new_host["original_monitor"] = original_monitor
    old_deployment = host.root / "deployment.json"
    new_host["original_host_process"] = (
        read(old_deployment)["processes"]["host"] if old_deployment.exists() else None
    )
    original_inputs = read(host.config["inputs_ref"])
    original_inputs["authorization"]["binding"]["registration_ref"] = str(
        destination / "continuation/registration.json"
    )
    original_inputs["authorization"]["operation_ids"] = [
        item["operation_id"] for item in core["operations"] if item["dynamic"]
    ]
    immutable_json(destination / "inputs.json", original_inputs)
    immutable_json(new_host_ref, new_host)
    # Register through the supported backend constructor; no progress is copied.
    successor = MidtermHost(new_host_ref)
    successor.coordinator()
    result = {
        "schema": "rsi-preparation-repair-registration-v1",
        "repair_id": repair_id,
        "scope_id": host.config["scope_id"],
        "configuration_ref": str(new_host_ref),
        "source_manifest_ref": str(source_ref),
        "proof_ref": str(proof_ref),
        "old_operation_id": operation_id,
        "new_operation_id": successor_id,
        "runtime_slots": [
            {key: item[key] for key in ("run_id", "task_key", "seed", "bundle_id")}
            for item in core["runtime_slots"]
        ],
        "actual_dispatch": False,
        "original_engineering_operation_repeated": False,
    }
    immutable_json(destination / "registration-receipt.json", result)
    return result


def activate_preparation_repair(configuration_ref):
    """Continue a registered repair through one real Autoloop control event."""
    host = MidtermHost(configuration_ref)
    config = host.config
    repair = config["preparation_repair"]
    proof = read(repair["proof_ref"])
    if (
        proof.get("submission_absent") is not True
        or proof.get("scope_id") != config["scope_id"]
    ):
        raise PermissionError("repair lacks its matching unsubmitted preparation proof")
    record = host.ledger.read()["reservations"][repair["original_operation_id"]]
    if (
        record["status"] != "settled"
        or record["actual"].get("runtime_starts") != 0
        or record["actual"].get("model_requests") != 0
    ):
        raise PermissionError("original preparation cost/starts not reconciled")
    activation_ref = host.root / "activation.json"
    for name, identity in (
        ("monitor", config.get("original_monitor")),
        ("serve", config.get("original_host_process")),
    ):
        if identity is None:
            continue
        identity_ref = host.root / ("original-" + name + ".json")
        immutable_json(identity_ref, identity)
        if process_matches(identity_ref):
            current = owned_command_identity(
                identity["pid"], config["original_host_configuration_ref"], name
            )
            if {
                key: current[key]
                for key in ("pid", "owner_uid", "start_ticks", "boot_id")
            } != {
                key: identity[key]
                for key in ("pid", "owner_uid", "start_ticks", "boot_id")
            }:
                raise PermissionError("owned original process birth identity changed")
            os.kill(identity["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 5
            while process_matches(identity_ref) and time.monotonic() < deadline:
                time.sleep(0.05)
            if process_matches(identity_ref):
                raise RuntimeError("owned original process did not stop: " + name)
        immutable_json(
            host.root / ("original-" + name + "-stopped.json"),
            {"identity": identity, "stopped": True},
        )
    if activation_ref.exists():
        activation = read(activation_ref)
        if not process_matches(host.root / "activation-process.json"):
            raise RuntimeError(
                "repair activation exists; inspect original host before resuming"
            )
    else:
        host.ledger.authority()
        with (host.root / "host.log").open("ab") as stream:
            process = subprocess.Popen(
                [
                    config["python"],
                    config["script_ref"],
                    "serve",
                    "--configuration",
                    str(configuration_ref),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        _children.append(process)
        activation = {
            "scope_id": config["scope_id"],
            "repair_id": repair["repair_id"],
            "host_process": process_identity(process.pid),
        }
        atomic_json(host.root / "activation-process.json", activation["host_process"])
        immutable_json(activation_ref, activation)
    atomic_json(
        Path(config["autoloop_home"]) / "rules.json", build_autoloop_rules(config)
    )
    # Wait for the existing hub's real watcher to persist the new command
    # binding before emitting the control event; the state API omits commands.
    deadline = time.monotonic() + 5
    applied = False
    while time.monotonic() < deadline:
        with sqlite3.connect(
            "file:" + str(Path(config["autoloop_home"]) / "hub.db") + "?mode=ro",
            uri=True,
        ) as database:
            facts = database.execute(
                "SELECT fact FROM facts ORDER BY seq DESC LIMIT 100"
            ).fetchall()
        if any(json.dumps(str(host.configuration_ref)) in row[0] for row in facts):
            applied = True
            break
        time.sleep(0.05)
    if not applied:
        raise RuntimeError(
            "hub has not applied the registered repair command; no continuation event sent"
        )
    event_ref = host.root / "activation-event.json"
    if not event_ref.exists():
        # The hub reloads the trusted rules before routing the next source event.
        event = {
            "id": config["scope_id"] + "-" + repair["repair_id"],
            "type": "midterm.preparation_repaired",
            "subject": config["scope_id"],
            "summary": "continue the original unsubmitted logical slot",
            "detail": repair["proof_ref"],
        }
        atomic_json(
            event_ref,
            {
                "event": event,
                "response": hub_request(config, "/v1/events", event=event),
            },
        )
    observer_ref = host.root / "observer-process.json"
    if not observer_ref.exists():
        with (host.root / "observer.log").open("ab") as stream:
            observer = subprocess.Popen(
                [
                    config["python"],
                    config["script_ref"],
                    "observe",
                    "--configuration",
                    str(configuration_ref),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _children.append(observer)
        atomic_json(observer_ref, process_identity(observer.pid))
    return {
        "scope_id": config["scope_id"],
        "repair_id": repair["repair_id"],
        "activation_ref": str(activation_ref),
        "event_ref": str(event_ref),
        "observer_ref": str(observer_ref),
    }


def owned_command_identity(pid, configuration_ref, command):
    """Bind the exact task-owned CLI process, UID and birth identity before stopping it."""
    identity = process_identity(pid)
    if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
        raise PermissionError("process is owned by another user")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    argv = [part for part in argv if part]
    if command not in argv or "--configuration" not in argv:
        raise PermissionError("process is not the expected task CLI")
    index = argv.index("--configuration")
    if (
        index + 1 >= len(argv)
        or Path(argv[index + 1]).resolve() != Path(configuration_ref).resolve()
        or not any(Path(part).name == "run_midterm_host.py" for part in argv)
    ):
        raise PermissionError(
            "process command belongs to a different task configuration"
        )
    return {
        **identity,
        "command": command,
        "configuration_ref": str(Path(configuration_ref).resolve()),
    }


def atomic_json_changed(path, value):
    """Keep an unchanged observer snapshot at its existing logical path."""
    path = Path(path)
    if not path.exists() or read(path) != value:
        atomic_json(path, value)


def observe_repair(configuration_ref):
    """Observe the repaired chain without repeating engineering work or dispatching nodes."""
    host = MidtermHost(configuration_ref)
    evidence = host.root / "observer"
    evidence.mkdir(parents=True, exist_ok=True)
    while utcnow() < timestamp(host.ledger.authority(active=False)["expires_at"]):
        view = hub_request(host.config, "/v1/state")
        atomic_json_changed(evidence / "hub-state.json", view)
        state = host.coordinator().status()
        atomic_json_changed(evidence / "core-state.json", state)
        for run in view["runs"]:
            target = evidence / ("run-" + str(run["id"]) + "-why.json")
            if run["state"] in {"done", "nothing"} and not target.exists():
                atomic_json(
                    target,
                    hub_request(host.config, "/v1/runs/" + str(run["id"]) + "/why"),
                )
        if state["status"] in {"completed", "terminated", "cancelled", "paused"}:
            atomic_json(
                evidence / "observed-terminal.json",
                {
                    "core": state,
                    "hub_seq": view["seq"],
                    "preparation_repair": host.config["preparation_repair"],
                    "manual_node_dispatches": 0,
                    "engineering_repeated": False,
                    "observer_did_not_dispatch": True,
                },
            )
            return
        time.sleep(0.5)


def launch_deployment(configuration_ref):
    """Start the host, real Autoloop and an evidence-only restart monitor once."""
    host = MidtermHost(configuration_ref)
    host.ledger.authority()
    path = host.root / "deployment.json"
    if path.exists():
        raise FileExistsError(
            "deployment already submitted; inspect original process identities"
        )
    config = host.config
    processes = {}
    for name, command in (
        (
            "host",
            [
                config["python"],
                config["script_ref"],
                "serve",
                "--configuration",
                str(configuration_ref),
            ],
        ),
        (
            "hub",
            [
                config["node"],
                str(Path(config["autoloop_root"]) / "bin/agent-autoloop.ts"),
                "up",
            ],
        ),
    ):
        with (host.root / (name + ".log")).open("ab") as stream:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "AGENT_AUTOLOOP_HOME": config["autoloop_home"]},
            )
        _children.append(process)
        processes[name] = process_identity(process.pid)
    atomic_json(
        path,
        {
            "scope_id": config["scope_id"],
            "started_at": utcnow().isoformat(),
            "processes": processes,
        },
    )
    with (host.root / "monitor.log").open("ab") as stream:
        process = subprocess.Popen(
            [
                config["python"],
                config["script_ref"],
                "monitor",
                "--configuration",
                str(configuration_ref),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _children.append(process)
    return {
        "scope_id": config["scope_id"],
        "processes": processes,
        "monitor_pid": process.pid,
        "deployment_ref": str(path),
    }


def monitor_deployment(configuration_ref):
    """Inject one host-only restart and retain real timer/event/run evidence."""
    host = MidtermHost(configuration_ref)
    config = host.config
    evidence = host.ledger.root / "validation/P1-autoloop"
    evidence.mkdir(parents=True, exist_ok=True)
    deployment = read(host.root / "deployment.json")
    deadline = time.monotonic() + 30
    while True:
        try:
            hub_request(config, "/v1/state")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError("owned Autoloop did not become ready")
            time.sleep(0.1)
    lease = Path(
        config["operations"][config["engineering_probe_id"]]["admission_lease_ref"]
    ).open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease_released = False
    atomic_json(
        evidence / "total-start.json",
        hub_request(
            config,
            "/v1/events",
            event={
                "id": config["scope_id"] + "-total-start",
                "type": "midterm.start",
                "subject": config["scope_id"],
                "summary": "one authorized total start",
            },
        ),
    )
    restarted = False
    while utcnow() < timestamp(host.ledger.authority(active=False)["expires_at"]):
        view = hub_request(config, "/v1/state")
        atomic_json_changed(evidence / "hub-state.json", view)
        completed_runs = [
            run for run in view["runs"] if run["state"] in {"done", "nothing"}
        ]
        for run in completed_runs:
            target = evidence / ("run-" + str(run["id"]) + "-why.json")
            if not target.exists():
                atomic_json(
                    target, hub_request(config, "/v1/runs/" + str(run["id"]) + "/why")
                )
        wait_ref = host.root / "resource-wait.json"
        if (
            not lease_released
            and wait_ref.exists()
            and read(wait_ref)["waited_seconds"] > 0
            and completed_runs
        ):
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()
            lease_released = True
            atomic_json(
                evidence / "engineering-lease-released.json",
                {
                    "at": utcnow().isoformat(),
                    "wait": read(wait_ref),
                    "owner_uid": os.getuid(),
                    "resource": "owned engineering fcntl lease; no GPU telemetry altered",
                },
            )
        probe = Path(config["engineering_probe_dir"])
        if not restarted and (probe / "submitted.json").exists() and completed_runs:
            original = deployment["processes"]["host"]
            identity_ref = evidence / "old-host.json"
            atomic_json(identity_ref, original)
            if not process_matches(identity_ref):
                raise RuntimeError("host identity changed before controlled restart")
            before = read(
                host.root / "jobs" / config["engineering_probe_id"] / "process.json"
            )
            os.kill(original["pid"], signal.SIGTERM)
            wait_until = time.monotonic() + 5
            while process_matches(identity_ref) and time.monotonic() < wait_until:
                time.sleep(0.05)
            if process_matches(identity_ref):
                raise RuntimeError("owned host did not stop cleanly")
            with (host.root / "host.log").open("ab") as stream:
                replacement = subprocess.Popen(
                    [
                        config["python"],
                        config["script_ref"],
                        "serve",
                        "--configuration",
                        str(configuration_ref),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            _children.append(replacement)
            deployment["processes"]["host"] = process_identity(replacement.pid)
            atomic_json(host.root / "deployment.json", deployment)
            atomic_json(
                evidence / "controlled-restart.json",
                {
                    "old_host": original,
                    "new_host": deployment["processes"]["host"],
                    "original_supervisor": before,
                    "supervisor_survived": process_matches(
                        host.root
                        / "jobs"
                        / config["engineering_probe_id"]
                        / "process.json"
                    ),
                    "at": utcnow().isoformat(),
                },
            )
            restarted = True
        state = host.coordinator().status()
        timer_runs = [
            run
            for run in completed_runs
            if any(event.get("type") == "timer.due" for event in run.get("events", []))
        ]
        if state["status"] in {"completed", "terminated", "cancelled", "paused"}:
            atomic_json(
                evidence / "observed-terminal.json",
                {
                    "core": state,
                    "controlled_restart": restarted,
                    "completed_timer_runs": [run["id"] for run in timer_runs],
                    "hub_seq": view["seq"],
                    "manual_node_dispatches": 0,
                    "engineering_result_ref": str(probe / "result.json"),
                },
            )
            return
        time.sleep(0.5)
