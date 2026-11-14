# Copyright 2026 RSI-Agent-VLA Authors.
"""Revise unsubmitted GPU bindings without replacing a protocol10 campaign."""

import fcntl
import json
import os
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from .storage import atomic_json, identifier


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _lock(path):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "protocol10 revision requires idle coordinator and dynamic lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _apply(journal_path, journal):
    # All old bytes are durable before changing a consumer. A repeated revision
    # completes a partial commit; it never re-creates a campaign or dispatches.
    for change in journal["changes"]:
        current = Path(change["path"]).read_text(encoding="utf-8")
        if current not in (change["before"], change["after"]):
            raise ValueError("revision target changed since checkpoint")
    for change in journal["changes"]:
        target = Path(change["path"])
        if target.read_text(encoding="utf-8") != change["after"]:
            temporary = target.with_name(target.name + ".revision-tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(change["after"])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
    journal["status"] = "completed"
    atomic_json(journal_path, journal)
    return journal


def revise_registration(
    configuration_ref: str | Path,
    root: str | Path,
    revision_id: str,
    *,
    devices: dict[str, int] | None = None,
    refresh_source_manifest: bool = False,
    dynamic_lock: str | Path = "/tmp/rsi-agent-vla-dynamic.lock",
) -> dict[str, Any]:
    """Apply only GPU choices and a pre-dispatch source-manifest refresh.

    Args:
        configuration_ref: Existing CLI configuration, kept at the same path.
        root: Existing Protocol10Backend root, containing backend/coordinator.
        revision_id: Stable administrative revision identity for safe resumption.
        devices: Explicit physical GPU per unsubmitted run; any nonnegative index.
        refresh_source_manifest: Rebind the existing manifest input before any
            dispatch; the caller preserves and prepares that manifest separately.
        dynamic_lock: Shared Runtime lock; CPU fixtures can supply a private lock.

    This does not select hardware, issue permits, grant authorization, alter
    frozen inputs or dispatch work. Callers still perform admission and supply
    authorization matching the returned registration binding to the real CLI.
    Existing permit identity, limits, expiry and authorization are retained.
    """
    root, configuration_ref = Path(root).resolve(), Path(configuration_ref).resolve()
    identifier(revision_id)
    devices = dict(devices or {})
    if any(type(gpu) is not int or gpu < 0 for gpu in devices.values()):
        raise ValueError("GPU must be a nonnegative physical index")
    request = {
        "configuration_ref": str(configuration_ref),
        "devices": devices,
        "refresh_source_manifest": refresh_source_manifest,
    }
    progress_ref = root / "coordinator/progress.json"
    registration_ref = root / "backend/registration/protocol10.json"
    journal_ref = root / "revisions" / (revision_id + ".json")
    with ExitStack() as stack:
        stack.enter_context(_lock(dynamic_lock))
        stack.enter_context(_lock(root / "coordinator/coordinator.lock"))
        stack.enter_context(_lock(root / "coordinator/signals.lock"))
        if journal_ref.exists():
            journal = _read(journal_ref)
            if journal["request"] != request:
                raise ValueError("conflicting revision identity")
            return _apply(journal_ref, journal)
        registration, state = _read(registration_ref), _read(progress_ref)
        config = _read(configuration_ref)
        if registration["configuration"] != config or state["binding"] != registration:
            raise ValueError("configuration/registration/coordinator binding mismatch")
        if state["status"] in {"completed", "cancelled", "terminated"}:
            raise ValueError("closed campaign cannot be revised")
        pending = state.get("pending")
        receipt = state.get("pending_receipt")
        if pending and (
            not receipt
            or receipt.get("status") != "not_submitted"
            or receipt.get("submission_absent") is not True
        ):
            raise ValueError("pending submission is not proven absent")
        backend = root / "backend"
        research = Path(config["research"]["store_ref"])
        role_requests = []
        if config.get("isolation_task_root"):
            for path in (
                Path(config["isolation_task_root"]) / "operator/role_requests"
            ).glob("*.json"):
                task = _read(path).get("task", {})
                if task.get("action") in {"execute", "runtime_plan"} and task.get(
                    "operation", {}
                ).get("dynamic"):
                    role_requests.append(task["operation"])

        def started(slot):
            run = slot["run_id"]
            if any(op["payload"].get("run_id") == run for op in role_requests):
                return True
            if Path(slot["output_ref"]).exists():
                return True
            if any(
                (backend / kind / (run + ".json")).exists()
                for kind in ("dispatches", "runtime_returns")
            ):
                return True
            if (
                any(research.glob("trial_dispatches/*"))
                and slot["side"] == "development"
            ):
                # The dispatch marker binds an experiment; the durable operation
                # is authoritative when it exists, otherwise do not guess.
                for path in research.glob("trial_dispatches/*.json"):
                    marker = _read(path)
                    if marker.get("output") == slot["output_ref"]:
                        return True
            return any(
                op["payload"].get("run_id") == run and key in state["receipts"]
                for key, op in state["operations"].items()
            )

        slots = config["runtime_slots"]
        by_run = {slot["run_id"]: slot for slot in slots}
        if set(devices) - by_run.keys():
            raise ValueError("GPU revision references an unregistered run")
        if any(started(by_run[run]) for run in devices):
            raise ValueError("started or completed runs cannot change GPU")
        phase = _read(config["phase_ref"])
        charged = {
            entry.get("run_id")
            for key in ("protocol10_trial_charges", "development_trial_charges")
            for entry in phase.get(key, [])
        }
        if charged.intersection(devices):
            raise ValueError("charged runs cannot change GPU")
        new_registration = deepcopy(registration)
        if refresh_source_manifest:
            if (
                any(started(slot) for slot in slots)
                or charged.intersection(by_run)
                or role_requests
                or Path(config["research"]["output_ref"]).exists()
                or any((backend / "dispatches").glob("*.json"))
                or any(research.glob("trial_dispatches/*.json"))
                or any(
                    op.get("dynamic") and key in state["receipts"]
                    for key, op in state["operations"].items()
                )
            ):
                raise ValueError("source refresh requires no prior dispatch")
            source_ref = config["source_manifest_ref"]
            new_registration["inputs"][source_ref] = _read(source_ref)
        if any(
            _read(ref) != content for ref, content in new_registration["inputs"].items()
        ):
            raise ValueError("non-source registered input changed")
        changes = {}

        def stage(path, value):
            path = Path(path).resolve()
            before, after = path.read_text(encoding="utf-8"), _encoded(value)
            if before != after:
                changes[str(path)] = {
                    "path": str(path),
                    "before": before,
                    "after": after,
                }

        new_config = deepcopy(config)
        for slot in new_config["runtime_slots"]:
            run = slot["run_id"]
            if run not in devices:
                continue
            gpu = devices[run]
            permit_ref = Path(slot["permit_ref"])
            if permit_ref.exists():
                permit = _read(permit_ref)
                if permit.get("run_id") != run or permit.get("gpu_devices") != [
                    slot.get("cuda_device", 0)
                ]:
                    raise ValueError("existing permit GPU/run binding mismatch")
                if (
                    pending
                    and pending["payload"].get("run_id") == run
                    and pending["payload"].get("registered_permit") != permit
                ):
                    raise ValueError(
                        "pending registered permit changed before revision"
                    )
                permit["gpu_devices"] = [gpu]
                stage(permit_ref, permit)
            slot["cuda_device"] = gpu
            if pending and pending["payload"].get("run_id") == run:
                pending["payload"]["cuda_device"] = gpu
                if permit_ref.exists():
                    pending["payload"]["registered_permit"] = permit
                state["operations"][pending["operation_id"]] = deepcopy(pending)
        new_registration["configuration"] = new_config
        state["binding"] = new_registration
        stage(configuration_ref, new_config)
        stage(registration_ref, new_registration)
        stage(progress_ref, state)
        journal = {
            "schema_version": 1,
            "revision_id": revision_id,
            "request": request,
            "status": "prepared",
            "dispatch": False,
            "before_binding": registration,
            "after_binding": new_registration,
            "changes": list(changes.values()),
        }
        atomic_json(journal_ref, journal)
        return _apply(journal_ref, journal)
