# ruff: noqa: UP017, FURB162 - Python 3.10 support
# Copyright 2026 RSI-Agent-VLA Authors.
"""Validate registered Protocol10 permits immediately before process dispatch."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("authorization expiry must be a timezone-aware timestamp")  # noqa: TRY004 - invalid authorization values share one public error contract
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("authorization expiry must have a timezone")
    return result


def _same_path(actual, expected):
    return (
        isinstance(actual, str) and Path(actual).resolve() == Path(expected).resolve()
    )


def validate_dispatch_admission(
    configuration: dict[str, Any],
    protocol: dict[str, Any],
    operation: dict[str, Any],
    *,
    authorization: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check the registered write domains, identity, source, limits and real UTC.

    This function only reads registration and permits. Coordinator simulations
    retain their explicit time input; production dispatch never accepts it here.
    Returns the validated permit for the caller's existing consumer.
    """
    now = datetime.now(timezone.utc)
    if (
        not isinstance(authorization, dict)
        or authorization.get("dynamic_authorized") is not True
        or operation["operation_id"] not in authorization.get("operation_ids", [])
        or now >= _timestamp(authorization.get("expires_at"))
    ):
        raise ValueError("dispatch authorization missing, expired or outside operation")
    projected = _read(configuration["runtime_protocol_ref"])
    authority_ref = projected["runtime_projection"]["authorization_ref"]
    authority = _read(authority_ref)
    if (
        authority.get("authorized") is not True
        or authority.get("scope") != protocol["scope_id"]
        or authority.get("protocol_id") != protocol["protocol_id"]
        or not _same_path(
            authority.get("phase_snapshot_ref"), configuration["phase_ref"]
        )
        or not _same_path(
            authority.get("frozen_protocol_ref"), configuration["frozen_protocol_ref"]
        )
        or now >= _timestamp(authority.get("expires_at"))
    ):
        raise ValueError("administrative authorization expired or binding mismatch")
    kind, payload = operation["kind"], operation["payload"]
    if kind not in {"runtime", "research", "development"}:
        raise ValueError("dispatch admission requires a registered dynamic operation")
    cfg = configuration["research"] if kind == "research" else payload
    if kind != "research":
        registered = [
            slot
            for slot in configuration["runtime_slots"]
            if slot["run_id"] == payload["run_id"]
        ]
        if len(registered) != 1 or any(
            registered[0].get(key) != payload.get(key)
            for key in (
                "side",
                "task_key",
                "seed",
                "bundle_id",
                "output_ref",
                "scratch_ref",
                "permit_ref",
            )
        ):
            raise ValueError("dispatch slot differs from registration")
    permit = _read(cfg["permit_ref"])
    if permit != payload.get("registered_permit"):
        raise ValueError("registered dispatch permit changed")
    if not _same_path(permit.get("administrative_authorization_ref"), authority_ref):
        raise ValueError("dispatch permit uses another administrative authorization")
    if not _same_path(
        permit.get("source_manifest_ref"), configuration["source_manifest_ref"]
    ):
        raise ValueError("dispatch permit uses another registered source manifest")
    if now >= _timestamp(permit.get("expires_at")):
        raise ValueError("single-operation permit expired")
    expected = {
        "authorized": True,
        "scope": protocol["scope_id"],
        "protocol_id": protocol["protocol_id"],
        "bundle_id": protocol["current_bundle_id"]
        if kind == "research"
        else cfg["bundle_id"],
    }
    if kind == "research":
        expected.update(
            job_id=cfg["job_id"],
            request_id=cfg["request_id"],
            **protocol["models"]["research"],
        )
    else:
        expected.update(
            run_id=cfg["run_id"],
            task_key=cfg["task_key"],
            seed=cfg["seed"],
            simulation_episodes=1,
            gpu_devices=[cfg.get("cuda_device", 0)],
        )
        if kind == "development":
            expected.update(
                job_id=configuration["research"]["job_id"],
                experiment_id=payload["experiment_id"],
            )
    if any(permit.get(key) != value for key, value in expected.items()):
        raise ValueError("dispatch permit identity/scope mismatch")
    if not _same_path(permit.get("phase_snapshot_ref"), configuration["phase_ref"]):
        raise ValueError("dispatch permit uses another phase ledger")
    output, scratch = (
        Path(cfg[key]).resolve() for key in ("output_ref", "scratch_ref")
    )
    if (
        output == scratch
        or output.is_relative_to(scratch)
        or scratch.is_relative_to(output)
    ):
        raise ValueError("separate output/scratch write domains required")
    if not _same_path(permit.get("output_dir"), output) or permit.get(
        "storage_write_roots"
    ) != [str(output), str(scratch)]:
        raise ValueError("dispatch permit write domains differ from registration")
    cap = protocol["budget"][
        "research_proposal" if kind == "research" else "runtime_per_start"
    ]
    required = (
        {"model_requests", "wall_seconds", "storage_bytes"}
        if kind == "research"
        else set(cap) - {"simulation_episodes"}
    )
    limits = permit.get("limits", {})
    if not required <= limits.keys():
        raise ValueError("dispatch permit omits frozen resource dimensions")
    allowed = dict(cap)
    if kind != "research":
        allowed["planner_seconds"] = 600
    if any(
        key not in allowed
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or value > allowed[key]
        or (key in required and value == 0)
        for key, value in limits.items()
    ):
        raise ValueError("dispatch permit exceeds frozen per-dimension limits")
    if kind != "research" and (
        "planner_seconds" not in limits or limits["planner_seconds"] <= 0
    ):
        raise ValueError("dispatch permit omits Planner timeout")
    phase = _read(configuration["phase_ref"])
    if (
        phase.get("unresolved_costs")
        or any(
            value > phase["remaining"].get(key, -1)
            for key, value in limits.items()
            if key in cap
        )
        or (kind != "research" and phase["remaining"].get("simulation_episodes", 0) < 1)
    ):
        raise ValueError("registered phase cannot admit dispatch resources")
    manifest_ref = Path(permit["source_manifest_ref"])
    manifest = _read(manifest_ref)
    if manifest.get("version") not in {
        "batch4-preparation",
        "5a-assets-offline",
    } or not manifest.get("files"):
        raise ValueError("reviewed dispatch source manifest required")
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest path must remain in repository")
        if (Path(configuration["repository_root"]) / relative).read_bytes() != (
            manifest_ref.parent / "source" / relative
        ).read_bytes():
            raise ValueError("reviewed dispatch source changed: " + name)
    return permit
