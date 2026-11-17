# Copyright 2026 RSI-Agent-VLA Authors.
"""Zero-model CPU commands still require scope admission, supervision and receipts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .midterm_host import MidtermHost

import os
from copy import deepcopy
from pathlib import Path

from .storage import atomic_json

KIND = "generic-cpu-model0"
PURPOSES = frozenset(
    {
        "analysis",
        "data_review",
        "baseline_freeze",
        "result_import",
        "scientific_evaluation",
        "qualification",
        "successor_prepare",
    }
)


def is_cpu(template: dict[str, Any]) -> bool:
    return template.get("permit_template", {}).get("kind") == KIND


def validate_supervision(context: dict[str, Any]) -> None:
    """Validate CPU authority at the actual supervisor process boundary."""
    from datetime import datetime, timezone

    from .midterm_host import read

    grant = read(context["cpu_authorization_ref"])
    scope = read(grant["scope_ref"])
    reservation = read(grant["accounting_ref"])["reservations"][grant["operation_id"]]
    if (
        grant.get("kind") != KIND
        or grant.get("authorized") is not True
        or scope.get("authorized") is not True
        or scope.get("cpu_authorized") is not True
        or scope.get("status") != "active"
        or scope.get("owner_uid") != os.getuid()
        or grant.get("owner_uid") != os.getuid()
        or grant["scope_id"] != scope["scope_id"]
        or grant["operation_id"] not in scope.get("cpu_operation_ids", [])
        or context["run_id"] != grant["operation_id"]
        or context.get("gpu_count") != 0
        or Path(context["authorization_ref"]).resolve()
        != Path(grant["scope_ref"]).resolve()
        or any(
            value > grant["limits"].get(key, -1)
            for key, value in context["limits"].items()
        )
        or reservation.get("status") != "active"
        or not reservation.get("submitted_at")
        or reservation.get("scope_id") != scope["scope_id"]
        or reservation.get("gpu_devices") != []
        or datetime.now(timezone.utc)
        >= datetime.fromisoformat(scope["expires_at"].replace("Z", "+00:00"))
    ):
        raise PermissionError(
            "CPU supervisor lacks its live zero-model operation grant"
        )


def permit(
    host: MidtermHost,
    operation: dict[str, Any],
    template: dict[str, Any],
    *,
    consuming: bool = False,
) -> dict[str, Any]:
    """Check the exact registered operation and live scope again at process launch."""
    from .midterm_host import read

    scope = host.ledger.authority()
    if (
        scope.get("cpu_authorized") is not True
        or operation["operation_id"] not in scope.get("cpu_operation_ids", [])
        or template.get("cpu_purpose") not in PURPOSES
    ):
        raise PermissionError(
            "explicit registered CPU operation authorization required"
        )
    if template.get("gpu_devices") or any(
        v
        for k, v in template["budget"].items()
        if k not in {"wall_seconds", "storage_bytes", "current_disk_bytes"}
    ):
        raise PermissionError(
            "CPU model0 cannot acquire model, GPU, training or device authority"
        )
    if set(template["supervisor_limits"]) - {"wall_seconds", "storage_bytes"}:
        raise PermissionError("CPU supervisor cannot acquire extra resource dimensions")
    for field in ("permit_ref", "result_ref"):
        if not Path(template[field]).resolve().is_relative_to(host.ledger.root):
            raise PermissionError("CPU permit/result escapes scope")
    if "command" not in template:
        from .task_completion import ACTION_ROLES

        if operation.get("dynamic") is not False or operation.get("kind") not in ACTION_ROLES:
            raise PermissionError("CPU role permit requires a registered nondynamic consumer")
    value = deepcopy(template["permit_template"])
    value.update(
        schema="rsi-cpu-operation-permit-v1",
        authorized=True,
        scope_id=scope["scope_id"],
        scope_ref=str(host.ledger.scope_ref),
        accounting_ref=str(host.ledger.path),
        operation_id=operation["operation_id"],
        operation=operation,
        owner_uid=os.getuid(),
        command=template.get("command"),
        cpu_purpose=template["cpu_purpose"],
        expires_at=scope["expires_at"],
        result_ref=template["result_ref"],
        limits=template["supervisor_limits"],
        version=template["version"],
        inputs_ref=str(host.root / "jobs" / operation["operation_id"] / "public-inputs.json"),
    )
    path = Path(template["permit_ref"])
    if path.exists() and read(path) != value:
        raise ValueError("conflicting CPU operation permit")
    if consuming:
        if not path.exists():
            raise PermissionError("CPU operation has no materialized permit")
        record = host.ledger.validate_reservation(
            operation["operation_id"],
            budget=template["budget"],
            phase_ref=host.core_config["phase_ref"],
            gpu_devices=[],
        )
        if not record.get("submitted_at"):
            raise PermissionError("CPU operation has no committed Host submission")
    else:
        atomic_json(path, value)
    return value


def result(
    host: MidtermHost, operation: dict[str, Any], template: dict[str, Any]
) -> dict[str, Any]:
    """Consume an original native terminal envelope; failure remains a business result."""
    from .midterm_host import read

    path = Path(template["result_ref"]).resolve(strict=True)
    value = read(path)
    if (
        not path.is_relative_to(host.ledger.root)
        or path.stat().st_uid != os.getuid()
        or value.get("schema") != "rsi-cpu-operation-result-v1"
        or value.get("scope_id") != host.config["scope_id"]
        or value.get("operation_id") != operation["operation_id"]
        or value.get("version") != template["version"]
        or value.get("status")
        not in {"completed", "failed", "defer", "reject", "inconclusive"}
    ):
        raise ValueError(
            "CPU native result identity, version or terminal state mismatch"
        )
    return value


def measurements(
    host: MidtermHost, identity: str, template: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Use supervisor observations; zero external costs follow the CPU capability grant."""
    actual = {}
    for key in template["budget"]:
        if key == "wall_seconds":
            actual[key] = report["wall_seconds"]
        elif key == "storage_bytes":
            actual[key] = report.get(
                "business_storage_delta_bytes", report["storage_bytes"]
            )
        elif key == "current_disk_bytes":
            actual[key] = report["storage_accounting"]["current_new_bytes"]
        else:
            # These dimensions are forbidden by the CPU capability boundary.
            actual[key] = 0
    return {
        "actual": actual,
        "unknown_dimensions": [],
        "evidence": {
            "supervisor_ref": str(host.root / "jobs" / identity / "supervisor.json")
        },
    }
