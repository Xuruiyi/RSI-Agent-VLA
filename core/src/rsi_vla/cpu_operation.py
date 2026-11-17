# Copyright 2026 RSI-Agent-VLA Authors.
"""Fixed CPU business consumers behind the Host's zero-model permit."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_json


def execute(permit_ref: str | Path, inputs_ref: str | Path) -> dict:
    """Run an existing consumer in its role; this entry never activates a release."""

    def read(path):
        return json.loads(Path(path).read_text())

    permit, inputs = read(permit_ref), read(inputs_ref)
    scope = read(permit["scope_ref"])
    expiry = datetime.fromisoformat(scope["expires_at"].replace("Z", "+00:00"))
    if (
        permit.get("schema") != "rsi-cpu-operation-permit-v1"
        or permit.get("authorized") is not True
        or scope.get("authorized") is not True
        or scope.get("cpu_authorized") is not True
        or scope.get("status") != "active"
        or datetime.now(timezone.utc) >= expiry
        or permit["scope_id"] != scope["scope_id"]
        or permit["owner_uid"] != os.getuid()
        or inputs["scope_id"] != scope["scope_id"]
        or inputs["operation"] != permit["operation"]
        or permit["operation_id"] not in scope.get("cpu_operation_ids", [])
    ):
        raise PermissionError("live CPU operation scope/input binding mismatch")
    reservation = read(permit["accounting_ref"])["reservations"][permit["operation_id"]]
    if (
        reservation.get("scope_id") != scope["scope_id"]
        or not reservation.get("submitted_at")
        or reservation.get("status") != "active"
        or reservation.get("gpu_devices") != []
    ):
        raise PermissionError("CPU consumer requires its committed reservation")
    payload = {**inputs["operation"]["payload"], **inputs["references"]}
    purpose = permit["cpu_purpose"]
    role = os.environ.get("RSI_ISOLATED_ROLE")
    roles = {
        "analysis": {"operator", "evaluation"},
        "baseline_freeze": {"operator"},
        "data_review": {"evaluation"},
        "result_import": {"evaluation"},
        "scientific_evaluation": {"evaluation"},
        "qualification": {"release"},
        "successor_prepare": {"runtime", "operator"},
    }
    if role not in roles[purpose]:
        raise PermissionError("CPU business consumer requires its independent role")
    result_ref = Path(permit["result_ref"])
    artifact_ref = result_ref.with_name("artifact.json")
    if purpose == "analysis":
        from .midterm_science import exact_weighted_paired

        value = exact_weighted_paired(read(payload["manifest_ref"])["differences"])
    elif purpose == "baseline_freeze":
        from .midterm_evaluation import freeze_midterm_b0

        value = freeze_midterm_b0(read(payload["manifest_ref"]))
    elif purpose == "data_review":
        from .adapters.pi05_training import audit_demonstration_payload

        value = audit_demonstration_payload(read(payload["manifest_ref"]))
    elif purpose == "result_import":
        from .contracts import ExperimentResult
        from .evaluation import EvaluationStore

        value = read(payload["manifest_ref"])
        EvaluationStore(payload["evaluation_store_ref"]).save_result(
            ExperimentResult(**value)
        )
    elif purpose == "scientific_evaluation":
        from .midterm_science import compute_admission

        value = compute_admission(
            payload["registration_ref"], payload.get("contrast", "theta")
        )
    elif purpose == "qualification":
        from .evaluation import EvaluationStore
        from .release import ReleaseManager
        from .storage import ResearchStore

        manager = ReleaseManager(
            payload["release_root"],
            ResearchStore(payload["research_store_ref"]),
            EvaluationStore(payload["evaluation_store_ref"]),
        )
        value = manager.qualify_initial(
            payload["qualification_id"],
            result_id=payload["result_id"],
            evidence_import_id=payload.get("evidence_import_id"),
        )
    else:
        from .midterm_deployment import inheritance_plan

        value = inheritance_plan(payload["registration_ref"])
    atomic_json(artifact_ref, value)
    result = {
        "schema": "rsi-cpu-operation-result-v1",
        "scope_id": permit["scope_id"],
        "operation_id": permit["operation_id"],
        "version": permit["version"],
        "status": "completed",
        "artifact_ref": str(artifact_ref),
        "value": value,
    }
    atomic_json(result_ref, result)
    return result
