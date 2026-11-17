# ruff: noqa: UP017, FURB162 - Python 3.10 support
"""Protected safety observations consumed by the existing single release writer.

Registrations and safety observations belong to operator; service loading facts
belong to runtime. Offline observations retain their synthetic evidence label.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .qualification_import import _readonly, require_role


def _read(reference):
    return json.loads(Path(reference).read_text(encoding="utf-8"))


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("safety evidence needs timezone-aware timestamps")
    return result


def registration(manager, *, role):
    """Resolve the operator registration under the current OS role boundary."""
    pointer = manager.records.path("publication_registration", "current")
    if not pointer.exists():
        return None
    saved = manager.records.get("publication_registration", "current")
    value = _read(saved["registration_ref"])
    if value != saved["registration_snapshot"]:
        raise ValueError("publication safety registration changed")
    _validate_registration(manager, value, saved["registration_ref"], role=role)
    return value


def _validate_registration(manager, value, reference, *, role):
    if (
        value.get("schema") != "rsi-release-safety-registration-v1"
        or value.get("authorized") is not True
        or _time(value["expires_at"]) <= datetime.now(timezone.utc)
    ):
        raise ValueError("publication safety registration missing or expired")
    require_role(value, role)
    domains = value["role_domains"]
    if (
        not Path(reference)
        .resolve(strict=True)
        .is_relative_to(Path(domains["operator"]).resolve())
        or not _readonly(reference)
        or Path(domains["release"]).resolve() != manager.records.root.resolve()
        or Path(domains["research"]).resolve() != manager.research.root.resolve()
        or Path(domains["evaluation"]).resolve() != manager.evaluations.root.resolve()
    ):
        raise ValueError("publication safety registration domains mismatch")
    for key, domain in (("safety_ref", "operator"), ("loaded_ref", "runtime")):
        path = Path(value[key]).resolve(strict=True)
        if not path.is_relative_to(Path(domains[domain]).resolve()):
            raise ValueError("publication evidence has wrong producer domain")


def register(manager, reference):
    """Bind a release domain to independently produced protected observations."""
    reference = str(Path(reference).resolve(strict=True))
    value = _read(reference)
    _validate_registration(manager, value, reference, role="release")
    manager.records.put(
        "publication_registration",
        "current",
        {"registration_ref": reference, "registration_snapshot": value},
    )
    return value


def _active_protocol(manager, active):
    """Use the accepted publication identity even when candidate artifacts fail."""
    decision = manager.records.path("decisions", active["activation"])
    if decision.exists():
        protocol_id = manager.records.get("decisions", active["activation"])[
            "protocol_id"
        ]
    elif active.get("qualification_ref"):
        protocol_id = manager.records.get(
            "qualifications", active["qualification_ref"]
        )["protocol_id"]
    else:
        protocol_id = manager.research.get("bundles", active["bundle_id"])[
            "protocol_id"
        ]
    return manager.evaluations.get("protocols", protocol_id)


def check(manager, *, active, operation, safe_point, queues_empty=True, role="release"):
    """Reject stale/unknown physical state, costs, cleanup, or loaded identities."""
    config = registration(manager, role=role)
    if config is None:
        protocol = _active_protocol(manager, active)
        if protocol.get("task_policy"):
            raise ValueError(
                "new-task publication requires independent safety registration"
            )
        # Legacy fixtures have only offline engineering semantics. Registered
        # domains, including synthetic domains, always use independent evidence.
        if (
            safe_point not in {"task_boundary", "approved_safe_point"}
            or queues_empty is not True
        ):
            raise ValueError("activation needs a safe point and cleared queues")
        return None
    safety = _read(config["safety_ref"])
    loaded = _read(config["loaded_ref"])
    if config.get("phase_ref") and _read(config["phase_ref"]).get("unresolved_costs"):
        raise ValueError("publication phase still has unresolved costs")
    now = datetime.now(timezone.utc)
    for ref, value in ((config["safety_ref"], safety), (config["loaded_ref"], loaded)):
        if (
            not _readonly(ref)
            or value.get("scope") != config["scope"]
            or value.get("active_snapshot") != active
            or not _time(value["observed_at"]) <= now < _time(value["expires_at"])
        ):
            raise ValueError("stale or unprotected publication evidence")
    if (
        safety.get("schema") != "rsi-release-safety-observation-v1"
        or safety.get("producer_role") != "operator"
        or operation not in safety.get("operations", [])
        or safety.get("safe_point") != safe_point
        or safe_point not in {"task_boundary", "approved_safe_point"}
        or queues_empty is not True
        or safety.get("queues_empty") is not True
        or safety.get("pending_actions") != []
        or safety.get("unresolved_costs") != []
        or safety.get("cleanup_complete") is not True
        or safety.get("physical_state") != "known_safe"
        or not safety.get("evidence_refs")
    ):
        raise ValueError("independent safety, queue, cost or cleanup evidence missing")
    if (
        loaded.get("schema") != "rsi-service-loading-observation-v1"
        or loaded.get("producer_role") != "runtime"
        or not loaded.get("evidence_refs")
    ):
        raise ValueError("runtime service loading evidence missing")
    protocol = _active_protocol(manager, active)
    expected_level = (
        "synthetic_offline" if protocol["scope"] == "offline" else protocol["scope"]
    )
    if (
        safety.get("evidence_level") != expected_level
        or loaded.get("evidence_level") != expected_level
    ):
        raise ValueError("publication evidence level mismatch")
    if loaded.get("service_stopped") is not True:
        from .runtime import execution_binding, resolve_reference

        bundle = manager.research.get("bundles", active["bundle_id"])
        expected = execution_binding(
            manager.research, bundle, resolve_reference(bundle["backend_ref"])
        )
        if loaded.get("loaded_binding") != expected:
            raise ValueError("actual service loading differs from effective version")
    return {
        "safety_ref": config["safety_ref"],
        "safety_snapshot": safety,
        "loaded_ref": config["loaded_ref"],
        "loaded_snapshot": loaded,
    }


def registered_publication_paths(manager):
    """Return exact immutable inputs to protect in a new Runtime namespace."""
    pointer = manager.records.path("publication_registration", "current")
    if not pointer.exists():
        return ()
    saved = manager.records.get("publication_registration", "current")
    config = saved["registration_snapshot"]
    return tuple(
        Path(item).resolve(strict=True)
        for item in (
            saved["registration_ref"],
            config["safety_ref"],
            config["loaded_ref"],
        )
    )
