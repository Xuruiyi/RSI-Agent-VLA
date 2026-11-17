# Copyright 2026 RSI-Agent-VLA Authors.
"""Receipt-bound inputs for registered Host consumers; never grant commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .midterm_host import MidtermHost

from copy import deepcopy
from pathlib import Path

from .storage import atomic_json, identifier

FIELDS = frozenset(
    {
        "theta_ref",
        "processor_ref",
        "model_path",
        "norm_stats_path",
        "dataset_root",
        "manifest_ref",
        "result_ref",
        "evaluation_ref",
        "qualification_ref",
        "handoff_ref",
        "bundle_id",
        "profile_ref",
        "artifact_ref",
        "admission_ref", "recipe_ref", "load_check_ref", "training_result_ref",
        "load_result_ref", "materialization_ref", "task_ref", "protocol_ref",
    }
)


def materialize_inputs(
    host: MidtermHost, operation: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Bind terminal native fields and settled costs into one immutable job input."""
    from .midterm_host import read

    template = host.config["operations"][operation["operation_id"]]
    references, sources = {}, {}
    for name, dependency in template.get("late_refs", {}).items():
        identifier(name)
        predecessor = dependency["operation_id"]
        if predecessor == operation["operation_id"]:
            raise ValueError("operation cannot consume its own pending output")
        registered = next(
            op
            for op in host.core_config["operations"]
            if op["operation_id"] == predecessor
        )
        producer = host.config["operations"][predecessor]
        export = producer.get("exports", {}).get(dependency["export"])
        if (
            not export
            or set(export) != {"field", "version", "type"}
            or export["field"] not in FIELDS
            or export["version"] != dependency["version"]
            or export["version"] != producer.get("version")
            or export["type"] not in {"path", "value"}
        ):
            raise PermissionError("late reference is not a registered versioned export")
        native_ref = (
            Path(host.config["continuation_root"])
            / "receipts"
            / (identifier(predecessor) + ".json")
        )
        native = read(native_ref)
        job = host.root / "jobs" / predecessor
        completion = read(job / "completion.json")
        report = read(job / "supervisor.json")
        reservation = host.ledger.read()["reservations"][predecessor]
        if (
            native.get("operation") != registered
            or native.get("status") != "completed"
            or completion.get("scope_id") != host.config["scope_id"]
            or completion.get("operation_id") != predecessor
            or completion.get("status") != "completed"
            or completion.get("actual") != reservation.get("actual")
            or completion.get("unknown_dimensions")
            or Path(completion["supervisor_ref"]).resolve()
            != (job / "supervisor.json").resolve()
            or reservation.get("scope_id") != host.config["scope_id"]
            or reservation.get("owner_uid")
            != host.config.get(
                "owner_uid", host.ledger.authority(active=False)["owner_uid"]
            )
            or reservation.get("status") != "settled"
            or reservation.get("cleanup_confirmed") is not True
            or report.get("identity", {}).get("run_id") != predecessor
            or report.get("status") != "completed"
            or report.get("exit_code") != 0
            or report.get("cleanup_confirmed") is not True
            or report.get("remaining_process_ids") != []
        ):
            raise PermissionError(
                "predecessor operation, scope, terminal or cost closure is unresolved"
            )
        value = native["value"][export["field"]]
        if export["type"] == "path":
            path = Path(value).resolve(strict=True)
            roots = [
                Path(root).resolve() for root in producer.get("storage_write_roots", [])
            ]
            if (
                not path.is_relative_to(host.ledger.root)
                or not (path.is_file() or path.is_dir())
                or not any(path.is_relative_to(root) for root in roots)
            ):
                raise PermissionError("late artifact escapes the task domain")
            value = str(path)
        references[name] = value
        sources[name] = {
            "dependency": dependency,
            "export": export,
            "native": native,
            "completion": completion,
            "supervision": report,
            "actual": reservation["actual"],
        }
    value = {
        "schema": "rsi-host-inputs-v1",
        "scope_id": host.config["scope_id"],
        "operation": operation,
        "version": template.get("version"),
        "references": references,
        "sources": sources,
    }
    # Freeze actual script input at the same boundary as the command and receipts.
    value["command_sources"] = {
        argument: Path(argument).read_text()
        for argument in template.get("command", [])
        if isinstance(argument, str)
        and argument.endswith(".py")
        and Path(argument).is_file()
    }
    if host.core_config.get("research_store_ref"):
        from .storage import ResearchStore
        from .version_binding import bundle_references

        store = ResearchStore(host.core_config["research_store_ref"])
        bundle = store.get("bundles", host.core_config["base_bundle_id"])
        value["version_references"] = bundle_references(bundle)
    path = host.root / "jobs" / identifier(operation["operation_id"]) / "inputs.json"
    if path.exists():
        if read(path) != value:
            raise ValueError("frozen Host input or predecessor receipt changed")
    else:
        atomic_json(path, value)
    public = {key: value[key] for key in ("schema", "scope_id", "operation", "version", "references")}
    public["version_references"] = value.get("version_references", {})
    public_path = path.with_name("public-inputs.json")
    if public_path.exists() and read(public_path) != public:
        raise ValueError("frozen public Host input changed")
    if not public_path.exists():
        atomic_json(public_path, public)
    return path, value


def consumer_inputs(
    host: MidtermHost, operation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill registered data fields only; commands, grants and paths stay host-owned."""
    template = deepcopy(host.config["operations"][operation["operation_id"]])
    if not template.get("late_refs"):
        return deepcopy(operation), template
    _path, saved = materialize_inputs(host, operation)
    resolved = deepcopy(operation)
    for name, dependency in template["late_refs"].items():
        target = dependency.get("target")
        if target is None:
            continue  # CPU commands consume the immutable input file directly.
        if (
            not isinstance(target, list)
            or len(target) != 2
            or target[0] not in {"payload", "permit_template"}
            or target[1] not in FIELDS
        ):
            raise PermissionError(
                "late reference cannot change commands or authorization"
            )
        destination = (
            resolved["payload"]
            if target[0] == "payload"
            else template["permit_template"]
        )
        value = saved["references"][name]
        if target[1] in destination and destination[target[1]] != value:
            raise ValueError("late reference conflicts with registered consumer input")
        destination[target[1]] = value
    return resolved, template


def resolved_role_configuration(
    configuration: dict[str, Any],
    operation: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Apply immutable receipt data after the registered public controls validate.

    A late input can fill only a registered data field in the payload. It cannot
    replace an existing input, command, operation identity or authorization.
    """
    registered = next(
        row
        for row in configuration["operations"]
        if row["operation_id"] == operation["operation_id"]
    )
    if registered != operation:
        raise PermissionError("late input operation differs from registered control")
    additions = {
        key: value
        for key, value in resolved.get("payload", {}).items()
        if key not in operation.get("payload", {})
    }
    expected = deepcopy(operation)
    expected.setdefault("payload", {}).update(additions)
    if not additions.keys() <= FIELDS or expected != resolved:
        raise PermissionError("late inputs cannot change registered role controls")
    value = deepcopy(configuration)
    value["operations"] = [
        resolved if row == registered else row for row in value["operations"]
    ]
    return value
