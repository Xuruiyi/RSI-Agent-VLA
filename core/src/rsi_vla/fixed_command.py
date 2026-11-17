"""Permit and receipt adapters for the registered CPU, teacher and native jobs."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_json
from .task_policy import select_gpu

FAMILIES = {
    "libero-public-data-acquire",
    "libero-public-data-materialize",
    "libero-scripted-teacher",
    "pi05-training-pilot",
    "pi05-training-learning",
    "pi05-load-check",
}
METER_FAMILIES = {
    "libero-public-data-acquire",
    "libero-public-data-materialize",
    "libero-scripted-teacher",
}


def read(path):
    return json.loads(Path(path).read_text())


def fixed_permit(host, operation, template):
    """Materialize only this operation's registered family permit within live authority."""
    if "permit_ref" not in template and "permit_template" not in template:
        return None
    if not {"permit_ref", "permit_template"} <= template.keys():
        raise ValueError("fixed command requires both permit_ref and permit_template")
    permit = deepcopy(template["permit_template"])
    kind = permit.get("kind")
    if kind not in FAMILIES:
        raise ValueError("fixed-command permit has no supported actual consumer")
    scope = host.ledger.authority()
    root = host.ledger.root.resolve()
    target = Path(template["permit_ref"]).resolve()
    if not target.is_relative_to(root):
        raise PermissionError("fixed permit escapes the current task write domain")
    command = template["command"]
    if (
        "--permit" not in command
        or Path(command[command.index("--permit") + 1]).resolve() != target
    ):
        raise PermissionError(
            "registered command does not consume its exact current-operation permit"
        )
    for field in (
        "result_ref",
        "cost_receipt_ref",
        "model_meter_ref",
        "download_ledger_ref",
        "download_meter_ref",
        "start_marker_ref",
    ):
        if template.get(field) and not Path(template[field]).resolve().is_relative_to(
            root
        ):
            raise PermissionError(
                "fixed receipt reference escapes task domain: " + field
            )
    for key in ("output_root", "artifact_root"):
        if key in permit and not Path(permit[key]).resolve().is_relative_to(root):
            raise PermissionError("fixed job writes outside its task domain: " + key)
    if kind in METER_FAMILIES and template["budget"].get("model_requests") != 0:
        raise PermissionError(
            "teacher/data require an explicit zero project-model request cap"
        )
    expiry = min(
        parse_time(scope["expires_at"]),
        parse_time(permit.get("expires_at", scope["expires_at"])),
    )
    if (expiry - datetime.now(timezone.utc)).total_seconds() < template[
        "supervisor_limits"
    ]["wall_seconds"]:
        raise PermissionError("complete fixed command does not fit task/permit expiry")
    permit.update(
        authorized=True,
        scope_id=scope["scope_id"],
        scope_ref=str(host.ledger.scope_ref),
        owner_uid=scope["owner_uid"],
        write_root=str(root),
        phase_ref=host.core_config["phase_ref"],
        accounting_ref=str(host.ledger.path),
        operation_id=operation["operation_id"],
        expires_at=expiry.isoformat(),
    )
    permit.setdefault(
        "protocol_id", read(host.core_config["protocol_ref"])["protocol_id"]
    )
    permit.setdefault("source_manifest_ref", host.core_config["source_manifest_ref"])
    outer = template["budget"]
    if kind == "pi05-training-learning":
        candidate = permit.get("candidate_id")
        if (
            not candidate
            or candidate != template.get("candidate_id")
            or not template.get("candidate_binding")
        ):
            raise PermissionError(
                "learning permit must bind its registered candidate identity and frozen recipe"
            )
        claims = host.ledger.read().get("learning_candidate_starts", {})
        expected = 0 if candidate in claims else 1
        if (
            type(outer.get("learning_candidates")) is not int
            or outer["learning_candidates"] != expected
        ):
            raise PermissionError(
                "candidate start reserve differs from the live once-per-candidate claim"
            )
    if kind == "libero-public-data-acquire" and permit.get(
        "max_download_bytes", -1
    ) > outer.get("download_bytes", -1):
        raise PermissionError(
            "download permit exceeds its registered outer reservation"
        )
    if "budget" in permit:
        inner = permit["budget"]
        if (
            inner["wall_seconds"] > template["supervisor_limits"]["wall_seconds"]
            or inner["gpu_device_seconds"] > outer.get("gpu_seconds", 0)
            or inner["artifact_bytes"]
            > min(outer.get("storage_bytes", 0), outer.get("current_disk_bytes", 0))
        ):
            raise PermissionError(
                "native permit exceeds the complete registered job envelope"
            )
    else:
        for key in ("storage_bytes", "current_disk_bytes"):
            if key in permit and permit[key] > outer.get(key, -1):
                raise PermissionError("CPU permit exceeds its outer reservation")
        if (
            permit.get("wall_seconds", 0)
            > template["supervisor_limits"]["wall_seconds"]
        ):
            raise PermissionError("CPU permit exceeds complete command wall limit")
    devices = template.get("gpu_devices", [])
    if kind.startswith("libero-public-data-") and devices:
        raise PermissionError(
            "registered public-data operations cannot acquire GPU devices"
        )
    if devices:
        if len(devices) != 1 or permit.get("cuda_device", devices[0]) != devices[0]:
            raise PermissionError(
                "fixed job CUDA binding differs from its registered device"
            )
        permit["cuda_device"] = devices[0]
    if target.exists() and read(target) != permit:
        raise ValueError(
            "fixed permit identity already binds different registered input"
        )
    atomic_json(target, permit)
    return permit


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("fixed permit expiry requires timezone")
    return parsed


def query_gpu_cards():
    returned = subprocess.run(
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
    for line in returned.stdout.splitlines():
        index, name, free, load, capability = [part.strip() for part in line.split(",")]
        cards.append(
            {
                "physical_gpu": int(index),
                "name": name,
                "free_memory_mib": int(free),
                "utilization_percent": int(load) if load.isdigit() else None,
                "compute_capability": capability,
            }
        )
    return cards


def probe_fixed_gpu(template, *, gpu_probe=None):
    """Fresh-check the registered device through the existing compatibility policy."""
    devices = template.get("gpu_devices", [])
    if not devices:
        return {"blockers": [], "gpu_telemetry": None}
    policy = template.get("gpu_admission")
    if not policy:
        raise ValueError("GPU fixed command requires a registered gpu_admission policy")
    for field in (
        "required_memory_mib",
        "safety_margin_mib",
        "minimum_compute_capability",
    ):
        if field not in policy:
            raise ValueError("fixed GPU policy is missing " + field)
    try:
        cards = (gpu_probe or query_gpu_cards)()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "blockers": ["gpu_telemetry_unavailable:" + type(exc).__name__],
            "gpu_telemetry": None,
        }
    assigned = [card for card in cards if card["physical_gpu"] in devices]
    blockers = [
        "registered_gpu_unavailable:" + str(device)
        for device in devices
        if select_gpu(
            [card for card in assigned if card["physical_gpu"] == device], policy
        )
        != device
    ]
    return {
        "blockers": blockers,
        "gpu_telemetry": {"devices": devices, "cards": assigned, "policy": policy},
    }


def _bound_record(path, scope_id, operation_id):
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError("fixed receipt is not a record")
    if value.get("scope_id") != scope_id or value.get("operation_id") != operation_id:
        raise ValueError(
            "fixed receipt belongs to another scope/operation: " + str(path)
        )
    return value


def fixed_measurements(host, operation_id, template, report):
    """Read registered native markers and explicit model/download producers."""
    from .cpu_command import is_cpu
    if is_cpu(template):
        from .cpu_command import measurements
        return measurements(host, operation_id, template, report)
    permit = read(template["permit_ref"])
    kind = permit["kind"]
    scope_id = host.config["scope_id"]
    actual, unknown, evidence = {}, [], {}
    cost_ref = template.get("cost_receipt_ref", template.get("result_ref"))
    cost = None
    if cost_ref and Path(cost_ref).exists():
        try:
            cost = _bound_record(cost_ref, scope_id, operation_id)
            if (
                Path(cost_ref).stat().st_uid != permit["owner_uid"]
                or cost.get("owner_uid", permit["owner_uid"]) != permit["owner_uid"]
            ):
                raise ValueError("fixed cost receipt owner mismatch")
            evidence["cost_receipt_ref"] = str(cost_ref)
        except (OSError, ValueError) as exc:
            evidence["cost_receipt_error"] = str(exc)
    marker_ref = template.get("start_marker_ref")
    for counter in (
        "runtime_starts",
        "training_starts",
        "training_pilots",
        "load_checks",
        "learning_candidates",
    ):
        if counter not in template["budget"]:
            continue
        if marker_ref is None:
            unknown.append(counter)
            continue
        if Path(marker_ref).exists():
            try:
                marker = _bound_record(marker_ref, scope_id, operation_id)
                schema = (
                    "rsi-libero-teacher-start-v1"
                    if kind == "libero-scripted-teacher"
                    else "rsi-native-training-start-v1"
                )
                if (
                    Path(marker_ref).stat().st_uid != permit["owner_uid"]
                    or marker.get("schema") != schema
                    or marker.get("owner_uid") != permit["owner_uid"]
                    or marker.get("run_id") != permit.get("run_id")
                    or marker.get("kind") != kind
                ):
                    raise ValueError(
                        "native start marker identity differs from fixed permit"
                    )
                if counter == "learning_candidates":
                    candidate = permit.get("candidate_id")
                    with host.ledger.lock():
                        state = host.ledger._read()
                        if candidate != template.get("candidate_id") or state.get(
                            "learning_candidate_bindings", {}
                        ).get(candidate, {}).get("binding") != template.get(
                            "candidate_binding"
                        ):
                            raise ValueError(
                                "native learning start has no matching frozen candidate registration"
                            )
                        claims = state.setdefault("learning_candidate_starts", {})
                        prior = claims.get(candidate)
                        if prior is None:
                            claims[candidate] = operation_id
                            atomic_json(host.ledger.path, state)
                        actual[counter] = (
                            1 if prior is None or prior == operation_id else 0
                        )
                else:
                    actual[counter] = template["budget"][counter]
            except (OSError, ValueError) as exc:
                unknown.append(counter)
                evidence["native_start_error"] = str(exc)
        else:
            actual[counter] = 0
    meter_ref = template.get("model_meter_ref") or (cost or {}).get("model_meter_ref")
    if meter_ref and not Path(meter_ref).resolve().is_relative_to(
        host.ledger.root.resolve()
    ):
        evidence["model_meter_error"] = "model meter escapes the task receipt domain"
        meter_ref = None
    if meter_ref and Path(meter_ref).exists():
        try:
            meter = _bound_record(meter_ref, scope_id, operation_id)
            if (
                Path(meter_ref).stat().st_uid != permit["owner_uid"]
                or meter.get("owner_uid", permit["owner_uid"]) != permit["owner_uid"]
            ):
                raise ValueError("model meter owner mismatch")
        except (OSError, ValueError) as exc:
            evidence["model_meter_error"] = str(exc)
            meter = {}
        schema = (
            "rsi-scripted-teacher-model-meter-v1"
            if kind == "libero-scripted-teacher"
            else "rsi-native-training-model-meter-v1"
            if kind.startswith("pi05-")
            else "rsi-public-data-model-meter-v1"
        )
        if (
            meter.get("schema") == schema
            and meter.get("status") == "closed"
            and (
                kind == "libero-scripted-teacher"
                or meter.get("owner_uid") == permit["owner_uid"]
            )
            and type(meter.get("project_model_requests")) is int
            and meter["project_model_requests"] == 0
            and meter.get("model_clients_created") == []
            and (
                not kind.startswith("pi05-")
                or meter.get("components")
                == [
                    "native_loader_one_prediction"
                    if kind == "pi05-load-check"
                    else "native_sft"
                ]
            )
            and report.get("cleanup_confirmed") is True
            and (
                meter.get("model_transports_created") == []
                or kind == "libero-scripted-teacher"
                and meter.get("enabled_components") == ["native_env"]
                and meter.get("request_cap") == 0
            )
        ):
            actual["model_requests"] = 0
            evidence["model_meter_ref"] = str(meter_ref)
    if "model_requests" in template["budget"] and "model_requests" not in actual:
        unknown.append("model_requests")
    if "download_bytes" in template["budget"]:
        download = download_measurement(
            host, operation_id, template, permit, cost, report
        )
        if download["lower_bound"] is not None:
            actual["download_bytes"] = download["lower_bound"]
        if not download["exact"]:
            unknown.append("download_bytes")
        evidence["download"] = download["evidence"]
    return {
        "actual": actual,
        "unknown_dimensions": sorted(set(unknown)),
        "evidence": evidence,
    }


def charge_fixed_phase(host, operation_id, measurements):
    """Charge receipt-only dimensions once; outer supervision already books resources."""
    path = host.core_config["phase_ref"]
    phase = read(path)
    observed = {}
    for key, value in measurements["actual"].items():
        dimension = (
            "simulation_episodes"
            if key == "runtime_starts" and "simulation_episodes" in phase["limits"]
            else key
        )
        if dimension in phase["limits"] and dimension not in {
            "wall_seconds",
            "gpu_seconds",
            "storage_bytes",
            "current_disk_bytes",
        }:
            observed[dimension] = value
    rows = phase.setdefault("fixed_command_receipt_charges", [])
    prior = next(
        (row for row in reversed(rows) if row["operation_id"] == operation_id), None
    )
    if (
        prior
        and prior["observed"] == observed
        and prior["unknown_dimensions"] == measurements["unknown_dimensions"]
    ):
        return
    previous = prior["observed"] if prior else {}
    if any(observed.get(key, -1) < value for key, value in previous.items()):
        raise ValueError(
            "fixed receipt reconciliation removes previously measured costs"
        )
    delta = {key: value - previous.get(key, 0) for key, value in observed.items()}
    for key, value in delta.items():
        phase["used"][key] += value
        phase["remaining"][key] = max(0, phase["limits"][key] - phase["used"][key])
    rows.append(
        {
            "operation_id": operation_id,
            "observed": observed,
            "newly_charged": delta,
            "unknown_dimensions": measurements["unknown_dimensions"],
            "evidence": measurements["evidence"],
        }
    )
    phase["unresolved_costs"] = [
        row
        for row in phase.get("unresolved_costs", [])
        if not (
            row.get("operation_id") == operation_id
            and row.get("source") == "fixed_command_receipts"
        )
    ]
    if measurements["unknown_dimensions"]:
        phase["unresolved_costs"].append(
            {
                "operation_id": operation_id,
                "missing": measurements["unknown_dimensions"],
                "source": "fixed_command_receipts",
            }
        )
    atomic_json(path, phase)


def fixed_storage_context(host, operation, template):
    """Keep one job's real writes separate from the shared scope inventory."""
    if not template.get("permit_ref"):
        return {}
    from .process_accounting import StorageMeter

    operation_id = operation["operation_id"]
    job = host.root / "jobs" / operation_id
    roots = template.get("storage_write_roots")
    if not roots:
        raise ValueError(
            "fixed command requires its actual registered storage_write_roots"
        )
    roots = list(
        dict.fromkeys(
            [
                *(str(Path(path).resolve()) for path in roots),
                str(job),
                str(Path(template["permit_ref"]).resolve()),
            ]
        )
    )
    if any(not Path(path).is_relative_to(host.ledger.root.resolve()) for path in roots):
        raise PermissionError("fixed command write root escapes scope")
    business_ref = job / "business-storage-state.json"
    scope_ref = host.root / "scope-storage-state.json"
    if not scope_ref.exists():
        atomic_json(scope_ref, StorageMeter([host.ledger.root]).state())
    if not business_ref.exists():
        atomic_json(business_ref, StorageMeter(roots).state())
    state = host.ledger.read()
    held = sum(
        host.ledger._held(row, "storage_bytes")
        for key, row in state["reservations"].items()
        if key != operation_id
    )
    reserve = sum(
        row["upper_bound_bytes"]
        for row in state.get("historical_control_uncertainties", [])
    )
    limits = dict(template["supervisor_limits"])
    if template.get("gpu_devices"):
        if template["budget"].get("gpu_seconds", 0) <= 0:
            raise ValueError(
                "GPU fixed command requires its complete device-time budget"
            )
        limits["gpu_seconds"] = template["budget"]["gpu_seconds"]
    return {
        "limits": limits,
        "storage_write_roots": roots,
        "storage_meter_state_ref": str(business_ref),
        "scope_accounting_roots": [str(host.ledger.root)],
        "scope_meter_state_ref": str(scope_ref),
        "scope_storage_remaining_bytes": state["limits"]["storage_bytes"]
        - state["used"]["storage_bytes"]
        - held
        - reserve,
        "scope_current_disk_limit_bytes": state["limits"]["current_disk_bytes"],
    }


def fixed_result(host, operation, template):
    """Consume the original native terminal envelope and optional frozen stage verdict."""
    from .cpu_command import is_cpu
    if is_cpu(template):
        from .cpu_command import result
        return result(host, operation, template)
    path = Path(template["result_ref"])
    permit = read(template["permit_ref"])
    cost_path = Path(template.get("cost_receipt_ref", path))
    value = _bound_record(path, host.config["scope_id"], operation["operation_id"])
    native = _bound_record(
        cost_path, host.config["scope_id"], operation["operation_id"]
    )
    schemas = {
        "libero-public-data-acquire": "rsi-libero-public-data-result-v1",
        "libero-public-data-materialize": "rsi-libero-public-data-result-v1",
        "libero-scripted-teacher": "rsi-libero-teacher-job-result-v1",
    }
    expected = schemas.get(permit["kind"], "rsi-native-training-job-result-v1")
    for target, receipt in ((path, value), (cost_path, native)):
        if (
            target.stat().st_uid != permit["owner_uid"]
            or receipt.get("owner_uid", permit["owner_uid"]) != permit["owner_uid"]
            or receipt.get("status") not in {"completed", "failed"}
        ):
            raise ValueError(
                "fixed native/stage result owner or terminal status differs"
            )
    if native.get("schema") != expected or value.get("schema") != template.get(
        "result_schema", expected
    ):
        raise ValueError("fixed native/stage result schema differs from registration")
    if path != cost_path and value.get("owner_uid") != permit["owner_uid"]:
        raise ValueError("registered stage result requires explicit owner identity")
    if permit["kind"] not in {
        "libero-public-data-acquire",
        "libero-public-data-materialize",
    } and (
        native.get("run_id") != permit["run_id"] or native.get("kind") != permit["kind"]
    ):
        raise ValueError("fixed native result run/family differs from permit")
    if permit["kind"] not in {
        "libero-public-data-acquire",
        "libero-public-data-materialize",
    } and not native.get("supervision_ref"):
        raise ValueError("native fixed command lacks its original supervision receipt")
    if native.get("supervision_ref"):
        supervision_ref = Path(native["supervision_ref"]).resolve()
        if (
            not supervision_ref.is_relative_to(host.ledger.root.resolve())
            or supervision_ref.stat().st_uid != permit["owner_uid"]
        ):
            raise ValueError("native supervision owner/domain differs from fixed job")
        supervision = read(supervision_ref)
        if (
            supervision.get("cleanup_confirmed") is not True
            or supervision.get("remaining_owned_pids") != []
        ):
            raise PermissionError("native fixed-command process cleanup is unresolved")
    if (
        native.get("source_manifest_ref", permit["source_manifest_ref"])
        != permit["source_manifest_ref"]
    ):
        raise ValueError("fixed native result source differs from registered permit")
    return value


def download_measurement(host, operation_id, template, permit, cost, report):
    """Keep measured returned payload distinct from an unresolved download tail."""
    scope_id, owner = host.config["scope_id"], permit["owner_uid"]
    ledger_ref = template.get("download_ledger_ref") or (cost or {}).get(
        "download_ledger_ref"
    )
    meter_ref = template.get("download_meter_ref") or (cost or {}).get(
        "download_meter_ref"
    )
    evidence, prefix, valid = {}, None, False

    def path_for(value):
        path = Path(value).resolve()
        if (
            not path.is_relative_to(host.ledger.root.resolve())
            or path.stat().st_uid != owner
        ):
            raise PermissionError(
                "download evidence owner/write root differs from current scope"
            )
        return path

    if ledger_ref:
        try:
            path = path_for(ledger_ref)
            prefix, valid = 0, True
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if (
                            not isinstance(row, dict)
                            or row.get("scope_id") != scope_id
                            or row.get("operation_id") != operation_id
                            or row.get("owner_uid") != owner
                            or not isinstance(row.get("url"), str)
                            or type(row.get("response_payload_bytes")) is not int
                            or row["response_payload_bytes"] < 0
                        ):
                            raise ValueError("download chunk identity/payload differs")
                        prefix += row["response_payload_bytes"]
                    except ValueError as error:
                        valid = False
                        evidence["ledger_error"] = str(error)
                        break
            evidence["download_ledger_ref"] = str(path)
        except (OSError, ValueError, PermissionError) as error:
            evidence["ledger_error"] = str(error)
    lower, exact = prefix, False
    if meter_ref:
        try:
            path = path_for(meter_ref)
            meter = _bound_record(path, scope_id, operation_id)
            if (
                meter.get("schema") != "rsi-public-data-download-meter-v1"
                or meter.get("owner_uid") != owner
            ):
                raise ValueError("download meter schema/owner differs")
            producer = meter.get("producer_returned_payload_bytes")
            if type(producer) is int and producer >= 0:
                lower = max(lower or 0, producer)
                evidence["producer_returned_payload_bytes_lower_bound"] = producer
            observed = meter.get("observed_download_payload_bytes_lower_bound")
            count = meter.get("download_response_payload_bytes")
            exact = (
                valid
                and prefix is not None
                and meter.get("download_meter_closed") is True
                and meter.get("status") == "closed"
                and meter.get("download_ledger_ref") is not None
                and Path(meter["download_ledger_ref"]).resolve()
                == Path(ledger_ref).resolve()
                and type(observed) is int
                and observed == prefix
                and type(producer) is int
                and producer == prefix
                and type(count) is int
                and count == prefix
                and report.get("cleanup_confirmed") is True
            )
            evidence.update(
                download_meter_ref=str(path),
                download_complete=meter.get("download_complete"),
                download_meter_closed=meter.get("download_meter_closed"),
                meter_exact_claim=count,
            )
        except (OSError, ValueError, PermissionError, TypeError) as error:
            evidence["meter_error"] = str(error)
    evidence.update(
        durable_payload_prefix_bytes=prefix, lower_bound_bytes=lower, exact=exact
    )
    return {"lower_bound": lower, "exact": exact, "evidence": evidence}


def fixed_end_report(host, operation, job):
    """Bind cleanup to the original committed scope and supervisor birth record."""
    from .midterm_host import process_matches

    template = host.config["operations"][operation["operation_id"]]
    submission, process, report = (
        read(job / "submission.json"),
        read(job / "process.json"),
        read(job / "supervisor.json"),
    )
    permit = read(template["permit_ref"])
    reservation = host.ledger.read()["reservations"][operation["operation_id"]]
    if (
        submission.get("operation") != operation
        or submission.get("owner_uid") != permit["owner_uid"]
        or process.get("owner_uid") != permit["owner_uid"]
        or not all(process.get(key) for key in ("pid", "start_ticks", "boot_id"))
        or reservation.get("scope_id") != host.config["scope_id"]
        or reservation.get("owner_uid") != permit["owner_uid"]
        or reservation.get("budget") != template["budget"]
        or not reservation.get("submitted_at")
        or reservation.get("phase_ref") != host.core_config["phase_ref"]
        or reservation.get("gpu_devices") != template.get("gpu_devices", [])
        or report.get("identity", {}).get("run_id") != operation["operation_id"]
        or report.get("status") != "completed"
        or report.get("cleanup_confirmed") is not True
        or report.get("remaining_process_ids") != []
        or process_matches(job / "process.json")
    ):
        raise PermissionError(
            "original fixed command scope/commit/birth/cleanup is unresolved"
        )
    return report


def reconcile_fixed_backend(backend, operation):
    """Import a proved original terminal return after the command process is gone."""
    from .midterm_host import immutable_json

    host = backend.host
    from .host_inputs import consumer_inputs
    _resolved, template = consumer_inputs(host, operation)
    path = backend._path(operation)
    if not path.exists():
        return None
    saved = read(path)
    if saved.get("operation") != operation:
        raise ValueError("original fixed backend intent differs from registration")
    if saved.get("status") == "completed":
        return None
    job = host.root / "jobs" / operation["operation_id"]
    fixed_end_report(host, operation, job)
    value = fixed_result(host, operation, template)
    immutable_json(job / "fixed-reconciliation/backend-before.json", saved)
    atomic_json(path, {"status": "completed", "operation": operation, "value": value})
    return backend._receipt(operation, value)


def recover_fixed_completion(host, job):
    """Reconcile an ended current command through Coordinator without executing again."""
    from .midterm_host import immutable_json, process_matches

    operation_id = job.name
    template = host.config["operations"].get(operation_id, {})
    if (
        not template.get("command")
        or not template.get("permit_ref")
        or process_matches(job / "process.json")
    ):
        return None
    operation = next(
        row
        for row in host.core_config["operations"]
        if row["operation_id"] == operation_id
    )
    try:
        fixed_end_report(host, operation, job)
        fixed_result(host, operation, template)
    except (OSError, ValueError, KeyError, PermissionError):
        return None
    coordinator = host.coordinator()
    immutable_json(
        job / "fixed-reconciliation/previous-core-state.json", coordinator.status()
    )
    state = coordinator.advance(host.inputs(), reconcile_only=True)
    if operation_id not in state["receipts"]:
        return None
    immutable_json(
        job / "fixed-reconciled.json",
        {
            "scope_id": host.config["scope_id"],
            "operation_id": operation_id,
            "original_completion_ref": str(job / "completion.json"),
            "supervisor_ref": str(job / "supervisor.json"),
            "receipt": state["receipts"][operation_id],
            "command_reexecuted": False,
        },
    )
    return {
        "status": "ready",
        "operation_id": operation_id,
        "core": state,
        "fixed_result_reconciled": True,
    }
