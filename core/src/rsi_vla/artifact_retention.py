# Copyright 2026 RSI-Agent-VLA contributors.

"""Pre-registered cleanup of unused derivatives after independent run closure.

Raw inputs, actions, results and audits remain intact. Removed derivatives are
unavailable for later reads; high-resolution depth is not independently saved
by the current producer and no reconstruction claim is made.
"""

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_json, identifier

IDENTITY = ("run_id", "episode_id", "bundle_id", "protocol_id")
WORLD = {"agentview_world_high.npz", "wrist_world_high.npz"}
SECONDARY_RGB = {"agentview_high.png", "wrist_high.png"}
POINTER = "artifact_retention.json"
SOURCES = ("execution_arms.py", "toolkit.py", "tools.py", "robot_spec.py")


def _read(path):
    return json.loads(Path(path).read_text())


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("retention timestamps require a timezone")
    return result


def _owned(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise PermissionError("retention evidence must be owned and not a symlink")
    return path.resolve(strict=True)


def _authority(policy_ref, run_id, *, active):
    policy_ref = _owned(policy_ref)
    policy = _read(policy_ref)
    scope_ref = _owned(policy["scope_authorization_ref"])
    scope = _read(scope_ref)
    root = Path(scope["write_root"]).resolve(strict=True)
    registered, expiry = _time(policy["registered_at"]), _time(policy["expires_at"])
    if (
        policy.get("schema") != "rsi-artifact-retention-policy-v1"
        or scope.get("schema") != "rsi-midterm-scope-v1"
        or scope.get("authorized") is not True
        or policy.get("scope_id") != scope.get("scope_id")
        or scope.get("owner_uid") != os.getuid()
        or policy.get("owner_uid") != os.getuid()
        or not _time(scope["accepted_at"])
        <= registered
        < expiry
        <= _time(scope["expires_at"])
        or not policy_ref.is_relative_to(root)
    ):
        raise PermissionError("retention scope, owner or registration mismatch")
    identifier(policy["policy_id"])
    if active and (
        scope.get("status") != "active" or datetime.now(timezone.utc) >= expiry
    ):
        raise PermissionError("retention execution scope is paused, closed or expired")
    binding = policy.get("runs", {}).get(identifier(run_id))
    if not binding:
        raise PermissionError("run is not registered for this retention policy")
    output = Path(binding["output_dir"]).resolve(strict=True)
    runtime_root = Path(policy["runtime_root"]).resolve(strict=True)
    receipt_root = Path(policy["receipt_root"]).resolve()
    audit = Path(binding["audit_ref"]).resolve(strict=True)
    if (
        not runtime_root.is_relative_to(root)
        or not output.is_relative_to(runtime_root)
        or not receipt_root.is_relative_to(root)
        or receipt_root.is_relative_to(runtime_root)
        or not audit.is_relative_to(root)
        or audit.is_relative_to(runtime_root)
        or policy_ref.is_relative_to(runtime_root)
    ):
        raise PermissionError(
            "retention policy, audit and receipts require independent task domains"
        )
    allowed = policy.get("allowed_artifacts", sorted(WORLD))
    if (
        not isinstance(allowed, list)
        or len(set(allowed)) != len(allowed)
        or not WORLD <= set(allowed) <= WORLD | SECONDARY_RGB
    ):
        raise ValueError(
            "retention allows only registered high-world and optional high-RGB derivatives"
        )
    return policy_ref, policy, binding, output, receipt_root, audit


def _identity(value, expected):
    observed = value.get("identity", value)
    return all(observed.get(key) == expected[key] for key in IDENTITY)


def _evidence(policy, binding, output, audit_ref, run_id):
    from robots.libero.execution_arms import validate_execution_arm

    context = _read(output / "runtime_context.json")
    expected = {
        "run_id": run_id,
        **{key: binding[key] for key in IDENTITY if key != "run_id"},
    }
    if not _identity(context, expected):
        raise PermissionError("retention runtime identity mismatch")
    permit = _read(context["authorization_ref"])
    if (
        permit.get("scope") != policy["scope_id"]
        or permit.get("authorized") is not True
        or Path(permit["approval_reference"]).resolve()
        != Path(policy["scope_authorization_ref"]).resolve()
        or any(
            permit.get(key) != expected[key]
            for key in ("run_id", "bundle_id", "protocol_id")
        )
    ):
        raise PermissionError(
            "retention run permit does not match the registered scope"
        )
    events = [
        _read_line(line)
        for line in (output / "action_events.jsonl").read_text().splitlines()
    ]
    if (
        not events
        or any(
            event.get("sequence") != index or not _identity(event, expected)
            for index, event in enumerate(events)
        )
        or events[0].get("kind") != "run_start"
        or datetime.fromtimestamp(events[0]["utc_ns"] / 1e9, timezone.utc)
        <= _time(policy["registered_at"])
    ):
        raise PermissionError(
            "retention policy must precede this exact run's event history"
        )
    spec = validate_execution_arm(context["execution_arm"])
    calls = [event["tool"] for event in events if event["kind"] == "tool_call"]
    starts = [event for event in events if event["kind"] == "episode_execution_start"]
    ends = [event for event in events if event["kind"] == "episode_execution_end"]
    if (
        spec["arm"] not in {"D", "S", "R"}
        or not calls
        or set(calls) != {"vla_execute"}
        or context.get("relation_control") is not None
        or context.get("planner", {}).get("backend") != "flash"
        or len(starts) != 1
        or len(ends) != 1
        or starts[0].get("execution_arm") != spec
        or ends[0].get("spec") != spec
        or list(output.glob("codex_*.stream.jsonl"))
    ):
        raise PermissionError(
            "only the reviewed frozen VLA path proves derivatives unused"
        )
    for name in SOURCES:
        relative = Path("RPent/robots/libero") / name
        if (output / "code_snapshot" / relative).read_bytes() != (
            Path(policy["reviewed_source_root"]) / relative
        ).read_bytes():
            raise PermissionError(
                "retention reviewed frozen consumer source changed: " + name
            )
    supervisor, service = (
        _read(output / "supervisor.json"),
        _read(output / "environment_service/runtime_costs.json"),
    )
    result, audit = _read(output / "independent_result.json"), _read(_owned(audit_ref))
    if (
        not all(
            _identity(value, expected) for value in (supervisor, service, result, audit)
        )
        or supervisor.get("status") != "completed"
        or supervisor.get("exit_code") != 0
        or supervisor.get("cleanup_confirmed") is not True
        or supervisor.get("remaining_process_ids")
        or service.get("service_closed") is not True
        or service.get("pending_action") is not False
        or service.get("environment_steps_unknown") != 0
        or result.get("attempt_closed") is not True
        or type(result.get("benchmark_success")) is not bool
        or result.get("validity") != "valid"
        or result.get("error")
        or audit.get("passed") is not True
        or audit.get("outcome") not in {"success", "failure"}
        or not audit.get("checks")
        or not all(value is True for value in audit["checks"].values())
    ):
        raise PermissionError(
            "retention requires complete independent audit and owned process closure"
        )
    states = _read(output / "states.json")["steps"]
    if (
        not states
        or any(
            type(row.get("step_idx")) is not int or row["step_idx"] < 0
            for row in states
        )
        or len({row["step_idx"] for row in states}) != len(states)
    ):
        raise PermissionError("retention requires unique complete state identities")
    protected = {max(row["step_idx"] for row in states)}
    marker = output / "native_completion.json"
    if marker.exists():
        native = _read(marker)
        if not _identity(native, expected) or type(native.get("step")) is not int:
            raise PermissionError("retention native completion identity mismatch")
        protected.add(native["step"])
    if not protected <= {row["step_idx"] for row in states}:
        raise PermissionError("protected completion state is missing")
    for state in states:
        if state["step_idx"] in protected:
            for name in state.get("artifacts", []):
                path = output / name / f"{state['step_idx']:02d}{Path(name).suffix}"
                if (
                    Path(name).name != name
                    or path.is_symlink()
                    or not path.is_file()
                    or not path.stat().st_size
                ):
                    raise PermissionError(
                        "all protected-state artifacts must remain present"
                    )
    _preserved_inputs(output, states, expected, events)
    if type(audit.get("costs", {}).get("artifact_bytes")) is not int:
        raise PermissionError(
            "retention requires the independently booked artifact cost"
        )
    return expected, states, protected, audit


def _read_line(line):
    return json.loads(line)


def _preserved_inputs(output, states, identity, events):
    """Require the actual two-camera policy inputs and aligned native actions."""
    import numpy as np

    from robots.libero.flywheel import LIBERO_SPEC
    from rpent.flywheel.episode import validate_episode

    names = {
        "agentview_policy.png",
        "agentview.png",
        "wrist.png",
        "agentview_depth.npz",
        "wrist_depth.npz",
        "agentview_metadata.json",
        "wrist_metadata.json",
    }
    for state in states:
        if not names <= set(state.get("artifacts", [])):
            raise PermissionError(
                "retention requires preserved raw observation artifacts"
            )
        for name in names:
            path = output / name / f"{state['step_idx']:02d}{Path(name).suffix}"
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise PermissionError(
                    "retention cannot substitute for missing raw observations"
                )
    manifest = _read(output / "dataset_manifest.json")
    if (
        not _identity(manifest, identity)
        or manifest.get("alignment_complete") is not True
        or len(manifest.get("episodes", [])) != 1
    ):
        raise PermissionError(
            "retention requires a closed aligned policy/action collection"
        )
    episode = Path(manifest["episodes"][0]["path"]).resolve(strict=True)
    if not episode.is_relative_to(output / "data"):
        raise PermissionError("retained collection must belong to this exact run")
    metadata = validate_episode(episode, spec=LIBERO_SPEC)
    count = sum(event["kind"] == "inference_start" for event in events)
    if type(metadata.get("input_count")) is not int or metadata["input_count"] != count:
        raise PermissionError("retention requires every original policy input")
    for index in range(count):
        with np.load(episode / f"input-{index:05d}.npz", allow_pickle=False) as data:
            for name, shape in (
                ("main_images", (256, 256, 3)),
                ("wrist_images", (256, 256, 3)),
                ("states", (8,)),
            ):
                if data[name].shape != shape or not np.isfinite(data[name]).all():
                    raise PermissionError(
                        "retained policy input shape or values are invalid"
                    )


def plan_retention(policy_ref, run_id):
    """Prepare a bounded file plan without writing or deleting any file."""
    policy_ref, policy, binding, output, receipt_root, audit_ref = _authority(
        policy_ref, run_id, active=True
    )
    identity, states, protected, audit = _evidence(
        policy, binding, output, audit_ref, run_id
    )
    files = []
    for state in states:
        step = state["step_idx"]
        if step in protected:
            continue
        for name in policy.get("allowed_artifacts", sorted(WORLD)):
            if name not in state.get("artifacts", []):
                continue
            path = output / name / f"{step:02d}{Path(name).suffix}"
            if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
                raise PermissionError(
                    "planned retention derivative is missing or not a regular file"
                )
            stat = path.stat()
            files.append(
                {
                    "path": str(path),
                    "artifact": name,
                    "step": step,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    from .process_accounting import accounted_storage_bytes, snapshot

    return {
        "schema": "rsi-artifact-retention-receipt-v1",
        "status": "prepared",
        "policy_ref": str(policy_ref),
        "policy_id": policy["policy_id"],
        "scope_id": policy["scope_id"],
        "identity": identity,
        "output_dir": str(output),
        "audit_ref": str(audit_ref),
        "receipt_ref": str(receipt_root / (identifier(run_id) + ".json")),
        "protected_steps": sorted(protected),
        "files": files,
        "removed_paths": [],
        "cumulative_bytes_floor": max(
            accounted_storage_bytes(output), audit["costs"]["artifact_bytes"]
        ),
        "retained_output_bytes_at_receipt": sum(snapshot([output]).values()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "availability": "policy-unavailable after deletion; no reconstruction guarantee; no independently saved 1024 depth",
        "deletion_refunds_cumulative_cost": False,
    }


def certified_retention_floor(output):
    """Read only independently owned, pre-registered retention cost evidence."""
    output = Path(output).resolve()
    pointer = output / POINTER
    if not pointer.exists():
        return None
    location = _read(pointer)
    policy_ref, policy, binding, registered, receipt_root, _ = _authority(
        location["policy_ref"], location["run_id"], active=False
    )
    receipt_path = _owned(location["receipt_ref"])
    receipt = _read(receipt_path)
    expected = {
        "run_id": location["run_id"],
        **{key: binding[key] for key in IDENTITY if key != "run_id"},
    }
    if (
        registered != output
        or receipt_path != receipt_root / (identifier(location["run_id"]) + ".json")
        or location.get("schema") != "rsi-artifact-retention-pointer-v1"
        or receipt.get("schema") != "rsi-artifact-retention-receipt-v1"
        or receipt.get("policy_ref") != str(policy_ref)
        or receipt.get("policy_id") != policy["policy_id"]
        or receipt.get("scope_id") != policy["scope_id"]
        or not _identity(receipt, expected)
        or receipt.get("output_dir") != str(output)
        or receipt.get("receipt_ref") != str(receipt_path)
        or receipt.get("audit_ref") != str(Path(binding["audit_ref"]).resolve())
        or receipt.get("status") not in {"prepared", "completed"}
        or receipt.get("deletion_refunds_cumulative_cost") is not False
        or any(
            type(receipt.get(key)) is not int or receipt[key] < 0
            for key in ("cumulative_bytes_floor", "retained_output_bytes_at_receipt")
        )
    ):
        raise PermissionError(
            "retention cumulative floor is not certified by the registered independent receipt"
        )
    return receipt["cumulative_bytes_floor"], receipt[
        "retained_output_bytes_at_receipt"
    ]


def policy_unavailable_media(output):
    """Describe certified removals without claiming missing files are present."""
    output = Path(output).resolve()
    if certified_retention_floor(output) is None:
        return None
    pointer = _read(output / POINTER)
    receipt = _read(pointer["receipt_ref"])
    return {
        "status": "policy-unavailable",
        "policy_ref": receipt["policy_ref"],
        "receipt_ref": pointer["receipt_ref"],
        "paths": [path for path in receipt["removed_paths"] if not Path(path).exists()],
        "reconstruction": "not_guaranteed; independently saved 1024 depth unavailable",
    }


def apply_retention(policy_ref, run_id):
    """Apply or reconcile one registered plan after audit; preserve all raw facts."""
    policy_ref, policy, binding, output, receipt_root, audit_ref = _authority(
        policy_ref, run_id, active=True
    )
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / (identifier(run_id) + ".json")
    with receipt_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            receipt = _read(receipt_path)
            if not (output / POINTER).exists():
                atomic_json(
                    output / POINTER,
                    {
                        "schema": "rsi-artifact-retention-pointer-v1",
                        "policy_ref": str(policy_ref),
                        "run_id": run_id,
                        "receipt_ref": str(receipt_path),
                    },
                )
            certified_retention_floor(output)
            if receipt["status"] == "completed":
                return receipt
            _evidence(policy, binding, output, audit_ref, run_id)
        else:
            receipt = plan_retention(policy_ref, run_id)
            atomic_json(receipt_path, receipt)
            atomic_json(
                output / POINTER,
                {
                    "schema": "rsi-artifact-retention-pointer-v1",
                    "policy_ref": str(policy_ref),
                    "run_id": run_id,
                    "receipt_ref": str(receipt_path),
                },
            )
        from .process_accounting import snapshot

        if not receipt["removed_paths"]:
            current = sum(snapshot([output]).values())
            receipt["cumulative_bytes_floor"] += max(
                0, current - receipt["retained_output_bytes_at_receipt"]
            )
            receipt["retained_output_bytes_at_receipt"] = current
            atomic_json(receipt_path, receipt)
        for item in receipt["files"]:
            path = Path(item["path"])
            if path.exists():
                stat = path.stat()
                if (
                    path.is_symlink()
                    or path.parent.is_symlink()
                    or path.parent.parent != output
                    or path
                    != output
                    / item["artifact"]
                    / f"{item['step']:02d}{Path(item['artifact']).suffix}"
                    or item["artifact"]
                    not in policy.get("allowed_artifacts", sorted(WORLD))
                    or item["step"] in receipt["protected_steps"]
                    or (stat.st_size, stat.st_mtime_ns)
                    != (item["bytes"], item["mtime_ns"])
                ):
                    raise PermissionError(
                        "retention file differs from its authorized prepared plan"
                    )
                path.unlink()
            if item["path"] not in receipt["removed_paths"]:
                receipt["removed_paths"].append(item["path"])
                atomic_json(receipt_path, receipt)
        receipt["status"] = "completed"
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["retained_output_bytes_at_receipt"] = sum(snapshot([output]).values())
        atomic_json(receipt_path, receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = (
        apply_retention(args.policy, args.run)
        if args.apply
        else plan_retention(args.policy, args.run)
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
