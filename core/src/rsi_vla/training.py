"""Bounded supervision for an explicitly authorized native training process."""
import fcntl
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DYNAMIC_LOCK = "/tmp/rsi-agent-vla-dynamic.lock"
JOB_MARKER = "RSI_TRAINING_JOB_ROOT"


@dataclass(frozen=True)
class TrainingBudget:
    wall_seconds: float
    gpu_device_seconds: float
    reserved_devices: int
    artifact_bytes: int
    max_steps: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite budget required: {name}")
        for name in ("reserved_devices", "artifact_bytes", "max_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"integer required: {name}")

    @property
    def deadline_seconds(self):
        return min(self.wall_seconds, self.gpu_device_seconds / self.reserved_devices)


def require_training_permit(permit, *, run_id, budget, dataset_root, model_path,
                            kind="pi05-training-pilot"):
    """A simulation permit never authorizes training."""
    if kind not in {"pi05-training-pilot", "pi05-training-learning", "pi05-load-check"} or permit.get("kind") != kind or permit.get("authorized") is not True:
        raise PermissionError(f"explicit {kind} authorization required")
    if not run_id or permit.get("run_id") != run_id:
        raise PermissionError("training authorization run_id mismatch")
    for key, expected in (("dataset_root", dataset_root), ("model_path", model_path)):
        if Path(permit.get(key, "")).resolve() != Path(expected).resolve():
            raise PermissionError(f"training authorization {key} mismatch")
    allowed = TrainingBudget(**permit["budget"])
    if any(value > asdict(allowed)[key] for key, value in asdict(budget).items()):
        raise PermissionError("requested training budget exceeds authorization")


def require_scoped_training_permit(permit, *, run_id, budget, dataset_root,
                                   model_path, artifact_root, kind, cuda_device,
                                   require_submission=False):
    """Check live task authority and host reservation at the process boundary."""
    require_training_permit(permit, run_id=run_id, budget=budget,
                            dataset_root=dataset_root, model_path=model_path, kind=kind)
    if kind != "pi05-load-check" and budget.artifact_bytes > 48 * 1024**3:
        raise PermissionError("task single-training artifact cap is 48 GiB")
    if type(permit.get("cuda_device")) is not int or permit["cuda_device"] != cuda_device:
        raise PermissionError("training GPU differs from the authorized physical device")
    scope_path = Path(permit["scope_ref"]).resolve()
    scope = json.loads(scope_path.read_text())
    if (scope.get("schema") != "rsi-midterm-scope-v1" or scope.get("status") != "active"
            or scope.get("authorized") is not True or scope.get("dynamic_authorized") is not True
            or permit.get("scope_id") != scope.get("scope_id")):
        raise PermissionError("active task-wide training scope required")
    if scope.get("owner_uid") != os.geteuid() or permit.get("owner_uid") != os.geteuid():
        raise PermissionError("training task owner mismatch")
    write_root = Path(scope["write_root"]).resolve()
    if (Path(permit["write_root"]).resolve() != write_root
            or not Path(artifact_root).resolve().is_relative_to(write_root)
            or not scope_path.is_relative_to(write_root)):
        raise PermissionError("training artifact/scope lies outside the task write domain")
    now = datetime.now(timezone.utc)
    def utc(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise PermissionError("training authority requires timezone-aware UTC expiry")
        return parsed
    expiry = min(utc(scope["expires_at"]), utc(permit["expires_at"]))
    if now < utc(scope["accepted_at"]) or (expiry - now).total_seconds() < budget.deadline_seconds:
        raise PermissionError("training complete lifecycle would exceed task/permit expiry")
    p1 = json.loads(Path(scope["p1_acceptance_ref"]).read_text())
    if (p1.get("schema") != "rsi-midterm-p1-acceptance-v1"
            or p1.get("scope_id") != scope["scope_id"] or p1.get("status") != "passed"):
        raise PermissionError("current-task unattended P1 acceptance required before training/loading")
    if kind == "pi05-training-pilot" and permit.get("pilot_disposition") != "resource_only_no_capability_selection":
        raise PermissionError("pilot must predeclare no capability-based checkpoint selection")
    from .midterm_host import MidtermLedger
    needed = {"gpu_seconds": budget.gpu_device_seconds, "storage_bytes": budget.artifact_bytes,
              "current_disk_bytes": budget.artifact_bytes}
    if kind == "pi05-load-check":
        needed["load_checks"] = 1
    else:
        needed["training_starts"] = 1
        if kind == "pi05-training-pilot":
            needed["training_pilots"] = 1
    reservation = MidtermLedger(scope_path, permit["accounting_ref"]).validate_reservation(
        permit["operation_id"], budget=needed, phase_ref=permit["phase_ref"],
        gpu_devices=[cuda_device])
    if require_submission and not reservation.get("submitted_at"):
        raise PermissionError("host must persist the submitted operation before the GPU consumer starts")
    return {"scope_id": scope["scope_id"], "operation_id": permit["operation_id"],
            "reservation": reservation, "expires_at": expiry.isoformat()}


@contextmanager
def dynamic_lock(path=DYNAMIC_LOCK):
    with open(path, "a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another dynamic Runtime/training job holds the shared lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def artifact_size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


def prepare_ray_temp_path(artifact_root):
    """Keep Ray sockets short while storing all Ray files in this artifact."""
    target = Path(artifact_root).resolve() / "ray"
    target.mkdir(parents=True, exist_ok=True)
    # Ray appends /ray to RAY_TMPDIR. Preserve this alias as a literal path;
    # resolving it would reintroduce the AF_UNIX path-length failure.
    short_parent = Path(tempfile.mkdtemp(prefix="rsit-", dir="/tmp"))
    alias = short_parent / "ray"
    alias.symlink_to(target, target_is_directory=True)
    paths = {"ray_tmpdir": str(short_parent), "ray_temp_alias": str(alias),
             "ray_temp_target": str(target)}
    (Path(artifact_root) / "ray-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    return paths


def _process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        owner = path.stat().st_uid
        stat = (path / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rfind(")") + 2:].split()
    return {"pid": pid, "uid": owner, "state": fields[0],
            "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def _refresh_owned(owned, marker):
    """Keep PID birth identity; adoption/setsid do not remove job ownership."""
    processes = {}
    uid = os.geteuid()
    tag = (JOB_MARKER + "=" + marker).encode()
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            current = _process_identity(int(path.name))
            if current is None or current["uid"] != uid:
                continue
            processes[current["pid"]] = current
            if tag in (path / "environ").read_bytes().split(b"\0"):
                owned[(current["pid"], current["start_ticks"])] = current
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    # Include observed descendants even if a child changes its environment.
    changed = True
    while changed:
        changed = False
        live_parents = {pid for pid, birth in owned if pid in processes
                        and processes[pid]["start_ticks"] == birth}
        for current in processes.values():
            key = (current["pid"], current["start_ticks"])
            if key not in owned and current["ppid"] in live_parents:
                owned[key] = current
                changed = True
    return _live_owned(owned)


def _live_owned(owned):
    """Check every observed PID's exact birth identity without rescanning the host."""
    uid = os.geteuid()
    live = []
    for pid, birth in owned:
        try:
            current = _process_identity(pid)
        except PermissionError:
            # A previously owned PID that cannot be inspected is unresolved.
            live.append({"pid": pid, "start_ticks": birth, "state": "unreadable"})
            continue
        if (current is not None and current["uid"] == uid
                and current["start_ticks"] == birth and current["state"] not in {"Z", "X"}):
            live.append(current)
    return live


def _signal_owned(identity, sig):
    """Signal only a still-matching same-UID PID, using pidfd when available."""
    pid, birth = identity["pid"], identity["start_ticks"]
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(pid)
        current = _process_identity(pid)
        if (current is None or current["start_ticks"] != birth
                or current["uid"] != os.geteuid() or current["state"] in {"Z", "X"}):
            return
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(pid, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _cleanup_owned(process, owned, marker, deadline, reserve, remaining=None):
    started = time.monotonic()
    term_until = min(deadline, started + min(5.0, reserve / 3))
    signaled = set()
    errors = []
    if not remaining:
        remaining = _refresh_owned(owned, marker)
    while True:
        if not remaining:
            break
        now = time.monotonic()
        sig = signal.SIGTERM if now < term_until else signal.SIGKILL
        for identity in remaining:
            key = (identity["pid"], identity["start_ticks"], int(sig))
            if key not in signaled:
                try:
                    _signal_owned(identity, sig)
                    signaled.add(key)
                except PermissionError as exc:
                    errors.append({"pid": identity["pid"], "error": str(exc)})
        process.poll()  # Reap our direct child before checking /proc again.
        if now >= deadline:
            break
        time.sleep(min(0.02, max(0.001, deadline - time.monotonic())))
        remaining = _live_owned(owned)
        if not remaining:
            # Discover late or detached descendants after observed members exit.
            # While they are alive, check their exact identities directly so a
            # host-wide scan does not delay TERM/KILL past the cleanup deadline.
            remaining = _refresh_owned(owned, marker)
    process.poll()
    return {"cleanup_confirmed": not remaining, "remaining_owned_pids": remaining,
            "cleanup_seconds": time.monotonic() - started, "cleanup_errors": errors,
            "owned_pid_identities": [{"pid": pid, "start_ticks": birth} for pid, birth in owned],
            "ownership_scope": "same UID, exact job marker, and observed descendants; PID birth checked",
            "zombie_semantics": "Z/X are exited, non-executing processes; GPU utilization is not measured"}


def supervise_process(command, *, budget, artifact_root, env=None, poll_seconds=0.1,
                      cleanup_seconds=None):
    """Supervise this job's marked/observed PIDs, including detached sessions.

    Device seconds are a reservation envelope (elapsed x reserved devices),
    not GPU utilization. Storage is sampled; this is not a filesystem quota.
    """
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    if poll_seconds <= 0 or poll_seconds > 1:
        raise ValueError("poll_seconds must be in (0, 1]")
    if artifact_size(root) >= budget.artifact_bytes:
        raise ValueError("artifact budget already exhausted")
    marker = str(root.resolve())
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("Linux /proc ownership tracking is required")
    discovery_started = time.monotonic()
    if _refresh_owned({}, marker):
        raise RuntimeError("a previous process still carries this job's ownership marker")
    discovery_reserve = time.monotonic() - discovery_started
    reserve = min(30.0, budget.deadline_seconds * 0.2) if cleanup_seconds is None else cleanup_seconds
    if not 0 < reserve < budget.deadline_seconds:
        raise ValueError("cleanup reserve must be positive and smaller than the total budget")
    ray_paths = prepare_ray_temp_path(root)
    child_env = dict(os.environ if env is None else env)
    child_env.update({JOB_MARKER: marker, "RAY_ADDRESS": "local", "RAY_TMPDIR": ray_paths["ray_tmpdir"]})
    started = time.monotonic()
    deadline = started + budget.deadline_seconds
    # Include the observed host-discovery cost in the existing hard envelope.
    # TERM/KILL grace alone does not cover the final complete member scan.
    work_deadline = deadline - reserve - discovery_reserve
    stop_reason = None
    owned = {}
    remaining = None
    raised = None
    with (root / "trainer.log").open("xb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=child_env, start_new_session=True)
        initial = _process_identity(process.pid)
        if initial is not None:
            owned[(process.pid, initial["start_ticks"])] = initial
        try:
            while True:
                remaining = _refresh_owned(owned, marker)
                if time.monotonic() >= work_deadline:
                    stop_reason = "wall_or_device_reservation_limit"
                elif artifact_size(root) >= budget.artifact_bytes:
                    stop_reason = "artifact_limit"
                if stop_reason or process.poll() is not None:
                    break
                time.sleep(min(poll_seconds, max(0.001, work_deadline - time.monotonic())))
        except BaseException as exc:
            raised = exc
            stop_reason = "supervisor_exception"
        finally:
            cleanup = _cleanup_owned(process, owned, marker, deadline, reserve, remaining)
    final_bytes = artifact_size(root)
    elapsed = time.monotonic() - started
    report = {"returncode": process.returncode, "stop_reason": stop_reason,
              "wall_seconds": elapsed,
              "gpu_device_seconds_envelope": elapsed * budget.reserved_devices,
              "reserved_devices": budget.reserved_devices,
              "artifact_bytes_before_report": final_bytes,
              "cleanup_reserve_seconds": reserve + discovery_reserve,
              "cleanup_discovery_reserve_seconds": discovery_reserve,
              "resource_limits_satisfied": elapsed <= budget.deadline_seconds and final_bytes <= budget.artifact_bytes,
              **ray_paths,
              **cleanup,
              "budget": asdict(budget), "command": list(command),
              "evidence_level": "process_supervision"}
    (root / "supervision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if raised is not None:
        raise raised
    return report


def training_handoff(
    result_ref: str | Path, *, scope_id: str | None = None, require_load: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a successful native receipt and revalidate its concrete model outputs."""
    from .adapters.pi05_training import inspect_checkpoint

    if not result_ref:
        raise PermissionError("actual completed native receipt reference is required")
    path = Path(result_ref).resolve()
    result = json.loads(path.read_text())
    if (
        result.get("schema") != "rsi-native-training-job-result-v1"
        or result.get("status") != "completed"
        or not result.get("handoff_ref")
        or (scope_id is not None and result.get("scope_id") != scope_id)
        or (require_load and result.get("kind") != "pi05-load-check")
    ):
        raise PermissionError("completed native training/load receipt is required")
    report = json.loads(Path(result["supervision_ref"]).read_text())
    if (
        report.get("returncode") != 0
        or report.get("stop_reason") is not None
        or report.get("cleanup_confirmed") is not True
        or report.get("resource_limits_satisfied") is not True
    ):
        raise PermissionError("native receipt process closure is invalid")
    handoff = json.loads(Path(result["handoff_ref"]).read_text())
    if (
        handoff.get("scope_id") != result["scope_id"]
        or handoff.get("run_id") != result["run_id"]
    ):
        raise PermissionError("native handoff scope/run mismatch")
    descriptor = handoff["model_descriptor"]
    if (
        result.get("theta_ref")
        != str(Path(result["handoff_ref"]).resolve()) + "#/model_descriptor"
        or result.get("processor_ref") != descriptor["norm_stats_path"]
    ):
        raise PermissionError(
            "native theta/processor export does not match its handoff"
        )
    layout = inspect_checkpoint(descriptor["model_path"], descriptor["norm_stats_path"])
    weights = Path(layout["selected_weights"])
    identity = {
        "path": str(weights.resolve()),
        "bytes": weights.stat().st_size,
        "mtime_ns": weights.stat().st_mtime_ns,
    }
    if (
        handoff.get("weights_identity") != identity
        or handoff.get("processor_snapshot")
        != json.loads(Path(layout["norm_stats"]).read_text())
        or result.get("model_path") != descriptor["model_path"]
        or result.get("norm_stats_path") != descriptor["norm_stats_path"]
    ):
        raise PermissionError("native saved model or processor identity changed")
    if require_load:
        from .adapters.pi05_training import verify_native_load_receipt

        if descriptor.get("actual_load_verified") is not True:
            raise PermissionError("actual RPent model load has not been verified")
        verify_native_load_receipt(
            json.loads(Path(result["load_check_ref"]).read_text()), handoff
        )
    return result, handoff


def finalize_training_job(
    *,
    prepared_root: str | Path,
    artifact_root: str | Path,
    permit: dict[str, Any],
    supervision: dict[str, Any],
    python_executable: str,
    training_result_ref: str | Path | None = None,
) -> dict[str, Any]:
    """Close a native job even when its save/load output is missing or incomplete."""
    from .adapters.pi05_training import (
        resolve_native_training_output,
        verify_native_load_receipt,
    )
    from .storage import atomic_json

    root, artifact = Path(prepared_root).resolve(), Path(artifact_root).resolve()
    result = {
        "schema": "rsi-native-training-job-result-v1",
        "scope_id": permit["scope_id"],
        "operation_id": permit["operation_id"],
        "run_id": permit["run_id"],
        "kind": permit["kind"],
        "status": "failed",
        "handoff_ref": None,
        "supervision_ref": str(artifact / "supervision.json"),
        "native_start_ref": str(artifact / "native-start.json"),
        "capability_evaluation": "not_run",
        "errors": [],
        "result_ref": str(artifact / "job-result.json"),
        "artifact_ref": str(artifact / "job-result.json"),
    }
    try:
        persisted = json.loads((artifact / "supervision.json").read_text())
        if persisted != supervision:
            raise PermissionError(
                "native supervision differs from its persisted original"
            )
        recipe = json.loads((root / "recipe.json").read_text())
        if recipe["runner"]["logger"]["experiment_name"] != permit["run_id"]:
            raise PermissionError("native parser recipe/permit run mismatch")
        expected_root = root / (
            permit["run_id"] + "-load-check"
            if permit["kind"] == "pi05-load-check"
            else permit["run_id"]
        )
        if artifact != expected_root:
            raise PermissionError(
                "native parser artifact path differs from its registered run"
            )
        start = json.loads((artifact / "native-start.json").read_text())
        entry = (
            "native_verify_load"
            if permit["kind"] == "pi05-load-check"
            else "native_train"
        )
        if (
            start.get("schema") != "rsi-native-training-start-v1"
            or start.get("entrypoint") != entry
            or any(
                start.get(key) != permit[key]
                for key in ("scope_id", "operation_id", "run_id", "kind")
            )
        ):
            raise PermissionError(
                "native parser start identity differs from its exact operation"
            )
        if (
            supervision.get("returncode") != 0
            or supervision.get("stop_reason") is not None
            or supervision.get("cleanup_confirmed") is not True
            or supervision.get("resource_limits_satisfied") is not True
        ):
            raise ValueError(
                "native job failed or exceeded its resource/cleanup limits"
            )
        if permit["kind"] == "pi05-load-check":
            if not training_result_ref:
                raise PermissionError(
                    "load check must consume the actual completed training result"
                )
            trained, handoff = training_handoff(
                training_result_ref, scope_id=permit["scope_id"]
            )
            if trained["run_id"] != permit["run_id"]:
                raise PermissionError(
                    "load check belongs to another native training run"
                )
            verified = json.loads((artifact / "load-check.json").read_text())
            verify_native_load_receipt(verified, handoff)
            handoff["actual_model_load"] = "passed"
            handoff["model_descriptor"].update(
                actual_load_verified=True,
                actual_load_evidence_ref=str(artifact / "load-check.json"),
            )
            handoff_ref = artifact / "deployment-handoff-verified.json"
            result.update(
                load_check_ref=str(artifact / "load-check.json"),
                training_result_ref=str(Path(training_result_ref).resolve()),
            )
        else:
            handoff = resolve_native_training_output(
                prepared_root=root,
                artifact_root=artifact,
                permit=permit,
                supervision=supervision,
                python_executable=python_executable,
            )
            handoff_ref = artifact / "deployment-handoff.json"
        atomic_json(handoff_ref, handoff)
        descriptor = handoff["model_descriptor"]
        result.update(
            status="completed",
            handoff_ref=str(handoff_ref),
            model_path=descriptor["model_path"],
            norm_stats_path=descriptor["norm_stats_path"],
            theta_ref=str(handoff_ref) + "#/model_descriptor",
            processor_ref=descriptor["norm_stats_path"],
            recipe_ref=str(root / "recipe.json"),
        )
    except (OSError, ValueError, KeyError, TypeError, PermissionError) as exc:
        result["errors"] = [str(exc)]
    atomic_json(artifact / "job-result.json", result)
    return result


def finalize_native_model_meter(
    *,
    artifact_root: str | Path,
    permit: dict[str, Any],
    supervision: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Close project-request accounting for the two fixed local native entries.

    Native SFT and the local RPent facade use no Planner/Research client or
    project LLM transport. Their start marker, supervised command and completed
    output must establish that exact boundary before its zero count is usable.
    """
    from .storage import atomic_json

    root = Path(artifact_root).resolve()
    meter = {
        "schema": "rsi-native-training-model-meter-v1",
        "scope_id": permit["scope_id"],
        "operation_id": permit["operation_id"],
        "owner_uid": os.geteuid(),
        "status": "unknown",
        "project_model_requests": None,
        "model_clients_created": None,
        "model_transports_created": None,
        "components": [],
        "native_start_ref": str(root / "native-start.json"),
        "supervision_ref": str(root / "supervision.json"),
        "result_ref": str(root / "job-result.json"),
        "count_scope": "project Planner/Research LLM requests and transports; local native policy and Ray collectives are separate components",
        "errors": [],
    }
    try:
        start = json.loads((root / "native-start.json").read_text())
        saved_supervision = json.loads((root / "supervision.json").read_text())
        saved_result = json.loads((root / "job-result.json").read_text())
        entrypoint = start.get("entrypoint")
        component = {
            "native_train": "native_sft",
            "native_verify_load": "native_loader_one_prediction",
        }.get(entrypoint)
        family = (
            "pi05-load-check" if entrypoint == "native_verify_load" else permit["kind"]
        )
        if (
            component is None
            or family
            not in {"pi05-training-pilot", "pi05-training-learning", "pi05-load-check"}
            or permit["kind"] != family
            or (entrypoint == "native_train" and family == "pi05-load-check")
            or start.get("schema") != "rsi-native-training-start-v1"
            or start.get("boundary")
            != "native_adapter_process_entry_before_optional_imports"
            or start.get("owner_uid") != os.geteuid()
            or permit.get("owner_uid") != os.geteuid()
            or any(
                start.get(key) != permit[key]
                for key in ("scope_id", "operation_id", "run_id", "kind")
            )
        ):
            raise PermissionError("native project-request boundary identity mismatch")
        command = supervision.get("command", [])
        suffix = (
            "native_verify_load(*sys.argv[1:])"
            if entrypoint == "native_verify_load"
            else "native_train(sys.argv[1])"
        )
        expected_code = (
            "from rsi_vla.adapters.pi05_training import "
            + entrypoint
            + "; import sys; "
            + suffix
        )
        owned_pids = {row["pid"] for row in supervision.get("owned_pid_identities", [])}
        if (
            len(command) < 4
            or command[1:3] != ["-c", expected_code]
            or Path(command[3]).resolve() != root.parent / "recipe.json"
            or start.get("pid") not in owned_pids
            or saved_supervision != supervision
            or saved_result != result
            or supervision.get("returncode") != 0
            or supervision.get("stop_reason") is not None
            or supervision.get("resource_limits_satisfied") is not True
            or supervision.get("cleanup_confirmed") is not True
            or supervision.get("remaining_owned_pids") != []
            or result.get("schema") != "rsi-native-training-job-result-v1"
            or result.get("status") != "completed"
            or any(
                result.get(key) != permit[key]
                for key in ("scope_id", "operation_id", "run_id", "kind")
            )
        ):
            raise PermissionError(
                "fixed native command or completed cleanup evidence missing"
            )
        meter.update(
            status="closed",
            project_model_requests=0,
            model_clients_created=[],
            model_transports_created=[],
            components=[component],
            observed_entrypoint=entrypoint,
            boundary_evidence="fixed local native entrypoint; no project LLM client or transport factory; exact owned PID and completed supervised native output",
        )
    except (OSError, ValueError, KeyError, TypeError, PermissionError) as exc:
        meter["errors"] = [str(exc)]
    atomic_json(root / "model-meter.json", meter)
    return meter
