"""Bounded supervision for an explicitly authorized native training process."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone


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
