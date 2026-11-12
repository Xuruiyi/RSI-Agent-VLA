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
    if kind not in {"pi05-training-pilot", "pi05-load-check"} or permit.get("kind") != kind or permit.get("authorized") is not True:
        raise PermissionError(f"explicit {kind} authorization required")
    if not run_id or permit.get("run_id") != run_id:
        raise PermissionError("training authorization run_id mismatch")
    for key, expected in (("dataset_root", dataset_root), ("model_path", model_path)):
        if Path(permit.get(key, "")).resolve() != Path(expected).resolve():
            raise PermissionError(f"training authorization {key} mismatch")
    allowed = TrainingBudget(**permit["budget"])
    if any(value > asdict(allowed)[key] for key, value in asdict(budget).items()):
        raise PermissionError("requested training budget exceeds authorization")


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


def _cleanup_owned(process, owned, marker, deadline, reserve):
    started = time.monotonic()
    term_until = min(deadline, started + min(5.0, reserve / 3))
    signaled = set()
    errors = []
    while True:
        remaining = _refresh_owned(owned, marker)
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
    process.poll()
    remaining = _refresh_owned(owned, marker)
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
    if _refresh_owned({}, marker):
        raise RuntimeError("a previous process still carries this job's ownership marker")
    reserve = min(30.0, budget.deadline_seconds * 0.2) if cleanup_seconds is None else cleanup_seconds
    if not 0 < reserve < budget.deadline_seconds:
        raise ValueError("cleanup reserve must be positive and smaller than the total budget")
    ray_paths = prepare_ray_temp_path(root)
    child_env = dict(os.environ if env is None else env)
    child_env.update({JOB_MARKER: marker, "RAY_ADDRESS": "local", "RAY_TMPDIR": ray_paths["ray_tmpdir"]})
    started = time.monotonic()
    deadline = started + budget.deadline_seconds
    work_deadline = deadline - reserve
    stop_reason = None
    owned = {}
    raised = None
    with (root / "trainer.log").open("xb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=child_env, start_new_session=True)
        initial = _process_identity(process.pid)
        if initial is not None:
            owned[(process.pid, initial["start_ticks"])] = initial
        try:
            while True:
                _refresh_owned(owned, marker)
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
            cleanup = _cleanup_owned(process, owned, marker, deadline, reserve)
    final_bytes = artifact_size(root)
    elapsed = time.monotonic() - started
    report = {"returncode": process.returncode, "stop_reason": stop_reason,
              "wall_seconds": elapsed,
              "gpu_device_seconds_envelope": elapsed * budget.reserved_devices,
              "reserved_devices": budget.reserved_devices,
              "artifact_bytes_before_report": final_bytes,
              "cleanup_reserve_seconds": reserve,
              "resource_limits_satisfied": elapsed <= budget.deadline_seconds and final_bytes <= budget.artifact_bytes,
              **ray_paths,
              **cleanup,
              "budget": asdict(budget), "command": list(command),
              "evidence_level": "process_supervision"}
    (root / "supervision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if raised is not None:
        raise raised
    return report
