# Copyright 2026 RSI-Agent-VLA Authors.
"""Isolated scripted LIBERO teacher using native RPent actions and episode export.

Only this standalone entry point reads registered private object geometry. It
installs no Runtime tools, memory or observation changes in D/S/R/A execution.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..training import TrainingBudget

TEACHER_KIND = "libero-scripted-teacher"
TEACHER_SOURCES = (
    "core/src/rsi_vla/adapters/libero_teacher.py",
    "core/src/rsi_vla/adapters/pi05_training.py",
    "core/src/rsi_vla/training.py",
    "core/src/rsi_vla/midterm_host.py",
    "scripts/run_libero_teacher.py",
    "RPent/robots/libero/tools.py",
    "RPent/robots/libero/env_server.py",
    "RPent/robots/libero/env_client.py",
    "RPent/robots/libero/action_spec.py",
    "RPent/robots/libero/runtime_evidence.py",
    "RPent/robots/libero/flywheel.py",
    "RPent/rpent/flywheel/episode.py",
    "RPent/rpent/flywheel/export.py",
    "RPent/rpent/session/runtime.py",
    "RPent/rpent/robots/components/env_client_base.py",
    "RPent/rpent/robots/components/env_facade_base.py",
    "RPent/rpent/robots/components/action_spec.py",
    "RPent/rpent/utils/egl.py",
)


def verify_teacher_sources(registration: dict[str, Any], permit: dict[str, Any]) -> str:
    """Compare the existing manifest's source snapshots directly, without hashes."""
    manifest_ref = Path(registration["source_manifest_ref"]).resolve()
    repository = Path(registration["repository_root"]).resolve()
    if (
        Path(permit["source_manifest_ref"]).resolve() != manifest_ref
        or repository != Path(__file__).resolve().parents[4]
    ):
        raise PermissionError("teacher source manifest/repository binding differs")
    manifest = _read(manifest_ref)
    if manifest.get("scope") != registration["scope_id"] or not set(
        TEACHER_SOURCES
    ) <= set(manifest.get("files", [])):
        raise PermissionError(
            "teacher manifest omits an actual teacher/Env/collector consumer"
        )
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest path must remain in repository")
        if (repository / relative).read_bytes() != (
            manifest_ref.parent / "source" / relative
        ).read_bytes():
            raise PermissionError("reviewed teacher source changed: " + name)
    return str(manifest_ref)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware teacher authorization required")
    return result


def validate_teacher_registration(registration: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen teacher, all attempts and quality rules before a job."""
    if (
        registration.get("schema") != "rsi-libero-teacher-registration-v1"
        or registration.get("status") != "frozen"
        or registration.get("authorized") is not True
    ):
        raise PermissionError("frozen, authorized teacher registration required")
    plan = registration["teacher_plan"]
    if plan.get("schema") != "rsi-libero-pick-place-teacher-v1":
        raise ValueError("supported scripted teacher plan required")
    for key in (
        "teacher_id",
        "version",
        "scope_id",
        "protocol_id",
        "native_config_ref",
    ):
        if not plan.get(key):
            raise ValueError("teacher registration missing: " + key)
    if (
        plan["scope_id"] != registration["scope_id"]
        or plan["protocol_id"] != registration["protocol_id"]
    ):
        raise ValueError("teacher registration identity mismatch")
    if plan.get("task") != {
        "task_key": "position_swap_salad",
        "suite": "libero_object_swap",
        "task_id": 2,
    } or plan.get("geometry_keys") != [
        "salad_dressing_1_pos",
        "salad_dressing_1_quat",
        "basket_1_pos",
    ]:
        raise ValueError(
            "teacher supports only the registered salad swap and three private geometry keys"
        )
    splits = registration["init_splits"]
    if splits.get("teacher_development") != [1, 2] or splits.get(
        "training_demonstration"
    ) != list(range(20, 30)):
        raise ValueError("teacher development/train initial-state registration differs")
    formal = splits.get("formal", [])
    if (
        not formal
        or any(type(i) is not int or not 0 <= i < 50 for i in formal)
        or set(formal).intersection(
            splits["teacher_development"] + splits["training_demonstration"]
        )
    ):
        raise ValueError("teacher and formal initial-state splits must be disjoint")
    assignments = registration["assignments"]
    if (
        not assignments
        or len({row["run_id"] for row in assignments}) != len(assignments)
        or len({(row["purpose"], row["seed"]) for row in assignments})
        != len(assignments)
    ):
        raise ValueError("unique teacher attempt identities required")
    for row in assignments:
        if (
            row["purpose"] not in {"teacher_development", "training_demonstration"}
            or row["seed"] not in splits[row["purpose"]]
        ):
            raise ValueError(
                "teacher attempt outside its registered initial-state split"
            )
    for purpose in ("teacher_development", "training_demonstration"):
        assigned = {row["seed"] for row in assignments if row["purpose"] == purpose}
        if assigned and assigned != set(splits[purpose]):
            raise ValueError(
                "register the whole teacher-purpose denominator before collection: "
                + purpose
            )
    script = plan["script"]
    for key in ("grasp_offset_object", "placement_offset_world"):
        if (
            not isinstance(script.get(key), list)
            or len(script[key]) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in script[key]
            )
        ):
            raise ValueError("finite registered teacher XYZ required: " + key)
    for key in (
        "approach_height",
        "lift_height",
        "lateral_segment",
        "position_tolerance",
    ):
        if (
            type(script.get(key)) not in (int, float)
            or not math.isfinite(script[key])
            or script[key] <= 0
        ):
            raise ValueError("positive finite teacher servo parameter required: " + key)
    if script["lateral_segment"] > 0.25:
        raise ValueError(
            "teacher lateral waypoint segments must not exceed 0.25 metres"
        )
    for key in ("grasp_pitch", "grasp_yaw"):
        if type(script.get(key)) not in (int, float) or not math.isfinite(script[key]):
            raise ValueError("explicit finite teacher orientation required: " + key)
    for key in ("stage_steps", "close_steps", "release_steps"):
        if type(script.get(key)) is not int or not 1 <= script[key] <= 2000:
            raise ValueError("finite teacher primitive step bound required: " + key)
    rules = registration["quality_rules"]
    for key in (
        "native_success_required",
        "all_attempts_preserved",
        "no_reset",
        "no_model_actions",
    ):
        if rules.get(key) is not True:
            raise ValueError("teacher quality rule required: " + key)
    if (
        type(rules.get("min_lift_m")) not in (int, float)
        or not 0 < rules["min_lift_m"] <= script["lift_height"]
    ):
        raise ValueError("registered object lift quality threshold required")
    return plan


def require_teacher_permit(
    registration_ref: str | Path,
    permit: dict[str, Any],
    *,
    output_root: str | Path,
    cuda_device: int,
    require_submission: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], TrainingBudget]:
    """Consume current scope, registered teacher privilege and live host lease."""
    from ..midterm_host import MidtermLedger

    ledger = MidtermLedger(permit["scope_ref"], permit["accounting_ref"])
    scope = ledger.authority()
    registration = _read(registration_ref)
    plan = validate_teacher_registration(registration)
    verify_teacher_sources(registration, permit)
    if (
        permit.get("kind") != TEACHER_KIND
        or permit.get("authorized") is not True
        or permit.get("scope_id") != scope["scope_id"]
        or registration["scope_id"] != scope["scope_id"]
        or permit.get("protocol_id") != registration["protocol_id"]
        or permit.get("teacher_plan") != plan
        or Path(permit["registration_ref"]).resolve()
        != Path(registration_ref).resolve()
        or permit.get("owner_uid") != os.geteuid()
        or registration.get("owner_uid") != os.geteuid()
        or permit.get("cuda_device") != cuda_device
        or type(cuda_device) is not int
        or cuda_device < 0
    ):
        raise PermissionError(
            "teacher permit identity/owner/registered plan/device mismatch"
        )
    root = Path(output_root).resolve()
    write_root = Path(scope["write_root"]).resolve()
    if (
        root != Path(permit["output_root"]).resolve()
        or not root.is_relative_to(write_root)
        or not Path(registration_ref).resolve().is_relative_to(write_root)
    ):
        raise PermissionError(
            "teacher writes and registration must stay in the task domain"
        )
    assignment = next(
        (
            row
            for row in registration["assignments"]
            if row["run_id"] == permit.get("run_id")
        ),
        None,
    )
    if assignment is None or any(
        permit.get(key) != assignment[key] for key in ("seed", "purpose")
    ):
        raise PermissionError("teacher attempt is not a registered run/seed/purpose")
    p1 = _read(scope["p1_acceptance_ref"])
    if (
        p1.get("schema") != "rsi-midterm-p1-acceptance-v1"
        or p1.get("scope_id") != scope["scope_id"]
        or p1.get("status") != "passed"
    ):
        raise PermissionError("P1 acceptance required before teacher simulation")
    budget = TrainingBudget(**permit["budget"])
    if (
        budget.reserved_devices != 1
        or budget.wall_seconds > 1200
        or budget.gpu_device_seconds > 1200
        or budget.max_steps > 2000
        or budget.artifact_bytes > 1024**3
        or type(permit.get("tool_call_limit")) is not int
        or not 1 <= permit["tool_call_limit"] <= 512
    ):
        raise ValueError("teacher exceeds Runtime single-job bounds")
    expiry = min(_utc(scope["expires_at"]), _utc(permit["expires_at"]))
    if (expiry - datetime.now(timezone.utc)).total_seconds() < budget.deadline_seconds:
        raise PermissionError("complete teacher lifecycle exceeds scope/permit expiry")
    record = ledger.validate_reservation(
        permit["operation_id"],
        phase_ref=permit["phase_ref"],
        gpu_devices=[cuda_device],
        budget={
            "runtime_starts": 1,
            "gpu_seconds": budget.gpu_device_seconds,
            "storage_bytes": budget.artifact_bytes,
            "current_disk_bytes": budget.artifact_bytes,
        },
    )
    if require_submission and not record.get("submitted_at"):
        raise PermissionError(
            "teacher host submission must be persisted before native launch"
        )
    return registration, assignment, budget


class _LocalTransport:
    """Use existing facade routes synchronously on the native rendering thread."""

    def __init__(self, facade: Any):
        self.facade = facade

    def call(
        self, method: str, args: tuple = (), kwargs: dict | None = None, **unused: Any
    ) -> Any:
        if method not in {
            "env.get_env_meta",
            "env.get_action_spec",
            "env.reset",
            "env.step",
            "env.raw_obs",
            "env.rsi_step",
        }:
            raise PermissionError(
                "teacher local transport route is not registered: " + method
            )
        return self.facade._rpc[method](*args, **(kwargs or {}))


class TeacherQualityFailure(RuntimeError):
    """The scripted demonstration did not meet its predeclared quality rule."""


def run_teacher_script(
    primitives: Any,
    *,
    plan: dict[str, Any],
    quality_rules: dict[str, Any],
    geometry_log: str | Path,
    tool_limit: int,
) -> dict[str, Any]:
    """Run one registered attempt; every control step uses the existing primitives."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    script = plan["script"]
    calls = 0
    stages = []

    def geometry(stage: str) -> dict[str, Any]:
        raw = primitives.env.raw_obs()
        selected = {
            key: np.asarray(raw[key], dtype=float) for key in plan["geometry_keys"]
        }
        if any(
            value.shape != ((4,) if key.endswith("quat") else (3,))
            or not np.isfinite(value).all()
            for key, value in selected.items()
        ):
            raise TeacherQualityFailure(
                "registered native private geometry unavailable"
            )
        with Path(geometry_log).open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "stage": stage,
                        "source": "native_raw_observation",
                        "privilege": "scripted_teacher_only",
                        "values": {
                            key: value.tolist() for key, value in selected.items()
                        },
                    }
                )
                + "\n"
            )
        return selected

    def call(stage: str, fn: Any, **kwargs: Any) -> None:
        nonlocal calls
        if primitives.env.terminated or primitives.env.truncated:
            return
        calls += 1
        if calls > tool_limit:
            raise TeacherQualityFailure("registered teacher tool-call limit reached")
        journal = primitives.runtime_journal
        if journal is not None:
            journal.admit_tool("scripted_teacher:" + stage)
        primitives.begin_primitive("scripted_teacher:" + stage)
        status = "failed"
        try:
            result = fn(**kwargs)
            data = result.data
            stages.append({"stage": stage, "result": data})
            if (
                data.get("error")
                or data.get("final_dist_m", 0) > script["position_tolerance"]
            ):
                raise TeacherQualityFailure("teacher servo did not converge: " + stage)
            status = "completed"
        finally:
            primitives.end_primitive()
            if journal is not None:
                journal.emit(
                    "tool_end", tool="scripted_teacher:" + stage, status=status
                )

    def move(stage: str, xyz: Any, gripper: float) -> None:
        target = np.asarray(xyz, dtype=float)
        current = primitives._last_obs_eef_pos.copy()
        count = max(
            1,
            int(
                math.ceil(
                    float(np.linalg.norm((target - current)[:2]))
                    / script["lateral_segment"]
                )
            ),
        )
        for index in range(1, count + 1):
            waypoint = current + (target - current) * index / count
            call(
                stage + f":{index}/{count}",
                primitives.move_to,
                xyz=waypoint.tolist(),
                gripper=gripper,
                max_steps=script["stage_steps"],
                tol=script["position_tolerance"],
            )

    initial = geometry("initial")
    target = initial["salad_dressing_1_pos"]
    grasp = target + Rotation.from_quat(initial["salad_dressing_1_quat"]).apply(
        script["grasp_offset_object"]
    )
    high = grasp.copy()
    high[2] += script["approach_height"]
    current = primitives._last_obs_eef_pos.copy()
    move("raise_before_approach", [current[0], current[1], high[2]], -1.0)
    move("approach", high, -1.0)
    call(
        "align_and_descend",
        primitives.move_pose,
        xyz=grasp.tolist(),
        target_pitch=script["grasp_pitch"],
        target_yaw=script["grasp_yaw"],
        gripper=-1.0,
        max_steps=script["stage_steps"],
        tol=script["position_tolerance"],
    )
    for index in range(script["close_steps"]):
        call(
            "close:" + str(index),
            primitives.execute_action,
            values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
    carry = grasp.copy()
    carry[2] += script["lift_height"]
    move("lift", carry, 1.0)
    lifted = geometry("after_lift")
    lift = float(lifted["salad_dressing_1_pos"][2] - target[2])
    if not primitives.env.terminated and lift < quality_rules["min_lift_m"]:
        raise TeacherQualityFailure(
            "target object did not rise by the registered minimum lift"
        )
    placement = lifted["basket_1_pos"] + np.asarray(script["placement_offset_world"])
    carry_z = max(float(carry[2]), float(placement[2] + script["approach_height"]))
    current = primitives._last_obs_eef_pos.copy()
    move("raise_before_carry", [current[0], current[1], carry_z], 1.0)
    move("carry", [placement[0], placement[1], carry_z], 1.0)
    move("lower_to_basket", placement, 1.0)
    call("release", primitives.release, max_steps=script["release_steps"])
    geometry("final")
    return {
        "stages": stages,
        "tool_calls": calls,
        "target_lift_m": lift,
        "lift_threshold_passed": lift >= quality_rules["min_lift_m"],
        "native_success": bool(primitives.env.terminated),
        "native_truncated": bool(primitives.env.truncated),
    }


def collect_native_teacher(
    registration_ref: str, permit_ref: str, context_ref: str
) -> None:
    """Create one native Env and preserve an explicitly sourced teacher attempt."""
    import shutil
    import signal

    from ..training import JOB_MARKER

    permit = _read(permit_ref)
    root = Path(os.environ[JOB_MARKER]).resolve()
    registration, assignment, budget = require_teacher_permit(
        registration_ref,
        permit,
        output_root=root,
        cuda_device=permit["cuda_device"],
        require_submission=True,
    )
    plan = registration["teacher_plan"]
    os.environ.update(
        ROBOT_PLATFORM="LIBERO", LIBERO_TYPE="pro", RSI_RUNTIME_CONTEXT=context_ref
    )
    config_root = root / "native_config"
    config_root.mkdir()
    shutil.copyfile(plan["native_config_ref"], config_root / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    # Follow env_server's actual EGL/CUDA mapping. No VLA server is created.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from rpent.utils.egl import configure_egl_device

    configure_egl_device(permit["cuda_device"])
    import torch

    torch.cuda.set_device(permit["cuda_device"])
    from rlinf.envs.sim.libero.utils import benchmark

    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade, make_env
    from robots.libero.flywheel import LIBERO_SPEC
    from robots.libero.tools import LiberoPrimitives
    from rpent.flywheel.episode import EpisodeWriter
    from rpent.session.runtime import RuntimeJournal

    def stop_owned_teacher(signum: int, frame: Any) -> None:
        raise TeacherQualityFailure("teacher stopped by its supervisor")

    signal.signal(signal.SIGTERM, stop_owned_teacher)
    suite = benchmark.get_benchmark(plan["task"]["suite"])(task_order_index=0)
    task_id, seed = plan["task"]["task_id"], assignment["seed"]
    trials = len(suite.get_task_init_states(task_id))
    if trials != 50 or seed % trials != seed:
        raise ValueError(
            "actual native teacher initial-state table differs from registered split"
        )
    reset_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id)) + seed
    _write(
        root / "init-binding.json",
        {
            "suite": plan["task"]["suite"],
            "task_id": task_id,
            "seed": seed,
            "trials": trials,
            "init_index": seed,
            "specific_reset_id": reset_id,
            "purpose": assignment["purpose"],
            "registration_ref": registration_ref,
        },
    )
    _write(
        root / "model-meter.json",
        {
            "schema": "rsi-scripted-teacher-model-meter-v1",
            "status": "open",
            "scope_id": permit["scope_id"],
            "operation_id": permit["operation_id"],
            "request_cap": 0,
            "enabled_components": ["native_env"],
            "model_clients_created": [],
        },
    )
    marker = {
        "schema": "rsi-libero-teacher-start-v1",
        "scope_id": permit["scope_id"],
        "operation_id": permit["operation_id"],
        "run_id": permit["run_id"],
        "kind": TEACHER_KIND,
        "owner_uid": os.geteuid(),
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "before_make_env_constructor",
        "source_kind": "authorized_demonstration",
        "teacher_kind": "scripted_teacher",
        "purpose": assignment["purpose"],
        "source_manifest_ref": registration["source_manifest_ref"],
    }
    # Re-read source bytes immediately before the native Env side effect.
    verify_teacher_sources(registration, permit)
    with (root / "native-start.json").open("x") as handle:
        handle.write(json.dumps(marker, indent=2) + "\n")
    context = _read(context_ref)
    journal = RuntimeJournal(context)
    facade = None
    writer = None
    primitives = None
    error = None
    quality = {}
    episode_ref = None
    try:
        native = make_env(
            task_id,
            seed,
            suite_name=plan["task"]["suite"],
            max_episode_steps=budget.max_steps,
        )
        meta = {
            "suite": plan["task"]["suite"],
            "task": task_id,
            "seed": seed,
            "max_episode_steps": budget.max_steps,
        }
        facade = LiberoEnvFacade(native, meta=meta)
        # LiberoEnvClient construction performs the single initial reset. Do
        # not call primitives.reset(), which would reset the same attempt twice.
        client = LiberoEnvClient(_LocalTransport(facade), expected_meta=meta)
        primitives = LiberoPrimitives(
            client, None, None, journal.check, runtime_journal=journal
        )
        primitives.set_obs(client.last_obs)
        metadata = {
            "source_kind": "authorized_demonstration",
            "teacher_kind": "scripted_teacher",
            "training_admission": "not_admitted",
            "teacher_purpose": assignment["purpose"],
            "teacher_id": plan["teacher_id"],
            "teacher_version": plan["version"],
            "scope_id": permit["scope_id"],
            "protocol_id": permit["protocol_id"],
            "run_id": permit["run_id"],
            "operation_id": permit["operation_id"],
            "suite": meta["suite"],
            "task_id": task_id,
            "seed": seed,
            "init_index": seed,
            "specific_reset_id": reset_id,
            "registration_ref": str(Path(registration_ref).resolve()),
            "source_manifest_ref": registration["source_manifest_ref"],
            "registration_snapshot_ref": str(
                root / "teacher-registration-snapshot.json"
            ),
            "quality_evidence_ref": str(root / "teacher-quality.json"),
            "privilege": {
                "scope": "teacher_only",
                "geometry_keys": plan["geometry_keys"],
            },
            "task_language": client.get_task_language(),
            "native_action_spec": client.action_specs,
        }
        writer = EpisodeWriter(
            root / "episodes",
            metadata=metadata,
            spec=LIBERO_SPEC,
            initial_observation=client.last_obs,
        )
        primitives._flywheel = writer
        quality = run_teacher_script(
            primitives,
            plan=plan,
            quality_rules=registration["quality_rules"],
            geometry_log=root / "teacher-private-geometry.jsonl",
            tool_limit=permit["tool_call_limit"],
        )
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for phase, cleanup in (
            ("episode_finalize", writer.finalize if writer is not None else None),
            ("native_env_close", facade.close if facade is not None else None),
            ("journal_finalize", journal.finalize),
        ):
            if cleanup is None:
                continue
            try:
                result = cleanup()
                if phase == "episode_finalize":
                    episode_ref = str(result)
            except Exception as exc:
                error = error or {
                    "type": type(exc).__name__,
                    "message": phase + ": " + str(exc),
                }
        _write(
            root / "model-meter.json",
            {
                "schema": "rsi-scripted-teacher-model-meter-v1",
                "status": "closed",
                "scope_id": permit["scope_id"],
                "operation_id": permit["operation_id"],
                "request_cap": 0,
                "enabled_components": ["native_env"],
                "model_clients_created": [],
                "project_model_requests": 0,
                "basis": "closed scripted-only component lifecycle; no Planner/VLA client or model transport instantiated",
            },
        )
        native_success = bool(primitives is not None and primitives.env.terminated)
        accepted = (
            error is None
            and native_success
            and quality.get("lift_threshold_passed") is True
        )
        _write(
            root / "teacher-quality.json",
            {
                "schema": "rsi-scripted-teacher-quality-v1",
                "scope_id": permit["scope_id"],
                "operation_id": permit["operation_id"],
                "registration_ref": registration_ref,
                "episode_ref": episode_ref,
                "native_success": native_success,
                "registered_collection_checks_passed": accepted,
                "checks": quality,
                "error": error,
                "training_admission": "not_admitted",
                "independent_quality_review": "required",
            },
        )
        _write(
            root / "native-result.json",
            {
                "schema": "rsi-libero-teacher-native-result-v1",
                "scope_id": permit["scope_id"],
                "operation_id": permit["operation_id"],
                "run_id": permit["run_id"],
                "kind": TEACHER_KIND,
                "status": "completed" if error is None else "failed",
                "native_success": native_success,
                "collection_quality_pass": accepted,
                "episode_ref": episode_ref,
                "error": error,
                "teacher_purpose": assignment["purpose"],
                "training_admission": "not_admitted",
            },
        )
    if error is not None:
        raise RuntimeError("native teacher attempt failed: " + error["message"])


def export_teacher_demonstrations(
    *,
    registration_ref: str | Path,
    episode_paths: list[str],
    output_root: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    """Use RPent's real exporter and preserve explicit teacher provenance separately.

    Registered producer checks select eligible training-source episodes. The
    resulting native dataset remains unadmitted until independent review.
    """
    import numpy as np

    from robots.libero.flywheel import LIBERO_SPEC
    from rpent.flywheel.episode import validate_episode
    from rpent.flywheel.export import export_lerobot

    from .pi05_training import (
        ACTION_CONTRACT,
        dataset_identity,
        validate_dataset_manifest,
    )

    registration = _read(registration_ref)
    plan = validate_teacher_registration(registration)
    assignments = {row["run_id"]: row for row in registration["assignments"]}
    verify_teacher_sources(
        registration, {"source_manifest_ref": registration["source_manifest_ref"]}
    )
    provenance = []
    for path in episode_paths:
        metadata = validate_episode(path, spec=LIBERO_SPEC)
        registered = assignments.get(metadata.get("run_id"))
        if (
            metadata.get("source_kind") != "authorized_demonstration"
            or metadata.get("teacher_kind") != "scripted_teacher"
            or metadata.get("teacher_purpose") != "training_demonstration"
            or registered is None
            or registered["purpose"] != "training_demonstration"
            or registered["seed"] != metadata.get("seed")
            or metadata.get("teacher_id") != plan["teacher_id"]
            or metadata.get("teacher_version") != plan["version"]
            or metadata.get("scope_id") != registration["scope_id"]
            or metadata.get("protocol_id") != registration["protocol_id"]
            or metadata.get("source_manifest_ref")
            != registration["source_manifest_ref"]
            or Path(metadata["registration_ref"]).resolve()
            != Path(registration_ref).resolve()
        ):
            raise PermissionError(
                "episode is not a registered training-purpose scripted teacher demonstration"
            )
        quality = _read(metadata["quality_evidence_ref"])
        if (
            quality.get("schema") != "rsi-scripted-teacher-quality-v1"
            or quality.get("registered_collection_checks_passed") is not True
            or quality.get("native_success") is not True
            or quality.get("scope_id") != metadata["scope_id"]
            or quality.get("operation_id") != metadata["operation_id"]
            or Path(quality["episode_ref"]).resolve() != Path(path).resolve()
        ):
            raise PermissionError(
                "teacher episode did not pass registered collection quality checks"
            )
        snapshot = _read(metadata["registration_snapshot_ref"])
        if (
            snapshot.get("teacher_plan") != plan
            or snapshot.get("quality_rules") != registration["quality_rules"]
            or metadata.get("init_index") != registered["seed"]
        ):
            raise PermissionError(
                "teacher episode source recipe/quality/initial-state binding differs"
            )
        provenance.append(metadata)
    if not provenance or len({row["episode_id"] for row in provenance}) != len(
        provenance
    ):
        raise ValueError("nonempty unique teacher source episodes required")
    expected = {"suite": plan["task"]["suite"], "task_id": plan["task"]["task_id"]}
    exported = export_lerobot(
        [Path(path) for path in episode_paths],
        spec=LIBERO_SPEC,
        expected_metadata=expected,
        repo_id_prefix="rsi-scripted-teacher/swap-salad",
        output_root=output_root,
        dataset_id=dataset_id,
        purpose="demonstration",
    )
    root = Path(exported["dataset_path"])
    mapping = [
        {
            "episode_index": index,
            "source_episode_index": row["seed"],
            "source_episode_id": row["episode_id"],
            "task": row["task_language"],
            "source_ref": str(Path(path).resolve()),
        }
        for index, (row, path) in enumerate(zip(provenance, episode_paths))
    ]
    source = {
        "schema": "rsi-scripted-teacher-source-v1",
        "source_kind": "authorized_demonstration",
        "teacher_kind": "scripted_teacher",
        "revision": plan["teacher_id"] + ":" + plan["version"],
        "registration_ref": str(Path(registration_ref).resolve()),
        "raw_teacher_episodes": provenance,
        "exporter_result": exported,
    }
    _write(root / "teacher-source.json", source)
    manifest = {
        "source_kind": "authorized_demonstration",
        "teacher_kind": "scripted_teacher",
        "dataset_root": str(root.resolve()),
        "revision": source["revision"],
        "provenance_ref": str(root / "teacher-source.json"),
        "episodes": mapping,
        "action_contract": ACTION_CONTRACT,
        "training_admission": "not_admitted",
        "registration_ref": str(Path(registration_ref).resolve()),
        "source_episode_original_init_mapping": "registered native init_index, exact teacher collection identity",
    }
    validate_dataset_manifest(manifest)
    _write(root / "demonstration-manifest.json", manifest)
    # Audit every exported frame against its original pre-action schema-1
    # observation/action. This is separate from the exporter's source labels.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reader = LeRobotDataset(exported["repo_id"], root=root, download_videos=False)
    offset, checks = 0, []
    for index, (row, path) in enumerate(zip(provenance, episode_paths)):
        count = row["training_step_count"]
        passed = True
        with np.load(Path(path) / "transitions.npz", allow_pickle=False) as raw:
            for frame in range(count):
                sample = reader[offset + frame]
                for camera, key in (
                    ("image", "main_images"),
                    ("wrist_image", "wrist_images"),
                ):
                    image = np.rint(
                        np.asarray(sample[camera]).transpose(1, 2, 0) * 255
                    ).astype(np.uint8)
                    passed &= bool(np.array_equal(image, raw[key][frame]))
                passed &= bool(
                    np.array_equal(np.asarray(sample["state"]), raw["states"][frame])
                )
                passed &= bool(
                    np.array_equal(np.asarray(sample["actions"]), raw["actions"][frame])
                )
                passed &= bool(np.isfinite(np.asarray(sample["state"])).all())
                actions = np.asarray(sample["actions"])
                passed &= bool(
                    np.isfinite(actions).all()
                    and (np.abs(actions) <= 1.00001).all()
                    and float(actions[6]) in {-1.0, 1.0}
                )
        checks.append(
            {
                "episode_index": index,
                "frames_checked": count,
                "checks": {
                    "all_pre_action_images_states_and_actions_equal_source": bool(
                        passed
                    )
                },
            }
        )
        offset += count
    audit = {
        "schema": "rsi-demonstration-payload-audit-v1",
        "dataset": dataset_identity(manifest),
        "frames_checked": offset,
        "episodes": checks,
        "checks_passed": all(all(row["checks"].values()) for row in checks),
        "training_admission": "pending_independent_quality_and_isolation_review",
    }
    _write(root / "payload-audit.json", audit)
    if not audit["checks_passed"]:
        raise ValueError("native teacher export changed source pre-action payloads")
    return {
        "manifest_ref": str(root / "demonstration-manifest.json"),
        "payload_audit_ref": str(root / "payload-audit.json"),
        "dataset_root": str(root),
        "exporter_ref": str(root / "meta/rpent_flywheel.json"),
        "training_admission": "not_admitted",
    }
