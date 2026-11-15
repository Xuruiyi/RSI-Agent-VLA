# Copyright 2026 RSI-Agent-VLA Authors.
"""Registered pinned public LIBERO episodes, native CPU export and payload audit."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..training import artifact_size
from .pi05_training import (
    ACTION_CONTRACT,
    _fetch_bounded,
    _safe_relative,
    audit_demonstration_payload,
    validate_dataset_manifest,
)

REPO_ID = "physical-intelligence/libero"
SOURCE_FILES = (
    "core/src/rsi_vla/adapters/libero_public_data.py",
    "scripts/prepare_libero_public_data.py",
    "core/src/rsi_vla/adapters/pi05_training.py",
    "core/src/rsi_vla/training.py",
    "core/src/rsi_vla/midterm_host.py",
    "core/src/rsi_vla/storage.py",
)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def registered_plan(registration: dict[str, Any]) -> dict[str, Any]:
    """Resolve only registered episode IDs from existing immutable metadata."""
    if registration.get("schema") != "rsi-libero-public-data-registration-v1":
        raise ValueError("registered public demonstration selection required")
    metadata = Path(registration["metadata_root"])
    source = _read(metadata / "source.json")
    if (
        registration.get("repo_id") != REPO_ID
        or source["repo_id"] != REPO_ID
        or source["revision"] != registration.get("revision")
        or not re.fullmatch("[0-9a-f]{40}", source["revision"])
    ):
        raise ValueError("public dataset identity differs from pinned metadata")
    info = _read(metadata / "meta/info.json")
    if info.get("codebase_version") not in {"v2.0", "v2.1"} or info.get("fps") != 10:
        raise ValueError(
            "this public source requires native LeRobot v2.0/v2.1 at fps10"
        )
    if info["features"]["state"]["shape"] != [8] or info["features"]["actions"][
        "shape"
    ] != [7]:
        raise ValueError("public source native state/action geometry mismatch")
    if any(
        info["features"][camera]["dtype"] != "image"
        for camera in ("image", "wrist_image")
    ):
        raise ValueError(
            "registered public source currently supports embedded RGB images"
        )
    tasks = {
        row["task_index"]: row["task"]
        for row in (
            json.loads(line)
            for line in (metadata / "meta/tasks.jsonl").read_text().splitlines()
            if line
        )
    }
    episodes = {
        row["episode_index"]: row
        for row in (
            json.loads(line)
            for line in (metadata / "meta/episodes.jsonl").read_text().splitlines()
            if line
        )
    }
    selected = []
    for group in registration["task_groups"]:
        if (
            group.get("role") not in {"target", "rehearsal"}
            or tasks.get(group["task_index"]) != group["task"]
        ):
            raise ValueError(
                "registered task/role differs from immutable source metadata"
            )
        for index in group["episode_indices"]:
            if (
                type(index) is not int
                or index not in episodes
                or episodes[index].get("tasks") != [group["task"]]
            ):
                raise ValueError(
                    "registered source episode does not belong to its exact task"
                )
            row = episodes[index]
            name = info["data_path"].format(
                episode_chunk=index // info["chunks_size"], episode_index=index
            )
            _safe_relative(name)
            size = source["file_sizes"].get(name)
            if (
                type(size) is not int
                or size <= 0
                or type(row.get("length")) is not int
                or row["length"] <= 0
            ):
                raise ValueError("pinned episode size/frame evidence missing")
            selected.append(
                {
                    "episode_index": index,
                    "task_index": group["task_index"],
                    "task": group["task"],
                    "role": group["role"],
                    "length": row["length"],
                    "file": name,
                    "bytes": size,
                }
            )
    if (
        not selected
        or not any(row["role"] == "target" for row in selected)
        or len({row["episode_index"] for row in selected}) != len(selected)
    ):
        raise ValueError("unique source episodes with a target group required")
    selected.sort(key=lambda row: row["episode_index"])
    frames = sum(row["length"] for row in selected)
    if (
        type(registration.get("max_frames")) is not int
        or frames > registration["max_frames"]
    ):
        raise ValueError("registered source frames exceed the finite selection budget")
    if (
        type(registration.get("max_output_bytes")) is not int
        or registration["max_output_bytes"] <= 0
    ):
        raise ValueError("positive finite native dataset artifact cap required")
    isolation = registration["evaluation_isolation"]
    if (
        isolation.get("public_original_init_ids") is not None
        or isolation.get("train_formal_init_overlap") != "unknown"
        or isolation.get("unseen_generalization_claim") is not False
        or isolation.get("claim_scope")
        != "familiar_task_under_registered_runtime_conditions"
    ):
        raise ValueError(
            "public original initial-state overlap is unknown; unseen/disjoint claims are unsupported"
        )
    return {
        "repo_id": REPO_ID,
        "revision": source["revision"],
        "fps": 10,
        "episodes": selected,
        "total_episodes": len(selected),
        "total_frames": frames,
        "episode_payload_bytes": sum(row["bytes"] for row in selected),
        "README_bytes": source["file_sizes"].get("README.md"),
        "evaluation_isolation": isolation,
        "training_admission": "not_admitted",
    }


def require_data_permit(
    registration_ref: str | Path,
    permit: dict[str, Any],
    *,
    action: str,
    output_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check current task authority and its actual CPU/download reservation."""
    from ..midterm_host import MidtermLedger

    if action not in {"acquire", "materialize"}:
        raise ValueError("unsupported registered public data action")
    registration = _read(registration_ref)
    plan = registered_plan(registration)
    ledger = MidtermLedger(permit["scope_ref"], permit["accounting_ref"])
    scope = ledger.authority()
    if (
        registration.get("status") != "frozen"
        or registration.get("authorized") is not True
        or permit.get("kind") != "libero-public-data-" + action
        or permit.get("authorized") is not True
        or permit.get("scope_id") != scope["scope_id"]
        or registration.get("scope_id") != scope["scope_id"]
        or permit.get("protocol_id") != registration.get("protocol_id")
        or permit.get("owner_uid") != os.geteuid()
        or registration.get("owner_uid") != os.geteuid()
        or Path(permit["registration_ref"]).resolve()
        != Path(registration_ref).resolve()
        or permit.get("registered_plan") != plan
    ):
        raise PermissionError(
            "public data operation lacks its exact frozen scoped selection"
        )
    root = Path(output_root).resolve()
    write_root = Path(scope["write_root"]).resolve()
    if root != Path(permit["output_root"]).resolve() or not root.is_relative_to(
        write_root
    ):
        raise PermissionError("public dataset operation is outside task write domain")
    p1 = _read(scope["p1_acceptance_ref"])
    if (
        p1.get("schema") != "rsi-midterm-p1-acceptance-v1"
        or p1.get("scope_id") != scope["scope_id"]
        or p1.get("status") != "passed"
    ):
        raise PermissionError(
            "actual P1 closure required before selected data operation"
        )
    if not registration["target_selection_evidence_refs"]:
        raise PermissionError("actual registered target-selection evidence required")
    for name in registration["target_selection_evidence_refs"] + [
        registration["license_evidence_ref"]
    ]:
        if not Path(name).is_file():
            raise PermissionError("registered target/license evidence unavailable")
    manifest_ref = Path(registration["source_manifest_ref"]).resolve()
    if (
        Path(permit["source_manifest_ref"]).resolve() != manifest_ref
        or Path(registration["repository_root"]).resolve()
        != Path(__file__).resolve().parents[4]
    ):
        raise PermissionError("public data code binding mismatch")
    manifest = _read(manifest_ref)
    if manifest.get("scope") != scope["scope_id"] or not set(SOURCE_FILES) <= set(
        manifest.get("files", [])
    ):
        raise PermissionError("public data manifest omits an actual consumer")
    for name in manifest["files"]:
        path = _safe_relative(name)
        if (Path(registration["repository_root"]) / path).read_bytes() != (
            manifest_ref.parent / "source" / path
        ).read_bytes():
            raise PermissionError("public data consumer source changed: " + name)
    wall = permit["wall_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError(
            "positive finite whole-command data preparation budget required"
        )
    expiry = min(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in (scope["expires_at"], permit["expires_at"])
    )
    if (
        expiry.tzinfo is None
        or (expiry - datetime.now(timezone.utc)).total_seconds() < wall
    ):
        raise PermissionError("complete data preparation exceeds authority expiry")
    budget = {key: permit[key] for key in ("storage_bytes", "current_disk_bytes")}
    if (
        action == "materialize"
        and registration["max_output_bytes"] > permit["storage_bytes"]
    ):
        raise PermissionError("native dataset artifact cap exceeds its CPU reservation")
    if action == "acquire":
        budget["download_bytes"] = permit["max_download_bytes"]
    ledger.validate_reservation(
        permit["operation_id"],
        budget=budget,
        phase_ref=permit["phase_ref"],
        gpu_devices=[],
    )
    return registration, plan


def acquire_registered_files(
    registration: dict[str, Any],
    output: str | Path,
    *,
    max_download_bytes: int,
    opener: Any = None,
    operation_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch exact pinned files, with existing public payload reuse and byte meter.

    The caller owns the complete CPU command supervisor and outer reservation.
    This function never resolves main, downloads a model or scans sealed data.
    """
    registered_plan(registration)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(
            "public acquisition identity already exists; reconcile its receipt"
        )
    root.mkdir(parents=True)
    meter = _PublicDownloadMeter(root, operation_identity or {}, opener)
    complete = False
    try:
        report = _acquire_registered_payloads(
            registration, root, max_download_bytes, meter
        )
        complete = True
        return report
    finally:
        meter.close(download_complete=complete)


class _PublicDownloadMeter:
    """Account returned payload chunks and distinguish closure from asset success."""

    def __init__(self, root: Path, identity: dict[str, Any], opener: Any):
        self.root, self.identity, self.opener = root, identity, opener
        self.payload_bytes = 0
        self.recorded_bytes = 0
        self.healthy = True
        self.read_inflight = False
        (root / "download-ledger.jsonl").touch(exist_ok=False)

    def record(self, url: str, count: int) -> None:
        self.payload_bytes += count
        try:
            with (self.root / "download-ledger.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {**self.identity, "url": url, "response_payload_bytes": count}
                    )
                    + "\n"
                )
        except BaseException:
            self.healthy = False
            raise
        else:
            self.recorded_bytes += count

    def fetch(self, url: str, *, remaining: int, expected_size: int) -> bytes:
        self.read_inflight = True
        try:
            data = _fetch_bounded(
                url,
                remaining=remaining,
                expected_size=expected_size,
                opener=self.opener,
                on_bytes=self.record,
            )
        except ValueError as exc:
            # These bounded-fetch rejections happen before a body read or after
            # all returned chunks have reached record(). Other read failures can
            # contain an unreturned partial chunk and retain unknown use.
            if str(exc) in {
                "download would exceed the authorized byte limit",
                "Content-Length exceeds remaining download budget",
                "incomplete HTTP response",
                "downloaded size differs from pinned file metadata",
                "response completeness unknown at the download cap",
            }:
                self.read_inflight = False
            raise
        else:
            self.read_inflight = False
            return data

    def close(self, *, download_complete: bool) -> None:
        closed = self.healthy and not self.read_inflight
        _write(
            self.root / "download-meter.json",
            {
                "schema": "rsi-public-data-download-meter-v1",
                **self.identity,
                "status": "closed" if closed else "unresolved",
                "download_complete": download_complete,
                "download_meter_closed": closed,
                "observed_download_payload_bytes_lower_bound": self.recorded_bytes,
                "producer_returned_payload_bytes": self.payload_bytes,
                "download_response_payload_bytes": self.payload_bytes
                if closed
                else None,
                "download_ledger_ref": str(self.root / "download-ledger.jsonl"),
                "basis": "every payload chunk returned to the bounded fetch callback; read/ledger failure remains unknown",
            },
        )


def _acquire_registered_payloads(
    registration: dict[str, Any],
    root: Path,
    max_download_bytes: int,
    meter: _PublicDownloadMeter,
) -> dict[str, Any]:
    plan = registered_plan(registration)
    metadata = Path(registration["metadata_root"])
    pinned = _read(metadata / "source.json")
    shutil.copytree(metadata / "meta", root / "meta")
    reused = {}
    for reference in registration.get("reuse_source_roots", []):
        candidate = Path(reference)
        origin = _read(candidate / "source.json")
        if origin["repo_id"] != REPO_ID or origin["revision"] != plan["revision"]:
            raise ValueError("reused public source revision differs from registration")
        for episode in plan["episodes"]:
            path = candidate / _safe_relative(episode["file"])
            if (
                path.is_file()
                and path.stat().st_size == episode["bytes"]
                and origin["file_sizes"].get(episode["file"]) == episode["bytes"]
            ):
                reused[episode["file"]] = path
    readme_bytes = plan["README_bytes"]
    if type(readme_bytes) is not int or readme_bytes <= 0:
        raise ValueError("pinned dataset card size required for license review")
    required = (
        sum(row["bytes"] for row in plan["episodes"] if row["file"] not in reused)
        + readme_bytes
    )
    if type(max_download_bytes) is not int or max_download_bytes < required:
        raise ValueError("exact registered public files exceed download envelope")
    total = 0
    for row in plan["episodes"] + [{"file": "README.md", "bytes": readme_bytes}]:
        path = root / _safe_relative(row["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if row["file"] in reused:
            shutil.copyfile(reused[row["file"]], path)
            continue
        data = meter.fetch(
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/{plan['revision']}/{row['file']}",
            remaining=max_download_bytes - total,
            expected_size=row["bytes"],
        )
        total += len(data)
        path.write_bytes(data)
    report = {
        **pinned,
        "selection": plan,
        "downloaded_demonstration_bytes": total,
        "reused_files": list(reused),
        "registration_ref": registration.get("registration_ref"),
        "scope_id": registration["scope_id"],
        "protocol_id": registration["protocol_id"],
        "source_manifest_ref": registration.get("source_manifest_ref"),
        "training_admission": "not_admitted",
        "license_document_ref": str(root / "README.md"),
    }
    _write(root / "source.json", report)
    return report


def materialize_registered_data(
    registration: dict[str, Any], source_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Convert all registered tasks with the native writer and audit every payload."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("native public materialization is local CPU only")
    plan = registered_plan(registration)
    source_root, output = Path(source_root), Path(output)
    source = _read(source_root / "source.json")
    if source.get("selection") != plan or source["revision"] != plan["revision"]:
        raise ValueError("acquired public files differ from registered exact selection")
    if output.exists():
        raise FileExistsError(
            "public subset identity already exists; reconcile prior evidence"
        )
    for row in plan["episodes"]:
        path = source_root / _safe_relative(row["file"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("registered public episode missing or size changed")
    info = _read(source_root / "meta/info.json")
    lookup = {row["episode_index"]: row for row in plan["episodes"]}
    reader = LeRobotDataset(
        REPO_ID,
        root=source_root,
        episodes=list(lookup),
        revision=plan["revision"],
        download_videos=False,
        video_backend="pyav",
    )
    defaults = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {
        key: {**value, "shape": tuple(value["shape"])}
        for key, value in info["features"].items()
        if key not in defaults
    }
    writer = LeRobotDataset.create(
        "rsi-local/libero-registered-public",
        root=output,
        fps=info["fps"],
        robot_type=info.get("robot_type"),
        features=features,
        use_videos=False,
    )
    mapping = []
    previous = None
    for index in range(len(reader)):
        sample = reader[index]
        original = int(sample["episode_index"])
        row = lookup[original]
        if previous is not None and original != previous:
            writer.save_episode()
        if original != previous:
            mapping.append(
                {
                    "episode_index": len(mapping),
                    "source_episode_index": original,
                    "source_task_index": row["task_index"],
                    "role": row["role"],
                    "task": row["task"],
                    "source_ref": f"hf://datasets/{REPO_ID}@{plan['revision']}/episode/{original}",
                }
            )
        if sample["task"] != row["task"]:
            raise ValueError(
                "actual downloaded payload task differs from registered metadata"
            )
        frame = {}
        for key in features:
            value = np.asarray(sample[key])
            if key in ("image", "wrist_image"):
                if value.shape[0] == 3:
                    value = value.transpose(1, 2, 0)
                if np.issubdtype(value.dtype, np.floating):
                    value = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
            if key in ("state", "actions") and not np.isfinite(value).all():
                raise ValueError("nonfinite public demonstrated state/action")
            frame[key] = value
        writer.add_frame(
            frame, task=sample["task"], timestamp=float(sample["timestamp"])
        )
        if artifact_size(output) > registration["max_output_bytes"]:
            raise ValueError("registered native subset artifact threshold exceeded")
        previous = original
    writer.save_episode()
    if (
        len(mapping) != plan["total_episodes"]
        or len(reader) != plan["total_frames"]
        or artifact_size(output) > registration["max_output_bytes"]
    ):
        raise ValueError("registered native subset episode/frame/byte bound differs")
    manifest = {
        "source_kind": "public_demonstration",
        "provenance_ref": str((source_root / "source.json").resolve()),
        "revision": plan["revision"],
        "dataset_root": str(output.resolve()),
        "action_contract": ACTION_CONTRACT,
        "episodes": mapping,
        "training_admission": "not_admitted",
        "source_repo_id": REPO_ID,
        "scope_id": registration["scope_id"],
        "protocol_id": registration["protocol_id"],
        "registration_ref": registration.get("registration_ref"),
        "source_manifest_ref": registration.get("source_manifest_ref"),
        "license_evidence_ref": registration["license_evidence_ref"],
        "attribution": registration["attribution"],
        "evaluation_isolation": registration["evaluation_isolation"],
        "conversion": "native LeRobot writer; original actions/states/tasks and decoded RGB pixels preserved; episodes/tasks reindexed",
    }
    validate_dataset_manifest(manifest)
    _write(output / "demonstration-manifest.json", manifest)
    audit = audit_demonstration_payload(manifest)
    _write(output / "payload-audit.json", audit)
    if not audit["checks_passed"]:
        raise ValueError("full registered public source payload audit failed")
    return {
        "dataset_root": str(output.resolve()),
        "manifest_ref": str(output / "demonstration-manifest.json"),
        "payload_audit_ref": str(output / "payload-audit.json"),
        "training_admission": "not_admitted",
        "episodes": len(mapping),
        "frames": len(reader),
    }
