# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export successful episodes to LeRobot using caller-supplied data rules."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.episode import read_episode_arrays, validate_episode

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SOURCE_KINDS = {
    "authorized_demonstration",
    "autonomous_execution",
    "public_demonstration",
    "teacher_data",
    "independent_evaluation",
}


def episode_source(metadata: dict[str, Any]) -> str:
    """Preserve declared provenance and isolate evaluation trajectories."""
    scope = metadata.get(
        "information_scope",
        metadata.get("collection_config", {}).get("information_scope"),
    )
    if scope in {"release", "evaluation", "sealed", "final", "test"}:
        return "independent_evaluation"
    source = metadata.get("source_kind", "legacy_unspecified")
    if source not in SOURCE_KINDS | {"legacy_unspecified"}:
        raise ValueError(f"unknown episode source_kind: {source!r}")
    return source


def _features(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": "image"
            if key in spec["image_fields"]
            else spec["arrays"][key]["dtype"],
            "shape": spec["arrays"][key]["shape"],
            "names": ["height", "width", "channel"]
            if key in spec["image_fields"]
            else [name],
        }
        for name, key in spec["export_fields"].items()
    }


def _successful_episodes(
    paths: Iterable[Path],
    *,
    spec: dict[str, Any],
    expected_metadata: dict[str, Any],
    purpose: str = "demonstration",
) -> list[tuple[Path, dict[str, Any]]]:
    episodes = []
    for path in paths:
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        metadata = validate_episode(path, spec=spec)
        if any(metadata[key] != value for key, value in expected_metadata.items()):
            raise ValueError(f"episode metadata does not match its directory: {path}")
        if (
            episode_source(metadata)
            in {"autonomous_execution", "independent_evaluation"}
            and purpose != "diagnostic"
        ):
            raise ValueError(
                "autonomous execution and independent evaluation require separate training admission"
            )
        if purpose == "diagnostic" or metadata["is_success"]:
            episodes.append((path, metadata))
    if not episodes:
        raise ValueError(f"no successful episodes found for {expected_metadata}")
    return episodes


def export_lerobot(
    episode_paths: Iterable[Path],
    *,
    spec: dict[str, Any],
    expected_metadata: dict[str, Any],
    repo_id_prefix: str,
    output_root: Path | str,
    dataset_id: str | None = None,
    purpose: str = "demonstration",
) -> dict[str, Any]:
    """Validate selected episodes and export their successful training prefixes."""
    if purpose not in {"demonstration", "diagnostic"}:
        raise ValueError("unknown export purpose")
    dataset_id = dataset_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _NAME.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset ID: {dataset_id!r}")

    episodes = _successful_episodes(
        episode_paths, spec=spec, expected_metadata=expected_metadata, purpose=purpose
    )
    languages = {metadata["task_language"] for _, metadata in episodes}
    if len(languages) != 1:
        raise ValueError("successful episodes have different task descriptions")
    language = languages.pop()

    parent = Path(output_root).expanduser().resolve()
    destination = parent / dataset_id
    partial = parent / f"{dataset_id}.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(f"dataset already exists: {destination}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("install RPent with the 'flywheel' extra") from exc
    parent.mkdir(parents=True, exist_ok=True)

    repo_id = f"{repo_id_prefix}-{dataset_id}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=partial,
        robot_type=spec["robot_type"],
        fps=spec["fps"],
        features=_features(spec),
        use_videos=False,
        image_writer_threads=2,
    )
    frame_count = 0
    source_ids = []
    for path, metadata in episodes:
        count = (
            metadata["step_count"]
            if purpose == "diagnostic"
            else metadata["training_step_count"]
        )
        with _episode_data(path) as data:
            for index in range(count):
                dataset.add_frame(
                    {
                        name: data[key][index]
                        for name, key in spec["export_fields"].items()
                    },
                    task=language,
                )
        dataset.save_episode()
        frame_count += count
        source_ids.append(metadata["episode_id"])

    reopened = LeRobotDataset(repo_id, root=partial)
    if len(reopened) != frame_count:
        raise RuntimeError("LeRobot frame count changed after reopening")
    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        **expected_metadata,
        "task_language": language,
        "source_episode_ids": source_ids,
        "source_episode_paths": [str(path) for path, _ in episodes],
        "purpose": purpose,
        "source_kind": next(iter({episode_source(m) for _, m in episodes}))
        if len({episode_source(m) for _, m in episodes}) == 1
        else "mixed",
        "sources": [
            {
                "episode_id": m["episode_id"],
                "path": str(path),
                "source_kind": episode_source(m),
                "provenance_refs": m.get("provenance_refs", []),
                "information_scope": m.get("information_scope"),
                "training_admission": m.get("training_admission", "not_admitted"),
            }
            for path, m in episodes
        ],
        "training_admission": "not_admitted"
        if purpose == "diagnostic"
        else "requires_separate_demonstration_manifest",
        "episode_count": len(source_ids),
        "frame_count": frame_count,
    }
    (partial / "meta" / "rpent_flywheel.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, destination)
    return {"dataset_path": str(destination), **manifest}


@contextmanager
def _episode_data(path: Path):
    if (path / "transitions.npz").exists():
        with np.load(path / "transitions.npz", allow_pickle=False) as data:
            yield data
    else:
        yield read_episode_arrays(path)
