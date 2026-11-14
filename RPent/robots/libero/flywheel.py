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

"""LIBERO data rules and paths for the shared Flywheel implementation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rpent.flywheel.episode import BatchedEpisodeWriter, EpisodeWriter

_SUITE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

LIBERO_ARRAYS = {
    "main_images": {"shape": (256, 256, 3), "dtype": "uint8"},
    "wrist_images": {"shape": (256, 256, 3), "dtype": "uint8"},
    "states": {"shape": (8,), "dtype": "float32"},
    "actions": {"shape": (7,), "dtype": "float32"},
}
LIBERO_EXPORT_FIELDS = {
    "image": "main_images",
    "wrist_image": "wrist_images",
    "state": "states",
    "actions": "actions",
}


def success_mask(transitions: Any) -> Any:
    """LIBERO uses environment termination to signal task success."""
    return transitions["terminated"]


LIBERO_SPEC = {
    "arrays": LIBERO_ARRAYS,
    "export_fields": LIBERO_EXPORT_FIELDS,
    "image_fields": ("main_images", "wrist_images"),
    "fps": 20,
    "robot_type": "panda",
    "success_mask": success_mask,
}


def _task_root(root: Path | str, suite: str, task_id: int) -> Path:
    if not _SUITE.fullmatch(suite):
        raise ValueError(f"invalid LIBERO suite: {suite!r}")
    if type(task_id) is not int or task_id < 0:
        raise ValueError("task_id must be a non-negative integer")
    return (
        Path(root).expanduser().resolve()
        / "raw"
        / "libero"
        / suite
        / f"task_{task_id:02d}"
    )


def create_episode_writer(config: dict[str, Any], obs: dict[str, Any]) -> EpisodeWriter:
    """Create a writer using the current LIBERO task and policy observation."""
    parent = _task_root(config["root"], config["suite"], config["task_id"])
    seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    language = obs.get("task_descriptions")
    if not isinstance(language, str) or not language:
        raise ValueError("LIBERO observation has no task description")
    bounded = config.get("collection_version") == "aligned-batches-v1"
    writer = BatchedEpisodeWriter if bounded else EpisodeWriter
    return writer(
        parent / f"seed_{seed:03d}",
        metadata={
            "suite": config["suite"],
            "task_id": config["task_id"],
            "seed": seed,
            "task_language": language,
            **config.get("identity", {}),
            "runtime_identity": config.get("identity", {}),
            "native_action_spec": config.get("native_action_spec"),
            "collection_config": {**config, "root": str(config["root"])},
            "information_scope": config.get("information_scope", "unspecified"),
            "action_contract": {
                "images": "RGB uint8 256x256; native agentview/wrist rotated 180deg; policy orientation",
                "state": "EEF xyz metres, axis-angle radians, two gripper qpos metres; float32[8]",
                "actions": "native OSC_POSE normalized xyz/axis-angle delta; gripper -1 open +1 close; float32[7]",
                "frame": "reference frame and scaling are in native_action_spec; no extra delta transform; not joint commands",
                "frequency": "nominal 20Hz, not measured; indices are returned control slots, wall clock separate",
                "exact_state": "input files and raw_states retain original dtype; states is float32 export view",
                "model_input": "exact predict() input before installed model transforms; processor bound by bundle",
            },
        },
        initial_observation=obs,
        spec=LIBERO_SPEC,
        **({"flush_steps": config.get("flush_steps", 25)} if bounded else {}),
    )


def export_options(
    data_root: Path | str,
    *,
    suite: str,
    task_id: int,
    output_root: Path | str | None = None,
) -> dict[str, Any]:
    """Select a LIBERO task without changing its raw or exported directory layout."""
    root = Path(data_root).expanduser().resolve()
    task_root = _task_root(root, suite, task_id)
    return {
        "episode_paths": sorted(task_root.glob("seed_*/episode_*")),
        "expected_metadata": {"suite": suite, "task_id": task_id},
        "repo_id_prefix": f"rpent/{suite}-task-{task_id:02d}",
        "output_root": (
            output_root
            if output_root is not None
            else root / "datasets" / "lerobot" / suite / f"task_{task_id:02d}"
        ),
    }
