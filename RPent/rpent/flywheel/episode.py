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

"""Episode collection and bounded persistence using caller-supplied data rules."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _array(value: Any, field: dict[str, Any]) -> np.ndarray:
    """Copy one policy input before the caller can reuse its buffer."""
    array = np.array(value, dtype=field["dtype"], copy=True, order="C")
    if array.shape != field["shape"]:
        raise ValueError(f"expected array shape {field['shape']}; got {array.shape}")
    return array


def _observation(obs: dict[str, Any], spec: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: _array(obs[key], field)
        for key, field in spec["arrays"].items()
        if key != "actions"
    }


def _training_step_count(transitions: Any, spec: dict[str, Any]) -> int:
    mask = np.asarray(spec["success_mask"](transitions))
    if mask.dtype != np.bool_ or mask.shape != (len(transitions["actions"]),):
        raise ValueError("success_mask must return one boolean per action")
    steps = np.flatnonzero(mask)
    return int(steps[0] + 1) if steps.size else 0


class EpisodeWriter:
    """Collect one normal evaluation episode and publish it on close."""

    def __init__(
        self,
        parent: Path | str,
        *,
        metadata: dict[str, Any],
        spec: dict[str, Any],
        initial_observation: dict[str, Any],
    ) -> None:
        self._spec = spec
        first_observation = _observation(initial_observation, spec)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.episode_id = f"episode_{stamp}_{uuid.uuid4().hex[:8]}"
        parent = Path(parent).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / self.episode_id
        self._partial = self.path.with_name(f"{self.episode_id}.partial")
        self._partial.mkdir()

        self._metadata = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
        }
        self._observations = [first_observation]
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._primitive_ids: list[int] = []
        self._vla_ids: list[int] = []
        self._proposal_indices: list[int] = []
        self._primitive_names: list[str] = []
        self._active_primitive = -1
        self._proposals: list[dict[str, Any]] = []
        self._closed = False

    @property
    def step_count(self) -> int:
        return len(self._actions)

    def begin_primitive(self, name: str) -> None:
        self._active_primitive = len(self._primitive_names)
        self._primitive_names.append(name)

    def end_primitive(self) -> None:
        self._active_primitive = -1

    def add_proposal(self, instruction: str, actions: Any) -> int:
        field = self._spec["arrays"]["actions"]
        proposal = np.array(actions, dtype=field["dtype"], copy=True, order="C")
        if proposal.shape[1:] != field["shape"] or not np.isfinite(proposal).all():
            raise ValueError(
                "VLA proposal must be finite with shape (horizon, *action_shape)"
            )
        vla_id = len(self._proposals)
        self._proposals.append(
            {
                "created_step": self.step_count,
                "primitive_id": self._active_primitive,
                "instruction": instruction,
                "actions": proposal,
            }
        )
        return vla_id

    def add_transition(
        self,
        action: Any,
        next_observation: dict[str, Any],
        reward: Any,
        terminated: Any,
        truncated: Any,
        *,
        vla_id: int = -1,
        proposal_index: int = -1,
    ) -> None:
        action = _array(action, self._spec["arrays"]["actions"])
        if not np.isfinite(action).all():
            raise ValueError("action must be finite")
        self._actions.append(action)
        self._observations.append(_observation(next_observation, self._spec))
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._primitive_ids.append(self._active_primitive)
        self._vla_ids.append(vla_id)
        self._proposal_indices.append(proposal_index)

    def finalize(self) -> Path:
        if self._closed:
            return self.path
        field = self._spec["arrays"]["actions"]
        actions = (
            np.stack(self._actions)
            if self._actions
            else np.empty((0, *field["shape"]), field["dtype"])
        )
        transitions = {
            key: np.stack([obs[key] for obs in self._observations])
            for key in self._observations[0]
        }
        transitions.update(
            actions=actions,
            rewards=np.asarray(self._rewards, np.float32),
            terminated=np.asarray(self._terminated, np.bool_),
            truncated=np.asarray(self._truncated, np.bool_),
            action_source=(np.asarray(self._vla_ids, np.int32) >= 0).astype(np.uint8),
            primitive_id=np.asarray(self._primitive_ids, np.int32),
            vla_chunk_id=np.asarray(self._vla_ids, np.int32),
            proposal_index=np.asarray(self._proposal_indices, np.int16),
        )
        with (self._partial / "transitions.npz").open("wb") as stream:
            np.savez_compressed(stream, **transitions)

        proposal_actions = (
            np.stack([item["actions"] for item in self._proposals])
            if self._proposals
            else np.empty((0, 0, *field["shape"]), field["dtype"])
        )
        with (self._partial / "proposals.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                actions=proposal_actions,
                created_step=np.asarray(
                    [item["created_step"] for item in self._proposals], np.int32
                ),
                primitive_id=np.asarray(
                    [item["primitive_id"] for item in self._proposals], np.int32
                ),
                instruction=np.asarray(
                    [item["instruction"] for item in self._proposals], dtype=np.str_
                ),
            )

        training_steps = _training_step_count(transitions, self._spec)
        metadata = {
            **self._metadata,
            "is_success": bool(training_steps),
            "stop_reason": (
                "env_terminated"
                if any(self._terminated)
                else "env_truncated"
                if any(self._truncated)
                else "agent_stopped"
            ),
            "step_count": self.step_count,
            "training_step_count": training_steps,
            "primitive_names": self._primitive_names,
            "proposal_count": len(self._proposals),
        }
        (self._partial / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_episode(self._partial, spec=self._spec)
        os.replace(self._partial, self.path)
        self._closed = True
        return self.path


def validate_episode(path: Path | str, *, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the alignment needed by the exporter and training loader."""
    root = Path(path)
    metadata = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") == 2:
        count = metadata["step_count"]
        total, first_success, boundary = 0, 0, None
        for item in metadata["batches"]:
            if item["start"] != total:
                raise ValueError("noncontiguous collection batch")
            with np.load(root / item["file"], allow_pickle=False) as data:
                for key, field in spec["arrays"].items():
                    expected = (
                        item["count"] if key == "actions" else item["count"] + 1,
                        *field["shape"],
                    )
                    values = data[key]
                    if values.shape != expected or values.dtype != np.dtype(
                        field["dtype"]
                    ):
                        raise ValueError(f"invalid collected {key}")
                    if (
                        np.issubdtype(values.dtype, np.floating)
                        and not np.isfinite(values).all()
                    ):
                        raise ValueError(f"non-finite collected {key}")
                    if (
                        key != "actions"
                        and boundary is not None
                        and not np.array_equal(boundary[key], values[0])
                    ):
                        raise ValueError("batch boundary observation changed")
                boundary = {
                    key: data[key][-1] for key in spec["arrays"] if key != "actions"
                }
                training_steps = _training_step_count(data, spec)
                if not first_success and training_steps:
                    first_success = total + training_steps
            total += item["count"]
        if total != count or metadata["training_step_count"] != first_success:
            raise ValueError("collection count or success boundary mismatch")
        for proposal_id in range(metadata["proposal_count"]):
            if not (root / f"proposal-{proposal_id:05d}.npz").exists():
                raise ValueError("missing proposal artifact")
        return metadata
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema in {root}")
    with np.load(root / "transitions.npz", allow_pickle=False) as data:
        count = int(metadata["step_count"])
        for key, field in spec["arrays"].items():
            array = data[key]
            length = count if key == "actions" else count + 1
            if array.shape != (length, *field["shape"]):
                raise ValueError(f"invalid {key} shape in {root}")
            if array.dtype != np.dtype(field["dtype"]):
                raise ValueError(f"invalid {key} dtype in {root}")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"non-finite {key} in {root}")
        for key in (
            "rewards",
            "terminated",
            "truncated",
            "action_source",
            "primitive_id",
            "vla_chunk_id",
            "proposal_index",
        ):
            if data[key].shape != (count,):
                raise ValueError(f"invalid {key} shape in {root}")
        expected = _training_step_count(data, spec)
        if (
            metadata["is_success"] != bool(expected)
            or metadata["training_step_count"] != expected
        ):
            raise ValueError(f"invalid training boundary in {root}")
    proposal_count = metadata["proposal_count"]
    with np.load(root / "proposals.npz", allow_pickle=False) as proposals:
        actions = proposals["actions"]
        field = spec["arrays"]["actions"]
        if (
            actions.ndim != len(field["shape"]) + 2
            or actions.shape[0] != proposal_count
            or actions.shape[2:] != field["shape"]
        ):
            raise ValueError(f"invalid proposal action shape in {root}")
        for key in ("created_step", "primitive_id", "instruction"):
            if proposals[key].shape != (proposal_count,):
                raise ValueError(f"invalid proposal {key} shape in {root}")
    return metadata


class BatchedEpisodeWriter(EpisodeWriter):
    """Persist bounded batches and input/send journals before finalization."""

    def __init__(self, *args: Any, flush_steps: int = 25, **kwargs: Any) -> None:
        if type(flush_steps) is not int or not 1 <= flush_steps <= 100:
            raise ValueError("flush_steps must be within 1..100")
        super().__init__(*args, **kwargs)
        self._observations[0]["raw_states"] = np.array(
            kwargs["initial_observation"]["states"], copy=True
        )
        self._flush_steps = flush_steps
        self._offset = 0
        self._batches: list[dict[str, Any]] = []
        self._inputs = 0
        self._dispatches = 0
        self._pending: dict[str, Any] | None = None
        self._last_input = -1
        self._status()

    @property
    def step_count(self) -> int:
        return self._offset + len(self._actions)

    def _json(self, name: str, value: Any) -> None:
        target = self._partial / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def _event(self, kind: str, **fields: Any) -> None:
        import time

        with (self._partial / "alignment.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": kind,
                        "step": self.step_count,
                        "monotonic_seconds": time.monotonic(),
                        **fields,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()

    def _status(self) -> None:
        self._json(
            "collection.json",
            {
                **self._metadata,
                "schema_version": 2,
                "collection_version": "aligned-batches-v1",
                "status": "open",
                "batches": self._batches,
                "persisted_steps": self._offset,
                "observed_steps": self.step_count,
                "unflushed_observed_steps": len(self._actions),
                "unflushed_tail_bound": self._flush_steps - 1,
                "pending_dispatch": self._pending,
                "input_count": self._inputs,
                "primitive_names": self._primitive_names,
                "missing": [
                    "controller_applied_action",
                    "measured_actuator_motion",
                    "reset_setup_images_and_actions",
                    "nonreturned_rpc_transitions",
                ],
                "source_kind": "autonomous_execution",
                "training_admission": "not_admitted",
                "allowed_uses": ["preservation", "diagnosis"],
            },
        )

    def record_input(self, observation: dict[str, Any], **links: Any) -> int:
        """Store the exact preprocessor input, including local subgoal language."""
        input_id = self._inputs
        with (self._partial / f"input-{input_id:05d}.npz").open("wb") as stream:
            arrays = {
                key: np.array(observation[key], copy=True)
                for key in self._spec["arrays"]
                if key != "actions"
            }
            if any(
                value.shape != self._spec["arrays"][key]["shape"]
                for key, value in arrays.items()
            ):
                raise ValueError("policy input shape mismatch")
            np.savez_compressed(stream, **arrays)
        self._inputs += 1
        self._last_input = input_id
        self._event(
            "policy_input",
            input_id=input_id,
            instruction=observation.get("task_descriptions"),
            **links,
        )
        self._status()
        return input_id

    def add_proposal(self, instruction: str, actions: Any, **links: Any) -> int:
        proposal_id = super().add_proposal(instruction, actions)
        with (self._partial / f"proposal-{proposal_id:05d}.npz").open("wb") as stream:
            np.savez_compressed(stream, actions=np.array(actions, copy=True))
        self._event(
            "proposal",
            proposal_id=proposal_id,
            input_id=self._last_input,
            instruction=instruction,
            **links,
        )
        return proposal_id

    def begin_dispatch(self, actions: Any, *, source: str, **links: Any) -> int:
        dispatch = self._dispatches
        self._dispatches += 1
        self._pending = {
            "dispatch_id": dispatch,
            "source": source,
            "sent": np.asarray(actions).tolist(),
            "expected_steps": len(actions) if np.asarray(actions).ndim > 1 else 1,
            "confirmed_steps": 0,
            **links,
        }
        self._event("send", **self._pending)
        self._status()
        return dispatch

    def end_dispatch(self) -> None:
        pending = self._pending or {}
        confirmed, expected = (
            pending.get("confirmed_steps", 0),
            pending.get("expected_steps", 0),
        )
        self._event(
            "feedback_complete",
            dispatch_id=pending.get("dispatch_id"),
            execution_mask=(
                [True] * confirmed
                + [False] * max(0, pending.get("proposal_length", expected) - confirmed)
            )
            if confirmed == expected
            else None,
            missing=["controller_applied_action"]
            if confirmed == expected
            else ["execution_mapping"],
        )
        if confirmed == expected:
            self._pending = None
        self._status()

    def add_transition(
        self,
        action: Any,
        next_observation: dict[str, Any],
        reward: Any,
        terminated: Any,
        truncated: Any,
        **kwargs: Any,
    ) -> None:
        if reward is None or terminated is None or truncated is None:
            self._event(
                "unusable_feedback",
                reward=None if reward is None else float(reward),
                terminated=None if terminated is None else bool(terminated),
                truncated=None if truncated is None else bool(truncated),
                missing=[
                    key
                    for key, value in (
                        ("reward", reward),
                        ("terminated", terminated),
                        ("truncated", truncated),
                    )
                    if value is None
                ],
            )
            self._status()
            raise ValueError(
                "returned transition lacks native feedback; pending transfer retained"
            )
        super().add_transition(
            action, next_observation, reward, terminated, truncated, **kwargs
        )
        self._observations[-1]["raw_states"] = np.array(
            next_observation["states"], copy=True
        )
        if self._pending is not None:
            self._pending["confirmed_steps"] += 1
        optional = {}
        for key in (
            "joint_positions",
            "joint_velocities",
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ):
            optional[key] = (
                np.asarray(next_observation[key]).tolist()
                if next_observation.get(key) is not None
                else None
            )
        self._event(
            "transition",
            dispatch_id=(self._pending or {}).get("dispatch_id"),
            observation_index=self.step_count,
            previous_observation_index=self.step_count - 1,
            proposal_id=kwargs.get("vla_id", -1),
            proposal_index=kwargs.get("proposal_index", -1),
            execution_mask=True,
            confirmation="returned_rpc_slot; actuator application unknown",
            source="vla"
            if kwargs.get("vla_id", -1) >= 0
            else "scripted"
            if self._active_primitive >= 0
            else "unknown",
            optional_state=optional,
        )
        if len(self._actions) >= self._flush_steps:
            self.flush()

    def flush(self) -> None:
        if not self._actions:
            self._status()
            return
        count = len(self._actions)
        arrays = {
            key: np.stack([obs[key] for obs in self._observations])
            for key in self._observations[0]
        }
        arrays.update(
            actions=np.stack(self._actions),
            rewards=np.asarray(self._rewards, np.float32),
            terminated=np.asarray(self._terminated, np.bool_),
            truncated=np.asarray(self._truncated, np.bool_),
            action_source=(np.asarray(self._vla_ids, np.int32) >= 0).astype(np.uint8),
            primitive_id=np.asarray(self._primitive_ids, np.int32),
            vla_chunk_id=np.asarray(self._vla_ids, np.int32),
            proposal_index=np.asarray(self._proposal_indices, np.int16),
        )
        name = f"batch-{len(self._batches):05d}.npz"
        temporary = self._partial / (name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._partial / name)
        self._batches.append({"file": name, "start": self._offset, "count": count})
        self._offset += count
        self._observations = self._observations[-1:]
        for values in (
            self._actions,
            self._rewards,
            self._terminated,
            self._truncated,
            self._primitive_ids,
            self._vla_ids,
            self._proposal_indices,
        ):
            values.clear()
        self._status()

    def finalize(self) -> Path:
        if self._closed:
            return self.path
        self.flush()
        metadata = json.loads((self._partial / "collection.json").read_text())
        training_steps, offset, truncated = 0, 0, False
        for batch in self._batches:
            with np.load(self._partial / batch["file"], allow_pickle=False) as arrays:
                steps = _training_step_count(arrays, self._spec)
                if not training_steps and steps:
                    training_steps = offset + steps
                truncated = truncated or bool(arrays["truncated"].any())
            offset += batch["count"]
        metadata.update(
            status="closed" if self._pending is None else "incomplete",
            step_count=self.step_count,
            is_success=bool(training_steps),
            training_step_count=training_steps,
            proposal_count=len(self._proposals),
            stop_reason="env_terminated"
            if training_steps
            else "env_truncated"
            if truncated
            else "agent_stopped",
            alignment_complete=self._pending is None,
            training_admission="not_admitted",
            allowed_uses=["preservation", "diagnosis"],
        )
        self._json("episode.json", metadata)
        validate_episode(self._partial, spec=self._spec)
        os.replace(self._partial, self.path)
        self._closed = True
        return self.path


def read_episode_arrays(path: Path | str) -> dict[str, np.ndarray]:
    """Read persisted schema-2 batches or legacy schema-1 transitions."""
    root = Path(path)
    if (root / "transitions.npz").exists():
        with np.load(root / "transitions.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    metadata = json.loads((root / "collection.json").read_text())
    parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    for index, item in enumerate(metadata["batches"]):
        if item["start"] != offset:
            raise ValueError("noncontiguous collection batch")
        with np.load(root / item["file"], allow_pickle=False) as data:
            for key in data.files:
                values = data[key]
                if key in {"main_images", "wrist_images", "states", "raw_states"}:
                    if len(values) != item["count"] + 1:
                        raise ValueError("observation/action batch misalignment")
                    if index and not np.array_equal(parts[key][-1][-1], values[0]):
                        raise ValueError("batch boundary observation changed")
                    values = values[1:] if index else values
                elif len(values) != item["count"]:
                    raise ValueError("transition batch length mismatch")
                parts.setdefault(key, []).append(values)
        offset += item["count"]
    if offset != metadata["persisted_steps"]:
        raise ValueError("persisted step mismatch")
    if not parts:
        return {
            "actions": np.empty((0, 7), np.float32),
            "terminated": np.empty(0, bool),
            "truncated": np.empty(0, bool),
        }
    return {key: np.concatenate(values) for key, values in parts.items()}
