# Copyright 2026 RSI-Agent-VLA Authors.
"""Derived LIBERO index clocks with per-episode origin and native-pair checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..training import artifact_size
from .pi05_training import ACTION_CONTRACT, dataset_identity, validate_dataset_manifest


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write(path: str | Path, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2) + "\n")


def _episode_origin(
    manifest: dict[str, Any], episode: dict[str, Any], scope_id: str
) -> dict[str, Any]:
    kind = manifest["source_kind"]
    provenance = _read(manifest["provenance_ref"])
    if kind == "public_demonstration":
        if (
            manifest.get("source_repo_id", provenance.get("repo_id"))
            != "physical-intelligence/libero"
            or provenance.get("revision") != manifest["revision"]
            or provenance.get("source_kind") == "autonomous_execution"
        ):
            raise PermissionError(
                "public origin must remain the actual pinned public source"
            )
    elif (
        kind == "authorized_demonstration"
        and manifest.get("teacher_kind") == "scripted_teacher"
    ):
        raw_ref = Path(episode["source_ref"]) / "episode.json"
        raw = _read(raw_ref)
        if (
            provenance.get("teacher_kind") != "scripted_teacher"
            or raw.get("source_kind") != "authorized_demonstration"
            or raw.get("teacher_kind") != "scripted_teacher"
            or raw.get("teacher_purpose") != "training_demonstration"
            or raw.get("scope_id") != scope_id
            or raw.get("episode_id") != episode.get("source_episode_id")
            or not any(
                row.get("episode_id") == raw["episode_id"] and row == raw
                for row in provenance.get("raw_teacher_episodes", [])
            )
        ):
            raise PermissionError(
                "derived teacher origin differs from its real training-purpose producer"
            )
    else:
        raise PermissionError(
            "only original public or registered scripted-teacher origins may be composed"
        )
    return {
        "source_kind": kind,
        "teacher_kind": manifest.get("teacher_kind"),
        "source_revision": manifest["revision"],
        "source_provenance_ref": manifest["provenance_ref"],
        "original_episode": episode,
    }


def registered_mixed_plan(registration: dict[str, Any]) -> dict[str, Any]:
    """Read original source identities; an index clock is not physical resampling."""
    if (
        registration.get("schema") != "rsi-libero-index-clock-registration-v1"
        or registration.get("clock_kind") != "per_episode_frame_index"
        or registration.get("index_clock_fps") != 20
        or not registration.get("derived_id")
    ):
        raise ValueError("registered 20-fps index-clock derivation required")
    if (
        type(registration.get("max_frames")) is not int
        or type(registration.get("max_output_bytes")) is not int
    ):
        raise ValueError("finite integer mixed-data frame and byte caps required")
    if registration["max_frames"] <= 0 or registration["max_output_bytes"] <= 0:
        raise ValueError("positive mixed-data caps required")
    if registration.get("unseen_generalization_claim") is not False:
        raise ValueError(
            "composition does not establish unseen-task/initial-state generalization"
        )
    sources = []
    seen = set()
    total = 0
    for entry in registration["sources"]:
        if not entry["episode_indices"]:
            raise ValueError("every declared source must contribute actual episodes")
        ref = str(Path(entry["manifest_ref"]).resolve())
        manifest = _read(ref)
        validate_dataset_manifest(manifest)
        info = _read(Path(manifest["dataset_root"]) / "meta/info.json")
        if (
            info["fps"] not in {10, 20}
            or any(
                info["features"][key]["dtype"] != "image"
                for key in ("image", "wrist_image")
            )
            or any(value.get("dtype") == "video" for value in info["features"].values())
        ):
            raise ValueError(
                "equivalence is supported only for original fps10/20 embedded-image datasets"
            )
        episodes = {row["episode_index"]: row for row in manifest["episodes"]}
        metadata = {
            row["episode_index"]: row
            for row in (
                json.loads(line)
                for line in (Path(manifest["dataset_root"]) / "meta/episodes.jsonl")
                .read_text()
                .splitlines()
                if line
            )
        }
        selected = []
        for index in entry["episode_indices"]:
            if (ref, index) in seen or index not in episodes:
                raise ValueError("unique registered source episodes required")
            seen.add((ref, index))
            origin = _episode_origin(
                manifest, episodes[index], registration["scope_id"]
            )
            selected.append(
                {
                    "episode_index": index,
                    "frames": metadata[index]["length"],
                    "task": episodes[index]["task"],
                    "origin": origin,
                }
            )
            total += metadata[index]["length"]
        sources.append(
            {
                "manifest_ref": ref,
                "dataset_root": manifest["dataset_root"],
                "source_fps": info["fps"],
                "episodes": selected,
            }
        )
    if not sources or total <= 0 or total > registration["max_frames"]:
        raise ValueError("registered mixed source frame cap exceeded or empty")
    return {
        "schema": "rsi-libero-index-clock-plan-v1",
        "derived_id": registration["derived_id"],
        "index_clock_fps": 20,
        "clock_kind": "per_episode_frame_index",
        "sources": sources,
        "total_frames": total,
        "total_episodes": len(seen),
        "physical_retiming": False,
        "unseen_generalization_claim": False,
        "public_original_init_overlap": "unknown",
        "frame_insertion_or_deletion": False,
        "training_admission": "not_admitted",
    }


def compose_index_clock_data(
    registration: dict[str, Any], output: str | Path
) -> dict[str, Any]:
    """Compose lossless source rows and verify the real native training inputs."""
    import os

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("mixed composition and equivalence checks are local CPU only")
    plan = registered_mixed_plan(registration)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("derived dataset identity already exists")
    first = _read(Path(plan["sources"][0]["dataset_root"]) / "meta/info.json")
    features = {
        key: {**first["features"][key], "shape": tuple(first["features"][key]["shape"])}
        for key in ("image", "wrist_image", "state", "actions")
    }
    extra = {
        "source_timestamp": "float64",
        "source_fps": "int64",
        "source_episode_index": "int64",
        "source_global_index": "int64",
        "source_dataset_index": "int64",
    }
    features.update(
        {
            key: {"dtype": dtype, "shape": (1,), "names": None}
            for key, dtype in extra.items()
        }
    )
    writer = LeRobotDataset.create(
        "rsi-local/libero-index-clock",
        root=output,
        fps=20,
        features=features,
        robot_type=first.get("robot_type"),
        use_videos=False,
    )
    mapping = []
    for source_index, source in enumerate(plan["sources"]):
        selected = {row["episode_index"]: row for row in source["episodes"]}
        reader = LeRobotDataset(
            source["dataset_root"],
            episodes=list(selected),
            download_videos=False,
            video_backend="pyav",
        )
        previous = None
        for index in range(len(reader)):
            sample = reader[index]
            original = int(sample["episode_index"])
            if previous is not None and original != previous:
                writer.save_episode()
            if original != previous:
                ep = selected[original]
                mapping.append(
                    {
                        "episode_index": len(mapping),
                        "task": ep["task"],
                        "source_episode_index": original,
                        "source_ref": source["manifest_ref"],
                        "source_manifest_ref": source["manifest_ref"],
                        "source_dataset_index": source_index,
                        "source_fps": source["source_fps"],
                        "source_kind": ep["origin"]["source_kind"],
                        "teacher_kind": ep["origin"].get("teacher_kind"),
                        "origin": ep["origin"],
                    }
                )
            frame = {}
            for key in ("image", "wrist_image", "state", "actions"):
                value = np.asarray(sample[key])
                if key in ("image", "wrist_image"):
                    if value.shape[0] == 3:
                        value = value.transpose(1, 2, 0)
                    value = np.rint(value * 255).astype(np.uint8)
                frame[key] = value
            values = {
                "source_timestamp": float(sample["timestamp"]),
                "source_fps": source["source_fps"],
                "source_episode_index": original,
                "source_global_index": int(sample["index"]),
                "source_dataset_index": source_index,
            }
            frame.update(
                {
                    key: np.asarray([value], dtype=features[key]["dtype"])
                    for key, value in values.items()
                }
            )
            writer.add_frame(frame, task=sample["task"])
            if artifact_size(output) > registration["max_output_bytes"]:
                raise ValueError("derived dataset artifact cap exceeded")
            previous = original
        writer.save_episode()
    _write(output / "derivation.json", plan)
    manifest = {
        "source_kind": "authorized_demonstration",
        "derivation_kind": "index_clock_composition",
        "revision": registration["derived_id"],
        "dataset_root": str(output),
        "provenance_ref": str(output / "derivation.json"),
        "episodes": mapping,
        "action_contract": ACTION_CONTRACT,
        "training_admission": "not_admitted",
        "requires_new_data_admission": True,
        "index_clock_fps": 20,
        "physical_retiming": False,
        "unseen_generalization_claim": False,
        "public_original_init_overlap": "unknown",
        "source_fps_and_timestamps": "preserved per frame; fps20 is the derived index clock, not public physical cadence",
        "scope_id": registration["scope_id"],
        "protocol_id": registration["protocol_id"],
        "native_pair_equivalence_ref": str(output / "native-pair-equivalence.json"),
    }
    validate_dataset_manifest(manifest)
    _write(output / "demonstration-manifest.json", manifest)
    proof = verify_native_training_pairs(manifest, registration)
    _write(output / "native-pair-equivalence.json", proof)
    audit = {
        "schema": "rsi-demonstration-payload-audit-v1",
        "dataset": dataset_identity(manifest),
        "frames_checked": proof["frames_checked"],
        "episodes": proof["episode_checks"],
        "checks_passed": True,
        "training_admission": "pending_independent_quality_and_isolation_review",
    }
    _write(output / "payload-audit.json", audit)
    validate_mixed_manifest(manifest, scope_id=registration["scope_id"])
    return {
        "dataset_root": str(output),
        "manifest_ref": str(output / "demonstration-manifest.json"),
        "payload_audit_ref": str(output / "payload-audit.json"),
        "native_pair_equivalence_ref": str(output / "native-pair-equivalence.json"),
        "training_admission": "not_admitted",
    }


def verify_native_training_pairs(
    manifest: dict[str, Any], registration: dict[str, Any]
) -> dict[str, Any]:
    """Check each native raw horizon/padding and all normalized model inputs."""
    import jax
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openpi.models.model import Observation
    from openpi.training.data_loader import (
        TorchDataLoader,
        create_torch_dataset,
        transform_dataset,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from torch.utils.data import Dataset

    from .pi05_training import inspect_native_config, verify_installed_sources

    plan = registered_mixed_plan(registration)
    inspect_native_config(
        {
            "actor": {
                "model": {
                    "model_path": registration["model_path"],
                    "openpi_data": {"norm_stats_path": registration["norm_stats_path"]},
                }
            },
            "data": {"train_data_paths": manifest["dataset_root"]},
        }
    )

    def native(root):
        cfg = get_openpi_config(
            "pi05_libero",
            model_path=registration["model_path"],
            batch_size=1,
            repo_id=root,
            data_kwargs={"norm_stats_path": registration["norm_stats_path"]},
        )
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        return create_torch_dataset(data, 10, cfg.model), data

    derived_prompt, derived_cfg = native(manifest["dataset_root"])
    derived = transform_dataset(derived_prompt, derived_cfg)
    derived_raw = LeRobotDataset(
        manifest["dataset_root"],
        delta_timestamps={"actions": [j / 20 for j in range(10)]},
        download_videos=False,
    )
    originals = []
    transformed = []
    raw_readers = []
    for source in plan["sources"]:
        dataset, cfg = native(source["dataset_root"])
        originals.append(dataset)
        transformed.append(transform_dataset(dataset, cfg))
        raw_readers.append(
            LeRobotDataset(
                source["dataset_root"],
                delta_timestamps={
                    "actions": [j / source["source_fps"] for j in range(10)]
                },
                download_videos=False,
            )
        )

    def equal(left, right):
        left_leaves, left_tree = jax.tree.flatten(left)
        right_leaves, right_tree = jax.tree.flatten(right)
        return left_tree == right_tree and all(
            np.asarray(a).dtype == np.asarray(b).dtype
            and np.asarray(a).shape == np.asarray(b).shape
            and np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(left_leaves, right_leaves)
        )

    expected = []
    per_episode = {row["episode_index"]: 0 for row in manifest["episodes"]}
    for index in range(len(derived)):
        item = derived_raw[index]
        source_index = int(item["source_dataset_index"].item())
        source_row = int(item["source_global_index"].item())
        source = raw_readers[source_index][source_row]
        if (
            not np.isfinite(np.asarray(source["state"])).all()
            or not np.isfinite(np.asarray(source["actions"])).all()
        ):
            raise ValueError(
                "nonfinite source state/action cannot qualify a derived payload audit"
            )
        if (
            not equal(item["actions"], source["actions"])
            or not equal(item["actions_is_pad"], source["actions_is_pad"])
            or not equal(item["state"], source["state"])
            or not equal(item["image"], source["image"])
            or not equal(item["wrist_image"], source["wrist_image"])
            or float(item["source_timestamp"].item()) != float(source["timestamp"])
            or int(item["source_fps"].item())
            != plan["sources"][source_index]["source_fps"]
            or int(item["source_episode_index"].item()) != int(source["episode_index"])
        ):
            raise ValueError(
                "derived raw observation/action horizon/padding/origin differs from source"
            )
        original = transformed[source_index][source_row]
        if not equal(derived[index], original):
            raise ValueError(
                "native normalized images/state/prompt/masks/future action training pair changed"
            )
        expected.append((source_index, source_row))
        per_episode[int(item["episode_index"])] += 1

    class SourceUnion(Dataset):
        def __len__(self):
            return len(expected)

        def __getitem__(self, index):
            source_index, source_row = expected[index]
            return transformed[source_index][source_row]

    reference = iter(
        TorchDataLoader(
            SourceUnion(),
            1,
            shuffle=True,
            num_batches=len(expected),
            seed=0,
            num_workers=0,
            framework="pytorch",
        )
    )
    candidate = iter(
        TorchDataLoader(
            derived,
            1,
            shuffle=True,
            num_batches=len(expected),
            seed=0,
            num_workers=0,
            framework="pytorch",
        )
    )
    for _ in range(len(expected)):
        before, after = next(reference), next(candidate)
        if not equal(before, after):
            raise ValueError(
                "same-seed actual native training shuffle/collate inputs changed"
            )
        if not equal(
            Observation.from_dict(before).to_dict(),
            Observation.from_dict(after).to_dict(),
        ):
            raise ValueError("actual native Observation construction changed")
    return {
        "schema": "rsi-libero-native-pair-equivalence-v1",
        "dataset": dataset_identity(manifest),
        "frames_checked": len(expected),
        "episode_checks": [
            {
                "episode_index": index,
                "frames_checked": count,
                "checks": {
                    "raw_payload_source_clock_and_future_horizon_padding_equal": True,
                    "native_normalized_training_pair_equal": True,
                    "same_seed_training_shuffle_collation_equal": True,
                },
            }
            for index, count in per_episode.items()
        ],
        "native_pair_equal": True,
        "exact_dtypes_and_shapes_compared": True,
        "index_clock_fps": 20,
        "source_fps": [source["source_fps"] for source in plan["sources"]],
        "physical_20Hz_claim": False,
        "action_horizon": 10,
        "model_action_dim": 32,
        "model_config": "pi05_libero",
        "native_installed_sources": verify_installed_sources(),
        "norm_stats_path": registration["norm_stats_path"],
        "models_loaded": 0,
        "training_steps": 0,
        "consumer_scope": "current embedded-image native dataset -> repack/normalize/tokenize/pad -> same-seed TorchDataLoader; no fps/timestamp supplied to SFT",
    }


def validate_mixed_manifest(manifest: dict[str, Any], *, scope_id: str) -> None:
    """Verify origin and full receipt coverage at actual data admission, without rerunning the audit."""
    if (
        manifest.get("derivation_kind") != "index_clock_composition"
        or manifest.get("source_kind") != "authorized_demonstration"
        or manifest.get("scope_id") != scope_id
        or manifest.get("requires_new_data_admission") is not True
        or manifest.get("index_clock_fps") != 20
        or manifest.get("physical_retiming") is not False
        or manifest.get("unseen_generalization_claim") is not False
    ):
        raise PermissionError("mixed dataset derivation/scope/clock binding mismatch")
    validate_dataset_manifest(manifest)
    root = Path(manifest["dataset_root"])
    info = _read(root / "meta/info.json")
    if info["fps"] != 20 or any(
        key not in info["features"]
        for key in (
            "source_timestamp",
            "source_fps",
            "source_episode_index",
            "source_global_index",
            "source_dataset_index",
        )
    ):
        raise PermissionError(
            "mixed dataset does not retain its source clock and episode indices"
        )
    plan = _read(manifest["provenance_ref"])
    if (
        plan.get("schema") != "rsi-libero-index-clock-plan-v1"
        or plan.get("derived_id") != manifest["revision"]
        or plan.get("index_clock_fps") != 20
        or plan.get("physical_retiming") is not False
        or plan.get("frame_insertion_or_deletion") is not False
        or plan.get("total_frames") != info["total_frames"]
        or plan.get("total_episodes") != info["total_episodes"]
    ):
        raise PermissionError(
            "mixed dataset origin plan does not match current native metadata"
        )
    expected = []
    for source_index, source in enumerate(plan["sources"]):
        original = _read(source["manifest_ref"])
        validate_dataset_manifest(original)
        original_info = _read(Path(original["dataset_root"]) / "meta/info.json")
        originals = {row["episode_index"]: row for row in original["episodes"]}
        if (
            original["dataset_root"] != source["dataset_root"]
            or original_info["fps"] != source["source_fps"]
        ):
            raise PermissionError("mixed dataset original source clock/root changed")
        for episode in source["episodes"]:
            origin = _episode_origin(
                original, originals[episode["episode_index"]], scope_id
            )
            if origin != episode["origin"]:
                raise PermissionError(
                    "mixed dataset original public/teacher provenance changed"
                )
            expected.append(
                {
                    "source_manifest_ref": source["manifest_ref"],
                    "source_dataset_index": source_index,
                    "source_fps": source["source_fps"],
                    "source_episode_index": episode["episode_index"],
                    "source_kind": origin["source_kind"],
                    "teacher_kind": origin["teacher_kind"],
                    "origin": origin,
                }
            )
    if len(expected) != len(manifest["episodes"]):
        raise PermissionError(
            "mixed origin coverage does not account for every consumed episode"
        )
    for episode, origin in zip(manifest["episodes"], expected):
        if any(episode.get(key) != value for key, value in origin.items()):
            raise PermissionError("mixed episode origin was relabelled or changed")
    identity = dataset_identity(manifest)
    metadata = {
        row["episode_index"]: row
        for row in (
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line
        )
    }
    proof = _read(manifest["native_pair_equivalence_ref"])
    audit = _read(root / "payload-audit.json")
    if (
        proof.get("schema") != "rsi-libero-native-pair-equivalence-v1"
        or proof.get("dataset") != identity
        or proof.get("native_pair_equal") is not True
        or proof.get("exact_dtypes_and_shapes_compared") is not True
        or proof.get("physical_20Hz_claim") is not False
        or proof.get("action_horizon") != 10
        or proof.get("model_action_dim") != 32
        or proof.get("model_config") != "pi05_libero"
        or audit.get("schema") != "rsi-demonstration-payload-audit-v1"
        or audit.get("dataset") != identity
        or audit.get("checks_passed") is not True
    ):
        raise PermissionError("mixed native-pair/full-payload receipt binding mismatch")
    for record, key in ((proof, "episode_checks"), (audit, "episodes")):
        rows = record.get(key, [])
        if (
            record.get("frames_checked") != info["total_frames"]
            or len(rows) != len(metadata)
            or {row["episode_index"] for row in rows} != set(metadata)
            or any(
                row["frames_checked"] != metadata[row["episode_index"]]["length"]
                or not row.get("checks")
                or not all(value is True for value in row["checks"].values())
                for row in rows
            )
        ):
            raise PermissionError(
                "mixed native-pair/payload receipt full-frame coverage incomplete"
            )
