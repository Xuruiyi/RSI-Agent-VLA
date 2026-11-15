# Copyright 2026 RSI-Agent-VLA Authors.
"""Real native CPU input equivalence for public10/teacher20 index composition."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from rsi_vla.adapters.libero_mixed_data import (
    compose_index_clock_data,
    registered_mixed_plan,
)
from rsi_vla.adapters.pi05_training import ACTION_CONTRACT


class MixedClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
            HF_DATASETS_CACHE=str(self.root / "datasets-cache"),
        )
        self.scope = "offline-index-clock-fixture"

    def source(self, name, fps, count, kind, task):
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = self.root / name
        features = {
            "state": {"dtype": "float32", "shape": (8,), "names": None},
            "actions": {"dtype": "float32", "shape": (7,), "names": None},
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
        }
        writer = LeRobotDataset.create(
            "fixture/" + name,
            root=root,
            fps=fps,
            features=features,
            use_videos=False,
            robot_type="panda",
        )
        for frame in range(count):
            image = (
                np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3) + frame
            ).astype(np.uint8)
            writer.add_frame(
                {
                    "state": np.arange(8, dtype=np.float32) * 0.01 + frame * 0.01,
                    "actions": np.array(
                        [
                            frame * 0.01,
                            -frame * 0.01,
                            0.01,
                            0,
                            0,
                            0,
                            -1 if frame % 2 == 0 else 1,
                        ],
                        np.float32,
                    ),
                    "image": image,
                    "wrist_image": image[::-1].copy(),
                },
                task=task,
            )
        writer.save_episode()
        provenance = root / "origin.json"
        episode = {
            "episode_index": 0,
            "source_episode_index": 399 if fps == 10 else 20,
            "task": task,
        }
        if kind == "public_demonstration":
            episode["source_ref"] = (
                "hf://datasets/physical-intelligence/libero@"
                + ("a" * 40)
                + "/episode/399"
            )
            provenance.write_text(
                json.dumps(
                    {"repo_id": "physical-intelligence/libero", "revision": "a" * 40}
                )
            )
        else:
            rawroot = self.root / "teacher-raw"
            rawroot.mkdir()
            raw = {
                "source_kind": "authorized_demonstration",
                "teacher_kind": "scripted_teacher",
                "teacher_purpose": "training_demonstration",
                "scope_id": self.scope,
                "episode_id": "teacher-init20",
            }
            (rawroot / "episode.json").write_text(json.dumps(raw))
            provenance.write_text(
                json.dumps(
                    {"teacher_kind": "scripted_teacher", "raw_teacher_episodes": [raw]}
                )
            )
            episode.update(source_ref=str(rawroot), source_episode_id="teacher-init20")
        manifest = {
            "dataset_root": str(root),
            "source_kind": kind,
            "revision": "a" * 40 if fps == 10 else "script-v1",
            "provenance_ref": str(provenance),
            "episodes": [episode],
            "action_contract": ACTION_CONTRACT,
            "training_admission": "not_admitted",
        }
        if kind == "authorized_demonstration":
            manifest["teacher_kind"] = "scripted_teacher"
        path = root / "demonstration-manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def registration(self):
        public = self.source(
            "public10",
            10,
            12,
            "public_demonstration",
            "open the middle drawer of the cabinet",
        )
        teacher = self.source(
            "teacher20",
            20,
            7,
            "authorized_demonstration",
            "pick up the salad dressing and place it in the basket",
        )
        model = Path(
            "/home/xzh/RSI-Agent-VLA/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
        )
        return {
            "schema": "rsi-libero-index-clock-registration-v1",
            "scope_id": self.scope,
            "protocol_id": "fixture-mixed-v1",
            "derived_id": "fixture-mixed-index20-v1",
            "clock_kind": "per_episode_frame_index",
            "index_clock_fps": 20,
            "sources": [
                {"manifest_ref": str(public), "episode_indices": [0]},
                {"manifest_ref": str(teacher), "episode_indices": [0]},
            ],
            "max_frames": 50,
            "max_output_bytes": 100000000,
            "unseen_generalization_claim": False,
            "model_path": str(model),
            "norm_stats_path": str(
                model / "physical-intelligence/libero/norm_stats.json"
            ),
        }

    def test_actual_native_pairs_raw_padding_and_training_shuffle_are_exact(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        reg = self.registration()
        originals = {
            row["manifest_ref"]: Path(row["manifest_ref"]).read_bytes()
            for row in reg["sources"]
        }
        result = compose_index_clock_data(reg, self.root / "mixed")
        manifest = json.loads(Path(result["manifest_ref"]).read_text())
        proof = json.loads(Path(result["native_pair_equivalence_ref"]).read_text())
        self.assertTrue(proof["native_pair_equal"])
        self.assertEqual(proof["frames_checked"], 19)
        self.assertEqual(proof["source_fps"], [10, 20])
        self.assertFalse(proof["physical_20Hz_claim"])
        self.assertEqual(
            [ep["source_kind"] for ep in manifest["episodes"]],
            ["public_demonstration", "authorized_demonstration"],
        )
        self.assertEqual(manifest["source_kind"], "authorized_demonstration")
        self.assertEqual(manifest["training_admission"], "not_admitted")
        self.assertTrue(manifest["requires_new_data_admission"])
        dataset = LeRobotDataset(result["dataset_root"], download_videos=False)
        self.assertEqual(dataset.fps, 20)
        sample = dataset[10]
        self.assertEqual(float(sample["source_timestamp"].item()), 1.0)
        self.assertEqual(float(sample["timestamp"]), 0.5)
        self.assertEqual(int(sample["source_fps"].item()), 10)
        self.assertTrue(
            all(Path(ref).read_bytes() == value for ref, value in originals.items())
        )
        self._check_real_admission_branch(manifest, result)

    def _check_real_admission_branch(self, manifest, result):
        from unittest.mock import patch

        from rsi_vla.adapters.pi05_training import (
            dataset_identity,
            require_training_data_admission,
        )

        identity = dataset_identity(manifest)
        common = {
            "scope_id": self.scope,
            "protocol_id": "fixture-mixed-v1",
            "dataset": identity,
            "reviewer_id": "offline-reviewer",
            "decision": "accepted",
            "fixture_only": True,
            "evidence_refs": [result["native_pair_equivalence_ref"]],
        }
        quality = {
            **common,
            "schema": "rsi-demonstration-quality-review-v1",
            "image_orientation_verified": True,
            "state_action_alignment_verified": True,
            "task_relevance_verified": True,
            "license_verified": True,
        }
        isolation = {
            **common,
            "schema": "rsi-training-evaluation-isolation-v1",
            "source_episode_mapping_checked": True,
            "development_formal_split_checked": True,
            "sealed_labels_not_consumed": True,
        }
        for name, record in (("quality", quality), ("isolation", isolation)):
            (self.root / (name + "-review.json")).write_text(json.dumps(record))
        receipt = {
            "schema": "rsi-training-data-admission-v1",
            "scope_id": self.scope,
            "protocol_id": "fixture-mixed-v1",
            "reviewer_id": "offline-reviewer",
            "decision": "admitted",
            "dataset": identity,
            "original_manifest_ref": result["manifest_ref"],
            "payload_audit_ref": result["payload_audit_ref"],
            "approved_training_tasks": [ep["task"] for ep in manifest["episodes"]],
            "quality_review_ref": str(self.root / "quality-review.json"),
            "evaluation_isolation_review_ref": str(self.root / "isolation-review.json"),
        }
        receipt_ref = self.root / "admission.json"
        receipt_ref.write_text(json.dumps(receipt))
        args = {
            "admission_ref": str(receipt_ref),
            "scope_id": self.scope,
            "protocol_id": "fixture-mixed-v1",
            "authority": "offline-reviewer",
        }
        with patch(
            "rsi_vla.adapters.libero_mixed_data.verify_native_training_pairs",
            side_effect=AssertionError("must reuse valid full audit"),
        ):
            require_training_data_admission(manifest, **args)
            changed = json.loads(json.dumps(manifest))
            changed["episodes"][0]["source_kind"] = "authorized_demonstration"
            receipt["dataset"] = dataset_identity(changed)
            receipt_ref.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(PermissionError, "origin was relabelled"):
                require_training_data_admission(changed, **args)
            receipt["dataset"] = identity
            receipt_ref.write_text(json.dumps(receipt))
            proof_ref = Path(result["native_pair_equivalence_ref"])
            proof = json.loads(proof_ref.read_text())
            proof["frames_checked"] -= 1
            proof_ref.write_text(json.dumps(proof))
            with self.assertRaisesRegex(PermissionError, "full-frame coverage"):
                require_training_data_admission(manifest, **args)

    def test_autonomous_origin_cannot_be_renamed_and_video_is_not_supported(self):
        reg = self.registration()
        rawpath = self.root / "teacher-raw/episode.json"
        raw = json.loads(rawpath.read_text())
        raw["source_kind"] = "autonomous_execution"
        rawpath.write_text(json.dumps(raw))
        with self.assertRaisesRegex(PermissionError, "teacher origin"):
            registered_mixed_plan(reg)
        public_ref = reg["sources"][0]["manifest_ref"]
        public = json.loads(Path(public_ref).read_text())
        info_path = Path(public["dataset_root"]) / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"]["image"]["dtype"] = "video"
        info_path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "embedded-image"):
            registered_mixed_plan(reg)


if __name__ == "__main__":
    unittest.main()
