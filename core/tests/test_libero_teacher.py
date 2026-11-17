# Copyright 2026 RSI-Agent-VLA Authors.
"""CPU-only tests for isolated teacher provenance and native data export."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rsi_vla.adapters.libero_teacher import (
    export_teacher_demonstrations,
    run_teacher_script,
    validate_teacher_registration,
)


def registration(root):
    return {
        "schema": "rsi-libero-teacher-registration-v1",
        "authorized": True,
        "status": "frozen",
        "scope_id": "offline-teacher-fixture",
        "protocol_id": "fixture-teacher-v1",
        "owner_uid": os.getuid(),
        "init_splits": {
            "teacher_development": [1, 2],
            "training_demonstration": list(range(20, 30)),
            "formal": list(range(10, 16)),
        },
        "assignments": (
            [
                {
                    "run_id": f"train{seed}",
                    "seed": seed,
                    "purpose": "training_demonstration",
                }
                for seed in range(20, 30)
            ]
            + [
                {"run_id": f"dev{seed}", "seed": seed, "purpose": "teacher_development"}
                for seed in (1, 2)
            ]
        ),
        "teacher_plan": {
            "schema": "rsi-libero-pick-place-teacher-v1",
            "teacher_id": "fixture-script",
            "version": "v1",
            "scope_id": "offline-teacher-fixture",
            "protocol_id": "fixture-teacher-v1",
            "native_config_ref": str(root / "fixture-config.yaml"),
            "task": {
                "task_key": "position_swap_salad",
                "suite": "libero_object_swap",
                "task_id": 2,
            },
            "geometry_keys": [
                "salad_dressing_1_pos",
                "salad_dressing_1_quat",
                "basket_1_pos",
            ],
            "script": {
                "grasp_offset_object": [0.0, 0.0, 0.0],
                "placement_offset_world": [0.0, 0.0, 0.0],
                "approach_height": 0.15,
                "lift_height": 0.20,
                "lateral_segment": 0.2,
                "position_tolerance": 0.012,
                "grasp_pitch": 0.0,
                "grasp_yaw": 0.0,
                "stage_steps": 80,
                "close_steps": 2,
                "release_steps": 2,
            },
        },
        "quality_rules": {
            "native_success_required": True,
            "all_attempts_preserved": True,
            "no_reset": True,
            "no_model_actions": True,
            "min_lift_m": 0.05,
        },
    }


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reg = registration(self.root)
        from unittest.mock import patch

        import rsi_vla.adapters.libero_teacher as teacher

        work = Path(__file__).resolve().parents[2]
        repository = self.root / "repository"
        source_ref = self.root / "freeze/manifest.json"
        for name in teacher.TEACHER_SOURCES:
            source = work / name
            if not source.exists():
                source = Path("/home/xzh/RSI-Agent-VLA") / name
            content = source.read_bytes()
            for target in (repository / name, source_ref.parent / "source" / name):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        source_ref.write_text(
            json.dumps(
                {
                    "version": "5a-assets-offline",
                    "scope": self.reg["scope_id"],
                    "files": list(teacher.TEACHER_SOURCES),
                }
            )
        )
        self.reg.update(
            repository_root=str(repository), source_manifest_ref=str(source_ref)
        )
        module_file = patch.object(
            teacher,
            "__file__",
            str(repository / "core/src/rsi_vla/adapters/libero_teacher.py"),
        )
        module_file.start()
        self.addCleanup(module_file.stop)
        self.reg_ref = self.root / "registration.json"
        self.reg_ref.write_text(json.dumps(self.reg))

    def test_teacher_geometry_and_purpose_splits_cannot_expand_into_formal(self):
        validate_teacher_registration(self.reg)
        invalid = deepcopy(self.reg)
        invalid["teacher_plan"]["geometry_keys"].append("hidden_success_label")
        with self.assertRaisesRegex(ValueError, "three private geometry"):
            validate_teacher_registration(invalid)
        invalid = deepcopy(self.reg)
        invalid["assignments"][0]["seed"] = 10
        with self.assertRaisesRegex(ValueError, "initial-state split"):
            validate_teacher_registration(invalid)
        invalid = deepcopy(self.reg)
        invalid["init_splits"]["formal"].append(20)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_teacher_registration(invalid)

    def test_registered_script_uses_actual_primitives_and_preserves_every_step(self):
        import numpy as np

        from robots.libero.flywheel import LIBERO_SPEC
        from robots.libero.tools import LiberoPrimitives
        from rpent.flywheel.episode import EpisodeWriter
        from rpent.robots.components.action_spec import box_action_spec, validate_action
        from rpent.session.runtime import RuntimeJournal

        class FakeEnv:
            terminated = truncated = False

            def __init__(self):
                self.pos = np.array([-0.2, 0.0, 0.3], dtype=float)
                self.target = np.array([0.3, 0.0, 0.05], dtype=float)
                self.basket = np.array([-0.3, 0.0, 0.05], dtype=float)
                self.held = False
                self.steps = 0

            def observation(self):
                pixels = np.full((256, 256, 3), self.steps % 255, np.uint8)
                return {
                    "main_images": pixels,
                    "wrist_images": pixels.copy(),
                    "states": np.asarray(
                        [*self.pos, 0.0, 0.0, 0.0, 0.02, -0.02], np.float32
                    ),
                    "task_descriptions": "pick up the salad dressing and place it in the basket",
                }

            def raw_obs(self):
                return {
                    "salad_dressing_1_pos": self.target.copy(),
                    "salad_dressing_1_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                    "basket_1_pos": self.basket.copy(),
                    "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                }

            def validate_action(self, values):
                return validate_action(
                    values, box_action_spec([-1.0] * 7, [1.0] * 7, "fixture")
                )

            def step(self, action):
                if action[6] > 0 and np.linalg.norm(self.pos - self.target) < 0.03:
                    self.held = True
                self.pos += np.asarray(action[:3]) * 0.05
                if self.held:
                    self.target = self.pos.copy()
                if action[6] < 0 and self.held:
                    self.held = False
                    if np.linalg.norm(self.target - self.basket) < 0.03:
                        self.terminated = True
                self.steps += 1
                return (
                    self.observation(),
                    float(self.terminated),
                    self.terminated,
                    False,
                    {},
                )

            def rsi_step(self, action, *, evidence):
                return self.step(action)

        env = FakeEnv()
        context = {
            "schema_version": "1.0",
            "run_id": "fixture",
            "episode_id": "fixture",
            "bundle_id": "scripted_teacher:fixture",
            "protocol_id": "fixture-protocol",
            "profile_ref": str(self.reg_ref),
            "output_dir": str(self.root / "journal"),
            "limits": {
                "environment_steps": 600,
                "tool_calls": 100,
                "wall_seconds": 20,
                "model_requests": 0,
            },
        }
        journal = RuntimeJournal(context)
        primitive = LiberoPrimitives(
            env, None, None, journal.check, runtime_journal=journal
        )
        primitive.set_obs(env.observation())
        writer = EpisodeWriter(
            self.root / "raw",
            metadata={
                "task_language": "fixture",
                "source_kind": "authorized_demonstration",
                "teacher_kind": "scripted_teacher",
                "training_admission": "not_admitted",
            },
            spec=LIBERO_SPEC,
            initial_observation=env.observation(),
        )
        primitive._flywheel = writer
        result = run_teacher_script(
            primitive,
            plan=self.reg["teacher_plan"],
            quality_rules=self.reg["quality_rules"],
            geometry_log=self.root / "private.jsonl",
            tool_limit=100,
        )
        path = writer.finalize()
        self.assertTrue(result["native_success"])
        self.assertTrue(result["lift_threshold_passed"])
        self.assertEqual(writer.step_count, env.steps)
        self.assertEqual(journal.confirmed_steps, env.steps)
        self.assertEqual(journal.reserved_steps, env.steps)
        self.assertEqual(journal.tool_calls, result["tool_calls"])
        self.assertEqual(journal.active_tools, 0)
        self.assertIsNone(primitive.model)
        with np.load(path / "transitions.npz", allow_pickle=False) as data:
            self.assertEqual(len(data["states"]), len(data["actions"]) + 1)
            self.assertTrue((data["vla_chunk_id"] == -1).all())
            self.assertEqual(set(data["actions"][:, 6]), {-1.0, 1.0})

    def source_episode(self, *, purpose="training_demonstration"):
        import numpy as np

        from robots.libero.flywheel import LIBERO_SPEC
        from rpent.flywheel.episode import EpisodeWriter

        seed = 20 if purpose == "training_demonstration" else 1
        run_id = "train20" if seed == 20 else "dev1"
        quality_ref = self.root / (run_id + "-quality.json")
        obs = {
            "main_images": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(
                256, 256, 3
            ),
            "wrist_images": np.full((256, 256, 3), 10, np.uint8),
            "states": np.arange(8, dtype=np.float32) / 20,
            "task_descriptions": "pick up the salad dressing and place it in the basket",
        }
        writer = EpisodeWriter(
            self.root / run_id,
            spec=LIBERO_SPEC,
            initial_observation=obs,
            metadata={
                "source_kind": "authorized_demonstration",
                "teacher_kind": "scripted_teacher",
                "teacher_purpose": purpose,
                "scope_id": self.reg["scope_id"],
                "protocol_id": self.reg["protocol_id"],
                "teacher_id": "fixture-script",
                "teacher_version": "v1",
                "run_id": run_id,
                "operation_id": run_id,
                "seed": seed,
                "init_index": seed,
                "suite": "libero_object_swap",
                "task_id": 2,
                "registration_ref": str(self.reg_ref),
                "source_manifest_ref": self.reg["source_manifest_ref"],
                "quality_evidence_ref": str(quality_ref),
                "registration_snapshot_ref": str(self.reg_ref),
                "training_admission": "not_admitted",
                "task_language": obs["task_descriptions"],
            },
        )
        writer.begin_primitive("scripted_teacher:fixture")
        writer.add_transition(
            np.array([0.1, 0.2, 0.3, 0, 0, 0, -1], np.float32), obs, 0, False, False
        )
        writer.add_transition(
            np.array([0, 0, -0.1, 0, 0, 0, 1], np.float32), obs, 1, True, False
        )
        path = writer.finalize()
        quality_ref.write_text(
            json.dumps(
                {
                    "schema": "rsi-scripted-teacher-quality-v1",
                    "native_success": True,
                    "registered_collection_checks_passed": True,
                    "scope_id": self.reg["scope_id"],
                    "operation_id": run_id,
                    "episode_ref": str(path),
                }
            )
        )
        return path

    def test_true_exporter_and_native_batch_preserve_source_actions_images_and_identity(
        self,
    ):
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
        )
        os.environ["HF_DATASETS_CACHE"] = str(self.root / "datasets-cache")
        from rsi_vla.adapters.pi05_training import build_recipe, read_native_batch
        from rsi_vla.training import TrainingBudget

        path = self.source_episode()
        result = export_teacher_demonstrations(
            registration_ref=self.reg_ref,
            episode_paths=[str(path)],
            output_root=self.root / "export",
            dataset_id="cpu-fixture",
        )
        manifest = json.loads(Path(result["manifest_ref"]).read_text())
        self.assertEqual(manifest["source_kind"], "authorized_demonstration")
        self.assertEqual(manifest["teacher_kind"], "scripted_teacher")
        self.assertEqual(manifest["training_admission"], "not_admitted")
        audit = json.loads(Path(result["payload_audit_ref"]).read_text())
        self.assertTrue(audit["checks_passed"])
        self.assertEqual(audit["frames_checked"], 2)
        exporter = json.loads(Path(result["exporter_ref"]).read_text())
        self.assertEqual(exporter["source_kind"], "authorized_demonstration")
        model = Path(
            "/home/xzh/RSI-Agent-VLA/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
        )
        recipe = build_recipe(
            dataset_root=result["dataset_root"],
            model_path=model,
            norm_stats_path=model / "physical-intelligence/libero/norm_stats.json",
            output_root=self.root,
            run_id="cpu-fixture",
            budget=TrainingBudget(60, 60, 1, 1000000, 3),
        )
        batch = read_native_batch(recipe, manifest)
        self.assertEqual(batch["action_shape"], [1, 10, 32])

    def test_successful_development_teacher_is_not_exported_for_training(self):
        path = self.source_episode(purpose="teacher_development")
        with self.assertRaisesRegex(PermissionError, "training-purpose"):
            export_teacher_demonstrations(
                registration_ref=self.reg_ref,
                episode_paths=[str(path)],
                output_root=self.root / "forbidden",
                dataset_id="cpu-fixture",
            )
        self.assertFalse((self.root / "forbidden").exists())

    def test_source_freeze_consumes_actual_action_and_collector_bytes(self):
        from rsi_vla.adapters.libero_teacher import verify_teacher_sources

        permit = {"source_manifest_ref": self.reg["source_manifest_ref"]}
        verify_teacher_sources(self.reg, permit)
        changed = Path(self.reg["repository_root"]) / "RPent/robots/libero/tools.py"
        original = changed.read_bytes()
        changed.write_bytes(original + b"\n# changed teacher primitive fixture\n")
        with self.assertRaisesRegex(PermissionError, "source changed"):
            verify_teacher_sources(self.reg, permit)
        changed.write_bytes(original)
        source_ref = Path(self.reg["source_manifest_ref"])
        manifest = json.loads(source_ref.read_text())
        manifest["files"].remove("RPent/rpent/flywheel/episode.py")
        source_ref.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(PermissionError, "omits"):
            verify_teacher_sources(self.reg, permit)

    def test_scope_p1_expiry_and_live_runtime_reservation_are_consumed(self):
        from datetime import datetime, timedelta, timezone

        from rsi_vla.adapters.libero_teacher import TEACHER_KIND, require_teacher_permit
        from rsi_vla.midterm_host import MidtermLedger
        from rsi_vla.training import TrainingBudget

        now = datetime.now(timezone.utc)

        def write(name, value):
            path = self.root / name
            path.write_text(json.dumps(value))
            return str(path)

        limits = {
            "runtime_starts": 192,
            "gpu_seconds": 10000,
            "storage_bytes": 10000000,
            "current_disk_bytes": 10000000,
            "concurrent_gpu_devices": 2,
        }
        p1_ref = write(
            "p1.json",
            {
                "schema": "rsi-midterm-p1-acceptance-v1",
                "scope_id": self.reg["scope_id"],
                "status": "passed",
            },
        )
        scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": self.reg["scope_id"],
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "status": "active",
            "authorized": True,
            "dynamic_authorized": True,
            "outer_limits": limits,
            "accepted_at": (now - timedelta(minutes=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "p1_acceptance_ref": p1_ref,
        }
        scope_ref = write("scope.json", scope)
        phase_ref = write(
            "phase.json",
            {"limits": limits, "remaining": limits, "unresolved_costs": []},
        )
        ledger = MidtermLedger(scope_ref)
        budget = TrainingBudget(30, 30, 1, 1000000, 200)
        ledger.reserve(
            "teacher-op",
            budget={
                "runtime_starts": 1,
                "gpu_seconds": 30,
                "storage_bytes": 1000000,
                "current_disk_bytes": 1000000,
            },
            phase_ref=phase_ref,
            gpu_devices=[0],
            receipt_ref=self.root / "receipt.json",
        )
        permit = {
            "kind": TEACHER_KIND,
            "authorized": True,
            "scope_ref": scope_ref,
            "accounting_ref": str(ledger.path),
            "phase_ref": phase_ref,
            "operation_id": "teacher-op",
            "scope_id": scope["scope_id"],
            "protocol_id": self.reg["protocol_id"],
            "owner_uid": os.getuid(),
            "teacher_plan": self.reg["teacher_plan"],
            "registration_ref": str(self.reg_ref),
            "source_manifest_ref": self.reg["source_manifest_ref"],
            "output_root": str(self.root / "teacher"),
            "cuda_device": 0,
            "run_id": "train20",
            "seed": 20,
            "purpose": "training_demonstration",
            "budget": budget.__dict__,
            "tool_call_limit": 100,
            "expires_at": scope["expires_at"],
        }
        with self.assertRaisesRegex(PermissionError, "persisted"):
            require_teacher_permit(
                self.reg_ref,
                permit,
                output_root=permit["output_root"],
                cuda_device=0,
                require_submission=True,
            )
        ledger.mark_submitted("teacher-op")
        require_teacher_permit(
            self.reg_ref,
            permit,
            output_root=permit["output_root"],
            cuda_device=0,
            require_submission=True,
        )
        write(
            "p1.json",
            {
                "schema": "rsi-midterm-p1-acceptance-v1",
                "scope_id": scope["scope_id"],
                "status": "pending",
            },
        )
        with self.assertRaisesRegex(PermissionError, "P1"):
            require_teacher_permit(
                self.reg_ref, permit, output_root=permit["output_root"], cuda_device=0
            )
        write(
            "p1.json",
            {
                "schema": "rsi-midterm-p1-acceptance-v1",
                "scope_id": scope["scope_id"],
                "status": "passed",
            },
        )
        write(
            "phase.json",
            {
                "limits": limits,
                "remaining": {**limits, "gpu_seconds": 0},
                "unresolved_costs": [],
            },
        )
        with self.assertRaisesRegex(PermissionError, "phase"):
            require_teacher_permit(
                self.reg_ref, permit, output_root=permit["output_root"], cuda_device=0
            )
        permit["expires_at"] = "2020-01-01T00:00:00Z"
        with self.assertRaisesRegex(PermissionError, "expiry"):
            require_teacher_permit(
                self.reg_ref, permit, output_root=permit["output_root"], cuda_device=0
            )


if __name__ == "__main__":
    unittest.main()
