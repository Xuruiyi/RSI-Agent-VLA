# Copyright 2026 RSI-Agent-VLA contributors.

"""Real temporary-file retention, native persistence and exact policy inputs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rsi_vla.adapters.runtime_acceptance import audit_native_persistence
from rsi_vla.artifact_retention import (
    apply_retention,
    plan_retention,
    policy_unavailable_media,
)
from rsi_vla.process_accounting import accounted_storage_bytes
from rsi_vla.storage import atomic_json

from robots.libero import execution_arms
from robots.libero.flywheel import create_episode_writer

REPO = Path(execution_arms.__file__).resolve().parents[3]


class ArtifactRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "runtime" / "future-D"
        self.output.mkdir(parents=True)
        self.identity = {
            "run_id": "future-D",
            "episode_id": "task-s1",
            "bundle_id": "D",
            "protocol_id": "registered400",
        }
        now = datetime.now(timezone.utc)
        self.scope_ref = self.root / "scope.json"
        atomic_json(
            self.scope_ref,
            {
                "schema": "rsi-midterm-scope-v1",
                "scope_id": "midterm-test",
                "authorized": True,
                "owner_uid": os.getuid(),
                "status": "active",
                "write_root": str(self.root),
                "accepted_at": (now - timedelta(hours=1)).isoformat(),
                "expires_at": (now + timedelta(days=1)).isoformat(),
            },
        )
        self.audit_ref = self.root / "release" / "audit.json"
        self.policy_ref = self.root / "operator" / "retention-policy.json"
        self.policy = {
            "schema": "rsi-artifact-retention-policy-v1",
            "policy_id": "future-policy",
            "scope_id": "midterm-test",
            "scope_authorization_ref": str(self.scope_ref),
            "owner_uid": os.getuid(),
            "registered_at": (now - timedelta(minutes=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "runtime_root": str(self.root / "runtime"),
            "receipt_root": str(self.root / "release" / "retention"),
            "reviewed_source_root": str(REPO),
            "allowed_artifacts": ["agentview_world_high.npz", "wrist_world_high.npz"],
            "runs": {
                "future-D": {
                    **{k: v for k, v in self.identity.items() if k != "run_id"},
                    "output_dir": str(self.output),
                    "audit_ref": str(self.audit_ref),
                }
            },
        }
        self.permit = self.root / "operator" / "permit.json"
        atomic_json(
            self.permit,
            {
                **{k: v for k, v in self.identity.items() if k != "episode_id"},
                "scope": "midterm-test",
                "authorized": True,
                "approval_reference": str(self.scope_ref),
            },
        )
        self.spec = {"schema": "libero-execution-arm-v1", "arm": "D", "max_chunks": 2}
        self.context = {
            **self.identity,
            "authorization_ref": str(self.permit),
            "execution_arm": self.spec,
            "planner": {"backend": "flash"},
        }
        atomic_json(self.output / "runtime_context.json", self.context)
        rows = [
            {
                "kind": "run_start",
                "utc_ns": int((now - timedelta(seconds=30)).timestamp() * 1e9),
            },
            {"kind": "episode_execution_start", "execution_arm": self.spec},
            {"kind": "tool_call", "tool": "vla_execute"},
            {"kind": "inference_start"},
            {"kind": "tool_end", "tool": "vla_execute", "status": "completed"},
            {"kind": "tool_call", "tool": "vla_execute"},
            {"kind": "inference_start"},
            {"kind": "tool_end", "tool": "vla_execute", "status": "completed"},
            {"kind": "native_completion", "step": 2, "reason": "native_success"},
            {"kind": "episode_execution_end", "spec": self.spec},
        ]
        self.events = [
            {**self.identity, **row, "sequence": i} for i, row in enumerate(rows)
        ]
        (self.output / "action_events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.events)
        )
        self.names = [
            "agentview_policy.png",
            "agentview.png",
            "wrist.png",
            "agentview_depth.npz",
            "wrist_depth.npz",
            "agentview_metadata.json",
            "wrist_metadata.json",
            "agentview_world_high.npz",
            "wrist_world_high.npz",
            "agentview_high.png",
            "wrist_high.png",
        ]
        states = []
        for step in range(3):
            for name in self.names:
                path = self.output / name / f"{step:02d}{Path(name).suffix}"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x" * (10000 if "world_high" in name else 100))
            states.append(
                {
                    "step_idx": step,
                    "terminated": step == 2,
                    "truncated": False,
                    "state": {"eef": [0, 0, 0]},
                    "artifacts": self.names,
                }
            )
        atomic_json(self.output / "states.json", {"steps": states})
        atomic_json(
            self.output / "native_completion.json",
            {
                "identity": self.identity,
                "step": 2,
                "sequence": 8,
                "reason": "native_success",
            },
        )
        atomic_json(
            self.output / "supervisor.json",
            {
                "identity": self.identity,
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
                "storage_bytes": 500000,
                "storage_output_bytes_at_sample": 0,
            },
        )
        atomic_json(
            self.output / "environment_service/runtime_costs.json",
            {
                "identity": self.identity,
                "service_closed": True,
                "pending_action": False,
                "environment_steps_unknown": 0,
            },
        )
        atomic_json(
            self.output / "independent_result.json",
            {
                **self.identity,
                "attempt_closed": True,
                "benchmark_success": True,
                "validity": "valid",
                "outcome": "success",
            },
        )
        for name in ("execution_arms.py", "toolkit.py", "tools.py", "robot_spec.py"):
            relative = Path("RPent/robots/libero") / name
            destination = self.output / "code_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, destination)
        obs = {
            "main_images": np.zeros((256, 256, 3), np.uint8),
            "wrist_images": np.ones((256, 256, 3), np.uint8),
            "states": np.zeros(8, np.float32),
            "task_descriptions": "native task",
        }
        config = {
            "root": str(self.output / "data"),
            "suite": "libero_object",
            "task_id": 2,
            "seed": 1,
            "collection_version": "aligned-batches-v1",
            "flush_steps": 5,
            "identity": self.identity,
            "information_scope": "development",
        }
        writer = create_episode_writer(config, obs)
        writer.begin_primitive("vla_execute")
        for i in range(2):
            writer.record_input(obs)
            actions = np.zeros((5, 7), np.float32)
            proposal = writer.add_proposal("native task", actions)
            writer.begin_dispatch(
                actions, source="vla", proposal_id=proposal, proposal_length=5
            )
            for j in range(5):
                writer.add_transition(
                    actions[j],
                    obs,
                    0,
                    i == 1 and j == 4,
                    False,
                    vla_id=proposal,
                    proposal_index=j,
                )
            writer.end_dispatch()
        self.episode = writer.finalize()
        atomic_json(
            self.output / "dataset_manifest.json",
            {
                "identity": self.identity,
                "alignment_complete": True,
                "episodes": [{"path": str(self.episode)}],
            },
        )
        atomic_json(
            self.audit_ref,
            {
                "identity": self.identity,
                "passed": True,
                "outcome": "success",
                "checks": {"clean_completion": True, "execution_arm_consumption": True},
                "costs": {"artifact_bytes": accounted_storage_bytes(self.output)},
            },
        )
        atomic_json(self.policy_ref, self.policy)

    def raw_snapshot(self):
        return {
            str(path.relative_to(self.output)): path.read_bytes()
            for path in self.output.rglob("*")
            if path.is_file()
            and ("world_high.npz/" not in str(path))
            and path.name != "artifact_retention.json"
        }

    def test_default_cleanup_preserves_raw_facts_final_artifacts_and_cumulative_cost(
        self,
    ):
        before = self.raw_snapshot()
        cost = accounted_storage_bytes(self.output)
        plan = plan_retention(self.policy_ref, "future-D")
        self.assertEqual(len(plan["files"]), 4)
        self.assertFalse((self.output / "artifact_retention.json").exists())
        receipt = apply_retention(self.policy_ref, "future-D")
        self.assertEqual(before, self.raw_snapshot())
        self.assertEqual(receipt["status"], "completed")
        self.assertGreaterEqual(accounted_storage_bytes(self.output), cost)
        self.assertEqual(
            audit_native_persistence(self.output, self.identity, self.events)["issues"],
            [],
        )
        self.assertEqual(apply_retention(self.policy_ref, "future-D"), receipt)
        before_late = accounted_storage_bytes(self.output)
        (self.output / "late-evaluation.json").write_bytes(b"x" * 123)
        self.assertEqual(accounted_storage_bytes(self.output), before_late + 123)
        self.assertEqual(len(policy_unavailable_media(self.output)["paths"]), 4)

    def test_original_600_attempt_and_late_policy_are_rejected_without_deletion(self):
        with self.assertRaisesRegex(PermissionError, "not registered"):
            apply_retention(self.policy_ref, "midterm16-dev-D-position_swap_salad-s2")
        self.policy["registered_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.policy_ref, self.policy)
        with self.assertRaisesRegex(PermissionError, "precede"):
            apply_retention(self.policy_ref, "future-D")
        self.assertTrue((self.output / "agentview_world_high.npz/00.npz").exists())

    def test_A_unknown_audit_and_live_process_each_refuse_cleanup(self):
        self.context["execution_arm"]["arm"] = "A"
        atomic_json(self.output / "runtime_context.json", self.context)
        with self.assertRaisesRegex(PermissionError, "frozen VLA"):
            apply_retention(self.policy_ref, "future-D")
        self.context["execution_arm"]["arm"] = "D"
        atomic_json(self.output / "runtime_context.json", self.context)
        audit = json.loads(self.audit_ref.read_text())
        audit["passed"] = False
        atomic_json(self.audit_ref, audit)
        with self.assertRaisesRegex(PermissionError, "independent audit"):
            apply_retention(self.policy_ref, "future-D")
        audit["passed"] = True
        atomic_json(self.audit_ref, audit)
        supervisor = json.loads((self.output / "supervisor.json").read_text())
        supervisor["cleanup_confirmed"] = False
        atomic_json(self.output / "supervisor.json", supervisor)
        with self.assertRaisesRegex(PermissionError, "process closure"):
            apply_retention(self.policy_ref, "future-D")

    def test_rgb_requires_explicit_registration_and_raw_policy_inputs(self):
        self.assertFalse(
            any(
                item["artifact"].endswith("_high.png")
                for item in plan_retention(self.policy_ref, "future-D")["files"]
            )
        )
        self.policy["allowed_artifacts"] += ["agentview_high.png", "wrist_high.png"]
        atomic_json(self.policy_ref, self.policy)
        plan = plan_retention(self.policy_ref, "future-D")
        self.assertEqual(len(plan["files"]), 8)
        saved = (self.episode / "input-00000.npz").read_bytes()
        (self.episode / "input-00000.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            apply_retention(self.policy_ref, "future-D")
        (self.episode / "input-00000.npz").write_bytes(saved)
        apply_retention(self.policy_ref, "future-D")
        self.assertFalse((self.output / "agentview_high.png/00.png").exists())
        self.assertTrue((self.output / "agentview_high.png/02.png").exists())
        self.assertTrue((self.output / "agentview_policy.png/00.png").exists())

    def test_missing_final_artifact_and_forged_floor_cannot_pass(self):
        final = self.output / "agentview_world_high.npz/02.npz"
        final.unlink()
        with self.assertRaisesRegex(PermissionError, "protected-state"):
            apply_retention(self.policy_ref, "future-D")
        self.assertTrue(
            audit_native_persistence(self.output, self.identity, self.events)["issues"]
        )
        atomic_json(
            self.output / "artifact_retention.json",
            {
                "policy_ref": str(self.policy_ref),
                "run_id": "future-D",
                "receipt_ref": str(self.output / "fake.json"),
            },
        )
        atomic_json(self.output / "fake.json", {"cumulative_bytes_floor": 99999999})
        with self.assertRaises(PermissionError):
            accounted_storage_bytes(self.output)

    def test_read_trajectory_keeps_media_missing_and_adds_policy_unavailable(self):
        apply_retention(self.policy_ref, "future-D")
        transcript = self.output / "transcript.json"
        atomic_json(transcript, {"runtime": {"identity": self.identity}})
        from rsi_vla.adapters.rpent import read_trajectory

        events = read_trajectory(
            self.output / "states.json", transcript, **self.identity
        )
        self.assertTrue(
            any(item.startswith("media_missing:") for item in events[0].missing)
        )
        self.assertEqual(
            events[0].flags["artifact_retention"]["status"], "policy-unavailable"
        )
        self.assertEqual(len(events[0].flags["artifact_retention"]["paths"]), 2)
        self.assertFalse(any("world_high" in path for path in events[0].media))

    def test_interrupted_deletion_reconciles_receipt_without_refunding(self):
        original = Path.unlink
        calls = []

        def interrupt(path, *args, **kwargs):
            if "world_high.npz" in str(path):
                calls.append(str(path))
                if len(calls) == 2:
                    raise OSError("controlled cleanup interruption")
            return original(path, *args, **kwargs)

        before = accounted_storage_bytes(self.output)
        with patch.object(Path, "unlink", interrupt):
            with self.assertRaisesRegex(OSError, "controlled cleanup"):
                apply_retention(self.policy_ref, "future-D")
        self.assertGreaterEqual(accounted_storage_bytes(self.output), before)
        receipt = apply_retention(self.policy_ref, "future-D")
        self.assertEqual(len(receipt["removed_paths"]), 4)
        self.assertEqual(len(set(receipt["removed_paths"])), 4)
        self.assertGreaterEqual(accounted_storage_bytes(self.output), before)

    def test_cost_floor_survives_closed_scope_and_changed_source_blocks_deletion(self):
        source = self.output / "code_snapshot/RPent/robots/libero/execution_arms.py"
        original = source.read_bytes()
        source.write_bytes(original + b"\n# changed consumer\n")
        with self.assertRaisesRegex(PermissionError, "source changed"):
            apply_retention(self.policy_ref, "future-D")
        source.write_bytes(original)
        apply_retention(self.policy_ref, "future-D")
        before = accounted_storage_bytes(self.output)
        scope = json.loads(self.scope_ref.read_text())
        scope["status"] = "closed"
        atomic_json(self.scope_ref, scope)
        self.assertEqual(accounted_storage_bytes(self.output), before)

    def test_unregistered_raw_artifact_is_rejected_not_silently_ignored(self):
        self.policy["allowed_artifacts"].append("agentview_policy.png")
        atomic_json(self.policy_ref, self.policy)
        with self.assertRaisesRegex(ValueError, "only registered high-world"):
            apply_retention(self.policy_ref, "future-D")
        self.assertTrue((self.output / "agentview_policy.png/00.png").exists())

    def test_cli_default_is_read_only(self):
        before = self.raw_snapshot()
        module_root = str(Path(__file__).parent)
        script = (
            "import rsi_vla; rsi_vla.__path__.insert(0,"
            + repr(module_root)
            + "); from rsi_vla.artifact_retention import main; main()"
        )
        command = [
            sys.executable,
            "-c",
            script,
            "--policy",
            str(self.policy_ref),
            "--run",
            "future-D",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "prepared")
        self.assertEqual(before, self.raw_snapshot())
        self.assertFalse((self.output / "artifact_retention.json").exists())


if __name__ == "__main__":
    unittest.main()
