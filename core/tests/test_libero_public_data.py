# Copyright 2026 RSI-Agent-VLA Authors.
"""Offline registered multi-task source selection and native CPU data contracts."""

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from rsi_vla.adapters.libero_public_data import (
    REPO_ID,
    acquire_registered_files,
    materialize_registered_data,
    registered_plan,
)

META = Path("/home/xzh/RSI-Agent-VLA/artifacts/authorized-20261110/dataset-metadata-01")
TASKS = [
    "open the middle drawer of the cabinet",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
]


def public_proposal():
    return {
        "schema": "rsi-libero-public-data-registration-v1",
        "status": "proposal",
        "authorized": False,
        "scope_id": "offline-public-data-fixture",
        "protocol_id": "offline-data-v1",
        "owner_uid": os.getuid(),
        "metadata_root": str(META),
        "repo_id": REPO_ID,
        "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "task_groups": [
            {
                "role": "target",
                "task_index": 19,
                "task": TASKS[0],
                "episode_indices": [399, 405, 410, 421, 437, 443, 463, 469],
            },
            {
                "role": "rehearsal",
                "task_index": 25,
                "task": TASKS[1],
                "episode_indices": [814, 817],
            },
            {
                "role": "rehearsal",
                "task_index": 26,
                "task": TASKS[2],
                "episode_indices": [816, 822],
            },
        ],
        "max_frames": 2000,
        "max_output_bytes": 1024**3,
        "attribution": "LIBERO authors; Physical Intelligence LeRobot conversion; reindexed local subset",
        "license_evidence_ref": "fixture-evidence-only",
        "evaluation_isolation": {
            "public_original_init_ids": None,
            "train_formal_init_overlap": "unknown",
            "claim_scope": "familiar_task_under_registered_runtime_conditions",
            "unseen_generalization_claim": False,
        },
    }


class RegisteredPublicDataTests(unittest.TestCase):
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

    def test_exact_pinned_drawer_and_rehearsal_selection_is_finite(self):
        plan = registered_plan(public_proposal())
        self.assertEqual(plan["total_episodes"], 12)
        self.assertEqual(plan["total_frames"], 1665)
        self.assertEqual(plan["episode_payload_bytes"], 206332000)
        self.assertEqual(plan["fps"], 10)
        self.assertEqual(
            [r["episode_index"] for r in plan["episodes"] if r["role"] == "target"],
            [399, 405, 410, 421, 437, 443, 463, 469],
        )
        self.assertEqual(plan["training_admission"], "not_admitted")

    def test_wrong_task_duplicate_episode_and_invented_isolation_are_rejected(self):
        for case in ["task", "duplicate", "isolation", "frames"]:
            with self.subTest(case=case):
                reg = public_proposal()
                if case == "task":
                    reg["task_groups"][0]["episode_indices"][0] = 814
                if case == "duplicate":
                    reg["task_groups"][0]["episode_indices"].append(399)
                if case == "isolation":
                    reg["evaluation_isolation"]["train_formal_init_overlap"] = (
                        "disjoint"
                    )
                if case == "frames":
                    reg["max_frames"] = 10
                with self.assertRaises(ValueError):
                    registered_plan(reg)

    def fixture_source(self):
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = self.root / "source"
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
            REPO_ID,
            root=root,
            fps=10,
            features=features,
            use_videos=False,
            robot_type="panda",
        )
        for index, task in enumerate(TASKS):
            for frame in range(2):
                image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
                writer.add_frame(
                    {
                        "state": np.arange(8, dtype=np.float32) / 20,
                        "actions": np.array(
                            [0.1, 0.2, 0.1, 0, 0, 0, -1 if frame == 0 else 1],
                            np.float32,
                        ),
                        "image": image,
                        "wrist_image": image[::-1].copy(),
                    },
                    task=task,
                )
            writer.save_episode()
        readme = b"---\nlicense: cc-by-4.0\n---\nCPU fixture only\n"
        (root / "README.md").write_bytes(readme)
        sizes = {
            str(p.relative_to(root)): p.stat().st_size
            for p in (root / "data").rglob("*.parquet")
        }
        sizes["README.md"] = len(readme)
        (root / "source.json").write_text(
            json.dumps({"repo_id": REPO_ID, "revision": "a" * 40, "file_sizes": sizes})
        )
        reg = public_proposal()
        reg.update(metadata_root=str(root), revision="a" * 40, max_frames=10)
        reg["task_groups"] = [
            {
                "role": "target" if i == 0 else "rehearsal",
                "task_index": i,
                "task": task,
                "episode_indices": [i],
            }
            for i, task in enumerate(TASKS)
        ]
        return root, reg

    def test_registered_multi_task_acquire_reuse_materialize_and_native_batch(self):
        source, reg = self.fixture_source()
        plan = registered_plan(reg)
        reuse = self.root / "reuse"
        (reuse / "data/chunk-000").mkdir(parents=True)
        shutil.copyfile(source / "source.json", reuse / "source.json")
        reused = plan["episodes"][-1]["file"]
        shutil.copyfile(source / reused, reuse / reused)
        reg["reuse_source_roots"] = [str(reuse)]
        seen = []

        class Response(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}

        def opener(request, **unused):
            name = request.full_url.split("/resolve/" + reg["revision"] + "/")[1]
            seen.append(name)
            return Response((source / name).read_bytes())

        cap = (
            sum(row["bytes"] for row in plan["episodes"] if row["file"] != reused)
            + plan["README_bytes"]
        )
        report = acquire_registered_files(
            reg, self.root / "download", max_download_bytes=cap, opener=opener
        )
        self.assertEqual(report["downloaded_demonstration_bytes"], cap)
        self.assertNotIn(reused, seen)
        self.assertEqual(
            sum(
                json.loads(l)["response_payload_bytes"]
                for l in (self.root / "download/download-ledger.jsonl")
                .read_text()
                .splitlines()
            ),
            cap,
        )
        result = materialize_registered_data(
            reg, self.root / "download", self.root / "subset"
        )
        manifest = json.loads(Path(result["manifest_ref"]).read_text())
        audit = json.loads(Path(result["payload_audit_ref"]).read_text())
        self.assertTrue(audit["checks_passed"])
        self.assertEqual(audit["frames_checked"], 6)
        self.assertEqual([row["task"] for row in manifest["episodes"]], TASKS)
        self.assertEqual(
            [row["role"] for row in manifest["episodes"]],
            ["target", "rehearsal", "rehearsal"],
        )
        self.assertEqual(manifest["training_admission"], "not_admitted")
        self.assertEqual(
            manifest["evaluation_isolation"]["train_formal_init_overlap"], "unknown"
        )
        from rsi_vla.adapters.pi05_training import build_recipe, read_native_batch
        from rsi_vla.training import TrainingBudget

        model = Path(
            "/home/xzh/RSI-Agent-VLA/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
        )
        recipe = build_recipe(
            dataset_root=result["dataset_root"],
            model_path=model,
            norm_stats_path=model / "physical-intelligence/libero/norm_stats.json",
            output_root=self.root,
            run_id="cpu-fixture",
            budget=TrainingBudget(60, 60, 1, 1024**3, 3),
        )
        self.assertEqual(
            read_native_batch(recipe, manifest)["action_shape"], [1, 10, 32]
        )

    def test_live_scope_source_phase_and_P1_admission(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        import rsi_vla.adapters.libero_public_data as module
        from rsi_vla.midterm_host import MidtermLedger

        reg = public_proposal()
        reg.update(status="frozen", authorized=True)

        def write(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return str(path)

        now = datetime.now(timezone.utc)
        limits = {
            "download_bytes": 8 * 1024**3,
            "storage_bytes": 16 * 1024**3,
            "current_disk_bytes": 8 * 1024**3,
            "concurrent_gpu_devices": 2,
        }
        p1_ref = write(
            "p1.json",
            {
                "schema": "rsi-midterm-p1-acceptance-v1",
                "scope_id": reg["scope_id"],
                "status": "passed",
            },
        )
        scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": reg["scope_id"],
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "status": "active",
            "authorized": True,
            "dynamic_authorized": True,
            "outer_limits": limits,
            "p1_acceptance_ref": p1_ref,
            "accepted_at": (now - timedelta(minutes=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        }
        scope_ref = write("scope.json", scope)
        phase_ref = write(
            "phase.json",
            {"limits": limits, "remaining": limits, "unresolved_costs": []},
        )
        ledger = MidtermLedger(scope_ref)
        ledger.reserve(
            "data-job",
            budget={
                "download_bytes": 220000000,
                "storage_bytes": 2 * 1024**3,
                "current_disk_bytes": 2 * 1024**3,
            },
            phase_ref=phase_ref,
            gpu_devices=[],
            receipt_ref=self.root / "receipt.json",
        )
        source_ref = self.root / "freeze/manifest.json"
        repository = self.root / "repository"
        work = Path(__file__).resolve().parents[2]
        for name in module.SOURCE_FILES:
            path = work / name
            if not path.is_file():
                path = Path("/home/xzh/RSI-Agent-VLA") / name
            content = path.read_bytes()
            for target in (repository / name, source_ref.parent / "source" / name):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        write(
            "freeze/manifest.json",
            {"scope": reg["scope_id"], "files": list(module.SOURCE_FILES)},
        )
        reg.update(
            repository_root=str(repository),
            source_manifest_ref=str(source_ref),
            target_selection_evidence_refs=[
                write(
                    "target.json", {"scope_id": reg["scope_id"], "fixture_only": True}
                )
            ],
            license_evidence_ref=write(
                "license.json", {"license": "cc-by-4.0", "fixture_only": True}
            ),
        )
        reg_ref = write("registration.json", reg)
        permit = {
            "kind": "libero-public-data-acquire",
            "authorized": True,
            "scope_id": reg["scope_id"],
            "scope_ref": scope_ref,
            "accounting_ref": str(ledger.path),
            "phase_ref": phase_ref,
            "operation_id": "data-job",
            "protocol_id": reg["protocol_id"],
            "owner_uid": os.getuid(),
            "registration_ref": reg_ref,
            "registered_plan": registered_plan(reg),
            "output_root": str(self.root / "download"),
            "source_manifest_ref": str(source_ref),
            "wall_seconds": 600,
            "expires_at": scope["expires_at"],
            "storage_bytes": 2 * 1024**3,
            "current_disk_bytes": 2 * 1024**3,
            "max_download_bytes": 220000000,
        }
        with patch.object(
            module,
            "__file__",
            str(repository / "core/src/rsi_vla/adapters/libero_public_data.py"),
        ):
            module.require_data_permit(
                reg_ref, permit, action="acquire", output_root=permit["output_root"]
            )
            write(
                "p1.json",
                {
                    "schema": "rsi-midterm-p1-acceptance-v1",
                    "scope_id": reg["scope_id"],
                    "status": "pending",
                },
            )
            with self.assertRaisesRegex(PermissionError, "P1"):
                module.require_data_permit(
                    reg_ref, permit, action="acquire", output_root=permit["output_root"]
                )
            write(
                "p1.json",
                {
                    "schema": "rsi-midterm-p1-acceptance-v1",
                    "scope_id": reg["scope_id"],
                    "status": "passed",
                },
            )
            changed = repository / "scripts/prepare_libero_public_data.py"
            changed.write_bytes(changed.read_bytes() + b"\n# changed fixture\n")
            with self.assertRaisesRegex(PermissionError, "source changed"):
                module.require_data_permit(
                    reg_ref, permit, action="acquire", output_root=permit["output_root"]
                )

    def test_download_failure_keeps_actual_partial_payload_in_byte_ledger(self):
        source, reg = self.fixture_source()
        plan = registered_plan(reg)

        class IncompleteResponse(io.BytesIO):
            def __init__(self):
                super().__init__(b"partial")
                self.headers = {"Content-Length": str(plan["episodes"][0]["bytes"])}

        with self.assertRaisesRegex(ValueError, "incomplete"):
            acquire_registered_files(
                reg,
                self.root / "partial-download",
                max_download_bytes=plan["episode_payload_bytes"] + plan["README_bytes"],
                opener=lambda *args, **kwargs: IncompleteResponse(),
            )
        lines = (
            (self.root / "partial-download/download-ledger.jsonl")
            .read_text()
            .splitlines()
        )
        self.assertEqual(
            sum(json.loads(l)["response_payload_bytes"] for l in lines), len(b"partial")
        )
        self.assertFalse((self.root / "partial-download/source.json").exists())
        meter = json.loads(
            (self.root / "partial-download/download-meter.json").read_text()
        )
        self.assertFalse(meter["download_complete"])
        self.assertTrue(meter["download_meter_closed"])
        self.assertEqual(meter["download_response_payload_bytes"], len(b"partial"))

    def test_unknown_read_failure_keeps_prefix_without_exact_download_charge(self):
        source, reg = self.fixture_source()
        plan = registered_plan(reg)
        identity = {
            "scope_id": "fixture",
            "operation_id": "read-fault",
            "owner_uid": os.getuid(),
        }

        class FailedRead(io.BytesIO):
            headers = {}

            def read(self, count=-1):
                if self.tell() == 0:
                    return super().read(2)
                raise OSError("read failed with an unknown unreturned tail")

        with self.assertRaisesRegex(OSError, "unknown"):
            acquire_registered_files(
                reg,
                self.root / "read-fault",
                max_download_bytes=plan["episode_payload_bytes"] + plan["README_bytes"],
                opener=lambda *args, **kwargs: FailedRead(b"abcdef"),
                operation_identity=identity,
            )
        row = json.loads((self.root / "read-fault/download-ledger.jsonl").read_text())
        self.assertEqual({key: row[key] for key in identity}, identity)
        meter = json.loads((self.root / "read-fault/download-meter.json").read_text())
        self.assertFalse(meter["download_meter_closed"])
        self.assertIsNone(meter["download_response_payload_bytes"])
        self.assertEqual(meter["observed_download_payload_bytes_lower_bound"], 2)

    def test_failed_CLI_publishes_terminal_and_closed_no_model_meter(self):
        import importlib.util
        from unittest.mock import patch

        script = (
            Path(__file__).resolve().parents[2]
            / "scripts/prepare_libero_public_data.py"
        )
        spec = importlib.util.spec_from_file_location("public_data_cli_fixture", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        reg_ref = self.root / "registration.json"
        reg_ref.write_text(json.dumps(public_proposal()))
        identity = {
            "scope_id": "fixture",
            "operation_id": "materialize-failure",
            "owner_uid": os.getuid(),
        }
        permit = {
            **identity,
            "source_manifest_ref": "fixture-manifest",
            "source_root": str(self.root / "source"),
        }
        permit_ref = self.root / "permit.json"
        permit_ref.write_text(json.dumps(permit))
        output = self.root / "failed-output"
        with (
            patch.object(
                cli, "require_data_permit", return_value=(public_proposal(), {})
            ),
            patch.object(
                cli,
                "materialize_registered_data",
                side_effect=ValueError("native writer fixture failure"),
            ),
        ):
            code = cli.main(
                [
                    "--registration",
                    str(reg_ref),
                    "--permit",
                    str(permit_ref),
                    "--action",
                    "materialize",
                    "--source-root",
                    permit["source_root"],
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 1)
        result = json.loads((output / "job-result.json").read_text())
        meter = json.loads(Path(result["model_meter_ref"]).read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertEqual(meter["status"], "closed")
        self.assertEqual(meter["project_model_requests"], 0)
        self.assertEqual(meter["model_transports_created"], [])
        self.assertEqual({key: meter[key] for key in identity}, identity)
        self.assertIsNone(result["download_response_payload_bytes"])

    def test_failed_acquire_CLI_closes_partial_meter_and_terminal_identity(self):
        import importlib.util
        from unittest.mock import patch

        source, reg = self.fixture_source()
        plan = registered_plan(reg)
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts/prepare_libero_public_data.py"
        )
        spec = importlib.util.spec_from_file_location(
            "public_data_acquire_cli_fixture", script
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        reg_ref = self.root / "registration.json"
        reg_ref.write_text(json.dumps(reg))
        identity = {
            "scope_id": "fixture",
            "operation_id": "partial-acquire",
            "owner_uid": os.getuid(),
        }
        permit = {
            **identity,
            "source_manifest_ref": "fixture-manifest",
            "max_download_bytes": plan["episode_payload_bytes"] + plan["README_bytes"],
        }
        permit_ref = self.root / "permit.json"
        permit_ref.write_text(json.dumps(permit))
        output = self.root / "partial-cli-output"

        class Partial(io.BytesIO):
            def __init__(self):
                super().__init__(b"partial")
                self.headers = {"Content-Length": str(plan["episodes"][0]["bytes"])}

        def acquire(*args, **kwargs):
            return acquire_registered_files(
                *args, **kwargs, opener=lambda *a, **k: Partial()
            )

        with (
            patch.object(cli, "require_data_permit", return_value=(reg, plan)),
            patch.object(cli, "acquire_registered_files", acquire),
        ):
            code = cli.main(
                [
                    "--registration",
                    str(reg_ref),
                    "--permit",
                    str(permit_ref),
                    "--action",
                    "acquire",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 1)
        result = json.loads((output / "job-result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["download_complete"])
        self.assertTrue(result["download_meter_closed"])
        self.assertEqual(result["download_response_payload_bytes"], 7)
        self.assertEqual(result["observed_download_payload_bytes_lower_bound"], 7)
        meter = json.loads(Path(result["model_meter_ref"]).read_text())
        self.assertEqual({key: meter[key] for key in identity}, identity)
        self.assertEqual(meter["project_model_requests"], 0)

    def test_download_ledger_write_failure_keeps_recorded_prefix_unknown(self):
        from unittest.mock import patch

        from rsi_vla.adapters.libero_public_data import _PublicDownloadMeter

        root = self.root / "write-fault"
        root.mkdir()
        meter = _PublicDownloadMeter(
            root,
            {
                "scope_id": "fixture",
                "operation_id": "write-fault",
                "owner_uid": os.getuid(),
            },
            None,
        )
        meter.record("https://fixture", 2)
        original_open = Path.open

        def fail_ledger(path, *args, **kwargs):
            if path.name == "download-ledger.jsonl":
                raise OSError("ledger write failed")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_ledger), self.assertRaises(OSError):
            meter.record("https://fixture", 3)
        meter.close(download_complete=False)
        result = json.loads((root / "download-meter.json").read_text())
        self.assertFalse(result["download_meter_closed"])
        self.assertIsNone(result["download_response_payload_bytes"])
        self.assertEqual(result["observed_download_payload_bytes_lower_bound"], 2)
        self.assertEqual(result["producer_returned_payload_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
