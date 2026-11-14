# Copyright 2026 RSI-Agent-VLA Authors.
"""CPU-only regression for physical GPU admission and durable unsubmitted revisions."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.protocol10_gpu import revise_registration
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "gpu_release_driver", ROOT / "scripts/run_release_comparison.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class GpuAdmissionTests(unittest.TestCase):
    def test_any_physical_index_and_least_occupied(self):
        with patch.object(
            driver.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(stdout="0, a\n3, b\n7, c\n"),
                SimpleNamespace(stdout="a, 10, 12289\nb, 20, 12000\nc, 30, 1000\n"),
            ],
        ):
            result = driver.gpu_admission()
        self.assertEqual(result["selected"], 7)
        self.assertEqual(len(result["cards"]), 3)
        self.assertEqual(result["threshold_mib"], 12288)

    def test_details_include_compatibility_and_load(self):
        with patch.object(
            driver.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(stdout="5, a\n"),
                SimpleNamespace(stdout=""),
                SimpleNamespace(stdout="5, a, NVIDIA A100, 40960, 39000, 0, 8.0\n"),
            ],
        ):
            result = driver.gpu_admission(details=True)
        self.assertEqual(result["selected"], 5)
        self.assertEqual(result["cards"][0]["compute_capability"], "8.0")
        self.assertEqual(result["cards"][0]["free_memory_mib"], 39000)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scope = self.root / "protocol10"
        (self.scope / "coordinator").mkdir(parents=True)
        self.manifest = self.root / "manifest.json"
        self.phase = self.root / "phase.json"
        atomic_json(self.manifest, {"snapshot": "before"})
        atomic_json(
            self.phase, {"used": {}, "remaining": {}, "protocol10_trial_charges": []}
        )
        self.slots = []
        for i in range(3):
            permit = self.root / f"permit{i}.json"
            atomic_json(
                permit,
                {
                    "run_id": f"run{i}",
                    "gpu_devices": [0],
                    "authorized": True,
                    "limits": {"wall_seconds": 900},
                    "expires_at": "expiry",
                },
            )
            self.slots.append(
                {
                    "run_id": f"run{i}",
                    "side": "baseline",
                    "cuda_device": 0,
                    "permit_ref": str(permit),
                    "output_ref": str(self.root / f"output{i}"),
                }
            )
        self.config = {
            "runtime_slots": self.slots,
            "source_manifest_ref": str(self.manifest),
            "phase_ref": str(self.phase),
            "research": {
                "store_ref": str(self.root / "research"),
                "output_ref": str(self.root / "research-output"),
            },
        }
        self.config_ref = self.root / "configuration.json"
        atomic_json(self.config_ref, self.config)
        self.registration = {
            "configuration": self.config,
            "inputs": {str(self.manifest): {"snapshot": "before"}},
        }
        atomic_json(
            self.scope / "backend/registration/protocol10.json", self.registration
        )
        coordinator = Coordinator(self.scope / "coordinator", self.registration, None)
        self.state = coordinator.status()
        op = {
            "operation_id": "runtime-run0",
            "dynamic": True,
            "kind": "runtime",
            "stage": "baseline",
            "payload": {
                **self.slots[0],
                "registered_permit": json.loads(
                    Path(self.slots[0]["permit_ref"]).read_text()
                ),
            },
        }
        self.state.update(
            pending=op,
            pending_receipt={
                "operation_id": op["operation_id"],
                "status": "not_submitted",
                "submission_absent": True,
            },
        )
        self.state["operations"][op["operation_id"]] = op
        self.progress = self.scope / "coordinator/progress.json"
        atomic_json(self.progress, self.state)

    def revise(self, **kwargs):
        return revise_registration(
            self.config_ref,
            self.scope,
            "revision1",
            dynamic_lock=self.root / "dynamic.lock",
            **kwargs,
        )

    def test_pending_and_future_slots_update_consistently_and_keep_old_bytes(self):
        before = self.config_ref.read_text()
        result = self.revise(devices={"run0": 7, "run2": 3})
        state = json.loads(self.progress.read_text())
        self.assertEqual(state["pending"]["operation_id"], "runtime-run0")
        self.assertEqual(state["pending"]["payload"]["cuda_device"], 7)
        self.assertEqual(
            state["pending"]["payload"]["registered_permit"]["gpu_devices"], [7]
        )
        self.assertEqual(
            state["binding"]["configuration"]["runtime_slots"][2]["cuda_device"], 3
        )
        self.assertEqual(
            next(c for c in result["changes"] if c["path"] == str(self.config_ref))[
                "before"
            ],
            before,
        )
        self.assertEqual(self.revise(devices={"run0": 7, "run2": 3}), result)
        self.assertEqual(
            Coordinator(
                self.scope / "coordinator", result["after_binding"], None
            ).status(),
            state,
        )
        permit = json.loads(Path(self.slots[0]["permit_ref"]).read_text())
        self.assertEqual(permit["limits"], {"wall_seconds": 900})
        self.assertEqual(permit["expires_at"], "expiry")
        self.assertEqual(json.loads(self.phase.read_text())["used"], {})

    def test_source_refresh_before_first_dispatch_preserves_other_inputs(self):
        atomic_json(self.manifest, {"snapshot": "after"})
        result = self.revise(refresh_source_manifest=True)
        self.assertEqual(
            result["before_binding"]["inputs"][str(self.manifest)],
            {"snapshot": "before"},
        )
        self.assertEqual(
            result["after_binding"]["inputs"][str(self.manifest)], {"snapshot": "after"}
        )
        self.assertEqual(
            result["before_binding"]["configuration"],
            result["after_binding"]["configuration"],
        )

    def test_started_outputs_dispatch_returns_and_charges_cannot_change_gpu(self):
        cases = [
            self.root / "output0",
            self.scope / "backend/dispatches/run0.json",
            self.scope / "backend/runtime_returns/run0.json",
        ]
        for path in cases:
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "started or completed"):
                    self.revise(devices={"run0": 2})
                path.unlink()
        atomic_json(self.phase, {"protocol10_trial_charges": [{"run_id": "run0"}]})
        with self.assertRaisesRegex(ValueError, "charged"):
            self.revise(devices={"run0": 2})

    def test_future_gpu_can_change_after_other_slot_started_but_source_cannot(self):
        (self.root / "output0").mkdir()
        result = self.revise(devices={"run2": 5})
        self.assertEqual(
            result["after_binding"]["configuration"]["runtime_slots"][0]["cuda_device"],
            0,
        )
        with self.assertRaisesRegex(ValueError, "no prior dispatch"):
            revise_registration(
                self.config_ref,
                self.scope,
                "source-revision",
                refresh_source_manifest=True,
                dynamic_lock=self.root / "dynamic.lock",
            )

    def test_unknown_pending_or_unrelated_input_change_rejected_without_mutation(self):
        before = self.config_ref.read_bytes()
        self.state["pending_receipt"]["status"] = "unknown"
        atomic_json(self.progress, self.state)
        with self.assertRaisesRegex(ValueError, "proven absent"):
            self.revise(devices={"run0": 2})
        self.state["pending_receipt"]["status"] = "not_submitted"
        atomic_json(self.progress, self.state)
        atomic_json(self.manifest, {"snapshot": "unregistered"})
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.revise(devices={"run0": 2})
        self.assertEqual(self.config_ref.read_bytes(), before)

    def test_source_without_pending_and_unissued_future_permit(self):
        self.state.update(pending=None, pending_receipt=None, operations={})
        atomic_json(self.progress, self.state)
        Path(self.slots[2]["permit_ref"]).unlink()
        result = self.revise(devices={"run2": 4}, refresh_source_manifest=True)
        self.assertEqual(
            result["after_binding"]["configuration"]["runtime_slots"][2]["cuda_device"],
            4,
        )
        self.assertFalse(Path(self.slots[2]["permit_ref"]).exists())

    def test_unreturned_role_dispatch_is_not_an_unsubmitted_future_run(self):
        self.config["isolation_task_root"] = str(self.root)
        self.registration["configuration"] = self.config
        self.state["binding"] = self.registration
        atomic_json(self.config_ref, self.config)
        atomic_json(
            self.scope / "backend/registration/protocol10.json", self.registration
        )
        atomic_json(self.progress, self.state)
        atomic_json(
            self.root / "operator/role_requests/runtime.json",
            {
                "task": {
                    "action": "execute",
                    "operation": {"dynamic": True, "payload": {"run_id": "run2"}},
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "started or completed"):
            self.revise(devices={"run2": 2})

    def test_partial_commit_resumes_same_revision(self):
        import rsi_vla.protocol10_gpu as module

        real_replace = module.os.replace
        calls = 0

        def fail_once(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected crash")
            return real_replace(*args)

        with (
            patch.object(module.os, "replace", side_effect=fail_once),
            self.assertRaises(OSError),
        ):
            self.revise(devices={"run0": 6})
        result = self.revise(devices={"run0": 6})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            json.loads(self.progress.read_text())["pending"]["payload"]["cuda_device"],
            6,
        )


class RealRuntimePreparationTests(unittest.TestCase):
    def test_revision_reopens_real_backend_and_prepares_selected_physical_device(self):
        from rsi_vla.protocol10_backend import Protocol10Backend
        from test_protocol10_integration import Protocol10IntegrationTests

        fixture = Protocol10IntegrationTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config_ref = fixture.root / "configuration.json"
        atomic_json(config_ref, fixture.config)
        coordinator = Coordinator(
            fixture.scope / "coordinator", fixture.backend.registration, fixture.backend
        )
        state = coordinator.status()
        slot = fixture.slots[0]
        op = fixture.backend._op(
            "runtime", "baseline", slot, dynamic=True, key="runtime-" + slot["run_id"]
        )
        state.update(
            pending=op,
            pending_receipt={"status": "not_submitted", "submission_absent": True},
        )
        state["operations"][op["operation_id"]] = op
        atomic_json(coordinator.path, state)
        report = revise_registration(
            config_ref,
            fixture.scope,
            "device7",
            devices={slot["run_id"]: 7},
            dynamic_lock=fixture.root / "dynamic.lock",
        )
        config = json.loads(config_ref.read_text())
        backend = Protocol10Backend(
            fixture.scope, config, runtime_execute=fixture.runtime_execute
        )
        self.assertEqual(backend.registration, report["after_binding"])
        plan = backend._prepare_runtime(config["runtime_slots"][0])
        self.assertEqual(plan["context"]["gpu_devices"], [7])
        self.assertEqual(
            plan["command"][plan["command"].index("--cuda-device") + 1], "7"
        )
        self.assertFalse(Path(slot["output_ref"]).exists())


if __name__ == "__main__":
    unittest.main()
