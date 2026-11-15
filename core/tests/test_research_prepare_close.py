"""CPU selector fault closure; no SDK, simulator, process or real scope writes."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.storage import atomic_json, ResearchStore
from rsi_vla.research_supervision import close_failed_research_preparation
from rsi_vla.midterm_host import MidtermContinuationBackend


class PreparationClosureTests(unittest.TestCase):
    def setUp(self):
        from test_midterm_host import LedgerTests

        LedgerTests.setUp(self)
        self.operation = {
            "operation_id": "r-prepare",
            "kind": "research_prepare",
            "dynamic": False,
            "payload": {"request_id": "r"},
            "stage": "development",
        }
        self.output = self.root / "native-output"
        self.store = ResearchStore(self.root / "store")
        self.job = self.root / "host/jobs/r-prepare"
        self.source = self.root / "frozen/manifest.json"
        atomic_json(self.source, {"files": []})
        snapshot = self.source.parent / "source"
        p = snapshot / "core/src/rsi_vla/research_transport.py"
        p.parent.mkdir(parents=True)
        p.write_text(
            'def select_input(request):\n    projection=request["sdk_projection"]\n    if projection not in (None,"terminal_history_v1"):\n        raise ValueError("unsupported explicit SDK field projection")\n'
        )
        p = snapshot / "core/src/rsi_vla/task_completion.py"
        p.write_text(
            'class Fixture:\n    def method(self):\n        if True:\n            if kind == "research_prepare":\n                transport = adapter.prepare_request(tools)\n                return transport\n            from rsi_vla.bounded_runner import execute\n'
        )
        p = snapshot / "scripts/run_isolated_role.py"
        p.parent.mkdir(parents=True)
        p.write_text('["--unshare-pid", "--die-with-parent"]\n')
        policy = self.root / "domain.json"
        atomic_json(
            policy,
            {
                "evidence_selection": {"sdk_projection": "terminal_history_catalog_v2"},
                "evidence_input": {"sdk_projection": "terminal_history_catalog_v2"},
            },
        )
        request = {
            "request_id": "r",
            "job_id": "job",
            "base_bundle_id": "base",
            "output_ref": str(self.output),
            "domain_policy_ref": str(policy),
        }
        core = {
            "operations": [self.operation],
            "research_requests": [request],
            "source_manifest_ref": str(self.source),
            "research_store_ref": str(self.store.root),
            "base_bundle_id": "base",
            "phase_ref": str(self.phase),
        }
        template = {
            "budget": {"storage_bytes": 1000, "current_disk_bytes": 1000},
            "gpu_devices": [],
        }
        self.host = SimpleNamespace(
            root=self.root / "host",
            ledger=self.ledger,
            core_config=core,
            config={
                "scope_id": self.scope["scope_id"],
                "operations": {"r-prepare": template},
            },
        )
        self.report = {
            "identity": {"run_id": "r-prepare"},
            "status": "completed",
            "exit_code": 1,
            "cleanup_confirmed": True,
            "remaining_process_ids": [],
            "storage_bytes": 300,
            "wall_seconds": 5,
        }
        atomic_json(
            self.job / "submission.json",
            {"operation": self.operation, "owner_uid": os.getuid()},
        )
        atomic_json(
            self.job / "process.json",
            {
                "pid": 999999999,
                "owner_uid": os.getuid(),
                "start_ticks": 1,
                "boot_id": "fixture",
            },
        )
        atomic_json(self.job / "supervisor.json", self.report)
        atomic_json(
            self.job / "completion.json",
            {"status": "unknown", "supervisor_ref": str(self.job / "supervisor.json")},
        )
        self.error_ref = (
            self.store.root / "role_receipts/r-prepare-execute-revision-00001.json"
        )
        atomic_json(
            self.error_ref,
            {
                "role": "research",
                "request": {
                    "configuration": core,
                    "task": {"operation": self.operation, "action": "execute"},
                },
                "error": "ValueError: unsupported explicit SDK field projection",
            },
        )
        self.store.put(
            "llm_requests",
            "r",
            {
                "request_id": "r",
                "job_id": "job",
                "bundle_id": "base",
                "all_facts": ["unchanged"],
            },
        )
        self.ledger.reserve(
            "r-prepare",
            budget=template["budget"],
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.job / "completion.json",
        )
        self.ledger.mark_submitted("r-prepare")
        self.ledger.settle(
            "r-prepare",
            actual={"storage_bytes": 300, "current_disk_bytes": 300},
            cleanup_confirmed=True,
        )

    def closure(self):
        with (
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={"live_processes": [], "unreadable_owned_processes": []},
            ),
            patch(
                "rsi_vla.midterm_host.capture_closure_sources",
                return_value=self.root / "new-control/manifest.json",
            ),
        ):
            return close_failed_research_preparation(
                self.host, "r-prepare", self.report
            )

    def test_original_cpu_error_partial_archive_and_settled_costs_survive(self):
        paths = [
            self.error_ref,
            self.store.path("llm_requests", "r"),
            self.job / "completion.json",
            self.job / "supervisor.json",
            self.ledger.path,
        ]
        before = {p: p.read_bytes() for p in paths}
        receipt = self.closure()
        self.assertEqual(self.closure(), receipt)
        value = receipt["value"]
        self.assertEqual(value["native_outcome"], "not_started")
        self.assertEqual(value["model_requests"], 0)
        self.assertFalse(value["original_lifecycle_reset"])
        for p, blob in before.items():
            self.assertEqual(p.read_bytes(), blob)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.store.path("llm_executions", "r").exists())
        backend = object.__new__(MidtermContinuationBackend)
        backend.host = self.host
        self.assertEqual(backend.reconcile(self.operation, {})["value"], value)
        self.assertEqual(
            backend.observe({"receipts": {"r-prepare": receipt}}, {}),
            {"terminal": "research_preparation_technical_stop"},
        )

    def test_native_or_complete_preparation_evidence_prevents_cpu_absence_claim(self):
        self.output.mkdir()
        with self.assertRaisesRegex(PermissionError, "native output"):
            self.closure()
        self.assertFalse((self.job / "host-ended-receipt.json").exists())

    def test_wrong_role_error_and_non_cpu_kind_are_rejected(self):
        error = json.loads(self.error_ref.read_text())
        error["error"] = "RuntimeError: different failure"
        atomic_json(self.error_ref, error)
        with self.assertRaisesRegex(PermissionError, "contradicts"):
            self.closure()
        self.operation["kind"] = "research"
        with self.assertRaisesRegex(PermissionError, "cannot admit"):
            self.closure()

    def test_cpu_marker_cannot_relax_interrupted_native_reconcile_guard(self):
        self.closure()
        self.operation["kind"] = "runtime"
        self.operation["dynamic"] = True
        target = self.job / "host-ended-receipt.json"
        receipt = json.loads(target.read_text())
        receipt["operation"] = self.operation
        atomic_json(target, receipt)
        backend = object.__new__(MidtermContinuationBackend)
        backend.host = self.host
        with self.assertRaisesRegex(PermissionError, "cannot close a native"):
            backend.reconcile(self.operation, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
