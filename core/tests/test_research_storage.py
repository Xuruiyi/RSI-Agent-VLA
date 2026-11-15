"""CPU-only regressions for complete Research envelopes and stopped-job closure."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.process_accounting import StorageMeter, snapshot
from rsi_vla.storage import atomic_json
from rsi_vla.bounded_runner import permitted_input
from rsi_vla.midterm_host import atomic_json_changed
from rsi_vla.research_supervision import (
    compact_input_binding,
    lifecycle,
    complete_context,
)
from rsi_vla.adapters.runtime_supervisor import supervise


class ResearchStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_atomic_replacement_growth_and_recreated_path(self):
        p = self.root / "a"
        p.write_bytes(b"a" * 10)
        meter = StorageMeter([self.root])
        q = self.root / "tmp"
        q.write_bytes(b"b" * 10)
        q.replace(p)
        self.assertEqual(meter.sample()["cumulative_new_bytes_lower_bound"], 0)
        q.write_bytes(b"c" * 20)
        q.replace(p)
        self.assertEqual(meter.sample()["cumulative_new_bytes_lower_bound"], 10)
        p.unlink()
        meter.sample()
        p.write_bytes(b"d" * 7)
        self.assertEqual(meter.sample()["cumulative_new_bytes_lower_bound"], 17)

    def test_retained_hardlink_credit_cannot_pay_for_atomic_replacement(self):
        for size in (10, 20):
            folder = self.root / str(size)
            folder.mkdir()
            a = folder / "a"
            b = folder / "b"
            a.write_bytes(b"a" * 10)
            os.link(a, b)
            meter = StorageMeter([folder])
            t = folder / "tmp"
            t.write_bytes(b"x" * size)
            t.replace(a)
            record = meter.sample()
            self.assertEqual(record["cumulative_new_bytes_lower_bound"], size)
            self.assertEqual(record["current_new_bytes"], size)
            self.assertEqual(sum(snapshot([folder]).values()), size + 10)

    def test_two_replacements_share_one_legacy_credit(self):
        a = self.root / "a"
        b = self.root / "b"
        a.write_bytes(b"a" * 10)
        os.link(a, b)
        meter = StorageMeter([self.root], baseline=snapshot([self.root]))
        for target in (a, b):
            t = self.root / "tmp"
            t.write_bytes(b"b" * 10)
            t.replace(target)
        self.assertEqual(meter.sample()["cumulative_new_bytes_lower_bound"], 10)

    def test_state_continuation_counts_prepare_and_later_growth_once(self):
        business = self.root / "business"
        business.mkdir()
        meter = StorageMeter([business])
        p = business / "request.json"
        p.write_bytes(b"a" * 100)
        meter.sample()
        state = meter.state()
        p.write_bytes(b"b" * 100)
        (business / "response.json").write_bytes(b"c" * 40)
        continued = StorageMeter([business], state=state)
        self.assertEqual(continued.sample()["cumulative_new_bytes_lower_bound"], 140)
        (business / "response.json").unlink()
        self.assertEqual(continued.sample()["cumulative_new_bytes_lower_bound"], 140)

    def test_compact_permit_is_consumed_and_rejects_replaced_archive(self):
        path = self.root / "transport.json"
        atomic_json(path, {"all": "facts", "value": [1, 2]})
        original = json.loads(path.read_text())
        permit = compact_input_binding(path)
        self.assertEqual(permitted_input(permit), original)
        self.assertEqual(permitted_input({"input": original}), original)
        atomic_json(path, {"all": "facts", "value": [1, 2, 3]})
        with self.assertRaisesRegex(PermissionError, "authorized content"):
            permitted_input(permit)

    def test_observer_only_writes_on_semantic_change(self):
        path = self.root / "observer.json"
        atomic_json_changed(path, {"seq": 1})
        old = path.stat().st_ino
        atomic_json_changed(path, {"seq": 1})
        self.assertEqual(path.stat().st_ino, old)
        atomic_json_changed(path, {"seq": 2})
        self.assertEqual(json.loads(path.read_text()), {"seq": 2})

    def test_complete_request_context_keeps_original_deadline_and_prepare_bytes(self):
        root = self.root
        store = root / "store"
        job = store / "jobs" / "job.json"
        atomic_json(job, {"old": "facts"})
        protocol = root / "protocol.json"
        atomic_json(
            protocol,
            {
                "task_policy": {
                    "research_per_request": {
                        "wall_seconds": 180,
                        "storage_bytes": 16 * 1024**2,
                    }
                }
            },
        )
        request = {
            "request_id": "r",
            "job_id": "job",
            "reserved_bundle_id": "proposal",
            **{
                key: str(root / key)
                for key in (
                    "output_ref",
                    "scratch_ref",
                    "permit_ref",
                    "preflight_ref",
                    "domain_policy_ref",
                )
            },
        }
        Path(request["domain_policy_ref"]).write_bytes(b"d" * 23)
        operation = {
            "operation_id": "prepare",
            "kind": "research_prepare",
            "payload": {"request_id": "r"},
        }
        ledger = SimpleNamespace(
            root=root,
            read=lambda: {
                "reservations": {},
                "used": {"storage_bytes": 0},
                "limits": {"storage_bytes": 10**8, "current_disk_bytes": 10**8},
            },
            _held=lambda *_: 0,
        )
        host = SimpleNamespace(
            root=root / "host",
            ledger=ledger,
            config={
                "scope_id": "scope",
                "continuation_root": str(root / "continuation"),
            },
            core_config={
                "research_request_envelope": "complete_v2",
                "research_requests": [request],
                "operations": [operation],
                "protocol_ref": str(protocol),
                "research_store_ref": str(store),
                "isolation_task_root": str(root),
            },
        )
        first = lifecycle(host, operation)
        Path(request["preflight_ref"]).write_bytes(b"p" * 75)
        context = complete_context(
            host,
            operation,
            {"limits": {"wall_seconds": 180, "storage_bytes": 64 * 1024**2}},
        )
        self.assertEqual(lifecycle(host, operation)["deadline"], first["deadline"])
        self.assertLessEqual(context["limits"]["wall_seconds"], 180)
        self.assertEqual(context["limits"]["storage_bytes"], 16 * 1024**2)
        state = json.loads(Path(first["business_state_ref"]).read_text())
        meter = StorageMeter(first["business_roots"], state=state)
        self.assertGreaterEqual(meter.sample()["cumulative_new_bytes_lower_bound"], 98)
        self.assertNotIn(str(root), first["business_roots"])

    def test_supervisor_separates_large_control_and_small_business_growth(self):
        business = self.root / "business"
        business.mkdir()
        control = self.root / "control"
        control.mkdir()
        state = self.root / "business-state.json"
        scope_state = self.root / "scope-state.json"
        atomic_json(state, StorageMeter([business]).state())
        atomic_json(scope_state, StorageMeter([business, control]).state())
        context = {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
            "output_dir": str(business),
            "dynamic_authorized": True,
            "authorization_ref": "offline",
            "limits": {"wall_seconds": 3, "storage_bytes": 50000},
            "storage_write_roots": [str(business)],
            "storage_meter_state_ref": str(state),
            "scope_accounting_roots": [str(business), str(control)],
            "scope_meter_state_ref": str(scope_state),
            "scope_storage_remaining_bytes": 1000000,
            "scope_current_disk_limit_bytes": 1000000,
        }
        context_ref = self.root / "context.json"
        atomic_json(context_ref, context)
        code = (
            "from pathlib import Path; Path("
            + repr(str(control / "snapshot"))
            + ').write_bytes(b"x"*200000); print("small business")'
        )
        result = supervise(
            [sys.executable, "-c", code],
            context_path=context_ref,
            poll_interval_s=0.01,
            terminate_grace_s=0.05,
        )
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["storage_bytes"], 50000)
        self.assertGreaterEqual(result["control_storage_delta_lower_bound"], 200000)
        before = result["storage_bytes"]
        next_meter = StorageMeter([business], state=json.loads(state.read_text()))
        self.assertGreaterEqual(
            next_meter.sample()["cumulative_new_bytes_lower_bound"], before
        )


class EndedResearchTests(unittest.TestCase):
    def setUp(self):
        from test_midterm_host import LedgerTests
        from rsi_vla.midterm_host import MidtermLedger

        LedgerTests.setUp(self)
        self.scope["outer_limits"].update(model_requests=10, research_requests=5)
        atomic_json(self.scope_ref, self.scope)
        if self.ledger.path.exists():
            self.ledger.path.unlink()
        self.ledger = MidtermLedger(self.scope_ref)
        self.op = {
            "operation_id": "r-model",
            "kind": "research",
            "payload": {"request_id": "r"},
            "dynamic": True,
        }
        self.job = self.root / "host/jobs/r-model"
        self.output = self.root / "output"
        self.store = self.root / "store"
        self.source = self.root / "source/manifest.json"
        atomic_json(self.source, {"files": [], "version": "5a-assets-offline"})
        import rsi_vla.bounded_runner

        repository = Path(rsi_vla.bounded_runner.__file__).parents[3]
        gate = self.source.parent / "source/core/src/rsi_vla/adapters/request_gate.py"
        gate.parent.mkdir(parents=True)
        gate.write_bytes(
            (repository / "core/src/rsi_vla/adapters/request_gate.py").read_bytes()
        )
        protocol = self.root / "protocol.json"
        atomic_json(protocol, {"protocol_id": "p"})
        self.request = {
            "request_id": "r",
            "job_id": "job",
            "base_bundle_id": "base",
            "permit_ref": str(self.root / "permit.json"),
            "output_ref": str(self.output),
        }
        self.host = SimpleNamespace(
            root=self.root / "host",
            ledger=self.ledger,
            config={
                "scope_id": self.scope["scope_id"],
                "repository_root": str(repository),
            },
            core_config={
                "research_requests": [self.request],
                "operations": [self.op],
                "source_manifest_ref": str(self.source),
                "base_bundle_id": "base",
                "protocol_ref": str(protocol),
                "research_store_ref": str(self.store),
            },
        )
        self.report = {
            "identity": {"run_id": "r-model"},
            "status": "interrupted",
            "cleanup_confirmed": True,
            "remaining_process_ids": [],
            "wall_seconds": 5.6,
            "storage_bytes": 31157082,
            "stop_reason": "storage_bytes",
        }
        atomic_json(
            self.job / "submission.json",
            {"operation": self.op, "owner_uid": os.getuid()},
        )
        atomic_json(self.job / "supervisor.json", self.report)
        atomic_json(
            self.job / "completion.json",
            {"status": "unknown", "missing": ["model_requests", "research_requests"]},
        )
        identity = {
            "run_id": "r",
            "episode_id": "research-only",
            "bundle_id": "base",
            "protocol_id": "p",
        }
        atomic_json(
            self.output / "runtime_context.json",
            {
                **identity,
                "job_id": "job",
                "authorization_ref": self.request["permit_ref"],
            },
        )
        atomic_json(self.output / "child-started.json", {"request_id": "r"})
        atomic_json(
            self.output / "request_gate_live.json",
            {"identity": identity, "requests_forwarded": 0},
        )
        atomic_json(
            self.store / "jobs/job.json",
            {
                "job_id": "job",
                "bundle_id": "base",
                "status": "running",
                "limits": {
                    "model_requests": 10,
                    "whole_job_wall_seconds": 100,
                    "artifact_bytes": 100000000,
                },
                "used": {
                    "model_requests": 0,
                    "whole_job_wall_seconds": 0,
                    "artifact_bytes": 0,
                },
                "charges": [],
            },
        )
        self.ledger.reserve(
            "r-model",
            budget={
                "model_requests": 1,
                "research_requests": 1,
                "storage_bytes": 1000000,
                "current_disk_bytes": 1000000,
            },
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.job / "completion.json",
        )
        self.ledger.mark_submitted("r-model")
        self.ledger.settle(
            "r-model",
            actual={"storage_bytes": 31157082, "current_disk_bytes": 9951945},
            cleanup_confirmed=True,
        )
        self.original = {
            p: p.read_bytes()
            for p in (
                self.job / "completion.json",
                self.job / "supervisor.json",
                self.output / "runtime_context.json",
            )
        }

    def test_zero_forward_closure_preserves_old_unknown_and_overlimit_cost(self):
        from rsi_vla.research_supervision import close_ended_research
        from rsi_vla.midterm_host import MidtermContinuationBackend

        with (
            patch(
                "rsi_vla.research_supervision.research_actor_evidence",
                return_value={"established": True},
            ),
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={"live_processes": [], "unreadable_owned_processes": []},
            ),
        ):
            receipt = close_ended_research(self.host, "r-model", self.report)
            self.assertEqual(
                close_ended_research(self.host, "r-model", self.report), receipt
            )
        self.assertEqual(receipt["value"]["research_outcome"], "unknown")
        self.assertTrue(receipt["value"]["research_technical_stop"])
        record = self.ledger.read()["reservations"]["r-model"]
        self.assertEqual(record["actual"]["model_requests"], 0)
        self.assertEqual(record["actual"]["research_requests"], 0)
        self.assertEqual(record["actual"]["storage_bytes"], 31157082)
        self.assertEqual(record["status"], "settled")
        for p, body in self.original.items():
            self.assertEqual(p.read_bytes(), body)
        self.assertFalse((self.output / "sdk-result.json").exists())
        self.assertFalse((self.store / "llm_executions/r.json").exists())
        backend = object.__new__(MidtermContinuationBackend)
        self.assertEqual(
            backend.observe({"receipts": {"r-model": receipt}}, {}),
            {"terminal": "research_pre_forward_technical_stop"},
        )
        self.assertEqual(
            json.loads((self.store / "jobs/job.json").read_text())["used"][
                "model_requests"
            ],
            0,
        )

    def test_new_or_unknown_actor_keeps_forwarding_unresolved(self):
        from rsi_vla.research_supervision import close_ended_research

        before = self.ledger.path.read_bytes()
        with (
            patch(
                "rsi_vla.research_supervision.research_actor_evidence",
                return_value={"established": True},
            ),
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={
                    "live_processes": [],
                    "unreadable_owned_processes": [999],
                },
            ),
        ):
            with self.assertRaisesRegex(PermissionError, "fully reconciled"):
                close_ended_research(self.host, "r-model", self.report)
        self.assertEqual(self.ledger.path.read_bytes(), before)
        self.assertFalse((self.job / "host-ended-receipt.json").exists())

    def test_prepare_and_feedback_overhead_charge_original_job_once(self):
        from rsi_vla.research_supervision import bill_research_overhead

        self.host.core_config["research_request_envelope"] = "complete_v2"
        entry = {"wall_seconds": 2.5, "storage_bytes": 123}
        bill_research_overhead(self.host, self.op, entry)
        before = (self.store / "jobs/job.json").read_bytes()
        bill_research_overhead(self.host, self.op, entry)
        self.assertEqual((self.store / "jobs/job.json").read_bytes(), before)
        job = json.loads(before)
        self.assertEqual(job["used"]["whole_job_wall_seconds"], 2.5)
        self.assertEqual(job["used"]["artifact_bytes"], 123)
        self.assertEqual(job["limits"]["model_requests"], 10)


class RecoveryPreparationTests(unittest.TestCase):
    def test_builder_freezes_new_identities_with_original_job_and_no_runtime_replay(
        self,
    ):
        from rsi_vla.research_supervision import prepare_research_storage_recovery
        import rsi_vla.midterm_host

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repository = Path(rsi_vla.midterm_host.__file__).parents[3]
            scope_ref = root / "scope.json"
            atomic_json(scope_ref, {"scope_id": "scope", "write_root": str(root)})
            protocol = root / "protocol.json"
            atomic_json(
                protocol,
                {
                    "task_policy": {
                        "research_per_request": {
                            "wall_seconds": 180,
                            "storage_bytes": 16 * 1024**2,
                        }
                    }
                },
            )
            domain = root / "old-domain.json"
            atomic_json(
                domain,
                {
                    "purpose": "drawer diagnosis",
                    "evidence_selection": {
                        "media_refs": "selected_events",
                        "sdk_projection": "terminal_history_v1",
                    },
                },
            )
            manifest = root / "old-source/manifest.json"
            atomic_json(
                manifest,
                {
                    "version": "5a-assets-offline",
                    "files": ["core/src/rsi_vla/bounded_runner.py"],
                },
            )
            phase = root / "phase.json"
            atomic_json(phase, {"unchanged": "old totals"})
            inputs = root / "old-inputs.json"
            atomic_json(
                inputs,
                {
                    "authorization": {
                        "binding": {"registration_ref": "old"},
                        "operation_ids": ["r-model"],
                    }
                },
            )
            old_ref = root / "old-host.json"
            atomic_json(old_ref, {"immutable": "original host"})
            ops = [
                {
                    "operation_id": "d0-runtime",
                    "kind": "runtime",
                    "dynamic": True,
                    "payload": {"run_id": "d0"},
                },
                {
                    "operation_id": "r-prepare",
                    "kind": "research_prepare",
                    "dynamic": False,
                    "payload": {"request_id": "r"},
                },
                {
                    "operation_id": "r-model",
                    "kind": "research",
                    "dynamic": True,
                    "payload": {"request_id": "r"},
                },
                {
                    "operation_id": "r-feedback",
                    "kind": "research_feedback",
                    "dynamic": False,
                    "payload": {"request_id": "r", "feedback_id": "r-feedback"},
                },
            ]
            # Real identities are deliberately non-prefix names in production.
            rid = "research-one"
            for op in ops[1:]:
                op["operation_id"] = op["operation_id"].replace("r-", rid + "-")
                op["payload"]["request_id"] = rid
            core = {
                "operations": ops,
                "scope_authorization_ref": str(scope_ref),
                "protocol_ref": str(protocol),
                "source_manifest_ref": str(manifest),
                "repository_root": str(repository),
                "phase_ref": str(phase),
                "base_bundle_id": "base",
                "research_store_ref": str(root / "store"),
                "research_requests": [
                    {
                        "request_id": rid,
                        "job_id": "same-job",
                        "job_limits": {"model_requests": 22},
                        "domain_policy_ref": str(domain),
                        "permit_ref": str(root / "old-permit.json"),
                        "preflight_ref": str(root / "old-preflight.json"),
                        "output_ref": str(root / "output" / rid),
                        "scratch_ref": str(root / "scratch" / rid),
                        "reserved_bundle_id": rid + "-proposal",
                    }
                ],
                "continuation": {
                    "start": "d0",
                    "nodes": {
                        "d0": {
                            "operation_id": "d0-runtime",
                            "transitions": [{"next": "prepare"}],
                        },
                        "prepare": {
                            "operation_id": rid + "-prepare",
                            "transitions": [{"next": "model"}],
                        },
                        "model": {
                            "operation_id": rid + "-model",
                            "transitions": [{"next": "feedback"}],
                        },
                        "feedback": {
                            "operation_id": rid + "-feedback",
                            "transitions": [{"next": "done"}],
                        },
                        "done": {"terminal": "done"},
                    },
                },
            }
            host_config = {
                "inputs_ref": str(inputs),
                "hub_configuration_ref": "hub",
                "python": sys.executable,
                "script_ref": str(repository / "scripts/run_midterm_host.py"),
                "poll_seconds": 30,
                "node": "node",
                "autoloop_root": "autoloop",
                "autoloop_home": "home",
            }
            previous = SimpleNamespace(
                core_config=core,
                ledger=SimpleNamespace(
                    root=root, authority=lambda: {"scope_id": "scope"}
                ),
                config=host_config,
                configuration_ref=old_ref,
                coordinator=lambda: SimpleNamespace(
                    status=lambda: {
                        "status": "completed",
                        "terminal": "research_pre_forward_technical_stop",
                    }
                ),
            )
            fresh = SimpleNamespace(coordinator=lambda: None)
            before = {p: p.read_bytes() for p in (old_ref, domain, phase, manifest)}
            with patch(
                "rsi_vla.midterm_host.MidtermHost", side_effect=[previous, fresh]
            ):
                result = prepare_research_storage_recovery(
                    old_ref, "research-storage-v2", "storage-v2"
                )
            cfg = json.loads(
                (
                    root
                    / "operator/research-storage-v2/continuation-configuration.json"
                ).read_text()
            )
            self.assertEqual(
                cfg["research_requests"][0]["request_id"], rid + "-storage-v2"
            )
            self.assertEqual(cfg["research_requests"][0]["job_id"], "same-job")
            self.assertEqual(
                cfg["research_requests"][0]["job_limits"], {"model_requests": 22}
            )
            self.assertEqual(cfg["phase_ref"], str(phase))
            self.assertEqual(cfg["continuation"]["start"], "prepare")
            self.assertNotIn("runtime", [op["kind"] for op in cfg["operations"]])
            self.assertEqual(cfg["research_request_envelope"], "complete_v2")
            self.assertFalse(result["actual_dispatch"])
            self.assertEqual(
                json.loads(
                    Path(cfg["research_requests"][0]["domain_policy_ref"]).read_text()
                ),
                json.loads(domain.read_text()),
            )
            for p, body in before.items():
                self.assertEqual(p.read_bytes(), body)


class CompactRunnerTests(unittest.TestCase):
    from test_batch4 import RunnerTest as _Original

    setUp = _Original.setUp
    result = _Original.result
    test_compact_execution_reconciles_without_duplicate_dispatch = _Original.test_one_dispatch_reconcile_no_repeat_charge_and_no_applicable_no_dispatch
    test_compact_child_validates_permit_before_sdk = (
        _Original.test_high_child_uses_actual_sdk_effort_without_network
    )

    def prepared(self):
        from test_batch4 import RunnerTest

        adapter, transport, output, permit, path = RunnerTest.prepared(self)
        permit.pop("input")
        permit.update(
            compact_input_binding(self.store.path("bounded_transports", "new-request"))
        )
        atomic_json(path, permit)
        return adapter, transport, output, permit, path


if __name__ == "__main__":
    unittest.main(verbosity=2)
