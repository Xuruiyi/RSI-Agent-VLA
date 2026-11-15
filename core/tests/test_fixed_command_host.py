"""Fixed jobs consume current permits and actual meters, including partial failures."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.fixed_command import fixed_measurements, fixed_permit, probe_fixed_gpu
from rsi_vla.midterm_host import MidtermHost, MidtermLedger
from rsi_vla.storage import atomic_json


class FixedCommandHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope_ref = self.root / "scope.json"
        limits = {
            "runtime_starts": 4,
            "training_starts": 2,
            "training_pilots": 2,
            "learning_candidates": 2,
            "load_checks": 2,
            "gpu_seconds": 100,
            "storage_bytes": 1000000,
            "current_disk_bytes": 1000000,
            "download_bytes": 100,
            "model_requests": 10,
            "concurrent_gpu_devices": 1,
        }
        limits["resource_wait_seconds"] = 10
        scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": "fixed-test",
            "status": "active",
            "authorized": True,
            "dynamic_authorized": True,
            "owner_uid": os.getuid(),
            "write_root": str(self.root),
            "storage_write_roots": [str(self.root)],
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "outer_limits": limits,
        }
        atomic_json(self.scope_ref, scope)
        self.ledger = MidtermLedger(self.scope_ref)
        phase_limits = {
            "wall_seconds": 500,
            "gpu_seconds": 100,
            "storage_bytes": 1000000,
            "model_requests": 10,
            "download_bytes": 100,
            "simulation_episodes": 4,
            "training_starts": 2,
            "training_pilots": 2,
            "load_checks": 2,
        }
        self.phase = self.root / "phase.json"
        atomic_json(
            self.phase,
            {
                "limits": phase_limits,
                "used": dict.fromkeys(phase_limits, 0),
                "remaining": phase_limits,
                "unresolved_costs": [],
            },
        )
        self.protocol = self.root / "protocol.json"
        atomic_json(self.protocol, {"protocol_id": "driver"})
        self.host = SimpleNamespace(
            ledger=self.ledger,
            config={"scope_id": "fixed-test", "operations": {}},
            core_config={
                "phase_ref": str(self.phase),
                "protocol_ref": str(self.protocol),
                "source_manifest_ref": str(self.root / "source.json"),
                "operations": [],
            },
            root=self.root / "host",
        )

    def template(self, kind="libero-public-data-acquire", operation_id="one"):
        output = self.root / operation_id
        target = self.root / "permits" / (operation_id + ".json")
        cpu = kind.startswith("libero-public-data-")
        body = {
            "kind": kind,
            "protocol_id": "family-protocol",
            "registration_ref": str(self.root / "registration.json"),
            "registered_plan": {"episodes": [1, 3]},
            "source_manifest_ref": str(self.root / "family-source.json"),
            "output_root": str(output),
            "run_id": operation_id,
        }
        budget = {"storage_bytes": 1000, "current_disk_bytes": 1000}
        if cpu:
            body.update(wall_seconds=8, storage_bytes=900, current_disk_bytes=900)
            devices = []
            if kind.endswith("acquire"):
                body["max_download_bytes"] = 100
                budget["download_bytes"] = 100
        else:
            body["budget"] = {
                "wall_seconds": 8,
                "gpu_device_seconds": 8,
                "reserved_devices": 1,
                "artifact_bytes": 900,
                "max_steps": 1,
            }
            budget["gpu_seconds"] = 10
            devices = [0]
            budget[
                "runtime_starts"
                if kind == "libero-scripted-teacher"
                else "load_checks"
                if kind == "pi05-load-check"
                else "training_starts"
            ] = 1
        if kind.startswith("libero-"):
            budget["model_requests"] = 0
        template = {
            "command": [
                sys.executable,
                "registered-consumer.py",
                "--permit",
                str(target),
            ],
            "permit_ref": str(target),
            "permit_template": body,
            "budget": budget,
            "gpu_devices": devices,
            "supervisor_limits": {"wall_seconds": 10, "storage_bytes": 1000},
            "result_ref": str(output / "job-result.json"),
            "model_meter_ref": str(output / "model-meter.json"),
            "download_ledger_ref": str(output / "download-ledger.jsonl"),
            "start_marker_ref": str(output / "native-start.json"),
            "output_ref": str(output),
            "download_meter_ref": str(output / "download-meter.json"),
            "storage_write_roots": [str(output)],
        }
        if kind == "pi05-training-learning":
            body["candidate_id"] = "candidate-one"
            template.update(
                candidate_id="candidate-one",
                candidate_binding={"recipe": "frozen-data-seed-stop"},
            )
            template["budget"]["learning_candidates"] = 1
        self.host.config["operations"][operation_id] = template
        operation = {
            "operation_id": operation_id,
            "kind": "host_command",
            "dynamic": True,
            "payload": {},
            "stage": "test",
        }
        self.host.core_config["operations"].append(operation)
        return operation, template

    def report(self, operation_id="one", **overrides):
        return {
            "identity": {"run_id": operation_id},
            "status": "completed",
            "exit_code": 0,
            "cleanup_confirmed": True,
            "wall_seconds": 3,
            "gpu_device_seconds_envelope": 0,
            "storage_bytes": 20,
            "storage_accounting": {"current_new_bytes": 20},
            **overrides,
        }

    def meter(self, template, operation_id="one", **overrides):
        schema = (
            "rsi-scripted-teacher-model-meter-v1"
            if template["permit_template"]["kind"] == "libero-scripted-teacher"
            else "rsi-public-data-model-meter-v1"
        )
        atomic_json(
            template["model_meter_ref"],
            {
                "schema": schema,
                "scope_id": "fixed-test",
                "operation_id": operation_id,
                "status": "closed",
                "model_clients_created": [],
                "model_transports_created": [],
                "project_model_requests": 0,
                "owner_uid": os.getuid(),
                **overrides,
            },
        )

    def write_download(
        self, template, counts, *, closed, complete=False, producer=None
    ):
        identity = {
            "scope_id": "fixed-test",
            "operation_id": "one",
            "owner_uid": os.getuid(),
        }
        Path(template["download_ledger_ref"]).parent.mkdir(parents=True, exist_ok=True)
        Path(template["download_ledger_ref"]).write_text(
            "".join(
                json.dumps(
                    {
                        **identity,
                        "url": "https://fixture.invalid/blob",
                        "response_payload_bytes": value,
                    }
                )
                + "\n"
                for value in counts
            )
        )
        atomic_json(
            template["download_meter_ref"],
            {
                "schema": "rsi-public-data-download-meter-v1",
                **identity,
                "status": "closed" if closed else "unresolved",
                "download_meter_closed": closed,
                "download_complete": complete,
                "download_response_payload_bytes": sum(counts) if closed else None,
                "observed_download_payload_bytes_lower_bound": sum(counts),
                "producer_returned_payload_bytes": sum(counts)
                if producer is None
                else producer,
                "download_ledger_ref": template["download_ledger_ref"],
            },
        )

    def test_actual_families_get_current_permits_without_replacing_family_bindings(
        self,
    ):
        for kind in (
            "libero-public-data-acquire",
            "libero-public-data-materialize",
            "libero-scripted-teacher",
            "pi05-training-pilot",
            "pi05-training-learning",
            "pi05-load-check",
        ):
            with self.subTest(kind=kind):
                operation, template = self.template(kind, kind)
                MidtermHost.materialize(self.host, operation)
                permit = json.loads(Path(template["permit_ref"]).read_text())
                self.assertEqual(permit["registered_plan"], {"episodes": [1, 3]})
                self.assertEqual(permit["protocol_id"], "family-protocol")
                self.assertEqual(
                    permit["source_manifest_ref"], str(self.root / "family-source.json")
                )
                self.assertEqual(permit["scope_ref"], str(self.scope_ref))
                self.assertEqual(permit["operation_id"], operation["operation_id"])
                self.assertEqual(permit["owner_uid"], os.getuid())
                self.assertEqual(permit["phase_ref"], str(self.phase))
                self.assertEqual(permit["accounting_ref"], str(self.ledger.path))
        self.assertFalse(self.ledger.read()["reservations"])

    def test_fixed_gpu_probe_uses_actual_free_memory_and_capability_on_registered_device(
        self,
    ):
        _, template = self.template("pi05-training-pilot")
        template["gpu_admission"] = {
            "required_memory_mib": 20000,
            "safety_margin_mib": 1000,
            "minimum_compute_capability": 8.0,
        }
        cards = [
            {
                "physical_gpu": 0,
                "name": "compatible",
                "free_memory_mib": 23000,
                "utilization_percent": 0,
                "compute_capability": "8.9",
            }
        ]
        self.assertEqual(
            probe_fixed_gpu(template, gpu_probe=lambda: cards)["blockers"], []
        )
        self.assertTrue(
            probe_fixed_gpu(
                template, gpu_probe=lambda: [{**cards[0], "free_memory_mib": 1000}]
            )["blockers"]
        )
        self.assertTrue(
            probe_fixed_gpu(
                template, gpu_probe=lambda: [{**cards[0], "compute_capability": "7.5"}]
            )["blockers"]
        )
        self.assertTrue(
            probe_fixed_gpu(
                template, gpu_probe=lambda: [{**cards[0], "physical_gpu": 1}]
            )["blockers"]
        )

    def test_fixed_gpu_unavailability_enters_existing_wait_before_permit_or_submission(
        self,
    ):
        operation, template = self.template("pi05-training-pilot")
        template["gpu_admission"] = {
            "required_memory_mib": 20000,
            "safety_margin_mib": 1000,
            "minimum_compute_capability": 8.0,
        }
        inputs_ref = self.root / "inputs.json"
        atomic_json(inputs_ref, {"resources": {"reserve": {}}})
        self.host.config["inputs_ref"] = str(inputs_ref)
        state = {
            "pending": operation,
            "status": "ready",
            "paused": False,
            "receipts": {},
        }
        self.host.coordinator = lambda: SimpleNamespace(status=lambda: state)
        self.host.inputs = lambda: MidtermHost.inputs(self.host)
        self.host.resource_wait = lambda: MidtermHost.resource_wait(self.host)
        phase = json.loads(self.phase.read_text())
        with (
            patch(
                "rsi_vla.task_continuation.probe_admission",
                return_value={
                    "resources": {
                        "available": phase["remaining"],
                        "required": {},
                        "blockers": [],
                        "unresolved_costs": [],
                    }
                },
            ),
            patch(
                "rsi_vla.fixed_command.query_gpu_cards",
                return_value=[
                    {
                        "physical_gpu": 0,
                        "name": "compatible",
                        "free_memory_mib": 1000,
                        "utilization_percent": 0,
                        "compute_capability": "8.9",
                    }
                ],
            ),
        ):
            result = MidtermHost.advance(self.host)
        self.assertEqual(result["status"], "waiting_resources")
        self.assertFalse(Path(template["permit_ref"]).exists())
        self.assertFalse(self.ledger.read()["reservations"])

    def test_failed_download_uses_all_durable_chunks_and_is_charged_only_once(self):
        operation, template = self.template()
        fixed_permit(self.host, operation, template)
        atomic_json(
            template["result_ref"],
            {
                "scope_id": "fixed-test",
                "operation_id": "one",
                "status": "failed",
                "download_ledger_ref": template["download_ledger_ref"],
            },
        )
        self.meter(template)
        self.write_download(template, [7, 5], closed=True, complete=False)
        self.ledger.reserve(
            "one",
            budget=template["budget"],
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.root / "completion.json",
        )
        self.ledger.mark_submitted("one")
        self.host.charge_phase = lambda identity, report: MidtermHost.charge_phase(
            self.host, identity, report
        )
        MidtermHost.finish(self.host, "one", self.report())
        MidtermHost.finish(self.host, "one", self.report())
        ledger = self.ledger.read()
        self.assertEqual(ledger["used"]["download_bytes"], 12)
        self.assertEqual(ledger["used"]["model_requests"], 0)
        self.assertEqual(ledger["reservations"]["one"]["status"], "settled")
        phase = json.loads(self.phase.read_text())
        self.assertEqual(phase["used"]["download_bytes"], 12)
        self.assertEqual(phase["used"]["wall_seconds"], 3)
        self.assertEqual(phase["used"]["storage_bytes"], 20)

    def test_hard_stopped_download_charges_lower_bound_and_retains_unknown_remainder(
        self,
    ):
        operation, template = self.template()
        fixed_permit(self.host, operation, template)
        Path(template["download_ledger_ref"]).parent.mkdir(parents=True)
        self.write_download(template, [7], closed=False)
        measured = fixed_measurements(
            self.host, "one", template, self.report(status="interrupted")
        )
        self.assertEqual(measured["actual"]["download_bytes"], 7)
        self.assertIn("download_bytes", measured["unknown_dimensions"])
        self.assertIn("model_requests", measured["unknown_dimensions"])
        self.ledger.reserve(
            "one",
            budget=template["budget"],
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.root / "completion.json",
        )
        self.ledger.settle(
            "one",
            actual={"download_bytes": 7, "storage_bytes": 20, "current_disk_bytes": 20},
            cleanup_confirmed=True,
            unknown_dimensions=measured["unknown_dimensions"],
        )
        record = self.ledger.read()["reservations"]["one"]
        self.assertEqual(record["status"], "unknown")
        self.assertEqual(self.ledger._held(record, "download_bytes"), 93)
        self.assertEqual(self.ledger._held(record, "storage_bytes"), 0)

    def test_teacher_marker_and_meter_require_exact_scope_operation_owner_and_lifecycle(
        self,
    ):
        operation, template = self.template("libero-scripted-teacher")
        permit = fixed_permit(self.host, operation, template)
        atomic_json(
            template["start_marker_ref"],
            {
                "schema": "rsi-libero-teacher-start-v1",
                "scope_id": "fixed-test",
                "operation_id": "one",
                "owner_uid": os.getuid(),
                "run_id": "one",
                "kind": permit["kind"],
            },
        )
        self.meter(template, request_cap=0, enabled_components=["native_env"])
        measured = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(measured["actual"]["runtime_starts"], 1)
        self.assertEqual(measured["actual"]["model_requests"], 0)
        self.meter(template, operation_id="another")
        measured = fixed_measurements(self.host, "one", template, self.report())
        self.assertNotIn("model_requests", measured["actual"])
        self.assertIn("model_requests", measured["unknown_dimensions"])

    def test_unknown_read_and_unlogged_returned_tail_keep_residual_reserved(self):
        for producer, expected in ((2, 2), (5, 5)):
            with self.subTest(producer=producer):
                operation, template = self.template()
                fixed_permit(self.host, operation, template)
                atomic_json(
                    template["result_ref"],
                    {
                        "scope_id": "fixed-test",
                        "operation_id": "one",
                        "status": "failed",
                    },
                )
                self.meter(template)
                self.write_download(template, [2], closed=False, producer=producer)
                result = fixed_measurements(self.host, "one", template, self.report())
                self.assertEqual(result["actual"]["download_bytes"], expected)
                self.assertIn("download_bytes", result["unknown_dimensions"])
                self.assertEqual(result["actual"]["model_requests"], 0)

    def test_missing_download_stream_is_unknown_and_empty_explicit_closed_stream_is_zero(
        self,
    ):
        operation, template = self.template()
        fixed_permit(self.host, operation, template)
        before = fixed_measurements(self.host, "one", template, self.report())
        self.assertNotIn("download_bytes", before["actual"])
        self.assertIn("download_bytes", before["unknown_dimensions"])
        self.write_download(template, [], closed=True)
        after = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(after["actual"]["download_bytes"], 0)
        self.assertNotIn("download_bytes", after["unknown_dimensions"])

    def test_failed_result_and_complete_ledger_do_not_override_unknown_meter(self):
        operation, template = self.template()
        fixed_permit(self.host, operation, template)
        self.write_download(template, [2], closed=False)
        atomic_json(
            template["result_ref"],
            {
                "scope_id": "fixed-test",
                "operation_id": "one",
                "status": "failed",
                "download_complete": False,
            },
        )
        result = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(result["actual"]["download_bytes"], 2)
        self.assertIn("download_bytes", result["unknown_dimensions"])
        meter = json.loads(Path(template["download_meter_ref"]).read_text())
        meter.update(
            status="closed",
            download_meter_closed=True,
            download_response_payload_bytes=2,
            producer_returned_payload_bytes=5,
        )
        atomic_json(template["download_meter_ref"], meter)
        result = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(result["actual"]["download_bytes"], 5)
        self.assertIn("download_bytes", result["unknown_dimensions"])

    def test_stage_verdict_preserves_native_failure_and_reconcile_never_relaunches(
        self,
    ):
        operation, template = self.template("libero-scripted-teacher")
        permit = fixed_permit(self.host, operation, template)
        native = Path(template["result_ref"])
        inner = native.with_name("supervision.json")
        atomic_json(inner, {"cleanup_confirmed": True, "remaining_owned_pids": []})
        atomic_json(
            native,
            {
                "schema": "rsi-libero-teacher-job-result-v1",
                "scope_id": "fixed-test",
                "operation_id": "one",
                "run_id": "one",
                "kind": permit["kind"],
                "status": "failed",
                "native_success": False,
                "supervision_ref": str(inner),
            },
        )
        before = native.read_bytes()
        template["cost_receipt_ref"] = str(native)
        template["result_ref"] = str(native.with_name("stage-result.json"))
        template["result_schema"] = "rsi-midterm-p2-input-stage-result-v1"
        stage = {
            "schema": template["result_schema"],
            "scope_id": "fixed-test",
            "operation_id": "one",
            "owner_uid": os.getuid(),
            "status": "completed",
            "stage_outcome": "reviewable_negative",
        }
        atomic_json(template["result_ref"], stage)
        with patch(
            "rsi_vla.midterm_host.subprocess.run",
            side_effect=AssertionError("must not rerun committed command"),
        ):
            value = MidtermHost.invoke(self.host, operation, "reconcile", {})
        self.assertEqual(value, stage)
        self.assertEqual(native.read_bytes(), before)
        atomic_json(template["result_ref"], {**stage, "operation_id": "foreign"})
        with self.assertRaises(ValueError):
            MidtermHost.invoke(self.host, operation, "reconcile", {})

    def test_business_state_excludes_history_and_scope_state_is_shared_reference(self):
        from rsi_vla.fixed_command import fixed_storage_context

        history = self.root / "historical"
        history.mkdir()
        for index in range(500):
            (history / str(index)).write_bytes(b"x")
        one, template = self.template(operation_id="first")
        first = fixed_storage_context(self.host, one, template)
        two, other = self.template(operation_id="second")
        second = fixed_storage_context(self.host, two, other)
        self.assertEqual(
            first["scope_meter_state_ref"], second["scope_meter_state_ref"]
        )
        self.assertNotIn("storage_baseline", first)
        business = json.loads(Path(first["storage_meter_state_ref"]).read_text())
        scope = json.loads(Path(first["scope_meter_state_ref"]).read_text())
        self.assertFalse(any("historical" in name for name in business["previous"]))
        self.assertTrue(any("historical" in name for name in scope["previous"]))

    def test_real_coordinator_recovers_lost_fixed_return_without_rerun_or_overwriting_unknown(
        self,
    ):
        from rsi_vla.midterm_host import MidtermContinuationBackend
        from rsi_vla.protocol10_coordinator import Coordinator

        operation, template = self.template()
        operation["resource_requirements"] = {"storage_bytes": 1}
        fixed_permit(self.host, operation, template)
        result = {
            "schema": "rsi-libero-public-data-result-v1",
            "scope_id": "fixed-test",
            "operation_id": "one",
            "owner_uid": os.getuid(),
            "status": "failed",
            "action": "acquire",
        }
        atomic_json(template["result_ref"], result)
        self.meter(template)
        self.write_download(template, [7], closed=True)
        native_before = Path(template["result_ref"]).read_bytes()
        self.host.config["continuation_root"] = str(self.root / "continuation")
        self.host.core_config.update(
            schema="rsi-task-completion-v1",
            scope_authorization_ref=str(self.scope_ref),
            continuation={
                "schema": "rsi-task-continuation-v1",
                "start": "job",
                "nodes": {
                    "job": {"operation_id": "one", "transitions": [{"next": "stop"}]},
                    "stop": {"terminal": "failed_operation_stop"},
                },
            },
        )
        calls = []

        def lost_return(op, action, inputs):
            calls.append(action)
            raise RuntimeError(
                "fixture worker died after original native terminal persisted"
            )

        self.host.invoke = lost_return
        backend = MidtermContinuationBackend(self.host)
        coordinator = Coordinator(
            self.root / "continuation/coordinator", backend.registration, backend
        )
        inputs = {
            "now": datetime.now(timezone.utc).isoformat(),
            "authorization": {
                "dynamic_authorized": True,
                "binding": backend.registration,
                "expires_at": json.loads(self.scope_ref.read_text())["expires_at"],
                "operation_ids": ["one"],
            },
            "resources": {
                "available": json.loads(self.phase.read_text())["remaining"],
                "required": {"storage_bytes": 1},
                "reserve": {},
            },
        }
        first = coordinator.advance(inputs)
        self.assertEqual(first["pending_receipt"]["status"], "unknown")
        job = self.host.root / "jobs/one"
        atomic_json(
            job / "submission.json", {"operation": operation, "owner_uid": os.getuid()}
        )
        atomic_json(
            job / "process.json",
            {
                "pid": 999999999,
                "owner_uid": os.getuid(),
                "start_ticks": 1,
                "boot_id": "offline-fixture",
            },
        )
        report = self.report(exit_code=1, remaining_process_ids=[])
        atomic_json(job / "supervisor.json", report)
        self.ledger.reserve(
            "one",
            budget=template["budget"],
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=job / "completion.json",
        )
        self.ledger.mark_submitted("one")
        self.host.charge_phase = lambda identity, report: MidtermHost.charge_phase(
            self.host, identity, report
        )
        MidtermHost.finish(self.host, "one", report)
        before = (job / "completion.json").read_bytes()
        self.host.coordinator = lambda: coordinator
        self.host.inputs = lambda: inputs
        self.host.backend = lambda: backend
        with patch("rsi_vla.midterm_host.process_matches", return_value=True):
            self.assertEqual(MidtermHost.advance(self.host)["status"], "running")
        recovered = MidtermHost.advance(self.host)
        self.assertTrue(recovered["fixed_result_reconciled"])
        self.assertEqual(calls, ["execute"])
        self.assertEqual((job / "completion.json").read_bytes(), before)
        self.assertEqual(Path(template["result_ref"]).read_bytes(), native_before)
        self.assertEqual(self.ledger.read()["used"]["download_bytes"], 7)
        self.assertTrue((job / "fixed-reconciliation/backend-before.json").exists())
        self.assertTrue((job / "fixed-reconciled.json").exists())
        self.assertEqual(MidtermHost.advance(self.host)["status"], "completed")
        self.assertEqual(calls, ["execute"])

    def test_native_candidate_claim_is_once_per_frozen_identity_and_not_reset_by_retry(
        self,
    ):
        operation, template = self.template("pi05-training-learning")
        self.ledger.register_candidate(
            template["candidate_id"], template["candidate_binding"]
        )
        fixed_permit(self.host, operation, template)
        atomic_json(
            template["start_marker_ref"],
            {
                "schema": "rsi-native-training-start-v1",
                "scope_id": "fixed-test",
                "operation_id": "one",
                "owner_uid": os.getuid(),
                "run_id": "one",
                "kind": "pi05-training-learning",
            },
        )
        first = fixed_measurements(self.host, "one", template, self.report())
        again = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(first["actual"]["learning_candidates"], 1)
        self.assertEqual(again["actual"]["learning_candidates"], 1)
        second, other = self.template("pi05-training-learning", "retry")
        with self.assertRaisesRegex(PermissionError, "once-per-candidate"):
            fixed_permit(self.host, second, other)
        other["budget"]["learning_candidates"] = 0
        fixed_permit(self.host, second, other)
        atomic_json(
            other["start_marker_ref"],
            {
                "schema": "rsi-native-training-start-v1",
                "scope_id": "fixed-test",
                "operation_id": "retry",
                "owner_uid": os.getuid(),
                "run_id": "retry",
                "kind": "pi05-training-learning",
            },
        )
        self.assertEqual(
            fixed_measurements(self.host, "retry", other, self.report("retry"))[
                "actual"
            ]["learning_candidates"],
            0,
        )
        with self.assertRaises(ValueError):
            self.ledger.register_candidate("candidate-one", {"recipe": "changed"})

    def test_native_training_load_model_zero_requires_its_closed_producer(self):
        operation, template = self.template("pi05-load-check")
        template["budget"]["model_requests"] = 0
        fixed_permit(self.host, operation, template)
        missing = fixed_measurements(self.host, "one", template, self.report())
        self.assertIn("model_requests", missing["unknown_dimensions"])
        atomic_json(
            template["model_meter_ref"],
            {
                "schema": "rsi-native-training-model-meter-v1",
                "scope_id": "fixed-test",
                "operation_id": "one",
                "owner_uid": os.getuid(),
                "status": "closed",
                "project_model_requests": 0,
                "model_clients_created": [],
                "model_transports_created": [],
                "components": ["native_loader_one_prediction"],
            },
        )
        closed = fixed_measurements(self.host, "one", template, self.report())
        self.assertEqual(closed["actual"]["model_requests"], 0)
        self.assertNotIn("model_requests", closed["unknown_dimensions"])


if __name__ == "__main__":
    unittest.main()
