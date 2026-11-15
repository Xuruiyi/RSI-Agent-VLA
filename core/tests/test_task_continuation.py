"""Actual Coordinator/receipt DAG continuation with CPU-only role effects."""

import contextlib
import importlib.util
import io
import json
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import test_task_completion
from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.protocol10_roles import ROLES
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskCompletionBackend
from rsi_vla.task_continuation import TaskContinuationBackend, drive, probe_admission

ROOT = Path(__file__).resolve().parents[2]


def operation(name, kind, *, dynamic=False, **payload):
    return {
        "operation_id": name,
        "stage": name,
        "kind": kind,
        "dynamic": dynamic,
        "payload": payload,
        "resource_requirements": {"simulation_episodes": 1} if dynamic else {},
    }


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        test_task_completion.ConsumerAdmissionTests.setUp(self)
        self.scope["runtime_technical_recovery_max"] = 4
        atomic_json(self.config["scope_authorization_ref"], self.scope)
        self.slot["split"] = "development"
        original_permit = json.loads(Path(self.slot["permit_ref"]).read_text())
        recovery = {
            **self.slot,
            "run_id": "recovery",
            "output_ref": str(self.root / "recovered"),
            "scratch_ref": str(self.root / "recovery-scratch"),
            "permit_ref": str(self.root / "recovery-permit.json"),
            "recovery_of_run_id": "run",
            "recovery_basis_ref": str(self.root / "fault-basis.json"),
        }
        self.config["runtime_slots"].append(recovery)
        atomic_json(
            recovery["permit_ref"],
            {
                **original_permit,
                "run_id": recovery["run_id"],
                "output_dir": recovery["output_ref"],
                "storage_write_roots": [
                    recovery["output_ref"],
                    recovery["scratch_ref"],
                ],
            },
        )
        self.config["isolation_task_root"] = str(self.root / "roles")
        self.config["artifacts_root"] = str(self.root / "artifacts")
        self.config["operations"] = [
            operation("run", "runtime", dynamic=True, run_id="run"),
            operation("inspect", "attempt_evaluation", run_id="run"),
            operation("recover", "runtime", dynamic=True, run_id="recovery"),
            operation("inspect-recovery", "attempt_evaluation", run_id="recovery"),
            operation("feedback", "research_feedback", request_id="next"),
        ]
        self.config["continuation"] = {
            "schema": "rsi-task-continuation-v1",
            "start": "execute",
            "resource_wait": {"max_wait_seconds": 10, "check_interval_seconds": 4},
            "nodes": {
                "execute": {
                    "operation_id": "run",
                    "transitions": [{"next": "inspect"}],
                },
                "inspect": {
                    "operation_id": "inspect",
                    "transitions": [
                        {
                            "when": {"path": ["failure_class"], "equals": "technical"},
                            "next": "recover",
                        },
                        {"next": "feedback"},
                    ],
                },
                "recover": {
                    "operation_id": "recover",
                    "transitions": [{"next": "inspect-recovery"}],
                },
                "inspect-recovery": {
                    "operation_id": "inspect-recovery",
                    "transitions": [{"next": "feedback"}],
                },
                "feedback": {
                    "operation_id": "feedback",
                    "transitions": [{"next": "done"}],
                },
                "done": {"terminal": "bounded_task_complete"},
            },
        }
        self.native = TaskCompletionBackend(self.config)
        self.calls = []
        self.verdict = {
            "run_id": "run",
            "validity": "valid",
            "outcome": "failure",
            "failure_class": None,
        }
        self.effects = []
        self.backend = self.new_backend()
        self.coordinator = self.new_coordinator()

    def new_backend(self, invoke=None):
        return TaskContinuationBackend(
            self.root / "continuation", self.config, invoke=invoke or self.invoke
        )

    def new_coordinator(self, backend=None):
        backend = backend or self.backend
        return Coordinator(
            self.root / "continuation/coordinator", backend.registration, backend
        )

    def values(self, available=10):
        return {
            "now": datetime.now(UTC).isoformat(),
            "authorization": {
                "dynamic_authorized": True,
                "binding": deepcopy(self.backend.registration),
                "expires_at": self.future,
                "operation_ids": [
                    op["operation_id"] for op in self.config["operations"]
                ],
            },
            "resources": {
                "available": {"simulation_episodes": available},
                "required": {"simulation_episodes": 1},
                "reserve": {"simulation_episodes": 1},
            },
            "wait_remaining_seconds": 60,
        }

    def invoke(self, op, action, inputs):
        self.calls.append((op["operation_id"], action))
        if op["kind"] == "runtime":
            self.assertEqual(action, "execute")

            # This uses the real TaskCompletion consumer and validate_dynamic;
            # only the final simulator launch is replaced with a CPU effect.
            def prepare(root, artifacts, **kwargs):
                return {
                    "context": {},
                    "run_id": kwargs["run_id"],
                    "output_dir": kwargs["output_dir"],
                }

            def execute(plan):
                self.effects.append(plan["run_id"])
                atomic_json(
                    Path(plan["output_dir"]) / "runtime_context.json",
                    {"run_id": plan["run_id"]},
                )

            with (
                patch("rsi_vla.runtime.prepare_runtime", side_effect=prepare),
                patch("rsi_vla.runtime.execute_runtime", side_effect=execute),
            ):
                return self.native.execute(op, inputs)
        if op["operation_id"] == "inspect":
            if self.verdict.get("failure_class") == "technical":
                atomic_json(self.root / "fault-basis.json", self.verdict)
            return deepcopy(self.verdict)
        if op["operation_id"] == "inspect-recovery":
            return {"run_id": "recovery", "validity": "valid", "outcome": "failure"}
        return {"feedback_recorded": True, "source_outcome": "failure"}

    def finish(self):
        return drive(
            self.coordinator,
            self.values,
            policy=self.config["continuation"]["resource_wait"],
            sleep=lambda delay: self.fail("no resource wait expected"),
        )

    def test_valid_failure_selects_feedback_without_recovery_or_replay(self):
        result = self.finish()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(result["receipts"]), ["run", "inspect", "feedback"])
        self.assertEqual(self.effects, ["run"])
        self.assertEqual(result["receipts"]["inspect"]["value"]["outcome"], "failure")
        calls = list(self.calls)
        restored = self.new_coordinator(self.new_backend())
        self.assertEqual(restored.advance(self.values())["status"], "completed")
        self.assertEqual(self.calls, calls)

    def test_technical_result_takes_registered_guarded_runtime_recovery_edge(self):
        self.verdict.update(
            validity="invalid",
            outcome="unknown",
            failure_class="technical",
            cost_closed=True,
            cleanup_complete=True,
            recovery_cause_resolved=True,
        )
        with patch.object(
            self.native, "validate_dynamic", wraps=self.native.validate_dynamic
        ) as guard:
            result = self.finish()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            list(result["receipts"]),
            ["run", "inspect", "recover", "inspect-recovery", "feedback"],
        )
        self.assertEqual(self.effects, ["run", "recovery"])
        self.assertEqual(
            [call.args for call in guard.call_args_list],
            [("runtime", {"run_id": "run"}), ("runtime", {"run_id": "recovery"})],
        )
        self.assertEqual(result["receipts"]["inspect"]["value"]["outcome"], "unknown")
        self.assertEqual(
            result["receipts"]["inspect-recovery"]["value"]["outcome"], "failure"
        )

    def test_unclosed_technical_fault_never_dispatches_recovery(self):
        self.verdict.update(
            validity="invalid",
            outcome="unknown",
            failure_class="technical",
            cost_closed=True,
            cleanup_complete=False,
            recovery_cause_resolved=True,
        )
        result = self.finish()
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["pending"]["operation_id"], "recover")
        self.assertIn(
            "closed original technical fault", result["pending_receipt"]["error"]
        )
        self.assertEqual(self.effects, ["run"])

    def role_return(self, op, value):
        role = "runtime" if op["kind"] == "runtime" else "evaluation"
        path = (
            Path(self.config["isolation_task_root"])
            / ROLES[role]
            / "role_receipts"
            / (op["operation_id"] + "-execute-revision-00001.json")
        )
        atomic_json(
            path,
            {
                "request": {"configuration": self.config, "task": {"operation": op}},
                "value": value,
            },
        )

    def test_lost_native_return_reconciles_original_role_receipt_after_restart(self):
        def lost(op, action, inputs):
            value = self.invoke(op, action, inputs)
            self.role_return(op, value)
            raise SystemExit("operator died after the native return")

        backend = self.new_backend(lost)
        coordinator = self.new_coordinator(backend)
        with self.assertRaises(SystemExit):
            coordinator.advance(self.values(), trigger_id="first-submit")
        self.assertEqual(self.effects, ["run"])
        restarted = self.new_coordinator(self.new_backend())
        recovered = restarted.advance(self.values())
        self.assertIsNone(recovered["pending"])
        self.assertEqual(
            recovered["receipts"]["run"]["value"], {"returned": True, "run_id": "run"}
        )
        self.assertEqual(self.effects, ["run"])
        self.assertEqual(
            restarted.advance(self.values(), trigger_id="first-submit"), recovered
        )

    def test_delayed_receipt_and_duplicate_events_do_not_resubmit_unknown_effect(self):
        delayed = []

        def lose(op, action, inputs):
            value = self.invoke(op, action, inputs)
            delayed.append((deepcopy(op), value))
            raise SystemExit("return not yet delivered")

        coordinator = self.new_coordinator(self.new_backend(lose))
        with self.assertRaises(SystemExit):
            coordinator.advance(self.values())
        restarted = self.new_coordinator(self.new_backend())
        waiting = restarted.advance(self.values())
        self.assertEqual(waiting["wait"]["reason"], "submission_unknown")
        self.assertEqual(self.effects, ["run"])
        self.role_return(*delayed[0])
        event = {
            "event_id": "native-ready",
            "type": "receipt_available",
            "operation_id": "run",
        }
        first = restarted.event(event)
        self.assertEqual(first, restarted.event(event))
        closed = restarted.advance(self.values(), reconcile_only=True)
        self.assertIsNone(closed["pending"])
        self.assertEqual(list(closed["receipts"]), ["run"])
        self.assertEqual(self.effects, ["run"])

    def test_manual_pause_cancel_and_expiry_do_not_poll_or_dispatch(self):
        for control, status in (("pause", "paused"), ("cancel", "cancelled")):
            with self.subTest(control=control):
                event = {"event_id": control, "type": control, "operation_id": None}
                self.coordinator.event(event)
                restored = self.new_coordinator(self.new_backend())
                result = drive(
                    restored,
                    self.values,
                    policy=self.config["continuation"]["resource_wait"],
                    sleep=lambda delay: self.fail("manual stop cannot be polled"),
                )
                self.assertEqual(result["status"], status)
                self.assertEqual(self.effects, [])
        # A distinct coordinator isolates expiry from the cancelled task.
        backend = TaskContinuationBackend(
            self.root / "expiry", self.config, invoke=self.invoke
        )
        coordinator = Coordinator(
            self.root / "expiry/coordinator", backend.registration, backend
        )
        values = self.values()
        values["authorization"]["binding"] = backend.registration
        values["authorization"]["expires_at"] = (
            datetime.now(UTC) - timedelta(seconds=1)
        ).isoformat()
        expired = drive(
            coordinator,
            lambda: values,
            policy=self.config["continuation"]["resource_wait"],
            sleep=lambda delay: self.fail("expired authority cannot be polled"),
        )
        self.assertEqual(expired["wait"]["reason"], "authorization_expired")
        self.assertEqual(self.effects, [])

    def test_resource_wait_is_bounded_persisted_across_restart_and_rechecks_fresh_inputs(
        self,
    ):
        clock, sleeps = [0.0], []

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        policy = self.config["continuation"]["resource_wait"]
        result = drive(
            self.coordinator,
            lambda: self.values(0),
            policy=policy,
            sleep=sleep,
            clock=lambda: clock[0],
        )
        self.assertEqual(result["wait"]["reason"], "budget_wait")
        self.assertEqual(sleeps, [4, 4, 2])
        wait_path = self.coordinator.root / "resource-wait.json"
        self.assertEqual(
            json.loads(wait_path.read_text()), {"waited_seconds": 10.0, "checks": 3}
        )
        self.assertEqual(self.effects, [])
        restored = self.new_coordinator(self.new_backend())
        drive(
            restored,
            lambda: self.values(0),
            policy=policy,
            clock=lambda: clock[0],
            sleep=lambda seconds: self.fail(
                "restart must not reset the wait allowance"
            ),
        )
        self.assertEqual(json.loads(wait_path.read_text())["waited_seconds"], 10)
        # A fresh availability check can advance without adding wait budget.
        advanced = drive(
            restored,
            self.values,
            policy=policy,
            sleep=lambda seconds: self.fail("resources are now available"),
        )
        self.assertEqual(advanced["status"], "completed")
        self.assertEqual(self.effects, ["run"])

    def test_resource_recovery_is_automatically_rechecked_after_one_bounded_wait(self):
        elapsed, sleeps = [0.0], []

        def sleep(seconds):
            sleeps.append(seconds)
            elapsed[0] += seconds

        result = drive(
            self.coordinator,
            lambda: self.values(10 if sleeps else 0),
            policy=self.config["continuation"]["resource_wait"],
            sleep=sleep,
            clock=lambda: elapsed[0],
        )
        self.assertEqual(sleeps, [4])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.effects, ["run"])

    def configure_gpu_probe(self):
        protocol = json.loads(Path(self.config["protocol_ref"]).read_text())
        protocol["task_policy"]["gpu_admission"] = {
            "required_memory_mib": 20000,
            "safety_margin_mib": 2000,
            "minimum_compute_capability": 8,
            "compatible_names": ["CPU fixture GPU"],
        }
        for key in ("protocol_ref", "frozen_protocol_ref"):
            atomic_json(self.config[key], protocol)
        self.native = TaskCompletionBackend(self.config)
        return {
            "physical_gpu": self.slot["cuda_device"],
            "name": "CPU fixture GPU",
            "free_memory_mib": 24000,
            "utilization_percent": 0,
            "compute_capability": "8.0",
        }

    def continuation_cli(self):
        module_spec = importlib.util.spec_from_file_location(
            "_continuation_admission_cli", ROOT / "scripts/run_task_continuation.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def cli_arguments(self, command, values):
        config_ref = self.root / "fresh-admission-config.json"
        input_ref = self.root / "stale-admission-input.json"
        atomic_json(config_ref, self.config)
        atomic_json(input_ref, values)
        return [
            command,
            "--root",
            str(self.backend.root),
            "--configuration",
            str(config_ref),
            "--inputs",
            str(input_ref),
        ]

    def test_cli_fresh_host_phase_and_gpu_override_stale_admission_input(self):
        card = self.configure_gpu_probe()
        phase = json.loads(Path(self.config["phase_ref"]).read_text())
        phase["remaining"]["simulation_episodes"] = 0
        atomic_json(self.config["phase_ref"], phase)
        stale = self.values(999)
        stale["wait_remaining_seconds"] = 99999
        module, requests, answers = self.continuation_cli(), [], []

        def host(request):
            requests.append(request)
            fresh = probe_admission(
                self.config, request["operation_id"], gpu_probe=lambda: [card]
            )
            answers.append(fresh)
            return fresh

        with (
            patch.dict("os.environ", {"RSI_ISOLATED_ROLE": "operator"}),
            patch.object(module, "TaskContinuationBackend", return_value=self.backend),
            patch("rsi_vla.protocol10_roles.broker_request", side_effect=host),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            module.main(self.cli_arguments("advance", stale))
        result = json.loads(output.getvalue())
        self.assertEqual(result["wait"]["reason"], "budget_wait")
        self.assertEqual(answers[0]["resources"]["available"], phase["remaining"])
        self.assertEqual(
            answers[0]["wait_remaining_seconds"], phase["remaining"]["wall_seconds"]
        )
        self.assertIs(answers[0]["authorization_granted"], False)
        self.assertEqual(
            requests, [{"action": "task_admission", "operation_id": "run"}]
        )
        self.assertEqual(self.effects, [])
        # The host resource response cannot replace the supplied authorization.
        self.assertEqual(stale["authorization"]["binding"], self.backend.registration)

    def test_cli_drive_rechecks_the_registered_gpu_and_advances_when_restored(self):
        card = self.configure_gpu_probe()
        telemetry = [
            {**card, "free_memory_mib": None},
            {**card, "physical_gpu": 1, "free_memory_mib": 64000},
        ]
        elapsed, sleeps, requests = [0.0], [], []
        module = self.continuation_cli()

        def host(request):
            self.assertEqual(set(request), {"action", "operation_id"})
            self.assertEqual(request["action"], "task_admission")
            requests.append(request)
            return probe_admission(
                self.config,
                request["operation_id"],
                gpu_probe=lambda: deepcopy(telemetry),
            )

        def sleep(seconds):
            sleeps.append(seconds)
            self.assertEqual(
                self.effects,
                [],
                "another free GPU must not replace the registered device",
            )
            elapsed[0] += seconds
            telemetry[:] = [card]

        def cpu_drive(coordinator, load_inputs, *, policy):
            return drive(
                coordinator,
                load_inputs,
                policy=policy,
                sleep=sleep,
                clock=lambda: elapsed[0],
            )

        stale = self.values(999)
        stale["resources"]["blockers"] = []
        with (
            patch.dict("os.environ", {"RSI_ISOLATED_ROLE": "operator"}),
            patch.object(module, "TaskContinuationBackend", return_value=self.backend),
            patch.object(module, "drive", side_effect=cpu_drive),
            patch("rsi_vla.protocol10_roles.broker_request", side_effect=host),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            module.main(self.cli_arguments("drive", stale))
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(sleeps, [4])
        self.assertEqual(self.effects, ["run"])
        self.assertGreaterEqual(len(requests), 3)
        wait = json.loads((self.coordinator.root / "resource-wait.json").read_text())
        self.assertEqual(wait["waited_seconds"], 4)

    def test_gpu_unknown_or_missing_registered_device_is_never_assumed_free(self):
        card = self.configure_gpu_probe()
        for cards in (
            [{**card, "free_memory_mib": None}],
            [{**card, "compute_capability": None}],
            [{**card, "physical_gpu": 1}],
            [{**card, "free_memory_mib": 21000}],
        ):
            with self.subTest(cards=cards):
                report = probe_admission(self.config, "run", gpu_probe=lambda: cards)
                self.assertIn(
                    "registered_gpu_unavailable", report["resources"]["blockers"]
                )
                self.assertIs(report["authorization_granted"], False)

        def unavailable():
            raise RuntimeError("CPU telemetry fixture unavailable")

        report = probe_admission(self.config, "run", gpu_probe=unavailable)
        self.assertTrue(report["resources"]["blockers"])
        self.assertIs(report["authorization_granted"], False)

    def test_unregistered_operation_is_rejected_before_any_gpu_probe(self):
        self.configure_gpu_probe()

        def unexpected():
            self.fail("unregistered operation must not reach a GPU probe")

        with self.assertRaisesRegex(ValueError, "exact registered operation"):
            probe_admission(self.config, "unregistered", gpu_probe=unexpected)
        self.assertEqual(self.effects, [])

    def test_interrupted_wait_restart_counts_elapsed_time_without_releasing_pause(self):
        elapsed = [10.0]
        policy = self.config["continuation"]["resource_wait"]

        def interrupt(seconds):
            elapsed[0] += 2
            raise SystemExit("operator interrupted during sleep")

        with self.assertRaises(SystemExit):
            drive(
                self.coordinator,
                lambda: self.values(0),
                policy=policy,
                sleep=interrupt,
                clock=lambda: elapsed[0],
            )
        path = self.coordinator.root / "resource-wait.json"
        self.assertEqual(json.loads(path.read_text())["active_wait_started"], 10)
        self.assertEqual(json.loads(path.read_text())["waited_seconds"], 0)
        self.coordinator.event(
            {"event_id": "interrupted-pause", "type": "pause", "operation_id": None}
        )
        elapsed[0] = 14
        restored = self.new_coordinator(self.new_backend())
        paused = drive(
            restored,
            lambda: self.values(0),
            policy=policy,
            clock=lambda: elapsed[0],
            sleep=lambda seconds: self.fail("restart must preserve the manual pause"),
        )
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(json.loads(path.read_text())["waited_seconds"], 4)
        self.assertNotIn("active_wait_started", json.loads(path.read_text()))
        restored.event(
            {"event_id": "timer-cannot-resume", "type": "timer", "operation_id": None}
        )
        self.assertEqual(restored.advance(self.values())["status"], "paused")
        restored.event(
            {"event_id": "explicit-resume", "type": "resume", "operation_id": None}
        )
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            elapsed[0] += seconds

        result = drive(
            restored,
            lambda: self.values(0),
            policy=policy,
            clock=lambda: elapsed[0],
            sleep=sleep,
        )
        self.assertEqual(result["wait"]["reason"], "budget_wait")
        self.assertEqual(sleeps, [4, 2])
        self.assertEqual(json.loads(path.read_text())["waited_seconds"], 10)
        self.assertEqual(self.effects, [])

    def test_interrupted_wait_restart_uses_only_the_remaining_wait_allowance(self):
        elapsed, sleeps = [0.0], []
        policy = self.config["continuation"]["resource_wait"]

        def interrupt(seconds):
            elapsed[0] += 2
            raise SystemExit("wait interrupted")

        with self.assertRaises(SystemExit):
            drive(
                self.coordinator,
                lambda: self.values(0),
                policy=policy,
                sleep=interrupt,
                clock=lambda: elapsed[0],
            )
        elapsed[0] = 7
        restored = self.new_coordinator(self.new_backend())

        def sleep(seconds):
            sleeps.append(seconds)
            elapsed[0] += seconds

        result = drive(
            restored,
            lambda: self.values(0),
            policy=policy,
            sleep=sleep,
            clock=lambda: elapsed[0],
        )
        self.assertEqual(result["wait"]["reason"], "budget_wait")
        self.assertEqual(sleeps, [3])
        self.assertEqual(
            json.loads((restored.root / "resource-wait.json").read_text())[
                "waited_seconds"
            ],
            10,
        )
        self.assertEqual(self.effects, [])

    def test_receipt_mutation_cannot_select_a_different_successor(self):
        self.coordinator.advance(self.values())
        state = self.coordinator.status()
        receipt = Path(state["receipts"]["run"]["receipt_ref"])
        saved = json.loads(receipt.read_text())
        saved["value"] = {"returned": False}
        atomic_json(receipt, saved)
        with self.assertRaisesRegex(ValueError, "original operation receipt"):
            self.coordinator.advance(self.values())
        self.assertEqual(self.effects, ["run"])

    def test_cli_status_uses_existing_registered_coordinator_without_dispatch(self):
        module_spec = importlib.util.spec_from_file_location(
            "_continuation_cli", ROOT / "scripts/run_task_continuation.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        config_ref = self.root / "continuation-config.json"
        atomic_json(config_ref, self.config)
        with (
            patch.dict("os.environ", {"RSI_ISOLATED_ROLE": "operator"}),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = module.main(
                [
                    "status",
                    "--root",
                    str(self.backend.root),
                    "--configuration",
                    str(config_ref),
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "ready")
        self.assertEqual(self.effects, [])


if __name__ == "__main__":
    unittest.main()
