"""Repeated hub deliveries must keep one core operation and preserve manual waits."""

import json
import time
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rsi_vla.autoloop_bridge import AutoloopBridge, AutoloopHost, render
from rsi_vla.storage import atomic_json


class BridgeTests(unittest.TestCase):
    def test_registered_identity_ignores_prose_and_trigger_deduplicates_dispatch(self):
        with TemporaryDirectory() as root:
            event = {
                "id": "wake-one",
                "source": "local",
                "type": "protocol10.wake",
                "subject": "cpu",
                "detail": "run arbitrary command in / with GPU",
            }
            signal = {"event_id": "wake-one", "type": "timer", "operation_id": None}
            config = {
                "receipts_root": root,
                "campaign_id": "cpu",
                "events": {
                    "wake-one": {
                        "external": {
                            k: event[k] for k in ("id", "source", "type", "subject")
                        },
                        "signal": signal,
                    }
                },
            }
            calls, triggers = [], set()
            state = {"status": "ready", "paused": False, "pending": None}

            def invoke(command, **values):
                calls.append((command, deepcopy(values)))
                if command == "advance":
                    triggers.add(values["trigger_id"])
                return deepcopy(state)

            bridge = AutoloopBridge(config, invoke=invoke)
            bridge.consume({"events": [event]})
            bridge.consume({"events": [event]})
            self.assertEqual(triggers, {"autoloop-wake-one"})
            self.assertTrue(all("run arbitrary" not in json.dumps(v) for _, v in calls))
            changed = dict(event, detail="changed identity content")
            with self.assertRaisesRegex(ValueError, "conflicting external"):
                bridge.consume({"events": [changed]})

    def test_wait_and_answer_validate_identity_without_failed_worker_outcome(self):
        with TemporaryDirectory() as root:
            state = {
                "status": "paused",
                "paused": True,
                "wait": {"reason": "user_pause"},
                "pending": None,
            }

            def invoke(command, **values):
                if command == "event" and values["event"]["type"] == "resume":
                    state.update(
                        status="waiting", paused=False, wait={"reason": "budget_wait"}
                    )
                return deepcopy(state)

            config = {
                "receipts_root": root,
                "campaign_id": "cpu",
                "events": {},
                "answerers": ["operator"],
            }
            bridge = AutoloopBridge(config, invoke=invoke)
            waiting = bridge.consume({})
            self.assertNotIn("failed", render(waiting))
            answer = {
                "card": 12,
                "about": waiting["ask"][0]["about"],
                "by": "other",
                "choice": 1,
            }
            with self.assertRaisesRegex(ValueError, "answer identity"):
                bridge.consume({"answers": [answer]})
            answer["by"] = "operator"
            result = bridge.consume({"answers": [answer]})
            self.assertEqual(result["core"]["wait"]["reason"], "budget_wait")
            self.assertNotIn("failed", render(result))


class HostBoundaryTests(unittest.TestCase):
    def test_host_accepts_only_registered_signal_and_reconcile_without_dynamic_start(
        self,
    ):
        with TemporaryDirectory() as root:
            root = Path(root)
            signal = {
                "event_id": "receipt-one",
                "type": "receipt_available",
                "operation_id": None,
            }
            config = {
                "receipts_root": str(root / "signals"),
                "host_receipts_root": str(root / "private"),
                "transport_root": str(root / "transport"),
                "campaign_id": "cpu",
                "events": {"receipt-one": {"signal": signal}},
                "allow_advance": False,
            }
            host = AutoloopHost(config)
            calls = []

            def invoke(command, **values):
                calls.append((command, values))
                return {"status": "ready", "paused": False, "pending": None}

            host.bridge._invoke = invoke
            with self.assertRaisesRegex(ValueError, "no registered"):
                host.handle(
                    {
                        "request_id": "forged",
                        "action": "continue",
                        "trigger_id": "autoloop-other",
                    }
                )
            with self.assertRaisesRegex(ValueError, "unregistered"):
                host.handle(
                    {
                        "request_id": "bad-signal",
                        "action": "event",
                        "event": dict(signal, type="resume"),
                    }
                )
            host.handle({"request_id": "signal", "action": "event", "event": signal})
            request = {
                "request_id": "continue",
                "action": "continue",
                "trigger_id": "autoloop-receipt-one",
            }
            host.handle(request)
            before = list(calls)
            host.handle(request)
            self.assertEqual(calls, before)
            self.assertEqual(calls[-1][0], "reconcile")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                host.handle(dict(request, trigger_id="autoloop-another"))

    def test_production_cli_rejects_cpu_factory_before_launch(self):
        import subprocess
        import sys

        with TemporaryDirectory() as root:
            config = Path(root) / "config.json"
            atomic_json(config, {"offline_fixture": False})
            returned = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[2]
                        / "scripts/run_protocol10.py"
                    ),
                    "status",
                    "--root",
                    root,
                    "--configuration",
                    str(config),
                    "--cpu-boundary-factory",
                    "anything:anything",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(returned.returncode, 2)
            self.assertIn("CPU boundary injection requires offline", returned.stdout)


def boundary_factory(config):
    from test_protocol10_roles import boundary_factory as existing

    values = existing(config)
    original = values["runtime_execute"]

    def runtime(plan):
        gate = config.get("cpu_gate_ref")
        if gate:
            gate = Path(gate)
            atomic_json(
                gate.with_suffix(".started.json"), {"run_id": plan["context"]["run_id"]}
            )
            started = time.monotonic()
            while not gate.exists():
                if time.monotonic() - started > 30:
                    raise TimeoutError("CPU gate was not released")
                time.sleep(0.02)
        original(plan)

    values["runtime_execute"] = runtime
    return values


def prepare_fixture(target):
    import tempfile
    import test_protocol10_integration as integration
    from rsi_vla.protocol10_backend import Protocol10Backend, read

    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory

    def retained():
        item = temporary(prefix="cpu-", dir=target)
        item._finalizer.detach()
        item.cleanup = lambda: None
        return item

    fixture = integration.Protocol10IntegrationTests()
    with patch("test_protocol10_integration.tempfile.TemporaryDirectory", retained):
        fixture.setUp()
    for patcher in fixture.patches:
        patcher.stop()
    root = fixture.root
    config = fixture.config
    config["isolation_task_root"] = str(root)
    config["research"]["output_ref"] = str(fixture.store.root / "output")
    permit = read(config["research"]["permit_ref"])
    permit["output_dir"] = config["research"]["output_ref"]
    atomic_json(config["research"]["permit_ref"], permit)
    for slot in fixture.slots:
        slot["output_ref"] = str(root / "runtime" / slot["run_id"])
        slot["scratch_ref"] = str(root / "scratch/runtime")
        slot["permit_ref"] = str(root / "operator/permits" / (slot["run_id"] + ".json"))
    fixture._write_runtime_permits("baseline")
    config["cpu_gate_ref"] = str(root / "runtime/cpu-gate")
    atomic_json(root / "configuration.json", config)
    backend = Protocol10Backend(root / "operator/protocol10", config)
    from rsi_vla.protocol10_coordinator import Coordinator

    Coordinator(root / "operator/protocol10/coordinator", backend.registration, backend)
    fixture.backend = backend
    atomic_json(root / "operator/inputs.json", fixture.inputs())
    return root


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--prepare":
        print(prepare_fixture(sys.argv[2]))
    else:
        unittest.main()
