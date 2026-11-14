"""Raw output nested in Research storage is billed separately from SDK scratch."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi_vla.artifact_accounting import begin, finish
from rsi_vla.storage import ResearchStore, atomic_json


class NestedOutputAccountingTests(unittest.TestCase):
    def test_nested_raw_output_and_scratch_keep_one_raw_growth_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResearchStore(root / "research")
            output = store.root / "output"
            output.mkdir(parents=True)
            (output / "raw").write_bytes(b"x" * 100)
            scratch = root / "scratch"
            scratch.mkdir()
            (scratch / "sdk").write_bytes(b"x" * 80)
            state = {
                "limits": {"artifact_bytes": 10000},
                "used": {"artifact_bytes": 180},
                "charges": [],
            }

            def record_storage(amount, reason):
                state["used"]["artifact_bytes"] += amount
                state["charges"].append(
                    {"reason": reason, "costs": {"artifact_bytes": amount}}
                )

            tools = SimpleNamespace(
                store=store,
                job=SimpleNamespace(state=state, record_storage=record_storage),
            )
            phase_ref = root / "phase.json"
            atomic_json(
                phase_ref,
                {
                    "limits": {"storage_bytes": 10000},
                    "used": {"storage_bytes": 180},
                    "remaining": {"storage_bytes": 9820},
                },
            )
            begin(tools, "request")
            (output / "validated").write_bytes(b"v" * 30)
            (store.root / "derived").write_bytes(b"d" * 20)
            receipt = finish(
                tools,
                "request",
                phase_path=phase_ref,
                output=output,
                billed_output_bytes=100,
            )
            self.assertEqual(receipt["booked_bytes"], 50)
            self.assertEqual(receipt["post_measurement_raw_growth"], 30)
            self.assertEqual(state["used"]["artifact_bytes"], 230)
            self.assertEqual(
                json.loads(phase_ref.read_text())["used"]["storage_bytes"], 230
            )
            saved = phase_ref.read_bytes()
            finish(
                tools,
                "request",
                phase_path=phase_ref,
                output=output,
                billed_output_bytes=100,
            )
            self.assertEqual(phase_ref.read_bytes(), saved)
            (output / "validated").write_bytes(b"v" * 40)
            finish(
                tools,
                "request",
                phase_path=phase_ref,
                output=output,
                billed_output_bytes=100,
            )
            self.assertEqual(state["used"]["artifact_bytes"], 240)

    def test_research_consumer_saves_raw_measurement_apart_from_scratch(self):
        from unittest.mock import patch

        import test_batch4
        from rsi_vla.bounded_runner import execute

        fixture = test_batch4.RunnerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prepared = fixture.prepared
        scratch = fixture.root / "sdk-scratch"
        scratch.mkdir()
        output = fixture.store.root / "research-output"

        def nested_prepared():
            adapter, transport, _, permit, path = prepared()
            permit["output_dir"] = str(output)
            permit["storage_write_roots"] = [str(output), str(scratch)]
            atomic_json(path, permit)
            return adapter, transport, output, permit, path

        fixture.prepared = nested_prepared

        def execute_with_sdk_scratch(*args, **kwargs):
            if kwargs.get("supervisor"):
                supervisor = kwargs["supervisor"]

                def with_scratch(*a, **kw):
                    (scratch / "sdk-journal").write_bytes(b"s" * 80)
                    return supervisor(*a, **kw)

                kwargs["supervisor"] = with_scratch
            return execute(*args, **kwargs)

        with patch("test_batch4.execute", execute_with_sdk_scratch):
            fixture.test_one_dispatch_reconcile_no_repeat_charge_and_no_applicable_no_dispatch()
        measurement = json.loads((output / "execution-costs.json").read_text())
        self.assertEqual(
            measurement["costs"]["artifact_bytes"],
            measurement["raw_output_bytes_at_measurement"] + 80,
        )
