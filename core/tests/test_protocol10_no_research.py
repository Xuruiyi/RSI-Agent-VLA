"""An empty legal domain terminates without a dispatch permit or model call."""

import unittest
from pathlib import Path

import test_protocol10_integration as fixture_module
from rsi_vla.protocol10_backend import Protocol10Backend, read
from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.storage import atomic_json


class EmptyDomainTests(unittest.TestCase):
    def test_complete_baseline_then_zero_research_without_request_permit(self):
        f = fixture_module.Protocol10IntegrationTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        atomic_json(f.config["research"]["preflight_ref"], [])
        Path(f.config["research"]["permit_ref"]).unlink()
        f.backend = Protocol10Backend(
            f.scope / "empty-domain",
            f.config,
            runtime_execute=f.runtime_execute,
            audit=f.audit,
            progress_inspector=f.progress,
            research_execute=f.research_execute,
            trial_collector=f.collect,
        )
        f.coordinator = Coordinator(
            f.scope / "empty-coordinator", f.backend.registration, f.backend
        )
        state = f.run_until(lambda s: s["status"] == "completed")
        self.assertEqual(state["terminal"], "research_stop")
        self.assertEqual((len(f.launches), f.research_launches), (8, 0))
        receipt = state["receipts"]["research"]
        self.assertTrue(receipt["stop"])
        self.assertEqual(receipt["request_count"], 0)
        self.assertFalse(state["operations"]["research"]["dynamic"])
        before = read(f.phase_ref)
        f.coordinator.advance(f.inputs())
        self.assertEqual(read(f.phase_ref), before)
        self.assertEqual(f.store.all("candidates"), [])

    def test_preflight_is_part_of_immutable_registration(self):
        f = fixture_module.Protocol10IntegrationTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        atomic_json(f.config["research"]["preflight_ref"], [])
        op = f.backend._op("matrix_evaluation", "baseline", {"side": "baseline"})
        with self.assertRaisesRegex(ValueError, "registered.*inputs changed"):
            f.backend.execute(op, {})
