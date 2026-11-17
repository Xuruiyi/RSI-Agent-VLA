# Copyright 2026 RSI-Agent-VLA Authors.
"""Frozen launch inputs and legacy Bundle references at actual consumers."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import test_proportionate_research
import test_runtime_plan
from rsi_vla.contracts import RunBundle
from rsi_vla.research_transport import unpack
from rsi_vla.runtime import execute_runtime, execution_binding, validate_runtime_binding
from rsi_vla.storage import ResearchStore
from rsi_vla.version_binding import bundle_references, validate_research_binding


class RuntimeVersionTests(unittest.TestCase):
    setUp = test_runtime_plan.RuntimePlanTests.setUp
    plan = test_runtime_plan.RuntimePlanTests.plan

    def test_legacy_bundle_roundtrip_and_conflicting_id(self):
        store = ResearchStore(self.artifacts / "research")
        bundle = RunBundle(**store.get("bundles", "initial-medium"))
        self.assertEqual(asdict(RunBundle(**asdict(bundle))), asdict(bundle))
        references = bundle_references(asdict(bundle))
        self.assertEqual(
            references["research"]["rho_ref"], references["execution"]["rho_ref"]
        )
        self.assertEqual(references["facts"], {"memory_ref": bundle.memory_ref})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            store.save_bundle(replace(bundle, theta_ref="different"))

    def test_prepared_runtime_retains_snapshot_and_rejects_live_input_change(self):
        plan = self.plan()
        validate_runtime_binding(plan)
        ref = Path(plan["context"]["profile_ref"])
        profile = json.loads(ref.read_text())
        profile["control"]["action_chunk"] = 1
        ref.write_text(json.dumps(profile))
        self.assertEqual(
            plan["context"]["execution_binding"]["profile"]["control"]["action_chunk"],
            5,
        )
        plan["blockers"] = []
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            execute_runtime(plan)
        self.assertFalse(Path(plan["context"]["output_dir"]).exists())

    def test_normalization_content_changes_invalidate_the_frozen_input(self):
        store = ResearchStore(self.artifacts / "research")
        plan = self.plan()
        norm = self.artifacts / "norm-stats.json"
        norm.write_text('{"mean": [0.0]}')
        binding = plan["context"]["execution_binding"]
        plan["context"]["execution_binding"] = execution_binding(
            store, binding["bundle"], binding["profile"], normalization_path=norm
        )
        validate_runtime_binding(plan)
        norm.write_text('{"mean": [1.0]}')
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            validate_runtime_binding(plan)


class ResearchVersionTests(unittest.TestCase):
    setUp = test_proportionate_research.ProportionateResearchTests.setUp
    result = test_proportionate_research.ProportionateResearchTests.result
    configure = test_proportionate_research.ProportionateResearchTests.configure
    prepare = test_proportionate_research.ProportionateResearchTests.prepare

    def test_new_policy_binds_new_request_without_rewriting_saved_request(self):
        _adapter, transport, _response = self.prepare()
        old = unpack(transport)
        validate_research_binding(self.tools, old)
        ref = self.store.path("psi", self.tools.base.psi_ref)
        psi = json.loads(ref.read_text())
        psi["new_rule"] = "check evidence first"
        ref.write_text(json.dumps(psi))
        with self.assertRaisesRegex(PermissionError, "starting records changed"):
            validate_research_binding(self.tools, old)
        new = unpack(
            _adapter.prepare_request(
                self.tools,
                request_id="new-policy",
                result_ids=[],
                max_input_tokens=100,
                max_output_tokens=10,
            )
        )
        validate_research_binding(self.tools, new)
        self.assertNotEqual(new["version_binding"], old["version_binding"])
        self.assertNotIn("new_rule", old["version_binding"]["records"]["psi_ref"])


if __name__ == "__main__":
    unittest.main()
