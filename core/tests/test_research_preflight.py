"""Preflight uses the current development split and real Runtime preparation."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.improvement_protocol import runtime_projection
from rsi_vla.research import ResearchJob
from rsi_vla.research_preflight import operator_preflight
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json, initialize

ROOT = Path(__file__).resolve().parents[2]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        frozen = json.loads((ROOT / "config/pilot_5a_protocol10.json").read_text())
        frozen_ref = self.root / "frozen.json"
        authority = self.root / "authority.json"
        atomic_json(frozen_ref, frozen)
        atomic_json(
            authority,
            {
                "authorized": True,
                "scope": frozen["scope_id"],
                "protocol_id": frozen["protocol_id"],
                "frozen_protocol_ref": str(frozen_ref),
                "allowed_runtime_splits": ["release", "development"],
            },
        )
        self.protocol = runtime_projection(
            frozen, frozen_ref=frozen_ref, authorization_ref=authority
        )
        self.protocol_ref = self.root / "protocol.json"
        atomic_json(self.protocol_ref, self.protocol)
        profile = json.loads(
            (ROOT / "config/run_profile_5a_protocol10.json").read_text()
        )
        profile["dynamic_authorized"] = True
        profile.pop("auxiliary_models", None)
        profile["model"]["checkpoint"] = str(self.root / "checkpoint")
        profile["model"]["normalization"] = str(self.root / "normalization.json")
        profile_ref = self.root / "profile.json"
        atomic_json(profile_ref, profile)
        self.base = RunBundle(
            "base",
            str(profile_ref) + "#/model",
            str(profile_ref) + "#/model/normalization",
            str(profile_ref) + "#/control",
            "rho",
            "psi",
            "memory",
            frozen["protocol_id"],
            str(profile_ref),
            "cpu-fixture",
        )
        self.store = ResearchStore(self.root / "research")
        initialize(
            self.store,
            self.base,
            psi={},
            memory={
                "legal_information": "development",
                "upstream": profile["memory"],
                "provenance": "isolated CPU fixture",
            },
        )
        self.job = ResearchJob(self.store, "preflight", "base", {"offline_calls": 1})
        self.tools = ResearchTools(
            self.store,
            self.job,
            self.protocol,
            domain_policy={
                "rule_operators": [
                    "finish_requires_native_success",
                    "planner_finish_reserve_calls",
                ],
            },
        )
        raw = self.root / "result.json"
        atomic_json(
            raw,
            {
                "bundle_id": "base",
                "protocol_id": frozen["protocol_id"],
                "planner": frozen["models"]["planner"],
                "planner_finish": {"status": "success"},
                "benchmark_success": False,
                "attempt_closed": True,
                "native_result_source": {"terminated": True},
            },
        )
        self.result = ExperimentResult(
            "legal",
            "base",
            frozen["protocol_id"],
            frozen["benchmark_id"],
            "development",
            "simulation",
            {"information_scope": "development"},
            [str(raw)],
            "native",
            {},
            {},
            {},
            assigned_count=1,
            outcome="failure",
            assignments=[
                {
                    "task_key": "position_swap_salad",
                    "seed": 2,
                    "outcome": "failure",
                    "validity": "valid",
                    "outcome_source": "native_benchmark",
                    "evidence_ref": str(raw),
                }
            ],
        )
        self.store.save_development_result(self.result)

    def preflight(self, ids=("legal",)):
        return {
            r["operator"]: r
            for r in operator_preflight(
                self.tools, ROOT, protocol_path=self.protocol_ref, result_ids=list(ids)
            )
        }

    def test_protocol10_seed2_reaches_real_consumer_without_dispatch(self):
        rows = self.preflight()
        guard = rows["finish_requires_native_success"]
        self.assertEqual(
            guard["development_assignment"],
            {"task_key": "position_swap_salad", "seed": 2},
        )
        self.assertTrue(guard["runtime_prepared"])
        self.assertTrue(guard["currently_trialable"])
        self.assertIn("native finish guard", guard["consumer"])
        self.assertFalse(guard["execution_authorized"])
        self.assertFalse(rows["planner_finish_reserve_calls"]["currently_trialable"])
        self.assertEqual(self.job.state["charges"], [])
        self.assertEqual(list(self.root.glob("**/runtime_context.json")), [])

    def test_domain_evidence_and_noop_remain_required(self):
        self.tools.domain_policy["rule_operators"] = []
        self.assertFalse(
            self.preflight()["finish_requires_native_success"]["currently_trialable"]
        )
        self.tools.domain_policy["rule_operators"] = ["finish_requires_native_success"]
        self.assertFalse(
            self.preflight([])["finish_requires_native_success"]["currently_trialable"]
        )
        rules = self.root / "rules.json"
        atomic_json(
            rules,
            {
                "schema": "rsi-runtime-rules-v2",
                "chunk_steps": 5,
                "baseline_chunk_steps": 5,
                "finish_requires_native_success": True,
            },
        )
        self.tools.base = replace(self.base, phi_ref=str(rules))
        row = self.preflight()["finish_requires_native_success"]
        self.assertFalse(row["currently_trialable"])
        self.assertIn("actual behavior change", row["runtime_error"])

    def test_old_seed1_evidence_is_not_admitted_as_protocol10_development(self):
        self.store.put(
            "development_results",
            "old-seed",
            replace(
                self.result,
                result_id="old-seed",
                assignments=[{**self.result.assignments[0], "seed": 1}],
            ),
        )
        with self.assertRaisesRegex(ValueError, "nondevelopment"):
            self.preflight(["old-seed"])

    def test_empty_current_split_is_explicitly_rejected(self):
        self.tools.protocol["task_sets"]["development"]["task_keys"] = []
        with self.assertRaisesRegex(ValueError, "no development assignment"):
            self.preflight([])
