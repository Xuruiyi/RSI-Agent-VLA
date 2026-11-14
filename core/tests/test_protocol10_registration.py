# ruff: noqa: UP017 - Python 3.10 support
"""Preparation consumes frozen production schemas without issuing authority."""

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rsi_vla.contracts import RunBundle
from rsi_vla.improvement_protocol import BASELINE, resource_envelope
from rsi_vla.storage import ResearchStore, atomic_json, initialize

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_protocol10_registration",
    ROOT / "scripts/prepare_protocol10_registration.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source-store"
        self.protocol_ref = ROOT / "config/pilot_5a_protocol10.json"
        self.frozen_ref = self.root / "frozen-protocol.json"
        protocol = json.loads(self.protocol_ref.read_text())
        atomic_json(self.frozen_ref, protocol)
        profile_ref = self.root / "source-profile.json"
        atomic_json(
            profile_ref,
            json.loads((ROOT / "config/run_profile_5a_protocol10.json").read_text()),
        )
        initialize(
            ResearchStore(self.source),
            RunBundle(
                BASELINE,
                "theta",
                "processor",
                "phi",
                "rho",
                "psi",
                "memory",
                protocol["protocol_id"],
                str(profile_ref),
                "cpu-preparation-source",
            ),
            psi={},
            memory={
                "provenance": "isolated CPU fixture",
                "legal_information": "development",
            },
        )
        caps = resource_envelope(protocol)["gross"]
        self.phase = self.root / "phase.json"
        atomic_json(
            self.phase,
            {
                "limits": caps,
                "used": dict.fromkeys(caps, 0),
                "remaining": caps,
                "unresolved_costs": [],
            },
        )
        self.authority = self.root / "authority.json"
        self.auth = {
            "authorized": True,
            "scope": protocol["scope_id"],
            "protocol_id": protocol["protocol_id"],
            "frozen_protocol_ref": str(self.frozen_ref),
            "allowed_runtime_splits": ["release", "development"],
            "phase_snapshot_ref": str(self.phase),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        atomic_json(self.authority, self.auth)
        self.config = self.root / "configuration.json"
        self.store = self.root / "state/research"
        atomic_json(
            self.config,
            {
                "protocol_ref": str(self.protocol_ref),
                "frozen_protocol_ref": str(self.frozen_ref),
                "phase_ref": str(self.phase),
                "runtime_protocol_ref": str(self.root / "operator/projection.json"),
                "profile_ref": str(self.store / "profiles" / f"{BASELINE}.json"),
                "research": {
                    "store_ref": str(self.store),
                    "job_id": "cpu-registration",
                    "job_limits": {
                        "model_requests": 41,
                        "whole_job_wall_seconds": 1980,
                        "tool_calls": 120,
                        "environment_steps": 4000,
                        "gpu_device_seconds_envelope": 1800,
                        "artifact_bytes": 2214592512,
                    },
                },
            },
        )

    def prepare(self):
        return module.prepare(
            self.config, self.authority, self.source, offline_seconds=0, offline_bytes=0
        )

    def test_unissued_expired_and_insufficient_authority_do_not_materialize(self):
        for changes in (
            {"authorized": False},
            {"expires_at": "2000-01-01T00:00:00+00:00"},
            {"phase_snapshot_ref": str(self.root / "other.json")},
        ):
            with self.subTest(changes=changes):
                atomic_json(self.authority, {**self.auth, **changes})
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.store.exists())
        atomic_json(self.authority, self.auth)
        phase = json.loads(self.phase.read_text())
        phase["limits"]["simulation_episodes"] = phase["remaining"][
            "simulation_episodes"
        ] = 1
        atomic_json(self.phase, phase)
        with self.assertRaisesRegex(ValueError, "envelope"):
            self.prepare()
        self.assertFalse(self.store.exists())

    def test_preparation_is_repeatable_without_permits_dispatch_or_phase_mutation(self):
        before = self.phase.read_bytes()
        first = self.prepare()
        self.assertEqual(self.prepare(), first)
        self.assertTrue(first["prepared"])
        self.assertFalse(first["dispatch"])
        self.assertEqual(first["permits_issued"], 0)
        self.assertEqual(before, self.phase.read_bytes())
        projected = json.loads((self.root / "operator/projection.json").read_text())
        self.assertEqual(projected["budget"]["token_policy"], "observed_only")
        job = json.loads((self.store / "jobs/cpu-registration.json").read_text())
        self.assertEqual(job["charges"], [])
        self.assertFalse((self.root / "runtime").exists())
