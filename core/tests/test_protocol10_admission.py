# ruff: noqa: UP017 - Python 3.10 support
# Copyright 2026 RSI-Agent-VLA Authors.
"""Production projection and pre-dispatch admission with isolated CPU data."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from rsi_vla.improvement_protocol import runtime_projection
from rsi_vla.protocol10_admission import validate_dispatch_admission
from rsi_vla.research_tools import Protocol10ResearchAdapter, ResearchTools
from rsi_vla.research_transport import unpack
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class DispatchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads(
            (ROOT / "config/pilot_5a_protocol10.json").read_text()
        )
        self.expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        source = self.root / "repo" / "module.py"
        source.parent.mkdir()
        source.write_text("registered source")
        frozen = self.root / "source/module.py"
        frozen.parent.mkdir()
        frozen.write_bytes(source.read_bytes())
        atomic_json(
            self.root / "manifest.json",
            {"version": "5a-assets-offline", "files": ["module.py"]},
        )
        self.slot = {
            "side": "baseline",
            "run_id": "baseline-0",
            "task_key": "base_salad",
            "seed": 0,
            "bundle_id": self.protocol["current_bundle_id"],
            "output_ref": str(self.root / "output"),
            "scratch_ref": str(self.root / "scratch"),
            "permit_ref": str(self.root / "permit.json"),
        }
        self.config = {
            "repository_root": str(source.parent),
            "phase_ref": str(self.root / "phase.json"),
            "runtime_slots": [self.slot],
            "research": {
                **self.slot,
                "job_id": "job",
                "request_id": "request",
            },
        }
        self.config.update(
            runtime_protocol_ref=str(self.root / "projection.json"),
            frozen_protocol_ref=str(self.root / "frozen.json"),
            source_manifest_ref=str(self.root / "manifest.json"),
        )
        self.authority_ref = self.root / "admin.json"
        self.authority = {
            "authorized": True,
            "scope": self.protocol["scope_id"],
            "protocol_id": self.protocol["protocol_id"],
            "expires_at": self.expiry,
            "phase_snapshot_ref": self.config["phase_ref"],
            "frozen_protocol_ref": self.config["frozen_protocol_ref"],
        }
        atomic_json(self.authority_ref, self.authority)
        atomic_json(
            self.config["runtime_protocol_ref"],
            {"runtime_projection": {"authorization_ref": str(self.authority_ref)}},
        )
        cap = self.protocol["budget"]["runtime_per_start"]
        atomic_json(
            self.config["phase_ref"],
            {
                "remaining": {key: value * 25 for key, value in cap.items()},
                "unresolved_costs": [],
            },
        )
        self.permit = {
            "authorized": True,
            "administrative_authorization_ref": str(self.authority_ref),
            "scope": self.protocol["scope_id"],
            "expires_at": self.expiry,
            "protocol_id": self.protocol["protocol_id"],
            "bundle_id": self.slot["bundle_id"],
            "run_id": self.slot["run_id"],
            "task_key": "base_salad",
            "seed": 0,
            "simulation_episodes": 1,
            "gpu_devices": [0],
            "output_dir": self.slot["output_ref"],
            "storage_write_roots": [self.slot["output_ref"], self.slot["scratch_ref"]],
            "phase_snapshot_ref": self.config["phase_ref"],
            "source_manifest_ref": str(self.root / "manifest.json"),
            "limits": {
                **{
                    key: value
                    for key, value in cap.items()
                    if key != "simulation_episodes"
                },
                "planner_seconds": 600,
            },
        }
        self.op = {
            "kind": "runtime",
            "operation_id": "run",
            "payload": copy.deepcopy(self.slot),
        }
        self.authorization = {
            "dynamic_authorized": True,
            "expires_at": self.expiry,
            "operation_ids": ["run"],
        }

    def admit(self, permit=None):
        permit = permit if permit is not None else self.permit
        atomic_json(self.slot["permit_ref"], permit)
        self.op["payload"]["registered_permit"] = copy.deepcopy(permit)
        return validate_dispatch_admission(
            self.config, self.protocol, self.op, authorization=self.authorization
        )

    def test_all_dispatch_kinds_accept_registered_permits(self):
        self.assertEqual(self.admit(), self.permit)
        self.op["kind"] = "development"
        self.op["payload"]["experiment_id"] = "exp"
        self.permit.update(job_id="job", experiment_id="exp")
        self.assertEqual(self.admit(), self.permit)
        self.op["kind"] = "research"
        self.permit.update(
            job_id="job", request_id="request", **self.protocol["models"]["research"]
        )
        self.permit["limits"] = {
            "model_requests": 1,
            "wall_seconds": 180,
            "storage_bytes": 67108864,
        }
        self.assertEqual(self.admit(), self.permit)

    def test_expired_authority_ignores_caller_past_time(self):
        self.authorization["expires_at"] = "2000-01-01T00:00:00Z"
        self.authorization["now"] = "1999-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "expired"):
            self.admit()

    def test_identity_phase_write_domains_and_expiry_reject(self):
        for key, value in {
            "phase_snapshot_ref": str(self.root / "other-phase.json"),
            "bundle_id": "other",
            "protocol_id": "other",
            "scope": "other",
            "run_id": "other",
            "task_key": "old_milk",
            "seed": 2,
            "output_dir": str(self.root / "other-output"),
            "storage_write_roots": [self.slot["output_ref"]],
            "expires_at": "2000-01-01T00:00:00Z",
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.admit({**self.permit, key: value})

    def test_administrative_expiry_and_registered_sources_cannot_be_replaced(self):
        self.authority["expires_at"] = "2000-01-01T00:00:00Z"
        atomic_json(self.authority_ref, self.authority)
        with self.assertRaisesRegex(ValueError, "administrative authorization expired"):
            self.admit()
        self.authority["expires_at"] = self.expiry
        atomic_json(self.authority_ref, self.authority)
        alternate = self.root / "alternate.json"
        atomic_json(alternate, {"version": "5a-assets-offline", "files": ["module.py"]})
        with self.assertRaisesRegex(ValueError, "registered source manifest"):
            self.admit({**self.permit, "source_manifest_ref": str(alternate)})
        with self.assertRaisesRegex(ValueError, "administrative authorization"):
            self.admit(
                {**self.permit, "administrative_authorization_ref": str(alternate)}
            )

    def test_each_runtime_dimension_is_bounded(self):
        for key, value in self.permit["limits"].items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "limits"):
                self.admit(
                    {**self.permit, "limits": {**self.permit["limits"], key: value + 1}}
                )

    def test_research_uses_frozen_180_seconds(self):
        self.op["kind"] = "research"
        self.permit.update(
            job_id="job", request_id="request", **self.protocol["models"]["research"]
        )
        self.permit["limits"] = {
            "model_requests": 1,
            "wall_seconds": 181,
            "storage_bytes": 10485760,
        }
        with self.assertRaisesRegex(ValueError, "limits"):
            self.admit()

    def test_changed_sources_and_unresolved_costs_reject(self):
        (self.root / "repo/module.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.admit()
        (self.root / "repo/module.py").write_text("registered source")
        phase = json.loads(Path(self.config["phase_ref"]).read_text())
        phase["unresolved_costs"] = ["unknown"]
        atomic_json(self.config["phase_ref"], phase)
        with self.assertRaisesRegex(ValueError, "phase"):
            self.admit()

    def test_real_frozen_projection_drives_research_preparation_and_cost_consumption(
        self,
    ):
        import test_protocol10_integration

        fixture = test_protocol10_integration.Protocol10IntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frozen_ref = self.root / "frozen.json"
        atomic_json(frozen_ref, self.protocol)
        authority_ref = self.root / "authority.json"
        atomic_json(
            authority_ref,
            {
                "authorized": True,
                "scope": self.protocol["scope_id"],
                "protocol_id": self.protocol["protocol_id"],
                "frozen_protocol_ref": str(frozen_ref),
                "allowed_runtime_splits": ["release", "development"],
            },
        )
        projected = runtime_projection(
            self.protocol, frozen_ref=frozen_ref, authorization_ref=authority_ref
        )
        job = fixture.backend._tools().job
        tools = ResearchTools(
            job.store,
            job,
            projected,
            domain_policy=fixture.backend._tools().domain_policy,
        )
        adapter = Protocol10ResearchAdapter()
        transport = adapter.prepare_request(
            tools,
            request_id="projected-request",
            result_ids=["legal-offline-development"],
            max_input_tokens=1000000,
            max_output_tokens=8000,
        )
        self.assertEqual(unpack(transport)["token_policy"], "observed_only")
        self.assertEqual(unpack(transport)["token_parameters_role"], "reference_only")
        record = adapter.record_execution(
            tools,
            "projected-request",
            {
                "model_requests": 1,
                "whole_job_wall_seconds": 2,
                "input_tokens": 1000001,
                "cached_input_tokens": 500,
                "output_tokens": 9000,
            },
            {
                "effective_model": "gpt-6.1-sol",
                "effective_reasoning_effort": "high",
                "evidence_ref": "temporary-cpu-transport",
            },
        )
        self.assertNotIn("input_tokens", record["hard_costs"])
        self.assertEqual(record["costs"]["input_tokens"], 1000001)
        self.assertEqual(job.state["used"]["model_requests"], 1)

    def test_cli_replaces_input_time_with_actual_utc(self):
        spec = importlib.util.spec_from_file_location(
            "protocol10_cli", ROOT / "scripts/run_protocol10.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roles = types.ModuleType("rsi_vla.protocol10_roles")
        roles.operator_command = Mock(return_value=None)
        backend = Mock(registration={})
        coordinator = Mock()
        coordinator.advance.return_value = {"status": "waiting"}
        earliest = datetime.now(timezone.utc)
        with (
            patch.dict(sys.modules, {"rsi_vla.protocol10_roles": roles}),
            patch.object(
                module, "read", side_effect=[{}, {"now": "2000-01-01T00:00:00Z"}]
            ),
            patch.object(module, "Protocol10Backend", return_value=backend),
            patch.object(module, "Coordinator", return_value=coordinator),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = module.main(
                [
                    "advance",
                    "--root",
                    str(self.root),
                    "--configuration",
                    str(self.root / "config.json"),
                    "--inputs",
                    str(self.root / "input.json"),
                ]
            )
        self.assertEqual(result, 0)
        supplied = datetime.fromisoformat(coordinator.advance.call_args.args[0]["now"])
        self.assertLessEqual(earliest, supplied)
        self.assertLessEqual(supplied, datetime.now(timezone.utc))
        self.assertIsNotNone(supplied.tzinfo)
