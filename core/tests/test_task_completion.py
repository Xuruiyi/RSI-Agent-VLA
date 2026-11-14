"""Offline regressions for new-task consumers, retaining real research state."""

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import test_research_tools
import test_task_policy
from rsi_vla.research_tools import BoundedResearchAdapter
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import (
    TaskCompletionBackend,
    TaskResearchAdapter,
    recovery_choice,
    selected_manifest,
)


class ConsumerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        self.limits = {
            "model_requests": 20,
            "tool_calls": 60,
            "environment_steps": 2000,
            "wall_seconds": 900,
            "gpu_seconds": 900,
            "storage_bytes": 1048576,
        }
        protocol = {
            "protocol_id": "completion-test",
            "scope_id": "completion-test-scope",
            "scope": "offline",
            "status": "frozen",
            "benchmark_id": "test",
            "anchor_bundle_id": "base",
            "metrics": {
                "success": {"direction": "max", "min": 0, "max": 1, "weight": 1}
            },
            "hard_constraints": ["resource_limits"],
            "min_gain": 0.1,
            "anchor_max_drop": 0,
            "task_policy": {
                "schema": "rsi-task-policy-v1",
                "research_per_request": {
                    "wall_seconds": 300,
                    "storage_bytes": 134217728,
                },
            },
            "budget": {
                "runtime_per_start": {"simulation_episodes": 1, **self.limits},
                "per_attempt_proposal": {"planner_timeout_seconds": 600},
            },
        }
        self.config = {
            "schema": "rsi-task-completion-v1",
            "base_bundle_id": "base",
            "research_store_ref": str(self.root / "research"),
            "evaluation_store_ref": str(self.root / "evaluation"),
            "repository_root": str(self.root / "repo"),
            "research_requests": [],
            "operations": [],
        }
        for key, name in (
            ("protocol_ref", "protocol.json"),
            ("frozen_protocol_ref", "frozen.json"),
            ("scope_authorization_ref", "scope.json"),
            ("administrative_authorization_ref", "admin.json"),
            ("phase_ref", "phase.json"),
            ("source_manifest_ref", "source-manifest.json"),
        ):
            self.config[key] = str(self.root / name)
        atomic_json(self.config["protocol_ref"], protocol)
        atomic_json(self.config["frozen_protocol_ref"], protocol)
        self.scope = {
            "authorized": True,
            "dynamic_authorized": True,
            "scope_id": "completion-test-scope",
            "expires_at": self.future,
            "dynamic_limits": {"simulation_episodes": 33},
            "research_requests_max": 4,
            "candidate_max": 2,
            "models": {
                "research": {"model": "gpt-6.1-sol", "reasoning_effort": "high"}
            },
        }
        self.admin = {
            "authorized": True,
            "dynamic_authorized": True,
            "scope": "completion-test-scope",
            "protocol_id": "completion-test",
            "expires_at": self.future,
            "phase_snapshot_ref": self.config["phase_ref"],
            "approval_reference": self.config["scope_authorization_ref"],
        }
        atomic_json(self.config["scope_authorization_ref"], self.scope)
        atomic_json(self.config["administrative_authorization_ref"], self.admin)
        phase = {
            "limits": {"simulation_episodes": 33, **self.limits},
            "remaining": {"simulation_episodes": 33, **self.limits},
            "unresolved_costs": [],
        }
        atomic_json(self.config["phase_ref"], phase)
        self.slot = {
            "run_id": "run",
            "bundle_id": "base",
            "task_key": "salad",
            "seed": 2,
            "cuda_device": 0,
            "output_ref": str(self.root / "output"),
            "scratch_ref": str(self.root / "scratch"),
            "permit_ref": str(self.root / "permit.json"),
        }
        self.config["runtime_slots"] = [self.slot]
        permit = {
            "authorized": True,
            "scope": "completion-test-scope",
            "protocol_id": "completion-test",
            "bundle_id": "base",
            "run_id": "run",
            "task_key": "salad",
            "seed": 2,
            "simulation_episodes": 1,
            "gpu_devices": [0],
            "expires_at": self.future,
            "output_dir": self.slot["output_ref"],
            "storage_write_roots": [self.slot["output_ref"], self.slot["scratch_ref"]],
            "phase_snapshot_ref": self.config["phase_ref"],
            "administrative_authorization_ref": self.config[
                "administrative_authorization_ref"
            ],
            "source_manifest_ref": self.config["source_manifest_ref"],
            "limits": {**self.limits, "planner_seconds": 600},
        }
        atomic_json(self.slot["permit_ref"], permit)
        atomic_json(self.config["source_manifest_ref"], {"files": ["consumer.py"]})
        for directory in (self.root / "repo", self.root / "source"):
            directory.mkdir()
            (directory / "consumer.py").write_text("fixed consumer\n")
        self.backend = TaskCompletionBackend(self.config)

    def test_deployment_identity_copies_do_not_consume_research_candidate_cap(self):
        from unittest.mock import Mock
        op = {"operation_id": "submit-next", "kind": "candidate_submit", "payload": {
            "request_id": "request", "candidate_id": "next", "experiment_ids": ["experiment"],
            "cost_snapshot_ref": "snapshot.json"}}
        self.config["operations"] = [op]
        for identity in ("deployment-a", "deployment-b"):
            self.backend.store.put("candidates", identity, {
                "candidate_id": identity, "protocol_id": identity})
        tools = Mock()
        tools.submit_candidate.return_value = {"candidate_id": "next"}
        self.backend.tools = lambda request_id: (tools, {})
        self.assertEqual(self.backend.execute(op, {}), {"candidate_id": "next"})
        for identity in ("source-a", "source-b"):
            self.backend.store.put("candidates", identity, {
                "candidate_id": identity, "protocol_id": self.backend.protocol["protocol_id"]})
        with self.assertRaisesRegex(PermissionError, "Candidate cap"):
            self.backend.execute(op, {})

    def test_registered_runtime_is_admitted_without_dispatch(self):
        permit = self.backend.validate_dynamic("runtime", {"run_id": "run"})
        self.assertEqual(permit["limits"]["model_requests"], 20)
        self.assertFalse(Path(self.slot["output_ref"]).exists())

    def test_recovery_consumer_requires_closed_original_fault_and_single_attempt(self):
        self.scope["runtime_technical_recovery_max"] = 4
        atomic_json(self.config["scope_authorization_ref"], self.scope)
        self.slot.update(
            recovery_of_run_id="original",
            recovery_basis_ref=str(self.root / "basis.json"),
        )
        basis = {
            "run_id": "original",
            "validity": "invalid",
            "failure_class": "technical",
            "cost_closed": True,
            "cleanup_complete": True,
            "recovery_cause_resolved": True,
        }
        atomic_json(self.slot["recovery_basis_ref"], basis)
        self.backend.validate_dynamic("runtime", {"run_id": "run"})
        atomic_json(self.slot["recovery_basis_ref"], {**basis, "validity": "valid"})
        with self.assertRaisesRegex(PermissionError, "closed original"):
            self.backend.validate_dynamic("runtime", {"run_id": "run"})
        atomic_json(self.slot["recovery_basis_ref"], basis)
        previous = {
            **self.slot,
            "run_id": "previous-recovery",
            "output_ref": str(self.root / "previous"),
        }
        self.config["runtime_slots"].append(previous)
        atomic_json(Path(previous["output_ref"]) / "runtime_context.json", {})
        with self.assertRaisesRegex(PermissionError, "recovery cap"):
            self.backend.validate_dynamic("runtime", {"run_id": "run"})

    def test_consumer_refuses_expired_authority_and_phase_or_source_drift(self):
        for fault in ("expired", "phase", "source", "write_roots", "budget"):
            with self.subTest(fault=fault):
                self.setUp()
                if fault == "expired":
                    atomic_json(
                        self.config["scope_authorization_ref"],
                        {**self.scope, "expires_at": "2000-01-01T00:00:00+00:00"},
                    )
                elif fault == "phase":
                    atomic_json(
                        self.config["administrative_authorization_ref"],
                        {**self.admin, "phase_snapshot_ref": "/tmp/other"},
                    )
                elif fault == "source":
                    (self.root / "repo/consumer.py").write_text("changed consumer\n")
                elif fault == "write_roots":
                    from rsi_vla.task_completion import read

                    permit = read(self.slot["permit_ref"])
                    atomic_json(
                        self.slot["permit_ref"],
                        {**permit, "storage_write_roots": ["/tmp"]},
                    )
                else:
                    from rsi_vla.task_completion import read

                    phase = read(self.config["phase_ref"])
                    phase["remaining"]["model_requests"] = 19
                    atomic_json(self.config["phase_ref"], phase)
                with self.assertRaises(PermissionError):
                    self.backend.validate_dynamic("runtime", {"run_id": "run"})


class NewTaskResearchTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    configure = test_task_policy.ExplorationTests.configure

    def request(self, request_id, *, defer=False):
        cfg = {
            "request_id": request_id,
            "reserved_bundle_id": request_id + "-candidate-bundle",
            "development_slots": [{"task_key": "salad", "seed": 2}],
        }
        adapter = TaskResearchAdapter(cfg)
        adapter.prepare_request(
            self.tools,
            request_id=request_id,
            result_ids=[],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        response = {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": request_id,
            "decision": "defer" if defer else "prepare_rule_trial",
            "source": "exploration",
            "result_ids": [],
            "reason": "test supervision",
            "external_value": "reduce wasted calls",
            "competing_explanations": self.explanations[:1],
            "intervention": None if defer else {"finish_requires_native_success": True},
            "distinguishing_prediction": "closure differs",
            "stop_condition": "one trial",
            "missing_evidence": ["need observation"],
            "development_slots": [] if defer else cfg["development_slots"],
        }
        return cfg, adapter, response

    def test_four_distinct_requests_produce_two_candidates_from_real_findings(self):
        self.configure()
        self.tools.domain_policy["rule_operators"].append(
            "planner_finish_reserve_calls"
        )
        self.protocol["task_policy"]["exploration"]["operators"].append(
            "planner_finish_reserve_calls"
        )
        for index in range(4):
            cfg, adapter, response = self.request(
                "round-" + str(index), defer=index % 2 == 0
            )
            if index == 3:
                response["intervention"] = {"planner_finish_reserve_calls": 1}
            trial = adapter.apply_response(
                self.tools,
                response,
                evidence_level="offline",
                cost_limit={"offline_calls": 1},
                synthetic=True,
            )
            if index % 2 == 0:
                self.assertEqual(trial["action"], "defer")
                continue
            experiment_id = cfg["request_id"] + "-experiment"
            self.assertEqual(trial["bundle_id"], cfg["reserved_bundle_id"])
            with self.assertRaises(FileNotFoundError):
                self.tools.submit_candidate(
                    "premature-" + str(index), experiment_id=experiment_id
                )
            self.tools.run_offline_experiment(
                experiment_id,
                lambda plan, artifact, index=index: replace(
                    self.result("result-" + str(index), artifact["bundle_id"], seed=2),
                    criteria_ref=plan["criterion_ref"],
                    intervention={
                        "information_scope": "development",
                        **plan["intervention"],
                    },
                ),
                cost={"offline_calls": 1},
            )
            self.tools.submit_candidate(
                "candidate-" + str(index), experiment_id=experiment_id
            )
        self.assertEqual(len(self.store.all("bounded_requests")), 4)
        self.assertEqual(len(self.store.all("candidates")), 2)
        self.assertEqual(self.job.state["used"]["offline_calls"], 2)
        self.assertEqual(len(self.store.all("task_completion_selections")), 4)

    def test_slot_injection_rejects_before_artifact_or_selection(self):
        self.configure()
        _, adapter, response = self.request("bad-slot")
        response["development_slots"] = [{"task_key": "salad", "seed": 5}]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            adapter.apply_response(
                self.tools,
                response,
                evidence_level="offline",
                cost_limit={"offline_calls": 1},
                synthetic=True,
            )
        self.assertEqual(self.store.all("rule_trials"), [])
        self.assertEqual(self.store.all("task_completion_selections"), [])

    def test_new_adapter_does_not_mutate_bounded_response_contract(self):
        self.configure()
        _, adapter, _ = self.request("schema")
        self.assertIn(
            "development_slots",
            adapter.response_schema("schema", tools=self.tools)["required"],
        )
        self.assertNotIn(
            "development_slots",
            BoundedResearchAdapter().response_schema("schema", tools=self.tools)[
                "required"
            ],
        )


class ActualFaultRecoveryTests(unittest.TestCase):
    def fault(self, run_id):
        return {
            "run_id": run_id,
            "validity": "invalid",
            "failure_class": "technical",
            "cost_closed": True,
            "cleanup_complete": True,
            "recovery_cause_resolved": True,
        }

    def test_first_four_actual_faults_recover_once_and_valid_failure_is_retained(self):
        attempts = [
            {"run_id": "valid-failure", "validity": "valid", "outcome": "failure"}
        ]
        attempts += [self.fault("fault-" + str(index)) for index in range(5)]
        for index in range(4):
            choice = recovery_choice(attempts)
            self.assertEqual(
                choice, {"state": "recovery_eligible", "run_id": "fault-" + str(index)}
            )
            attempts.append(
                {
                    "run_id": "recovered-" + str(index),
                    "recovery_of_run_id": choice["run_id"],
                }
            )
        self.assertIsNone(recovery_choice(attempts))

    def test_unclosed_first_fault_cannot_be_skipped_to_choose_another(self):
        first = {**self.fault("first"), "cleanup_complete": False}
        self.assertEqual(
            recovery_choice([first, self.fault("second")]),
            {"state": "waiting_fault_closure", "run_id": "first"},
        )

    def test_manifest_retains_first_valid_failure_and_missing_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("failed", "later-success"):
                atomic_json(
                    root / name / "runtime_context.json",
                    {"episode_id": name + "-episode"},
                )
            slots = [
                {"slot_id": "first", "task_key": "salad", "seed": 0},
                {"slot_id": "missing", "task_key": "salad", "seed": 1},
            ]
            attempts = [
                {
                    "slot_id": "first",
                    "run_id": name,
                    "output_ref": str(root / name),
                    "validity": "valid",
                    "outcome": outcome,
                }
                for name, outcome in (
                    ("failed", "failure"),
                    ("later-success", "success"),
                )
            ]
            manifest = selected_manifest(slots, attempts)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["run_id"], "failed")
            # The evaluator owns the frozen denominator and assigns unknown to
            # omitted slots. No unsuccessful slot is removed from the protocol.
            self.assertEqual(len(slots), 2)


if __name__ == "__main__":
    unittest.main()
