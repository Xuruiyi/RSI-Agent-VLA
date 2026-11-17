# ruff: noqa: UP017 - Python 3.10 support
# Copyright 2026 RSI-Agent-VLA Authors.
"""Bounded opportunities, trusted plans and actual isolated rule consumption."""

import tempfile
import unittest
from pathlib import Path

from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json, initialize
from rsi_vla.task_continuation import TaskContinuationBackend, install_research_plan


class F07ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f07-research-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResearchStore(self.root / "state/research/store")
        self.base = RunBundle(
            "base",
            "theta",
            "processor",
            "phi",
            "rho",
            "psi",
            "memory",
            "protocol",
            "backend",
            "code",
        )
        initialize(
            self.store,
            self.base,
            psi={},
            memory={"provenance": "synthetic", "legal_information": "development"},
        )
        self.protocol = {
            "protocol_id": "protocol",
            "scope": "offline",
            "benchmark_id": "synthetic",
            "task_policy": {"schema": "rsi-task-policy-v1"},
        }
        self.job = ResearchJob(
            self.store,
            "job",
            "base",
            {"whole_job_wall_seconds": 60, "artifact_bytes": 10_000_000},
        )
        self.tools = ResearchTools(
            self.store,
            self.job,
            self.protocol,
            domain_policy={
                "domain": "synthetic",
                "allowed_goal_kinds": ["manipulation"],
                "policy_ref": "domain",
            },
        )
        self.hypotheses = [
            {
                "explanation": "extra inspection repeats known evidence",
                "prediction": "one inspection remains sufficient",
                "distinguishing_test": "bounded development inspection",
            }
        ]
        self.store.save_development_result(
            ExperimentResult(
                "result",
                "base",
                "protocol",
                "synthetic",
                "development",
                "offline",
                {"information_scope": "development"},
                ["synthetic-development"],
                "criterion",
                {},
                {},
                {},
                outcome="success",
            )
        )
        self.config = {"isolation_task_root": str(self.root), "offline_fixture": True}
        self.policy = {
            "max_steps": 2,
            "max_versions": 2,
            "capabilities": {
                "inspect": {
                    "operation": {
                        "operation_id": "trusted",
                        "kind": "research_prepare",
                        "dynamic": False,
                        "stage": "research",
                        "payload": {"request_id": "fixed"},
                    },
                    "input_refs": ["result"],
                    "cost_limit": {"whole_job_wall_seconds": 3},
                    "expected_artifacts": ["request.json"],
                }
            },
            "payload_modality": "text_only",
        }
        self.opportunity("inefficient_success")

    def opportunity(self, kind, *, identity="opportunity"):
        target = {
            "domain": "synthetic",
            "kind": "manipulation",
            "description": "new legal subgoal",
            "value_criterion_ref": "external-value",
            "independent_value_test_ref": "independent-test",
        }
        return self.tools.discover_research_opportunity(
            identity,
            kind=kind,
            result_ids=["result"],
            external_value="less repeated inspection",
            hypotheses=self.hypotheses,
            expected_cost={"whole_job_wall_seconds": 3},
            target=target if kind == "new_capability" else None,
        )

    def proposal(self, identity="plan", **changes):
        value = {
            "plan_id": identity,
            "opportunity_id": "opportunity",
            "base_bundle_id": "base",
            "parent_plan_id": None,
            "distinguishing_prediction": "same result with less inspection",
            "stop_condition": "one bounded inspection",
            "requires_visual_evidence": False,
            "steps": [
                {
                    "step_id": "inspect",
                    "capability": "inspect",
                    "input_refs": ["result"],
                    "cost_limit": {"whole_job_wall_seconds": 3},
                    "expected_artifacts": ["request.json"],
                }
            ],
        }
        return {**value, **changes}

    def test_four_opportunity_purposes_preserve_evidence_and_unknown_gain(self):
        for kind in (
            "failure_unknown",
            "inefficient_success",
            "robustness_retention",
            "new_capability",
        ):
            if kind == "failure_unknown":
                result = self.store.get("development_results", "result")
                self.store.save_development_result(
                    ExperimentResult(
                        **{**result, "result_id": "failed", "outcome": "unknown"}
                    )
                )
                original = self.tools.query_development
                self.tools.query_development = lambda ids, original=original: original(
                    ["failed"]
                )
            value = self.opportunity(kind, identity=kind)
            self.assertEqual(value["kind"], kind)
            self.assertIsNone(value["capability_gain"])
            if kind == "failure_unknown":
                self.tools.query_development = original

    def test_plan_rejects_commands_unregistered_inputs_and_operator(self):
        for change in ({"command": "anything"}, {"base_bundle_id": "other"}):
            with self.assertRaises(ValueError):
                self.tools.consume_research_plan(
                    self.proposal(**change), policy=self.policy
                )
        for name, value in (("capability", "unregistered"), ("input_refs", ["sealed"])):
            proposal = self.proposal()
            proposal["steps"][0][name] = value
            with self.assertRaises(ValueError):
                self.tools.consume_research_plan(proposal, policy=self.policy)

    def test_no_budget_visual_gap_unknown_effect_and_training_never_dispatch(self):
        for name, policy, proposal in (
            (
                "unknown",
                {**self.policy, "unresolved_operations": ["original"]},
                self.proposal("unknown"),
            ),
            (
                "visual",
                self.policy,
                self.proposal("visual", requires_visual_evidence=True),
            ),
            (
                "training",
                {
                    **self.policy,
                    "capabilities": {
                        "inspect": {
                            **self.policy["capabilities"]["inspect"],
                            "training": True,
                        }
                    },
                },
                self.proposal("training"),
            ),
        ):
            result = self.tools.consume_research_plan(proposal, policy=policy)
            self.assertEqual(result["operations"], [], name)
            self.assertIn(result["action"], {"defer", "reconcile"})
        self.job.consume({"whole_job_wall_seconds": 60}, "exhausted")
        result = self.tools.consume_research_plan(
            self.proposal("budget"), policy=self.policy
        )
        self.assertEqual(result["action"], "defer")

    def test_registration_executes_existing_continuation_then_feedback_and_revision(
        self,
    ):
        registration = self.tools.consume_research_plan(
            self.proposal(), policy=self.policy
        )
        config = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": "scope",
            "protocol_ref": "protocol",
            "operations": [],
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "end",
                "nodes": {"end": {"terminal": "done"}},
            },
        }
        with self.assertRaises(RuntimeError):
            install_research_plan(
                config,
                registration,
                progress={"pending": {"operation_id": "old"}},
                after_node="end",
            )
        installed = install_research_plan(
            config, registration, progress={}, after_node="end"
        )
        self.assertEqual(config["operations"], [])
        backend = TaskContinuationBackend(
            self.root / "continuation",
            installed,
            invoke=lambda op, action, inputs: self.tools.query_development(["result"])[
                0
            ],
        )
        from rsi_vla.protocol10_coordinator import Coordinator

        coordinator = Coordinator(
            self.root / "coordinator", backend.registration, backend
        )
        state = coordinator.advance({})
        self.assertIsNone(state["pending"])
        self.assertEqual(coordinator.advance({})["status"], "completed")
        receipt = state["receipts"]["plan-inspect"]
        feedback = self.tools.record_plan_feedback("plan", receipts=[receipt])
        self.assertEqual(feedback["action"], "research_followup")
        revision = self.tools.consume_research_plan(
            self.proposal("plan-v2", parent_plan_id="plan"),
            policy=self.policy,
        )
        self.assertEqual(revision["version"], 2)
        self.assertEqual(
            self.tools.record_plan_feedback("plan", receipts=[receipt]), feedback
        )

    def prepare_code(self, identity="candidate", source=None):
        baseline = self.root / "baseline.py"
        baseline.write_text('def select_action(context):\n    return "defer"\n')
        source = source or (
            "def select_action(context):\n"
            '    return "bounded_experiment" if context["budget_available"] and context["evidence_available"] else "defer"\n'
        )
        return self.tools.prepare_code_candidate(
            identity,
            base_source_ref=baseline,
            source=source,
            declared_paths=["research_rule.py"],
            inputs={"evidence_available": True, "budget_available": True},
            config=self.config,
            timeout_seconds=3,
        )

    def test_code_scope_and_interface_reject_governance(self):
        baseline = self.root / "baseline.py"
        baseline.write_text('def select_action(context):\n return "defer"\n')
        with self.assertRaises(ValueError):
            self.tools.prepare_code_candidate(
                "invalid",
                base_source_ref=baseline,
                source="def modify_gate(context):\n return True\n",
                declared_paths=["scoring.py"],
                inputs={},
                config=self.config,
                timeout_seconds=3,
            )
        self.assertFalse(self.store.all("research_code_proposals"))

    def test_real_candidate_diff_costs_and_new_job_inherited_program_consumption(self):
        proposal = self.prepare_code()
        self.assertIn("bounded_experiment", Path(proposal["diff_ref"]).read_text())
        feedback = self.tools.execute_code_candidate("candidate", config=self.config)
        self.assertTrue(feedback["technical_valid"], feedback)
        self.assertEqual(feedback["selection"]["action"], "bounded_experiment")
        self.assertIsNone(feedback["capability_gain"])
        used = dict(self.job.state["used"])
        self.assertEqual(
            self.tools.execute_code_candidate("candidate", config=self.config),
            feedback,
        )
        self.assertEqual(self.job.state["used"], used)
        candidate = feedback["candidate"]
        new_job = ResearchJob(
            self.store,
            "next-job",
            candidate["bundle_id"],
            {"whole_job_wall_seconds": 30, "artifact_bytes": 1_000_000},
        )
        new_tools = ResearchTools(
            self.store, new_job, self.protocol, domain_policy=self.tools.domain_policy
        )
        result = new_tools.run_inherited_research_rule(
            "next-decision",
            inputs={"evidence_available": False, "budget_available": True},
            config=self.config,
        )
        self.assertTrue(result["technical_valid"], result)
        self.assertEqual(result["selection"]["action"], "defer")
        self.assertEqual(result["psi_ref"], candidate["changes"]["psi_ref"])
        with self.assertRaises(ValueError):
            new_tools.run_inherited_research_rule(
                "next-decision",
                inputs={"evidence_available": True, "budget_available": True},
                config=self.config,
            )
        with self.assertRaises(ValueError):
            self.tools.execute_code_candidate(
                "candidate", config={**self.config, "changed": True}
            )
        context = self.store.get("candidate_context", "candidate")
        self.assertEqual(context["job_ids"], ["job"])
        self.assertEqual(context["origin"]["kind"], "program")
        from rsi_vla.task_completion import TaskCompletionBackend

        protocol_ref = self.root / "frozen-protocol.json"
        atomic_json(
            protocol_ref,
            {
                **self.protocol,
                "status": "frozen",
                "scope_id": "f07",
                "anchor_bundle_id": "base",
                "metrics": {
                    "success": {"direction": "max", "min": 0, "max": 1, "weight": 1}
                },
                "hard_constraints": ["resource_limits"],
                "min_gain": 0,
                "anchor_max_drop": 0,
            },
        )
        domain_ref = self.root / "domain.json"
        atomic_json(domain_ref, self.tools.domain_policy)
        operation = {
            "operation_id": "prepare-real-next-job",
            "kind": "research_prepare",
            "dynamic": False,
            "stage": "research",
            "payload": {"request_id": "next-request"},
        }
        config = {
            **self.config,
            "schema": "rsi-task-completion-v1",
            "protocol_ref": str(protocol_ref),
            "research_store_ref": str(self.store.root),
            "evaluation_store_ref": str(self.root / "evaluation"),
            "base_bundle_id": candidate["bundle_id"],
            "operations": [operation],
            "research_requests": [
                {
                    "request_id": "next-request",
                    "job_id": "next-job",
                    "base_bundle_id": candidate["bundle_id"],
                    "domain_policy_ref": str(domain_ref),
                    "result_ids": [],
                }
            ],
        }
        production = TaskCompletionBackend(config).execute(operation, {})
        self.assertEqual(production["action"], "defer")
        self.assertEqual(production["request_count"], 0)
        self.assertTrue(production["policy_decision"]["technical_valid"])
        self.assertFalse((self.root / "release").exists())

    def test_unknown_execution_effects_block_plan_and_revision(self):
        self.store.put(
            "evidence_views",
            "view",
            {
                "bundle_id": "base",
                "l0": [{"event_id": "lost-feedback", "action_feedback_unknown": True}],
            },
        )
        result = self.store.get("development_results", "result")
        self.store.save_development_result(
            ExperimentResult(
                **{
                    **result,
                    "result_id": "unknown-effects",
                    "intervention": {
                        "information_scope": "development",
                        "evidence_view_id": "view",
                    },
                }
            )
        )
        self.tools.discover_research_opportunity(
            "unknown-opportunity",
            kind="robustness_retention",
            result_ids=["unknown-effects"],
            external_value="reconcile before action",
            hypotheses=self.hypotheses,
            expected_cost={"whole_job_wall_seconds": 3},
        )
        plan = self.tools.consume_research_plan(
            self.proposal(opportunity_id="unknown-opportunity"),
            policy=self.policy,
        )
        self.assertEqual(plan["action"], "reconcile")
        self.assertEqual(plan["unknown_effect_event_ids"], ["lost-feedback"])
        self.assertEqual(plan["operations"], [])

    def test_code_rejects_source_drift_and_host_calls(self):
        proposal = self.prepare_code()
        Path(proposal["source_ref"]).write_text(
            'def select_action(context):\n return "defer"\n'
        )
        with self.assertRaises(ValueError):
            self.tools.execute_code_candidate("candidate", config=self.config)
        with self.assertRaises(ValueError):
            self.prepare_code(
                "host-call",
                'def select_action(context):\n return open("/etc/passwd").read()\n',
            )

    def test_candidate_cost_snapshot_keeps_prior_failed_program_and_storage_dimensions(
        self,
    ):
        self.prepare_code(
            "failed", 'def select_action(context):\n return "change_score"\n'
        )
        failed = self.tools.execute_code_candidate("failed", config=self.config)
        self.assertFalse(failed["technical_valid"])
        spent = dict(self.job.state["used"])
        self.prepare_code("valid")
        valid = self.tools.execute_code_candidate("valid", config=self.config)
        for dimension, amount in spent.items():
            self.assertGreater(valid["candidate"]["costs"][dimension], amount)
        self.assertIsNone(
            valid["storage_observation"]["persistent"]["exact_cumulative_bytes"]
        )
        self.assertGreater(valid["storage_observation"]["persistent_current_bytes"], 0)
        limited = ResearchJob(
            self.store, "missing-storage", "base", {"whole_job_wall_seconds": 30}
        )
        tools = ResearchTools(
            self.store, limited, self.protocol, domain_policy=self.tools.domain_policy
        )
        baseline = self.root / "baseline.py"
        tools.prepare_code_candidate(
            "no-storage",
            base_source_ref=baseline,
            source='def select_action(context):\n return "inspect_evidence"\n',
            declared_paths=["research_rule.py"],
            inputs={"evidence_available": True, "budget_available": True},
            config=self.config,
            timeout_seconds=3,
        )
        with self.assertRaises(PermissionError):
            tools.execute_code_candidate("no-storage", config=self.config)
        self.assertFalse(
            self.store.path("research_code_launches", "no-storage").exists()
        )

    def test_invalid_program_feedback_keeps_failure_without_candidate(self):
        self.prepare_code(
            source='def select_action(context):\n    return "change_score"\n'
        )
        result = self.tools.execute_code_candidate("candidate", config=self.config)
        self.assertFalse(result["technical_valid"])
        self.assertIsNone(result["candidate"])
        self.assertGreater(result["costs"]["whole_job_wall_seconds"], 0)

    def test_four_plans_real_host_analysis_and_revision_reuse_original_receipts(self):
        import os
        import sys
        import time
        from copy import deepcopy
        from datetime import datetime, timedelta, timezone

        from rsi_vla.midterm_host import MidtermHost, build_configuration, read

        operator = self.root / "operator"
        operator.mkdir(exist_ok=True)
        scope_ref = operator / "scope.json"
        phase_ref = operator / "phase.json"
        protocol_ref = operator / "protocol.json"
        manifest_ref = operator / "differences.json"
        atomic_json(manifest_ref, {"differences": [1, 0, 1, -1]})
        atomic_json(protocol_ref, {"protocol_id": "f07-host-offline"})
        kinds = (
            "failure_unknown",
            "inefficient_success",
            "robustness_retention",
            "new_capability",
        )
        names = ["host-" + kind for kind in kinds]
        limits = {
            "wall_seconds": 1000,
            "storage_bytes": 512 * 1024**2,
            "current_disk_bytes": 512 * 1024**2,
            "model_requests": 0,
            "gpu_seconds": 0,
            "concurrent_gpu_devices": 0,
        }
        atomic_json(
            scope_ref,
            {
                "schema": "rsi-midterm-scope-v1",
                "scope_id": "f07-host-offline",
                "status": "active",
                "authorized": True,
                "dynamic_authorized": False,
                "cpu_authorized": True,
                "cpu_operation_ids": [
                    name + "-inspect" for name in [*names, "host-revision"]
                ],
                "owner_uid": os.getuid(),
                "write_root": str(self.root),
                "storage_write_roots": [str(self.root)],
                "outer_limits": limits,
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat(),
            },
        )
        phase_limits = {
            k: v for k, v in limits.items() if k != "concurrent_gpu_devices"
        }
        atomic_json(
            phase_ref,
            {
                "limits": phase_limits,
                "used": dict.fromkeys(phase_limits, 0),
                "remaining": phase_limits,
                "unresolved_costs": [],
            },
        )
        core = {
            "schema": "rsi-task-completion-v1",
            "scope_authorization_ref": str(scope_ref),
            "protocol_ref": str(protocol_ref),
            "phase_ref": str(phase_ref),
            "repository_root": str(Path(__file__).resolve().parents[2]),
            "base_bundle_id": "base",
            "isolation_task_root": str(self.root),
            "operations": [],
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": "start",
                "nodes": {"start": {"terminal": "idle"}},
            },
        }
        policy = deepcopy(self.policy)
        capability = policy["capabilities"]["inspect"]
        capability.update(
            operation={
                "operation_id": "trusted",
                "stage": "analysis",
                "kind": "host_command",
                "dynamic": False,
                "cpu_purpose": "analysis",
                "version": "f07-synthetic-v1",
                "payload": {"manifest_ref": str(manifest_ref)},
            },
            input_refs=[str(manifest_ref)],
            expected_artifacts=["artifact.json"],
        )
        registrations = []
        terminal = "start"
        for kind, name in zip(kinds, names, strict=True):
            if kind == "failure_unknown":
                result = self.store.get("development_results", "result")
                self.store.save_development_result(
                    ExperimentResult(
                        **{
                            **result,
                            "result_id": "unknown-result",
                            "outcome": "unknown",
                        }
                    )
                )
                self.tools.discover_research_opportunity(
                    kind,
                    kind=kind,
                    result_ids=["unknown-result"],
                    external_value="unknown outcome needs analysis",
                    hypotheses=self.hypotheses,
                    expected_cost={"whole_job_wall_seconds": 3},
                )
            else:
                self.opportunity(kind, identity=kind)
            proposal = self.proposal(name, opportunity_id=kind)
            proposal["steps"][0].update(
                input_refs=[str(manifest_ref)], expected_artifacts=["artifact.json"]
            )
            registered = self.tools.consume_research_plan(proposal, policy=policy)
            registrations.append(registered)
            core = install_research_plan(
                core, registered, progress={}, after_node=terminal
            )
            terminal = name + "-feedback"
        core_ref, host_ref = operator / "core.json", operator / "host.json"
        atomic_json(core_ref, core)
        config = build_configuration(
            core_ref,
            host_ref,
            operator / "hub.json",
            python=sys.executable,
            poll_seconds=0.05,
        )
        atomic_json(host_ref, config)
        atomic_json(config["inputs_ref"], {})

        def finish(path):
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                host = MidtermHost(path)
                state = host.advance()
                if state["status"] == "completed":
                    return host
                if state["status"] == "unknown":
                    job = host.root / "jobs" / state["operation_id"]
                    self.fail(
                        repr(state)
                        + "\n"
                        + repr(host.coordinator().status())
                        + "\n"
                        + (
                            (job / "backend.log").read_text()
                            if (job / "backend.log").exists()
                            else "no backend launch"
                        )
                    )
                time.sleep(0.03)
            self.fail("bounded real Host analysis did not finish: " + repr(state))

        host = finish(host_ref)
        progress = host.coordinator().status()
        self.assertEqual(len(progress["receipts"]), 4)
        old_outputs = {}
        for registered in registrations:
            identity = registered["operations"][0]["operation_id"]
            receipt = progress["receipts"][identity]
            feedback = self.tools.record_plan_feedback(
                registered["plan_id"], receipts=[receipt]
            )
            self.assertEqual(feedback["action"], "research_followup")
            native = read(config["operations"][identity]["result_ref"])
            self.assertEqual(native["status"], "completed")
            self.assertTrue(Path(native["artifact_ref"]).exists())
            old_outputs[identity] = (
                host.root / "jobs" / identity / "completion.json"
            ).read_bytes()
        self.assertEqual(host.ledger.read()["used"]["model_requests"], 0)
        self.assertEqual(host.ledger.read()["used"]["gpu_seconds"], 0)
        self.assertGreater(host.ledger.read()["used"]["wall_seconds"], 0)
        revision = self.proposal(
            "host-revision", opportunity_id=kinds[0], parent_plan_id=names[0]
        )
        revision["steps"][0].update(
            input_refs=[str(manifest_ref)], expected_artifacts=["artifact.json"]
        )
        registered = self.tools.consume_research_plan(revision, policy=policy)
        revised = install_research_plan(
            core, registered, progress=progress, after_node=terminal
        )
        second = operator / "revision"
        second.mkdir()
        second_core, second_host = second / "core.json", second / "host-v2.json"
        atomic_json(second_core, revised)
        config2 = build_configuration(
            second_core,
            second_host,
            second / "hub.json",
            python=sys.executable,
            poll_seconds=0.05,
        )
        atomic_json(second_host, config2)
        atomic_json(config2["inputs_ref"], {})
        restarted = finish(second_host)
        self.assertEqual(len(restarted.coordinator().status()["receipts"]), 1)
        self.assertEqual(len(restarted.ledger.read()["reservations"]), 5)
        for identity, before in old_outputs.items():
            self.assertEqual(
                (restarted.root / "jobs" / identity / "completion.json").read_bytes(),
                before,
            )
        self.assertEqual(len(revised["prior_receipts"]), 4)


if __name__ == "__main__":
    unittest.main()
