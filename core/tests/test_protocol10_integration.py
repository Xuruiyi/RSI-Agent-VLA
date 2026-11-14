# Copyright 2026 RSI-Agent-VLA Authors.
"""CPU integration of the production protocol10 backend and durable coordinator.

Only Runtime processes, model transport and their independent boundary reports
are fake. Runtime preparation, Research response validation, development cycles,
phase accounting, matrix evaluation, Candidate submission and Gate are real.
"""

import contextlib
import io
import json
import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rsi_vla.__main__ import main as core_main
from rsi_vla.bounded_runner import execute as execute_research
from rsi_vla.contracts import ExperimentResult
from rsi_vla.improvement_protocol import BASELINE, TASKS
from rsi_vla.protocol10_backend import CANDIDATE, Protocol10Backend, read
from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import Protocol10ResearchAdapter, ResearchTools
from rsi_vla.runtime import prepare_runtime
from rsi_vla.storage import ResearchStore, atomic_json

ROOT = Path(__file__).resolve().parents[2]


class Protocol10IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scope = self.root / "scope"
        self.artifacts = self.root / "state"
        self.protocol = read(ROOT / "config/pilot_5a_protocol10.json")
        # This is a new offline engineering fixture, never a simulation record.
        self.protocol.update(
            scope="offline",
            dynamic_authorized=True,
            allowed_runtime_splits=["release", "development"],
        )
        self.protocol["runtime_config"] = {
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
        }
        self.protocol["budget"]["token_policy"] = "observed_only"
        self.protocol["budget"]["per_attempt_proposal"] = read(
            ROOT / "config/pilot_v7.json"
        )["budget"]["per_attempt_proposal"]
        self.protocol["task_sets"]["release"]["benchmark_id"] = self.protocol[
            "benchmark_id"
        ]
        self.protocol_ref = self.root / "protocol.json"
        atomic_json(self.protocol_ref, self.protocol)
        profile = read(ROOT / "config/run_profile_v2.json")
        profile.update(dynamic_authorized=True, token_policy="observed_only")
        profile["research_llm"]["reasoning_effort"] = "high"
        # Existing matched-prefix support permits the offline finite chunk trial.
        profile["control"]["executed_prefix_steps"] = 2
        self.profile_ref = self.root / "profile.json"
        atomic_json(self.profile_ref, profile)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(
                [
                    "init",
                    "--artifacts",
                    str(self.artifacts),
                    "--profile",
                    str(self.profile_ref),
                    "--protocol",
                    str(self.protocol_ref),
                    "--bundle",
                    BASELINE,
                ]
            )
        self.store = ResearchStore(self.artifacts / "research")
        self.phase_ref = self.root / "phase.json"
        caps = {
            "simulation_episodes": 22,
            "model_requests": 441,
            "tool_calls": 1320,
            "environment_steps": 44000,
            "wall_seconds": 21180,
            "gpu_seconds": 19800,
            "storage_bytes": 24763170816,
        }
        atomic_json(
            self.phase_ref,
            {
                "limits": caps,
                "remaining": caps,
                "used": dict.fromkeys(caps, 0),
                "unresolved_costs": [],
            },
        )
        policy = {
            "domain": "tabletop",
            "allowed_goal_kinds": ["manipulation"],
            "policy_ref": "core/tests/test_batch4.py:BoundedRunnerTests.prepared",
            "rule_operators": ["chunk_steps"],
        }
        self.policy_ref = self.root / "policy.json"
        atomic_json(self.policy_ref, policy)
        self.job_limits = {
            "model_requests": 441,
            "whole_job_wall_seconds": 21180,
            "tool_calls": 1320,
            "environment_steps": 44000,
            "gpu_device_seconds_envelope": 19800,
            "artifact_bytes": 24763170816,
        }
        job = ResearchJob(self.store, "protocol10-job", BASELINE, self.job_limits)
        tools = ResearchTools(self.store, job, self.protocol, domain_policy=policy)
        historical = ExperimentResult(
            "legal-offline-development",
            BASELINE,
            self.protocol["protocol_id"],
            self.protocol["benchmark_id"],
            "development",
            "offline",
            {"information_scope": "development"},
            ["offline-development-input"],
            self.protocol["protocol_id"],
            {},
            {},
            {},
            assigned_count=1,
            outcome="success",
            assignments=[
                {
                    "task_key": "position_swap_salad",
                    "seed": 2,
                    "outcome": "success",
                    "validity": "valid",
                }
            ],
        )
        self.store.save_development_result(historical)
        adapter = Protocol10ResearchAdapter()
        transport = adapter.prepare_request(
            tools,
            request_id="protocol10-research",
            result_ids=[historical.result_id],
            max_input_tokens=64000,
            max_output_tokens=4000,
        )
        manifest = self.root / "sources.json"
        source = "core/src/rsi_vla/research_transport.py"
        source_copy = self.root / "source" / source
        source_copy.parent.mkdir(parents=True)
        source_copy.write_bytes((ROOT / source).read_bytes())
        atomic_json(manifest, {"version": "batch4-preparation", "files": [source]})
        research_output = self.root / "research-output"
        research_permit = self.root / "research-permit.json"
        atomic_json(
            research_permit,
            {
                "schema": "rsi-bounded-research-permit-v1",
                "authorized": True,
                "job_id": job.state["job_id"],
                "request_id": "protocol10-research",
                "bundle_id": BASELINE,
                "protocol_id": self.protocol["protocol_id"],
                "output_dir": str(research_output),
                "model": "gpt-6.1-sol",
                "reasoning_effort": "high",
                "input": transport,
                "job_limits": self.job_limits,
                "limits": {
                    "model_requests": 1,
                    "wall_seconds": 180,
                    "storage_bytes": 10485760,
                },
                "phase_snapshot_ref": str(self.phase_ref),
                "source_manifest_ref": str(manifest),
            },
        )
        self.slots = []
        for side in ("baseline", "candidate", "development"):
            for task in TASKS:
                for seed in [2] if side == "development" else [0, 1]:
                    run_id = f"{side}-{task}-s{seed}"
                    self.slots.append(
                        {
                            "side": side,
                            "task_key": task,
                            "seed": seed,
                            "run_id": run_id,
                            "bundle_id": BASELINE if side == "baseline" else CANDIDATE,
                            "output_ref": str(self.root / "outputs" / run_id),
                            "scratch_ref": str(self.root / "scratch" / run_id),
                            "permit_ref": str(
                                self.root / "permits" / (run_id + ".json")
                            ),
                        }
                    )
        self.config = {
            "offline_fixture": True,
            "repository_root": str(ROOT),
            "protocol_ref": str(self.protocol_ref),
            "runtime_protocol_ref": str(self.protocol_ref),
            "profile_ref": str(self.profile_ref),
            "artifacts_root": str(self.artifacts),
            "phase_ref": str(self.phase_ref),
            "runtime_slots": self.slots,
            "research": {
                "store_ref": str(self.store.root),
                "job_id": "protocol10-job",
                "request_id": "protocol10-research",
                "domain_policy_ref": str(self.policy_ref),
                "output_ref": str(research_output),
                "permit_ref": str(research_permit),
                "preflight_ref": str(self.root / "preflight.json"),
                "cost_limit": {
                    "model_requests": 20,
                    "whole_job_wall_seconds": 900,
                    "tool_calls": 60,
                    "environment_steps": 2000,
                    "gpu_device_seconds_envelope": 900,
                    "artifact_bytes": 1073741824,
                },
            },
        }
        atomic_json(
            self.root / "preflight.json",
            [{"operator": "chunk_steps", "currently_trialable": True}],
        )
        self.launches, self.research_launches = [], 0
        self.ceiling = False
        self.candidate_failure = False
        self.evaluation_fault = False
        self.technical_fault_run = None
        self.missing_cost = False
        self.cleanup_unknown = False
        self.incomplete_candidate_evidence = False
        self.invalid_response = False
        self.recovery_resolved = {}
        self._write_runtime_permits("baseline")
        self.backend = Protocol10Backend(
            self.scope,
            self.config,
            runtime_execute=self.runtime_execute,
            audit=self.audit,
            progress_inspector=self.progress,
            research_execute=self.research_execute,
            trial_collector=self.collect,
        )
        self.coordinator = Coordinator(
            self.scope / "coordination", self.backend.registration, self.backend
        )
        self.patches = [
            patch("rsi_vla.evaluator.inspect_attempt", side_effect=self.audit),
            patch("rsi_vla.evaluator.inspect_progress", side_effect=self.progress),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_runtime_permits(self, side):
        for slot in self.slots:
            if slot["side"] != side:
                continue
            draft = prepare_runtime(
                ROOT,
                self.artifacts,
                protocol_path=self.protocol_ref,
                run_id=slot["run_id"],
                bundle_id=slot["bundle_id"],
                task_key=slot["task_key"],
                seed=slot["seed"],
                split="development" if side == "development" else "release",
                output_dir=slot["output_ref"],
                storage_write_roots=[slot["scratch_ref"]],
            )
            context = draft["context"]
            permit = {
                "authorized": True,
                "run_id": slot["run_id"],
                "bundle_id": slot["bundle_id"],
                "protocol_id": self.protocol["protocol_id"],
                "task_key": slot["task_key"],
                "seed": slot["seed"],
                "simulation_episodes": 1,
                "gpu_devices": [0],
                "output_dir": slot["output_ref"],
                "storage_write_roots": [slot["output_ref"], slot["scratch_ref"]],
                "limits": context["limits"],
                "phase_snapshot_ref": str(self.phase_ref),
            }
            if side == "development":
                tasks = ["position_swap_salad", "old_milk"]
                index = (
                    tasks.index(slot["task_key"]) if slot["task_key"] in tasks else 0
                )
                experiment_id = "protocol10-research-experiment" + (
                    "" if index == 0 else "-slot02"
                )
                if slot.get("recovery_of"):
                    experiment_id += "-recovery"
                trial = self.store.get("rule_trials", "protocol10-research-experiment")
                experiment = self.store.get(
                    "research_experiments", "protocol10-research-experiment"
                )
                permit.update(
                    schema="rsi-development-trial-permit-v1",
                    experiment_id=experiment_id,
                    job_id="protocol10-job",
                    intervention=experiment["intervention"],
                    runtime_rules=read(trial["runtime_rules_ref"]),
                    collection=context.get("collection"),
                    code_ref=context["code_ref"],
                )
            atomic_json(slot["permit_ref"], permit)

    def runtime_execute(self, plan):
        context = plan["context"]
        output = Path(context["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        self.launches.append(context["run_id"])
        atomic_json(output / "runtime_context.json", context)
        success = not (
            context["bundle_id"] == BASELINE
            and context["task"]["suite"] == "libero_object_swap"
            and context["task"]["seed"] == 0
            and not self.ceiling
        )
        if (
            self.candidate_failure
            and context["bundle_id"] == CANDIDATE
            and plan["split"] == "release"
        ):
            success = False
        identity = {
            k: context[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        raw = dict(
            **identity,
            task=context["task"],
            planner=context["planner"],
            benchmark_success=success,
            terminated=success,
            truncated=not success,
            native_result_source={
                "terminated": "fake native boundary",
                "truncated": "fake native boundary",
            },
        )
        atomic_json(output / "independent_result.json", raw)
        atomic_json(
            output / "supervisor.json",
            {"status": "completed", "exit_code": 0, "cleanup_confirmed": True},
        )
        atomic_json(
            output / "launcher_costs.json",
            {"identity": identity, "whole_job_wall_seconds": 1.0},
        )

    def audit(self, output, **kwargs):
        context = read(Path(output) / "runtime_context.json")
        raw = read(Path(output) / "independent_result.json")
        identity = {
            k: context[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        costs = {
            "model_requests": 1,
            "tool_calls": 1,
            "environment_steps": 5,
            "whole_job_wall_seconds": 1.0,
            "gpu_device_seconds_envelope": 1.0,
            "artifact_bytes": 4096,
        }
        checks = dict.fromkeys(
            [
                "identity",
                "model_reference",
                "vla_actual_loading_and_prediction",
                "tool_budget",
                "request_budget",
                "step_budget",
                "wall_gpu_storage_budget",
                "all_tools_use_counted_dispatcher",
                "no_mid_attempt_reset",
                "clean_completion",
                "owner_cleanup",
                "complete_usage",
            ],
            True,
        )
        if self.missing_cost:
            costs.pop("model_requests")
        if self.cleanup_unknown:
            checks["clean_completion"] = None
            checks["owner_cleanup"] = None
        if self.incomplete_candidate_evidence and context["bundle_id"] == CANDIDATE:
            checks["model_reference"] = None
        valid = context["run_id"] != self.technical_fault_run
        if not valid:
            checks["clean_completion"] = False
        return {
            "identity": identity,
            "passed": valid,
            "outcome": "success" if raw["benchmark_success"] else "failure",
            "checks": checks,
            "costs": costs,
            "confirmed_violations": [],
            "notes": [],
        }

    def progress(self, output, identity):
        if self.evaluation_fault:
            raise ValueError("offline evaluator repair required")
        return {
            "value": True,
            "evidence_refs": [str(Path(output) / "independent_result.json")],
        }

    def collect(self, tools, experiment_id, output):
        context = read(Path(output) / "runtime_context.json")
        plan = tools._owned("research_experiments", experiment_id)
        task = next(
            t["task_key"]
            for t in self.protocol["tasks"]
            if t["suite"] == context["task"]["suite"]
            and t["task_id"] == context["task"]["task"]
        )
        result = ExperimentResult(
            experiment_id + "-result",
            CANDIDATE,
            self.protocol["protocol_id"],
            self.protocol["benchmark_id"],
            "development",
            "offline",
            {
                "information_scope": "development",
                "runtime_loaded": True,
                **plan["intervention"],
            },
            [str(Path(output) / "independent_result.json")],
            self.protocol["protocol_id"],
            {},
            {},
            self.audit(output)["costs"],
            assigned_count=1,
            outcome="success",
            assignments=[
                {"task_key": task, "seed": 2, "outcome": "success", "validity": "valid"}
            ],
        )
        if context["run_id"] == self.technical_fault_run:
            result = replace(
                result,
                outcome="unknown",
                invalid_count=1,
                unknown_count=1,
                assignments=[
                    {
                        "task_key": task,
                        "seed": 2,
                        "outcome": "unknown",
                        "validity": "invalid",
                    }
                ],
            )
            return result, {
                "technical_valid": False,
                "native_success": False,
                "runtime_loaded": True,
            }
        return result, {
            "technical_valid": True,
            "native_success": True,
            "runtime_loaded": True,
        }

    def research_execute(self, tools, adapter, transport, **kwargs):
        self.research_launches += 1
        output = Path(kwargs["output"])
        fixture = self

        class FakeGate:
            def __init__(self, context):
                self.url = "offline-fake-gate"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def supervisor(command, context_path):
            response = {
                "schema": "rsi-bounded-research-response-v2",
                "request_id": "protocol10-research",
                "decision": "prepare_rule_trial",
                "source": "normal",
                "result_ids": ["legal-offline-development"],
                "reason": "bounded offline prefix test",
                "external_value": "tabletop completion",
                "competing_explanations": [
                    {
                        "explanation": "overshoot",
                        "prediction": "prefix helps",
                        "distinguishing_test": "paired test",
                    },
                    {
                        "explanation": "observation",
                        "prediction": "no gain",
                        "distinguishing_test": "paired test",
                    },
                ],
                "intervention": {"chunk_steps": 2},
                "distinguishing_prediction": "fewer overshoots",
                "stop_condition": "one response",
                "missing_evidence": [],
                "development_slots": [
                    {"task_key": "position_swap_salad", "seed": 2},
                    {"task_key": "old_milk", "seed": 2},
                ],
            }
            if self.invalid_response:
                response["development_slots"][0]["seed"] = 3
            atomic_json(
                output / "sdk-result.json",
                {"status": "completed", "final_response": json.dumps(response)},
            )
            events = [
                {
                    "kind": "request_admitted",
                    "model": "gpt-6.1-sol",
                    "reasoning_effort": "high",
                },
                {
                    "kind": "response_terminal",
                    "backend_model": "gpt-6.1-sol",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "cached_input_tokens": 0,
                    },
                },
            ]
            (output / "model_requests.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )
            return {"status": "completed", "exit_code": 0, "cleanup_confirmed": True}

        result = execute_research(
            tools,
            adapter,
            transport,
            **kwargs,
            gate_factory=FakeGate,
            supervisor=supervisor,
            lock_factory=nullcontext,
        )
        fixture._write_runtime_permits("candidate")
        fixture._write_runtime_permits("development")
        return result

    def inputs(self):
        state = self.coordinator.status()
        facts = (
            self.backend.observe(
                state, {"recovery_cause_resolved": self.recovery_resolved}
            )
            if state["pending"] is None
            else {"next_operation": state["pending"]}
        )
        op = facts.get("next_operation")
        return {
            "now": "2026-11-13T00:00:00Z",
            "authorization": {
                "dynamic_authorized": True,
                "binding": self.backend.registration,
                "expires_at": "2026-11-14T00:00:00Z",
                "operation_ids": [op["operation_id"]] if op else [],
            },
            "resources": {
                "available": read(self.phase_ref)["remaining"],
                "required": {},
                "reserve": {},
            },
            "recovery_cause_resolved": self.recovery_resolved,
            "offline_used_seconds": 0,
            "offline_used_bytes": 0,
        }

    def run_until(self, predicate, limit=90):
        for _ in range(limit):
            state = self.coordinator.advance(self.inputs())
            if predicate(state):
                return state
            if state["status"] == "waiting":
                self.fail(str(state.get("pending_receipt") or state["wait"]))
        self.fail("protocol10 did not reach the expected checkpoint")

    def test_non_ceiling_full_chain_two_development_slots_and_no_activation(self):
        state = self.run_until(lambda s: s["status"] == "completed")
        self.assertEqual(state["terminal"], "comparison_complete")
        self.assertEqual((len(self.launches), self.research_launches), (18, 1))
        self.assertEqual(state["receipts"]["gate"]["decision"]["verdict"], "accept")
        self.assertFalse(state["receipts"]["qualification-handoff"]["release_eligible"])
        candidate = self.store.get("candidates", CANDIDATE)
        self.assertEqual(len(candidate["development_results"]), 2)
        self.assertEqual(len(list(self.store.root.glob("candidates/*.json"))), 1)
        self.assertEqual(len(read(self.phase_ref)["protocol10_trial_charges"]), 16)
        self.assertEqual(len(read(self.phase_ref)["development_trial_charges"]), 2)
        self.assertFalse((self.scope / "active.json").exists())
        # Resume twice: no Runtime, Research, cost or submission replay.
        before = read(self.phase_ref)
        self.coordinator.advance(self.inputs())
        self.coordinator.advance(self.inputs())
        self.assertEqual(read(self.phase_ref), before)
        self.assertEqual((len(self.launches), self.research_launches), (18, 1))
        request = self.store.get("bounded_requests", "protocol10-research")
        self.assertTrue(all(r["role"] == "development" for r in request["evidence"]))
        self.assertEqual(set(self.store.root.glob("release_results/*")), set())

    def test_ceiling_stops_before_research_or_candidate(self):
        self.ceiling = True
        state = self.run_until(lambda s: s["status"] == "completed")
        self.assertEqual(state["terminal"], "freeze_stop_ceiling")
        self.assertEqual((len(self.launches), self.research_launches), (8, 0))
        self.assertFalse(self.store.path("bundles", CANDIDATE).exists())

    def test_runtime_return_evaluation_repair_and_native_completion_reconciliation(
        self,
    ):
        self.coordinator.advance(self.inputs())
        self.assertEqual(len(self.launches), 1)
        run = self.slots[0]["run_id"]
        # Lose only the adapter acknowledgement; native output still proves return.
        self.backend.records.path("runtime_returns", run).unlink()
        self.backend.records.path("receipts", run + "-runtime").unlink()
        op = self.backend._op(
            "runtime", "baseline", self.slots[0], True, run + "-runtime"
        )
        receipt = self.backend.reconcile(op, {})
        self.assertEqual(receipt["status"], "completed")
        self.coordinator.advance(self.inputs())  # independent fee settlement
        before = deepcopy(read(self.phase_ref))
        self.evaluation_fault = True
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(read(self.phase_ref), before)
        self.evaluation_fault = False
        self.coordinator.advance(self.inputs())
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(read(self.phase_ref), before)

    def test_no_authority_waits_and_untyped_event_cannot_dispatch(self):
        state = self.coordinator.advance({})
        self.assertEqual(state["wait"]["reason"], "authorization_required")
        self.assertFalse(self.launches)
        with self.assertRaises(ValueError):
            self.coordinator.event(
                {
                    "type": "shell",
                    "event_id": "bad",
                    "operation_id": None,
                    "command": "true",
                }
            )
        self.assertFalse(self.launches)

    def test_configuration_and_domain_are_bound_to_scope(self):
        modified = deepcopy(self.config)
        modified["research"]["output_ref"] += "-other"
        with self.assertRaisesRegex(ValueError, "conflicting registration"):
            Protocol10Backend(self.scope, modified)

    def add_recovery(self, run_id):
        original = next(slot for slot in self.slots if slot["run_id"] == run_id)
        recovery = {
            **original,
            "run_id": run_id + "-recovery",
            "recovery_of": run_id,
            "output_ref": original["output_ref"] + "-recovery",
            "scratch_ref": original["scratch_ref"] + "-recovery",
            "permit_ref": original["permit_ref"].replace(".json", "-recovery.json"),
        }
        self.slots.append(recovery)
        self.config["runtime_slots"] = self.slots
        if original["side"] == "baseline":
            self._write_runtime_permits(original["side"])
        self.scope = self.root / "recovery-scope"
        self.backend = Protocol10Backend(
            self.scope,
            self.config,
            runtime_execute=self.runtime_execute,
            audit=self.audit,
            progress_inspector=self.progress,
            research_execute=self.research_execute,
            trial_collector=self.collect,
        )
        self.coordinator = Coordinator(
            self.scope / "coordination", self.backend.registration, self.backend
        )
        return recovery

    def test_technical_recovery_requires_closed_cost_cleanup_and_resolved_cause(self):
        run_id = self.slots[0]["run_id"]
        recovery = self.add_recovery(run_id)
        self.technical_fault_run = run_id
        self.run_until(
            lambda state: run_id + "-attempt_evaluation" in state["receipts"]
        )
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["wait"]["reason"], "technical_recovery_cause")
        self.assertEqual(len(self.launches), 1)
        self.recovery_resolved[run_id] = True
        state = self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(len(self.launches), 19)
        self.assertEqual(self.launches[:2], [run_id, recovery["run_id"]])
        self.assertEqual(len(read(self.phase_ref)["protocol10_trial_charges"]), 17)
        self.assertEqual(state["receipts"]["gate"]["decision"]["verdict"], "accept")
        self.assertEqual(read(self.phase_ref)["used"]["simulation_episodes"], 19)

    def test_valid_failure_is_never_replaced_by_registered_recovery(self):
        run_id = "baseline-position_swap_salad-s0"
        recovery = self.add_recovery(run_id)
        self.recovery_resolved[run_id] = True
        self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(len(self.launches), 18)
        self.assertNotIn(recovery["run_id"], self.launches)

    def test_missing_cost_and_cleanup_wait_for_same_attempt_reconciliation(self):
        self.coordinator.advance(self.inputs())
        self.missing_cost = True
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(len(self.launches), 1)
        phase = read(self.phase_ref)
        self.assertEqual(phase["used"]["simulation_episodes"], 1)
        self.assertTrue(phase["unresolved_costs"])
        self.missing_cost = False
        self.cleanup_unknown = True
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(read(self.phase_ref)["used"]["simulation_episodes"], 1)
        self.cleanup_unknown = False
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["status"], "ready")
        phase = read(self.phase_ref)
        self.assertEqual(
            (phase["used"]["simulation_episodes"], phase["used"]["model_requests"]),
            (1, 1),
        )
        self.assertFalse(phase["unresolved_costs"])
        self.assertEqual(len(self.launches), 1)

    def test_gate_reject_keeps_active_and_fixed_denominator(self):
        self.candidate_failure = True
        state = self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(state["receipts"]["gate"]["decision"]["verdict"], "reject")
        candidate = self.backend.evaluations.get("results", "candidate-protocol10")
        self.assertEqual(
            (candidate["assigned_count"], candidate["invalid_count"]), (8, 0)
        )
        self.assertFalse(state["receipts"]["qualification-handoff"]["release_eligible"])
        self.assertFalse((self.scope / "active.json").exists())
        self.assertEqual(len(self.launches), 18)

    def test_gate_inconclusive_does_not_create_qualification(self):
        self.incomplete_candidate_evidence = True
        state = self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(
            state["receipts"]["gate"]["decision"]["verdict"], "inconclusive"
        )
        self.assertFalse(state["receipts"]["qualification-handoff"]["release_eligible"])
        self.assertFalse((self.scope / "active.json").exists())

    def test_invalid_single_research_response_stops_and_retains_request_charge(self):
        self.invalid_response = True
        state = self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(state["terminal"], "research_stop")
        self.assertEqual((len(self.launches), self.research_launches), (8, 1))
        self.assertEqual(len(read(self.phase_ref)["research_charges"]), 1)
        self.assertFalse(self.store.path("bundles", CANDIDATE).exists())

    def test_unknown_submission_keeps_same_identity_without_dispatch(self):
        output = Path(self.slots[0]["output_ref"])
        output.mkdir(parents=True)
        state = self.coordinator.advance(self.inputs())
        pending = state["pending"]
        self.assertEqual(state["pending_receipt"]["status"], "unknown")
        for _ in range(2):
            state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["pending"], pending)
        self.assertFalse(self.launches)

    def test_pause_and_reconcile_do_not_submit_a_new_candidate(self):
        self.run_until(
            lambda state: "development-development-old_milk-s2" in state["receipts"]
        )
        self.coordinator.event(
            {"type": "pause", "event_id": "pause-before-submit", "operation_id": None}
        )
        self.coordinator.advance(self.inputs(), reconcile_only=True)
        self.assertFalse(self.store.path("candidates", CANDIDATE).exists())
        self.coordinator.event(
            {"type": "resume", "event_id": "resume-submit", "operation_id": None}
        )
        self.run_until(lambda state: state["status"] == "completed")
        self.assertTrue(self.store.path("candidates", CANDIDATE).exists())

    def test_complete_remaining_envelope_blocks_single_episode_surplus(self):
        phase = read(self.phase_ref)
        phase["limits"]["simulation_episodes"] = 1
        phase["remaining"]["simulation_episodes"] = 1
        atomic_json(self.phase_ref, phase)
        state = self.coordinator.advance(self.inputs())
        self.assertEqual(state["wait"]["reason"], "budget_wait")
        self.assertEqual(
            state["pending_receipt"]["admission"]["additional_required"][
                "simulation_episodes"
            ],
            21,
        )
        self.assertFalse(self.launches)

    def test_absent_offline_accounting_does_not_dispatch(self):
        inputs = self.inputs()
        inputs.pop("offline_used_seconds")
        state = self.coordinator.advance(inputs)
        self.assertEqual(state["wait"]["reason"], "phase_resource_evidence")
        self.assertFalse(self.launches)

    def test_development_technical_recovery_retains_original_attempt_and_costs(self):
        run_id = "development-position_swap_salad-s2"
        recovery = self.add_recovery(run_id)
        self.technical_fault_run = run_id
        self.recovery_resolved[run_id] = True
        state = self.run_until(lambda state: state["status"] == "completed")
        self.assertEqual(state["terminal"], "comparison_complete")
        self.assertEqual((len(self.launches), self.research_launches), (19, 1))
        self.assertIn(recovery["run_id"], self.launches)
        self.assertEqual(len(read(self.phase_ref)["development_trial_charges"]), 3)
        candidate = self.store.get("candidates", CANDIDATE)
        self.assertEqual(len(candidate["development_results"]), 2)
        self.assertEqual(len(list(self.store.root.glob("research_findings/*.json"))), 3)

    def test_research_terminal_response_failures_close_but_unknown_requests_wait(self):
        cfg = self.config["research"]
        output = Path(cfg["output_ref"])
        output.mkdir(parents=True)
        transport = self.store.get("bounded_transports", cfg["request_id"])
        atomic_json(output / "request.json", transport)
        context = {
            "job_id": cfg["job_id"],
            "bundle_id": BASELINE,
            "protocol_id": self.protocol["protocol_id"],
            "authorization_ref": cfg["permit_ref"],
            "limits": {
                "model_requests": 1,
                "wall_seconds": 180,
                "storage_bytes": 10485760,
            },
        }
        atomic_json(output / "runtime_context.json", context)
        admitted = {
            "kind": "request_admitted",
            "model": "gpt-6.1-sol",
            "reasoning_effort": "high",
        }
        terminal = {
            "kind": "response_terminal",
            "backend_model": "gpt-6.1-sol",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        cases = [
            ("missing", None, [admitted, terminal], True, "completed"),
            (
                "failed",
                {"status": "failed", "final_response": None},
                [admitted, terminal],
                True,
                "completed",
            ),
            (
                "empty",
                {"status": "completed", "final_response": ""},
                [admitted, terminal],
                True,
                "completed",
            ),
            (
                "unparseable",
                {"status": "completed", "final_response": "{invalid"},
                [admitted, terminal],
                True,
                "completed",
            ),
            (
                "cleanup-unknown",
                {"status": "failed", "final_response": None},
                [admitted, terminal],
                False,
                "waiting",
            ),
            ("request-unknown", None, [admitted], True, "waiting"),
        ]
        for label, sdk, events, cleanup, expected in cases:
            with self.subTest(label=label):
                atomic_json(
                    output / "supervisor.json",
                    {
                        "status": "completed",
                        "exit_code": 0,
                        "cleanup_confirmed": cleanup,
                    },
                )
                costs = {
                    "whole_job_wall_seconds": 1,
                    "model_requests": 1,
                    "tool_calls": 0,
                    "artifact_bytes": 1234,
                }
                atomic_json(output / "execution-costs.json", {"costs": costs})
                (output / "model_requests.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in events)
                )
                sdk_path = output / "sdk-result.json"
                if sdk is None:
                    sdk_path.unlink(missing_ok=True)
                else:
                    atomic_json(sdk_path, sdk)
                operation = self.backend._op(
                    "research", "research", dynamic=True, key="research-" + label
                )
                receipt = self.backend.reconcile(operation, {})
                self.assertEqual(receipt["status"], expected)
                if expected == "completed":
                    self.assertTrue(receipt["stop"])
                    self.assertTrue(receipt["cost_closed"])
                self.assertEqual(self.research_launches, 0)
        self.assertEqual(len(read(self.phase_ref)["research_charges"]), 1)

    def test_closed_zero_admitted_research_startup_is_terminal_without_retry(self):
        cfg = self.config["research"]
        output = Path(cfg["output_ref"])
        output.mkdir(parents=True)
        atomic_json(
            output / "request.json",
            self.store.get("bounded_transports", cfg["request_id"]),
        )
        atomic_json(
            output / "runtime_context.json",
            {
                "job_id": cfg["job_id"],
                "bundle_id": BASELINE,
                "protocol_id": self.protocol["protocol_id"],
                "authorization_ref": cfg["permit_ref"],
                "limits": {
                    "model_requests": 1,
                    "wall_seconds": 180,
                    "storage_bytes": 10485760,
                },
            },
        )
        atomic_json(
            output / "supervisor.json",
            {"status": "interrupted", "exit_code": 1, "cleanup_confirmed": True},
        )
        atomic_json(
            output / "execution-costs.json",
            {
                "costs": {
                    "whole_job_wall_seconds": 1,
                    "model_requests": 0,
                    "tool_calls": 0,
                    "artifact_bytes": 1234,
                }
            },
        )
        (output / "model_requests.jsonl").write_text("")
        operation = self.backend._op("research", "research", dynamic=True)
        receipt = self.backend.reconcile(operation, {})
        self.assertEqual(receipt["status"], "completed")
        self.assertTrue(receipt["stop"])
        self.assertEqual(receipt["request_count"], 0)
        self.assertEqual(
            read(self.phase_ref)["research_charges"][0]["costs"]["model_requests"], 0
        )
        self.assertEqual(self.research_launches, 0)

    def test_concurrent_continue_and_duplicate_receipts_keep_one_runtime_and_charge(
        self,
    ):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        entered, release = Event(), Event()
        execute_boundary = self.runtime_execute

        def blocked_runtime(plan):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("offline Runtime boundary was not released")
            return execute_boundary(plan)

        self.backend.runtime_execute = blocked_runtime
        peer_backend = Protocol10Backend(
            self.scope,
            self.config,
            runtime_execute=self.runtime_execute,
            audit=self.audit,
            progress_inspector=self.progress,
            research_execute=self.research_execute,
            trial_collector=self.collect,
        )
        peer = Coordinator(
            self.scope / "coordination", peer_backend.registration, peer_backend
        )
        admission = self.inputs()
        with ThreadPoolExecutor(max_workers=1) as workers:
            in_flight = workers.submit(self.coordinator.advance, admission)
            try:
                self.assertTrue(entered.wait(timeout=5))
                with self.assertRaisesRegex(
                    RuntimeError, "another protocol10 coordinator is active"
                ):
                    peer.advance(admission)
            finally:
                release.set()
            returned = in_flight.result(timeout=5)
        run_id = self.slots[0]["run_id"]
        runtime_id, settlement_id = run_id + "-runtime", run_id + "-settlement"
        self.assertEqual(self.launches, [run_id])
        self.assertEqual(set(returned["receipts"]), {runtime_id})
        receipt_event = {
            "type": "receipt_available",
            "event_id": "runtime-return-observed",
            "operation_id": runtime_id,
            "reference": returned["receipts"][runtime_id]["receipt_ref"],
        }
        observed = peer.event(receipt_event)
        self.assertEqual(peer.event(receipt_event), observed)
        self.assertEqual(observed["receipts"], returned["receipts"])
        self.assertEqual(len(observed["events"]), 1)
        settled = peer.advance(admission)
        self.assertIn(settlement_id, settled["receipts"])
        phase = read(self.phase_ref)
        self.assertEqual(phase["used"]["simulation_episodes"], 1)
        self.assertEqual(
            [entry["run_id"] for entry in phase["protocol10_trial_charges"]], [run_id]
        )

        restarted_backend = Protocol10Backend(
            self.scope,
            self.config,
            runtime_execute=self.runtime_execute,
            audit=self.audit,
            progress_inspector=self.progress,
            research_execute=self.research_execute,
            trial_collector=self.collect,
        )
        restarted = Coordinator(
            self.scope / "coordination",
            restarted_backend.registration,
            restarted_backend,
        )
        self.assertEqual(restarted.event(receipt_event), settled)
        self.assertEqual(restarted.advance(admission, reconcile_only=True), settled)
        # Native receipt reconciliation is equally idempotent after a process restart.
        for operation_id in (runtime_id, settlement_id):
            receipt = restarted_backend.reconcile(
                settled["operations"][operation_id], admission
            )
            self.assertEqual(receipt, settled["receipts"][operation_id])
        self.assertEqual(restarted.event(receipt_event), settled)
        self.assertEqual(self.launches, [run_id])
        self.assertEqual(read(self.phase_ref), phase)


if __name__ == "__main__":
    unittest.main()
