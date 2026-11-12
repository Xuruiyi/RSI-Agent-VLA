"""Only changed diagnostic/progression/transport/dispatch behavior, CPU in-process fakes."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import test_research_tools
import test_core
from rsi_vla.research_transport import pack, unpack, model_prompt
from rsi_vla.critical_progress import inspect_progress
from rsi_vla.bounded_runner import execute, reconcile, validate_permit
from rsi_vla.research_tools import BoundedResearchAdapter
from rsi_vla.storage import atomic_json


class DiagnosticsTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def opportunity(self, key, source, results):
        return self.tools.discover_opportunity(
            key,
            source=source,
            result_ids=results,
            external_value="diagnose",
            competing_explanations=self.explanations,
            expected_cost={},
        )

    def test_valid_failure_success_technical_unknown_and_no_evidence(self):
        self.store.save_development_result(
            self.result("failed", self.base.bundle_id, outcome="failure")
        )
        self.opportunity("failure", "failure", ["failed"])
        self.opportunity("normal", "normal", ["base-result"])
        self.store.save_development_result(
            replace(self.result("empty-success", self.base.bundle_id), data_refs=[])
        )
        with self.assertRaises(ValueError):
            self.opportunity("empty-success-opportunity", "normal", ["empty-success"])

        technical = replace(
            self.result("technical", self.base.bundle_id, outcome="unknown"),
            invalid_count=1,
            assignments=[
                {
                    "task_key": "salad",
                    "seed": 1,
                    "outcome": "unknown",
                    "validity": "invalid",
                    "evidence_ref": "fault-source.json",
                }
            ],
            notes=["inference failure; native success fact retained separately"],
            data_refs=["fault-source.json"],
        )
        self.store.save_development_result(technical)
        self.assertEqual(
            self.tools.query_development(["technical"])[0]["outcome"], "unknown"
        )
        self.opportunity("diagnostic", "technical_diagnostic", ["technical"])
        with self.assertRaises(ValueError):
            self.opportunity("wrong-failure", "failure", ["technical"])
        with self.assertRaises(ValueError):
            self.opportunity("empty", "technical_diagnostic", [])
        no_ref = replace(
            technical,
            result_id="no-ref",
            assignments=[{**technical.assignments[0], "evidence_ref": None}],
        )
        self.store.save_development_result(no_ref)
        with self.assertRaises(ValueError):
            self.opportunity("unreferenced", "technical_diagnostic", ["no-ref"])
        with self.assertRaisesRegex(ValueError, "does not establish"):
            self.tools.plan_rule_experiment(
                "bad",
                opportunity_id="diagnostic",
                bundle_id="trial",
                chunk_steps=2,
                criterion_ref="criterion",
                distinguishing_prediction="test",
                cost_limit={"offline_calls": 1},
            )
        for key, record in (
            ("release", replace(technical, result_id="release", role="release")),
            (
                "sealed",
                replace(
                    technical,
                    result_id="sealed",
                    assignments=[{**technical.assignments[0], "seed": 5}],
                ),
            ),
        ):
            self.store.put("development_results", key, record)
            with self.assertRaises(ValueError):
                self.opportunity("bad-" + key, "technical_diagnostic", [key])


class ProgressTest(unittest.TestCase):
    def fixture(self, root):
        identity = {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
        }
        data = [
            {"kind": "tool_call", "tool": "vla_execute"},
            {"kind": "vla_segment_start", "instruction": "task", "max_chunks": 24},
            {"kind": "observation_update"},
            {
                "kind": "inference_start",
                "instruction": "task",
                "segment_sequence": 1,
                "source": "vla",
                "observation_sequence": 2,
                "observation_mode": "latest",
            },
            {"kind": "tool_end", "tool": "vla_execute"},
        ]
        events = [{**identity, "sequence": i, **e} for i, e in enumerate(data)]
        (root / "states.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "command": {
                                "action": "vla_execute",
                                "instruction": "task",
                                "max_chunks": 24,
                            }
                        }
                    ]
                }
            )
        )
        source = root / "code_snapshot/RPent/robots/libero/tools.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            (
                Path(__file__).resolve().parents[2] / "RPent/robots/libero/tools.py"
            ).read_text()
        )
        return identity, events

    def test_independent_true_false_unknown_and_actual_inapplicability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity, events = self.fixture(root)

            def check():
                (root / "action_events.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in events)
                )
                return inspect_progress(root, identity)

            self.assertIs(check()["value"], True)
            self.assertIn("holding_or_settled", check()["not_applicable"])
            events[-2]["observation_sequence"] = 0
            self.assertIs(check()["value"], False)
            del events[-2]["observation_sequence"]
            self.assertIsNone(check()["value"])
            events[0]["tool"] = "move_to"
            self.assertIsNone(check()["value"])

    def test_evaluator_and_gate_three_states(self):
        from test_evaluator import EvaluatorTests

        fixture = EvaluatorTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Actual independent progress report is integrated, not a caller hard-constraint boolean.
        for value in (True, False, None):
            with patch(
                "rsi_vla.evaluator.inspect_progress",
                return_value={
                    "value": value,
                    "evidence_refs": ["independent-path"],
                    "missing": [],
                },
            ):
                result = fixture.evaluate(fixture.native())
                self.assertIs(
                    result.hard_constraints["critical_progress"],
                    False if value is False else None,
                )
                self.assertEqual(
                    result.intervention["critical_progress_evidence"][0]["value"], value
                )
        # Complete matching matrix allows True to reach the formal result.
        manifest = []
        for task in fixture.protocol["tasks"]:
            for seed in fixture.protocol["task_sets"]["release"]["seeds"]:
                folder = Path(fixture.temp.name) / (task["task_key"] + str(seed))
                folder.mkdir()
                path = folder / "result.json"
                atomic_json(
                    path,
                    {
                        "run_id": folder.name,
                        "episode_id": folder.name,
                        "bundle_id": "initial-medium",
                        "protocol_id": fixture.protocol["protocol_id"],
                        "task": {
                            "suite": task["suite"],
                            "task": task["task_id"],
                            "seed": seed,
                        },
                        "planner": fixture.profile["planner"],
                        "benchmark_success": True,
                        "terminated": True,
                        "truncated": False,
                        "native_result_source": {"terminated": "native"},
                    },
                )
                manifest.append(
                    {
                        "task_key": task["task_key"],
                        "seed": seed,
                        "run_id": folder.name,
                        "episode_id": folder.name,
                        "result_path": str(path),
                    }
                )
        for value in (True, False, None):
            with patch(
                "rsi_vla.evaluator.inspect_progress",
                return_value={
                    "value": value,
                    "evidence_refs": ["independent-path"],
                    "missing": [],
                },
            ):
                self.assertIs(
                    fixture.evaluate(manifest).hard_constraints["critical_progress"],
                    value,
                )
        f = test_core.CoreTest()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.p["hard_constraints"]["critical_progress"] = "independent progress"
        for value, verdict in (
            (True, "accept"),
            (False, "reject"),
            (None, "inconclusive"),
        ):
            candidate = replace(
                f.result(),
                hard_constraints={
                    "safety": True,
                    "compatibility": True,
                    "critical_progress": value,
                },
            )
            self.assertEqual(f.evaluate(candidate=candidate).verdict, verdict)


class TransportTest(unittest.TestCase):
    def test_exact_all_json_types_and_real_prompt_uses_packed_full_facts(self):
        value = {
            "identity": "a" * 80,
            "l0": [
                {
                    "missing": ["fault"],
                    "unknown": None,
                    "valid": False,
                    "source": "vla",
                    "a": [1, 1.0, True],
                },
                {
                    "missing": ["fault"],
                    "unknown": None,
                    "valid": False,
                    "source": "vla",
                    "a": [1, 1.0, True],
                },
            ],
        }
        encoded = pack(value)
        self.assertEqual(unpack(encoded), value)
        self.assertIn('"r":', model_prompt(encoded))
        self.assertNotIn('"evidence_views":', model_prompt(encoded))


class RunnerTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def prepared(self):
        self.protocol["budget"] = {"token_policy": "observed_only"}
        from rsi_vla.research import ResearchJob

        self.job = ResearchJob(
            self.store,
            "new-job",
            self.base.bundle_id,
            {
                "model_requests": 1,
                "whole_job_wall_seconds": 300,
                "artifact_bytes": 10485760,
            },
        )
        self.tools.job = self.job
        self.tools.domain_policy["rule_operators"] = ["chunk_steps"]
        adapter = BoundedResearchAdapter(model="gpt-6.1-sol", reasoning_effort="medium")
        transport = adapter.prepare_request(
            self.tools,
            request_id="new-request",
            result_ids=["base-result"],
            max_input_tokens=64000,
            max_output_tokens=4000,
        )
        phase = self.root / "phase.json"
        limits = {
            "model_requests": 10,
            "wall_seconds": 1000,
            "storage_bytes": 100000000,
            "tool_calls": 60,
            "environment_steps": 2000,
            "gpu_seconds": 900,
            "simulation_episodes": 1,
        }
        atomic_json(
            phase,
            {"limits": limits, "remaining": limits, "used": {k: 0 for k in limits}},
        )
        manifest = self.root / "sources.json"
        atomic_json(
            manifest, {"version": "batch4-preparation", "files": ["consumer.py"]}
        )
        (self.root / "consumer.py").write_text("pass\n")
        (self.root / "source").mkdir()
        (self.root / "source/consumer.py").write_text("pass\n")
        output = self.root / "dispatch"
        permit = {
            "schema": "rsi-bounded-research-permit-v1",
            "authorized": True,
            "job_id": "new-job",
            "request_id": "new-request",
            "bundle_id": self.base.bundle_id,
            "protocol_id": self.protocol["protocol_id"],
            "output_dir": str(output),
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "input": transport,
            "job_limits": self.job.state["limits"],
            "limits": {
                "model_requests": 1,
                "wall_seconds": 300,
                "storage_bytes": 10485760,
            },
            "phase_snapshot_ref": str(phase),
            "source_manifest_ref": str(manifest),
        }
        permit_path = self.root / "permit.json"
        atomic_json(permit_path, permit)
        return adapter, transport, output, permit, permit_path

    def test_one_dispatch_reconcile_no_repeat_charge_and_no_applicable_no_dispatch(
        self,
    ):
        from contextlib import nullcontext

        adapter, transport, output, permit, path = self.prepared()
        with self.assertRaises(PermissionError):
            execute(
                self.tools,
                adapter,
                transport,
                root=self.root,
                output=output,
                permit_path=path,
                preflight=[],
                cost_limit={},
            )
        self.assertFalse(output.exists())

        class Gate:
            def __init__(self, context):
                self.url = "in-process-double"
                self.context = context

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def supervisor(command, context_path):
            response = {
                "schema": "rsi-bounded-research-response-v2",
                "request_id": "new-request",
                "decision": "defer",
                "source": "normal",
                "result_ids": ["base-result"],
                "reason": "missing matched data",
                "external_value": "cost",
                "competing_explanations": self.explanations,
                "intervention": None,
                "distinguishing_prediction": "matched comparison",
                "stop_condition": "one request",
                "missing_evidence": ["matching comparison"],
            }
            atomic_json(
                output / "sdk-result.json",
                {"status": "completed", "final_response": json.dumps(response)},
            )
            events = [
                {
                    "kind": "request_admitted",
                    "model": "gpt-6.1-sol",
                    "reasoning_effort": "medium",
                },
                {
                    "kind": "response_terminal",
                    "backend_model": "gpt-6.1-sol",
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                },
            ]
            (output / "model_requests.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )
            return {"status": "completed", "exit_code": 0, "cleanup_confirmed": True}

        result = execute(
            self.tools,
            adapter,
            transport,
            root=self.root,
            output=output,
            permit_path=path,
            preflight=[{"currently_trialable": True}],
            cost_limit={},
            gate_factory=Gate,
            supervisor=supervisor,
            lock_factory=nullcontext,
        )
        self.assertEqual(result["action"], "defer")
        self.assertEqual(len(self.job.state["charges"]), 1)
        saved = json.loads((output / "sdk-result.json").read_text())
        atomic_json(
            output / "sdk-result.json", {**saved, "final_response": "{malformed"}
        )
        with self.assertRaises(ValueError):
            reconcile(self.tools, adapter, output, cost_limit={})
        self.assertEqual(len(self.job.state["charges"]), 1)
        atomic_json(output / "sdk-result.json", saved)
        reconcile(self.tools, adapter, output, cost_limit={})
        self.assertEqual(len(self.job.state["charges"]), 1)
        self.assertEqual(
            len(json.loads((self.root / "phase.json").read_text())["research_charges"]),
            1,
        )
        with self.assertRaises(PermissionError):
            execute(
                self.tools,
                adapter,
                transport,
                root=self.root,
                output=output,
                permit_path=path,
                preflight=[{"currently_trialable": True}],
                cost_limit={},
            )

    def test_old_mismatched_permit_and_source_refused(self):
        adapter, transport, output, permit, path = self.prepared()
        for field, value in (
            ("schema", "old"),
            ("job_id", "old"),
            ("input", {}),
            ("authorized", False),
        ):
            bad = {**permit, field: value}
            with self.assertRaises(PermissionError):
                validate_permit(
                    self.tools, transport, bad, root=self.root, output=output
                )
        altered = unpack(transport)
        altered["job_id"] = "different-job"
        with self.assertRaises(PermissionError):
            validate_permit(
                self.tools,
                pack(altered),
                {**permit, "input": pack(altered)},
                root=self.root,
                output=output,
            )
        (self.root / "consumer.py").write_text("changed\n")
        with self.assertRaises(PermissionError):
            validate_permit(
                self.tools, transport, permit, root=self.root, output=output
            )
