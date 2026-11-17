"""A real controlled Host chain materializes an unlisted task and binds RPent."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_host_receipt_inputs as host_fixture
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import Gate
from rsi_vla.evaluator import evaluate_matrix
from rsi_vla.midterm_host import build_configuration, read
from rsi_vla.storage import ResearchStore, atomic_json
from test_trusted_goal import goal_fixture

ROOT = Path(__file__).resolve().parents[2]


class GoalHostTests(unittest.TestCase):
    host = host_fixture.HostReceiptInputsTests.host
    end_job = host_fixture.HostReceiptInputsTests.end_job

    def setUp(self):
        make_temp = tempfile.TemporaryDirectory
        with patch(
            "test_host_receipt_inputs.tempfile.TemporaryDirectory",
            lambda **kwargs: make_temp(dir="/var/tmp", **kwargs),
        ):
            host_fixture.HostReceiptInputsTests.setUp(self)
        protocol, opportunity, policy = goal_fixture(self.operator)
        protocol["hidden_exam_labels"] = "F12-PRIVATE-EXAM-LABEL"
        protocol.update(
            scope_id=self.scope["scope_id"],
            anchor_bundle_id="base",
            metrics={
                "success_base": {"direction": "max", "min": 0, "max": 1, "weight": 1}
            },
            hard_constraints=["safety"],
            min_gain=0.1,
            anchor_max_drop=0,
            task_policy={"schema": "rsi-task-policy-v1"},
        )
        atomic_json(self.core["protocol_ref"], protocol)
        self.core.update(
            repository_root=str(ROOT),
            offline_fixture=True,
            research_store_ref=str(self.root / "state/research"),
            evaluation_store_ref=str(self.root / "state/evaluation"),
            frozen_protocol_ref=str(self.operator / "frozen.json"),
            administrative_authorization_ref=str(self.operator / "admin.json"),
        )
        atomic_json(self.core["frozen_protocol_ref"], protocol)
        atomic_json(
            self.core["administrative_authorization_ref"],
            {
                "authorized": True,
                "scope": self.scope["scope_id"],
                "protocol_id": protocol["protocol_id"],
                "expires_at": self.scope["expires_at"],
                "phase_snapshot_ref": self.core["phase_ref"],
                "approval_reference": self.core["scope_authorization_ref"],
            },
        )
        ResearchStore(self.core["research_store_ref"]).put(
            "opportunities", opportunity["opportunity_id"], opportunity
        )
        ResearchStore(self.core["research_store_ref"]).put(
            "bundles",
            "synthetic-v1",
            RunBundle(
                "synthetic-v1",
                "theta",
                "processor",
                "phi",
                "rho",
                "psi",
                "memory",
                protocol["protocol_id"],
                "backend",
                "code",
            ),
        )
        materialization = self.root / "state/evaluation/goals"
        task_binding = self.root / "runtime/task-inputs"
        self.operations[0].update(
            kind="goal_materialization",
            payload={
                "policy_ref": policy["policy_ref"],
                "opportunity_id": opportunity["opportunity_id"],
                "output_root": str(materialization),
                "new_protocol_id": "goal-v2",
            },
            version="synthetic-v1",
            exports={
                "goal": {
                    "field": "materialization_ref",
                    "version": "synthetic-v1",
                    "type": "path",
                }
            },
            storage_write_roots=[str(materialization)],
        )
        self.operations[1].update(
            kind="materialized_task_prepare",
            payload={"output_root": str(task_binding), "seed": 0},
            late_refs={
                "goal": {
                    "operation_id": "produce",
                    "export": "goal",
                    "version": "synthetic-v1",
                    "target": ["payload", "materialization_ref"],
                }
            },
            storage_write_roots=[str(task_binding)],
        )
        atomic_json(self.core_ref, self.core)
        self.config = build_configuration(
            self.core_ref, self.host_ref, self.operator / "hub.json"
        )
        self.config["inputs_ref"] = str(self.operator / "admission.json")
        for name in ("produce", "consume"):
            self.config["operations"][name].update(
                budget={
                    "wall_seconds": 8,
                    "storage_bytes": 2000000,
                    "current_disk_bytes": 2000000,
                    "model_requests": 0,
                    "gpu_seconds": 0,
                },
                supervisor_limits={"wall_seconds": 8, "storage_bytes": 2000000},
            )
        atomic_json(self.host_ref, self.config)
        self.protocol = protocol

    def test_unlisted_goal_host_to_rpent_binding_to_independent_result(self):
        self.assertEqual(self.host().advance()["status"], "running")
        host, _completion = self.end_job("produce")
        native = read(Path(self.config["continuation_root"]) / "receipts/produce.json")[
            "value"
        ]
        receipt = read(native["materialization_ref"])
        self.assertEqual(receipt["task"]["task_key"], "joint-packing")
        self.assertEqual(self.host().advance()["status"], "running")
        host, _ = self.end_job("consume")
        self.assertEqual(host.advance()["status"], "completed")
        prepared = read(
            Path(self.config["continuation_root"]) / "receipts/consume.json"
        )["value"]
        binding = prepared["task_binding"]
        self.assertEqual(binding["materialization_ref"], native["materialization_ref"])
        self.assertEqual(prepared["environment_initialization"], "unmeasured")
        self.assertIn("--materialized-task", prepared["runtime_arguments"])
        self.assertEqual(host.ledger.read()["used"]["model_requests"], 0)
        # Independently consume isolated synthetic native evidence. Neither this
        # test nor Host initializes an Env or claims physical task success.
        self.assertNotIn("protocol", receipt)
        self.assertNotIn("F12-PRIVATE-EXAM-LABEL", json.dumps(receipt))
        self.assertNotIn("F12-PRIVATE-EXAM-LABEL", json.dumps(prepared))
        p = read(native["protocol_ref"])
        self.assertEqual(p["hidden_exam_labels"], "F12-PRIVATE-EXAM-LABEL")
        profile = {
            "planner": {
                "backend": "fixture",
                "model": "none",
                "reasoning_effort": "none",
            }
        }
        manifest = []
        for task in p["tasks"]:
            root = self.root / "synthetic-native" / task["task_key"]
            root.mkdir(parents=True)
            desc = {"suite": task["suite"], "task": task["task_id"], "seed": 0}
            if task["suite"] == "materialized":
                desc = binding
            result = {
                "run_id": task["task_key"],
                "episode_id": task["task_key"],
                "bundle_id": "changed",
                "protocol_id": p["protocol_id"],
                "task": desc,
                "planner": profile["planner"],
                "benchmark_success": True,
                "terminated": True,
                "truncated": False,
                "native_result_source": {"terminated": "synthetic native fixture"},
            }
            path = root / "result.json"
            atomic_json(path, result)
            manifest.append(
                {
                    "task_key": task["task_key"],
                    "seed": 0,
                    "run_id": task["task_key"],
                    "episode_id": task["task_key"],
                    "result_path": str(path),
                }
            )
        independent = evaluate_matrix(
            p,
            profile,
            manifest,
            result_id="independent",
            bundle_id="changed",
            candidate_id="candidate",
        )
        self.assertEqual(independent.metrics["success_new"], 1)
        self.assertEqual(independent.unknown_count, 0)
        self.assertIsNone(independent.hard_constraints["safety"])
        candidate = Candidate(
            "candidate",
            "base",
            "changed",
            p["protocol_id"],
            p["benchmark_id"],
            {},
            [],
            [],
            {},
            {},
            "offline",
        )
        decision = Gate(p).evaluate(
            "goal-gate",
            candidate,
            current_bundle_id="base",
            candidate_result=independent,
        )
        self.assertEqual(decision.verdict, "inconclusive")
        self.assertFalse(decision.release_eligible)
        # Same files under a different materialization identity cannot fill a slot.
        path = Path(manifest[-1]["result_path"])
        tampered = read(path)
        tampered["task"]["materialization_ref"] = "other"
        atomic_json(path, tampered)
        invalid = evaluate_matrix(
            p,
            profile,
            manifest,
            result_id="invalid",
            bundle_id="changed",
            candidate_id="candidate",
        )
        self.assertEqual(invalid.invalid_count, 1)


if __name__ == "__main__":
    unittest.main()
