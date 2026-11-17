# Copyright 2026 RSI-Agent-VLA Authors.
"""The actual Host -> isolated Research -> trusted ψ broker -> successor path."""

import os
import unittest
from pathlib import Path

import test_host_receipt_inputs as host_fixture
from rsi_vla.contracts import RunBundle
from rsi_vla.midterm_host import read
from rsi_vla.research import ResearchJob
from rsi_vla.storage import ResearchStore, atomic_json, initialize


def isolation_probe(config):
    """Check permissions inside the production Research namespace."""
    root = Path(config["isolation_task_root"])
    for path in (
        root / "operator/forbidden",
        root / "scratch/candidate/forbidden",
        root / "release/forbidden",
    ):
        try:
            path.write_text("forbidden")
        except OSError:
            pass
        else:
            raise AssertionError(str(path))
    for path in (root / "hidden/labels.json", root / "scratch/evaluation/secret"):
        try:
            value = path.read_text()
        except OSError:
            continue
        assert "SYNTHETIC-HIDDEN" not in value
    atomic_json(
        root / "state/research/isolation-proof.json",
        {"pid": os.getpid(), "denied": True},
    )
    return {}


class InheritedRuleHostTests(unittest.TestCase):
    setUp = host_fixture.HostReceiptInputsTests.setUp
    host = host_fixture.HostReceiptInputsTests.host
    end_job = host_fixture.HostReceiptInputsTests.end_job

    def configure(self, source='def select_action(context):\n return "defer"\n'):
        store = ResearchStore(self.root / "state/research/store")
        program = store.root / "programs/psi.py"
        program.parent.mkdir(parents=True)
        program.write_text(source)
        bundle = RunBundle(
            "synthetic-v1",
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
            store,
            bundle,
            psi={
                "research_rule_plugin": {
                    "schema": "research-rule-select-action-v1",
                    "source_ref": str(program),
                    "source": source,
                }
            },
            memory={"provenance": "synthetic", "legal_information": "development"},
        )
        self.store = store
        ResearchJob(
            store,
            "job",
            "synthetic-v1",
            {"whole_job_wall_seconds": 30, "artifact_bytes": 2_000_000},
        )
        protocol_ref = self.operator / "protocol.json"
        atomic_json(
            protocol_ref,
            {
                "protocol_id": "protocol",
                "scope_id": "f03-offline",
                "scope": "offline",
                "status": "frozen",
                "benchmark_id": "synthetic",
                "anchor_bundle_id": "synthetic-v1",
                "task_policy": {"schema": "rsi-task-policy-v1"},
                "metrics": {
                    "success": {"direction": "max", "min": 0, "max": 1, "weight": 1}
                },
                "hard_constraints": ["resource_limits"],
                "min_gain": 0,
                "anchor_max_drop": 0,
            },
        )
        domain = self.operator / "domain.json"
        atomic_json(
            domain,
            {
                "domain": "synthetic",
                "allowed_goal_kinds": ["manipulation"],
                "policy_ref": "synthetic-domain",
            },
        )
        self.operations[0].update(
            kind="research_prepare", payload={"request_id": "request"}
        )
        self.core.update(
            research_store_ref=str(store.root),
            evaluation_store_ref=str(self.root / "state/evaluation/store"),
            repository_root=str(host_fixture.ROOT),
            offline_fixture=True,
            role_isolation={"hidden_paths": [str(self.root / "hidden")]},
            research_requests=[
                {
                    "request_id": "request",
                    "job_id": "job",
                    "base_bundle_id": "synthetic-v1",
                    "domain_policy_ref": str(domain),
                    "result_ids": [],
                }
            ],
        )
        self.config["operations"]["produce"].pop("command")
        self.config["operations"]["produce"].pop("exports")
        self.config["operations"]["consume"].pop("late_refs")
        command = self.config["operations"]["consume"]["command"]
        command[-1] = command[-1].replace(
            "assert json.loads(Path(i['references']['theta']).read_text())['theta']=='synthetic-test-only'; ",
            "",
        )
        self.core["continuation"]["nodes"]["produce"]["transitions"] = [
            {
                "when": {"path": ["action"], "equals": "inspect_evidence"},
                "next": "consume",
            },
            {"next": "consume"},
        ]
        for name in ("hidden", "scratch/evaluation", "scratch/candidate"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "hidden/labels.json").write_text("SYNTHETIC-HIDDEN")
        (self.root / "scratch/evaluation/secret").write_text("SYNTHETIC-HIDDEN")
        atomic_json(self.core_ref, self.core)
        atomic_json(self.host_ref, self.config)

    def run_chain(self):
        host = self.host()
        first = host.advance()
        self.assertEqual(first["status"], "running", first)
        self.assertEqual(self.host().advance()["status"], "running")
        _, completion = self.end_job("produce")
        role = list(
            (self.root / "state/research/role_receipts").glob("produce-execute-*.json")
        )
        self.assertEqual(len(role), 1)
        receipt = read(role[0])
        self.assertNotIn("error", receipt, receipt)
        self.assertTrue(receipt["value"]["policy_decision"]["technical_valid"])
        self.assertEqual(receipt["value"]["request_count"], 0)
        candidate = (
            self.root / "operator/code_candidate_receipts/request-psi-selection.json"
        )
        self.assertTrue(candidate.exists())
        self.assertGreater(read(candidate)["wall_seconds"], 0)
        used = self.store.get("jobs", "job")["used"].copy()
        self.assertGreater(used["whole_job_wall_seconds"], 0)
        self.assertEqual(self.host().advance()["status"], "running")
        self.end_job("consume")
        self.assertEqual(self.host().advance()["status"], "completed")
        self.assertEqual(self.counter.read_text(), "consume\n")
        self.assertEqual(self.host().advance()["status"], "completed")
        self.assertEqual(self.store.get("jobs", "job")["used"], used)
        self.assertEqual(
            len(
                list(
                    (self.root / "operator/code_candidate_receipts").glob(
                        "*.started.json"
                    )
                )
            ),
            1,
        )
        return receipt, completion

    def test_real_host_isolated_rule_receipt_cost_successor_restart_and_duplicates(
        self,
    ):
        self.configure()
        receipt, _ = self.run_chain()
        self.assertEqual(receipt["value"]["action"], "defer")
        self.assertEqual(receipt["role"], "research")
        self.assertFalse((self.root / "release/forbidden").exists())

    def test_evidence_selection_uses_same_broker_and_legal_successor(self):
        self.configure('def select_action(context):\n return "inspect_evidence"\n')
        receipt, _ = self.run_chain()
        self.assertEqual(receipt["value"]["next_action"], "prepare_evidence_followup")

    def test_unknown_candidate_launch_blocks_replay_without_new_native_attempt(self):
        self.configure()
        self.host().advance()
        self.end_job("produce")
        receipt = (
            self.root / "operator/code_candidate_receipts/request-psi-selection.json"
        )
        original = receipt.read_bytes()
        receipt.unlink()
        # Drop only the outer return after the original candidate side effect.
        for path in (self.root / "state/research/role_receipts").glob(
            "produce-execute-*.json"
        ):
            path.unlink()
        self.store.path("research_rule_consumptions", "request-psi-selection").unlink()
        from rsi_vla.protocol10_roles import run_role

        original_request = read(
            next((self.root / "operator/role_requests").glob("produce-execute-*.json"))
        )
        result = run_role(self.core, original_request["task"])
        self.assertIn("reconcile before resubmission", result["error"])
        self.assertFalse(receipt.exists())
        self.assertEqual(len(list(receipt.parent.glob("*.started.json"))), 1)
        receipt.write_bytes(original)

    def test_frozen_job_refuses_a_new_isolated_selection(self):
        self.configure()
        state = self.store.get("jobs", "job")
        atomic_json(self.store.path("jobs", "job"), {**state, "status": "stopped"})
        self.host().advance()
        self.end_job("produce", expected="unknown")
        self.assertFalse(
            (
                self.root
                / "operator/code_candidate_receipts/request-psi-selection.started.json"
            ).exists()
        )

    def test_wall_timeout_preserves_original_receipt_and_cost(self):
        self.configure("def select_action(context):\n while True:\n  pass\n")
        self.core["research_requests"][0]["research_rule_timeout_seconds"] = 0.05
        atomic_json(self.core_ref, self.core)
        self.host().advance()
        self.end_job("produce")
        receipt = read(
            self.root / "operator/code_candidate_receipts/request-psi-selection.json"
        )
        self.assertEqual(
            receipt["error"], "code candidate exceeded its registered timeout"
        )
        self.assertGreater(receipt["wall_seconds"], 0)
        role = read(
            next(
                (self.root / "state/research/role_receipts").glob(
                    "produce-execute-*.json"
                )
            )
        )
        self.assertEqual(role["value"]["action"], "defer")
        self.assertFalse(role["value"]["policy_decision"]["technical_valid"])
        self.host().advance()
        self.end_job("consume")
        self.assertEqual(self.host().advance()["status"], "completed")

    def test_nonterminating_rule_records_failure_cost_and_legal_defer(self):
        self.configure("def select_action(context):\n while True:\n  pass\n")
        self.host().advance()
        self.end_job("produce")
        role = read(
            next(
                (self.root / "state/research/role_receipts").glob(
                    "produce-execute-*.json"
                )
            )
        )
        self.assertEqual(role["value"]["action"], "defer")
        self.assertFalse(role["value"]["policy_decision"]["technical_valid"])
        self.assertGreater(
            self.store.get("jobs", "job")["used"]["whole_job_wall_seconds"], 0
        )
        self.host().advance()
        self.end_job("consume")
        self.assertEqual(self.host().advance()["status"], "completed")


class InheritedRuleAutoloopTests(unittest.TestCase):
    """One operator event; actual timers and native receipts drive every successor."""

    setUp = host_fixture.HostReceiptInputsTests.setUp
    configure = InheritedRuleHostTests.configure
    isolated_rule_producer = True

    def _owned_processes(self):
        from test_host_autoloop_receipts import HostAutoloopReceiptTests

        return HostAutoloopReceiptTests._owned_processes(self)

    def test_initial_event_timer_host_restart_and_isolated_rule_finish(self):
        from test_host_autoloop_receipts import HostAutoloopReceiptTests

        self.configure("def select_action(context):\n while True:\n  pass\n")
        HostAutoloopReceiptTests.test_real_hub_timer_restart_finishes_two_cpu_receipts_without_manual_advance(
            self
        )
        role = read(
            next(
                (self.root / "state/research/role_receipts").glob(
                    "produce-execute-*.json"
                )
            )
        )
        self.assertFalse(role["value"]["policy_decision"]["technical_valid"])
        self.assertEqual(role["value"]["action"], "defer")
        self.assertGreater(
            self.store.get("jobs", "job")["used"]["whole_job_wall_seconds"], 0
        )
        self.assertEqual(
            len(
                list(
                    (self.root / "operator/code_candidate_receipts").glob(
                        "*.started.json"
                    )
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
