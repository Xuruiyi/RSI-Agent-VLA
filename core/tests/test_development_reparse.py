"""Versioned evaluation consumes original facts without rerunning or overwriting them."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_execution_arm_acceptance as arms
from rsi_vla.adapters.runtime_acceptance import inspect_attempt
from rsi_vla.adapters.runtime_closure import finalize_runtime
from rsi_vla.contracts import RunBundle
from rsi_vla.development import import_development
from rsi_vla.development_reparse import (
    SOURCES,
    reparse_development_attempt,
    remaining_reparse_plan,
    resolve_development_evaluation,
    configure_reparse_research,
)
from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.task_completion import ACTION_ROLES, TaskCompletionBackend


class DevelopmentEvaluationRevisionTests(unittest.TestCase):
    fixture = arms.ExecutionArmAcceptanceTests.fixture

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.context, _ = self.fixture(
            self.root / "runtime", outcome="truncation"
        )
        raw = json.loads((self.run / "result.json").read_text())
        self.protocol = {
            "protocol_id": self.context["protocol_id"],
            "scope": "offline",
            "benchmark_id": "CPU-fixture",
            "tasks": [
                {
                    "task_key": "slot",
                    "suite": self.context["task"]["suite"],
                    "task_id": self.context["task"]["task"],
                    "group": "old",
                }
            ],
            "task_sets": {
                "development": {
                    "task_keys": ["slot"],
                    "seeds": [self.context["task"]["seed"]],
                }
            },
        }
        atomic_json(
            self.run / "transcript_fixture.json",
            {
                **self.context["task"],
                "stats": raw.get("planner_diagnostics", {}),
                "runtime": {"identity": raw["runtime"]["identity"]},
            },
        )
        finalize_runtime(self.context)
        self.evaluation = self.root / "evaluation"
        self.old_audit = self.evaluation / "original-audit.json"
        atomic_json(self.old_audit, inspect_attempt(self.run))
        self.assertTrue(json.loads(self.old_audit.read_text())["passed"])
        protocol_ref = self.root / "protocol.json"
        atomic_json(protocol_ref, self.protocol)
        repository = Path(__file__).resolve().parents[2]
        source_ref = self.root / "reviewed/manifest.json"
        for name in SOURCES:
            target = source_ref.parent / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, target)
        atomic_json(
            source_ref, {"version": "5a-assets-offline", "files": list(SOURCES)}
        )
        self.config = {
            "repository_root": str(repository),
            "source_manifest_ref": str(source_ref),
            "protocol_ref": str(protocol_ref),
            "frozen_protocol_ref": str(protocol_ref),
            "evaluation_store_ref": str(self.evaluation),
            "runtime_slots": [
                {"run_id": self.context["run_id"], "output_ref": str(self.run)}
            ],
        }
        self.revision_ref = (
            self.evaluation
            / "development_revisions"
            / self.context["run_id"]
            / "v2/revision.json"
        )
        self.payload = {
            "output_ref": str(self.run),
            "revision_id": "v2",
            "revision_ref": str(self.revision_ref),
            "previous_audit_ref": str(self.old_audit),
        }
        self.before = {
            name: (self.run / name).read_bytes()
            for name in (
                "result.json",
                "independent_result.json",
                "cost_breakdown.json",
            )
        }

    def test_evaluation_override_audit_and_import_preserve_original_bytes_and_costs(
        self,
    ):
        with patch.dict(os.environ, RSI_ISOLATED_ROLE="evaluation"):
            value = reparse_development_attempt(self.config, self.payload)
            self.assertEqual(
                reparse_development_attempt(self.config, self.payload), value
            )
        self.assertTrue(value["passed"])
        self.assertEqual(value["outcome"], "failure")
        revised = inspect_attempt(
            self.run,
            evaluation_ref=self.revision_ref,
            evaluation_store_ref=self.evaluation,
        )
        self.assertTrue(revised["passed"])
        self.assertEqual(
            revised["costs"], json.loads(self.old_audit.read_text())["costs"]
        )
        store = ResearchStore(self.root / "research")
        store.put(
            "bundles",
            self.context["bundle_id"],
            RunBundle(
                self.context["bundle_id"],
                "theta",
                "processor",
                "phi",
                "rho",
                "psi",
                "memory",
                self.protocol["protocol_id"],
                "backend",
                "code",
            ),
        )
        result = import_development(
            store,
            self.protocol,
            self.run,
            result_id="revised-result",
            evaluation_ref=self.revision_ref,
            evaluation_store_ref=self.evaluation,
        )
        self.assertEqual(result.outcome, "failure")
        self.assertEqual(result.assignments[0]["outcome_source"], "native_benchmark")
        self.assertEqual(
            result.assignments[0]["evidence_ref"],
            str(self.revision_ref.parent / "independent_result.json"),
        )
        self.assertIn(str(self.run / "independent_result.json"), result.data_refs)
        self.assertEqual(
            {name: (self.run / name).read_bytes() for name in self.before}, self.before
        )

    def test_verified_budget_failure_keeps_its_actual_outcome_source(self):
        run, context, _ = self.fixture(self.root / "budget-runtime", outcome="budget")
        raw = json.loads((run / "result.json").read_text())
        atomic_json(
            run / "transcript_fixture.json",
            {
                **context["task"],
                "stats": raw.get("planner_diagnostics", {}),
                "runtime": {"identity": raw["runtime"]["identity"]},
            },
        )
        finalize_runtime(context)
        audit_ref = self.root / "budget-audit.json"
        atomic_json(audit_ref, inspect_attempt(run))
        store = ResearchStore(self.root / "budget-research")
        store.put(
            "bundles",
            context["bundle_id"],
            RunBundle(
                context["bundle_id"],
                "theta",
                "processor",
                "phi",
                "rho",
                "psi",
                "memory",
                self.protocol["protocol_id"],
                "backend",
                "code",
            ),
        )
        result = import_development(
            store, self.protocol, run, result_id="budget-result", audit_path=audit_ref
        )
        self.assertEqual(result.outcome, "failure")
        self.assertEqual(
            result.assignments[0]["outcome_source"], "verified_budget_exhaustion"
        )

    def test_untrusted_store_or_changed_protocol_cannot_supply_revised_result(self):
        with patch.dict(os.environ, RSI_ISOLATED_ROLE="evaluation"):
            reparse_development_attempt(self.config, self.payload)
        with self.assertRaises(PermissionError):
            resolve_development_evaluation(
                self.revision_ref, self.run, evaluation_store_ref=self.root / "other"
            )
        with self.assertRaises(ValueError):
            resolve_development_evaluation(
                self.revision_ref,
                self.run,
                evaluation_store_ref=self.evaluation,
                protocol={**self.protocol, "min_gain": 0},
            )
        changed = json.loads((self.run / "result.json").read_text())
        changed["task"]["seed"] = 99
        atomic_json(self.run / "result.json", changed)
        with self.assertRaisesRegex(ValueError, "original Runtime facts"):
            resolve_development_evaluation(
                self.revision_ref, self.run, evaluation_store_ref=self.evaluation
            )

    def test_registered_task_consumer_runs_reparse_only_in_evaluation_role(self):
        operation = {
            "operation_id": "reparse-v2",
            "kind": "development_reparse",
            "payload": self.payload,
        }
        backend = TaskCompletionBackend.__new__(TaskCompletionBackend)
        backend.config = {**self.config, "operations": [operation]}
        authority_calls = []
        backend.authority = lambda: authority_calls.append(True)
        self.assertEqual(ACTION_ROLES["development_reparse"], "evaluation")
        with patch.dict(os.environ, RSI_ISOLATED_ROLE="research"):
            with self.assertRaises(PermissionError):
                backend.execute(operation, {})
        self.assertFalse(self.revision_ref.exists())
        with patch.dict(os.environ, RSI_ISOLATED_ROLE="evaluation"):
            result = backend.execute(operation, {})
        self.assertTrue(result["passed"])
        self.assertEqual(authority_calls, [True, True])


class RemainingReparsePlanTests(unittest.TestCase):
    def test_only_unstarted_runtime_tail_survives_and_no_progress_is_rewritten(self):
        operations, nodes = [], {}
        slots = []
        receipts = {}
        for index in range(8):
            run = f"run{index}"
            slots.append({"run_id": run, "output_ref": "/runtime/" + run})
            for kind, suffix in (
                ("runtime", "runtime"),
                ("development_base_collect", "collect"),
                ("host_command", "retention"),
                ("development_base_import", "import"),
            ):
                identity = run + "-" + suffix
                operations.append(
                    {
                        "operation_id": identity,
                        "kind": kind,
                        "dynamic": kind == "runtime",
                        "stage": "development",
                        "payload": {"run_id": run}
                        if kind == "runtime"
                        else {
                            "output_ref": "/runtime/" + run,
                            "audit_ref": "/evaluation/" + run,
                        },
                    }
                )
                next_node = (
                    run
                    + "-"
                    + {
                        "runtime": "collect",
                        "collect": "retention",
                        "retention": "import",
                    }.get(suffix, "unused")
                )
                if suffix == "import":
                    next_node = f"run{index + 1}-runtime" if index < 7 else "done"
                nodes[identity] = (
                    {
                        "operation_id": identity,
                        "transitions": [
                            {
                                "when": {"path": ["passed"], "equals": True},
                                "next": next_node,
                            },
                            {"next": "stopped"},
                        ],
                    }
                    if suffix == "collect"
                    else {
                        "operation_id": identity,
                        "transitions": [{"next": next_node}],
                    }
                )
                if index < 3 or index == 3 and suffix in {"runtime", "collect"}:
                    receipts[identity] = {
                        "value": {
                            "passed": False
                            if index == 3 and suffix == "collect"
                            else True
                        }
                    }
        nodes.update(done={"terminal": "done"}, stopped={"terminal": "stopped"})
        config = {
            "operations": operations,
            "runtime_slots": slots,
            "evaluation_store_ref": "/evaluation",
            "continuation": {"start": "run0-runtime", "nodes": nodes},
        }
        progress = {"status": "completed", "paused": False, "receipts": receipts}
        before = json.dumps(progress, sort_keys=True)
        remaining = remaining_reparse_plan(config, progress, "run3", "v2")
        self.assertEqual(
            sum(op["kind"] == "runtime" for op in remaining["operations"]), 4
        )
        self.assertEqual(remaining["runtime_slots"], config["runtime_slots"])
        self.assertFalse(
            {op["operation_id"] for op in remaining["operations"]}.intersection(
                receipts
            )
        )
        self.assertNotIn("run3-retention", remaining["continuation"]["nodes"])
        self.assertEqual(
            remaining["continuation"]["nodes"]["run3-collect"]["transitions"][0][
                "next"
            ],
            "run3-import",
        )
        self.assertEqual(json.dumps(progress, sort_keys=True), before)


class ReparseResearchConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.domain = self.root / "original-domain.json"
        self.policy = {
            "purpose": "original task",
            "rule_operators": ["planner_task_hint"],
            "evidence_selection": {
                "kinds": ["task_result"],
                "first_matches": 6,
                "last_events": 2,
            },
        }
        atomic_json(self.domain, self.policy)
        protocol = self.root / "protocol.json"
        atomic_json(
            protocol,
            {
                "task_sets": {
                    "development": {
                        "task_keys": ["old_drawer", "salad"],
                        "seeds": [1, 2],
                    }
                }
            },
        )
        self.config = {
            "protocol_ref": str(protocol),
            "runtime_slots": [{"task_key": "old_drawer", "seed": 2}],
            "research_requests": [
                {
                    "request_id": f"request{index}",
                    "result_ids": ["drawer-s1", "drawer-s2", "salad-s1"],
                    "domain_policy_ref": str(self.domain),
                    "development_slots": [{"task_key": "salad", "seed": 2}],
                }
                for index in range(2)
            ],
        }
        self.stage = self.root / "new-stage"

    def test_drawer_selection_and_projection_are_applied_before_any_domain_write(self):
        original = self.domain.read_bytes()
        domains = configure_reparse_research(
            self.config,
            self.stage,
            result_ids=["drawer-s1", "drawer-s2"],
            development_task="old_drawer",
            media_refs="selected_events",
            sdk_projection="terminal_history_v1",
        )
        self.assertEqual(len(domains), 1)
        for request in self.config["research_requests"]:
            self.assertEqual(request["result_ids"], ["drawer-s1", "drawer-s2"])
            self.assertEqual(
                request["development_slots"], [{"task_key": "old_drawer", "seed": 2}]
            )
            self.assertTrue(
                Path(request["domain_policy_ref"]).is_relative_to(self.stage)
            )
            policy = domains[request["domain_policy_ref"]]
            self.assertEqual(
                policy["evidence_selection"]["sdk_projection"], "terminal_history_v1"
            )
            self.assertEqual(
                policy["evidence_selection"]["media_refs"], "selected_events"
            )
            self.assertIn("old_drawer", policy["purpose"])
        self.assertEqual(self.domain.read_bytes(), original)
        self.assertFalse(self.stage.exists())

    def test_invalid_result_or_task_or_projection_does_not_change_registration(self):
        original = json.dumps(self.config, sort_keys=True)
        for options in (
            {"result_ids": ["outside-result"]},
            {"development_task": "unregistered"},
            {"media_refs": "all"},
            {"sdk_projection": "unknown"},
        ):
            with self.assertRaises((PermissionError, ValueError)):
                configure_reparse_research(self.config, self.stage, **options)
            self.assertEqual(json.dumps(self.config, sort_keys=True), original)
            self.assertFalse(self.stage.exists())

    def test_defaults_copy_domain_without_changing_input_or_policy_values(self):
        domains = configure_reparse_research(self.config, self.stage)
        self.assertEqual(list(domains.values()), [self.policy])
        for request in self.config["research_requests"]:
            self.assertEqual(
                request["result_ids"], ["drawer-s1", "drawer-s2", "salad-s1"]
            )
            self.assertEqual(
                request["development_slots"], [{"task_key": "salad", "seed": 2}]
            )
        self.assertEqual(json.loads(self.domain.read_text()), self.policy)


if __name__ == "__main__":
    unittest.main()
