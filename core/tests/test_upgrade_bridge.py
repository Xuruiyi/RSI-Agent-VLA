"""Frozen mapping and release consumers, including actual Linux mount boundaries."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rsi_vla.contracts import Candidate, ExperimentResult, RunBundle
from rsi_vla.evaluation import EvaluationStore, Gate
from rsi_vla.release import ReleaseManager
from rsi_vla.runtime import execution_binding
from rsi_vla.storage import ResearchStore, atomic_json, initialize
from rsi_vla.upgrade_bridge import (
    BEHAVIOR_FIELDS,
    import_candidate_evidence,
    isolation_paths,
    prepare_candidate_deployment,
    qualify_upgrade,
)

ROOT = Path(__file__).resolve().parents[2]


class UpgradeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="upgrade-bridge-", dir=ROOT / "RPent"
        )
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / "task"
        self.domains = {
            name: str(self.task / relative)
            for name, relative in {
                "operator": "operator",
                "research": "state/research",
                "evaluation": "state/evaluation",
                "runtime": "runtime",
                "release": "../actual-release",
            }.items()
        }
        self.domains = {
            key: str(Path(value).resolve()) for key, value in self.domains.items()
        }
        for value in self.domains.values():
            Path(value).mkdir(parents=True)
        for role in self.domains:
            (self.task / "scratch" / role).mkdir(parents=True)
        self.operator = Path(self.domains["operator"])
        self.ref = self.operator / "registration.json"
        self.research = ResearchStore(self.domains["research"])
        self.evaluation = EvaluationStore(self.domains["evaluation"])
        self.manager = ReleaseManager(
            self.domains["release"], self.research, self.evaluation
        )
        self.profile_ref = self.operator / "source-profile.json"
        atomic_json(
            self.profile_ref,
            {"planner": {"model": "fixed", "reasoning_effort": "medium"}},
        )
        descriptor = self.operator / "model.json"
        atomic_json(
            descriptor, {"model": {"checkpoint": "fixture"}, "processor": "norm"}
        )
        self.phi = self.operator / "phi.json"
        self.new_phi = Path(self.domains["research"]) / "new-phi.json"
        rules = {
            "schema": "rsi-runtime-rules-v2",
            "chunk_steps": 5,
            "baseline_chunk_steps": 5,
            "finish_requires_native_success": False,
        }
        atomic_json(self.phi, rules)
        atomic_json(self.new_phi, {**rules, "finish_requires_native_success": True})
        self.active_bundle = RunBundle(
            "real-active",
            str(descriptor) + "#/model",
            str(descriptor) + "#/processor",
            str(self.phi),
            "rho",
            "psi",
            "memory",
            "old-deployment",
            str(self.profile_ref),
            "code",
        )
        initialize(
            self.research,
            self.active_bundle,
            psi={},
            memory={"provenance": "fixture", "legal_information": []},
        )
        source_bundle = RunBundle(
            **{
                **asdict(self.active_bundle),
                "bundle_id": "comparison-base",
                "protocol_id": "comparison",
            }
        )
        self.research.save_bundle(source_bundle)
        trial = RunBundle(
            **{
                **asdict(source_bundle),
                "bundle_id": "comparison-trial",
                "parent_bundle_id": "comparison-base",
                "phi_ref": str(self.new_phi),
            }
        )
        self.research.save_bundle(trial)
        self.source_candidate = Candidate(
            "comparison-candidate",
            "comparison-base",
            "comparison-trial",
            "comparison",
            "benchmark",
            {"phi_ref": str(self.new_phi)},
            [str(self.new_phi)],
            [],
            {},
            {},
            "offline",
        )
        self.research.save_candidate(self.source_candidate)
        self.source_protocol = {
            "protocol_id": "comparison",
            "status": "frozen",
            "scope": "offline",
            "benchmark_id": "benchmark",
            "anchor_bundle_id": "comparison-base",
            "comparison_only": True,
            "metrics": {
                f"success_{group}": {
                    "direction": "max",
                    "min": 0,
                    "max": 1,
                    "weight": weight,
                }
                for group, weight in [
                    ("base", 0.4),
                    ("controlled", 0.35),
                    ("old", 0.25),
                ]
            },
            "hard_constraints": ["safety", "critical_old_preserved"],
            "min_gain": 0.02,
            "anchor_max_drop": 0.02,
            "task_sets": {
                "release": {"task_keys": ["base", "controlled", "old"], "seeds": [0]}
            },
            "tasks": [
                {"task_key": key, "group": key} for key in ["base", "controlled", "old"]
            ],
            "initial_requirements": {
                "metric_minima": {"success_base": 1, "success_old": 1},
                "required_old_tasks": ["old"],
            },
        }
        self.source_ref = self.operator / "comparison-protocol.json"
        atomic_json(self.source_ref, self.source_protocol)
        self.target_ref = self.operator / "deployment-protocol.json"
        self.target_protocol = deepcopy(self.source_protocol)
        self.target_protocol.update(
            protocol_id="new-deployment",
            anchor_bundle_id="real-active",
            comparison_only=False,
            deployment_derivation={
                "kind": "candidate_upgrade",
                "authorization_ref": str(self.ref),
            },
            candidate_upgrade_derivation={
                "registration_ref": str(self.ref),
                "source_protocol_id": "comparison",
            },
        )
        atomic_json(self.target_ref, self.target_protocol)
        self.active = {
            "bundle_id": "real-active",
            "scope": "offline",
            "activation": "qualified_initial",
            "qualification_ref": "original",
            "revision": 7,
        }
        atomic_json(self.manager.active_path, self.active)
        code = self.operator / "execution.py"
        code.write_text("fixed execution source\n")
        manifest = self.operator / "manifest.json"
        atomic_json(manifest, {"files": [str(code)]})
        self.raw = Path(self.domains["runtime"]) / "candidate" / "native.json"
        self.base_raw = Path(self.domains["runtime"]) / "baseline" / "native.json"
        for path, bundle in ((self.raw, trial), (self.base_raw, source_bundle)):
            atomic_json(path, {"native": True})
            atomic_json(
                path.parent / "runtime_context.json",
                {
                    "bundle_id": bundle.bundle_id,
                    "protocol_id": bundle.protocol_id,
                    "execution_binding": execution_binding(
                        self.research,
                        asdict(bundle),
                        json.loads(self.profile_ref.read_text()),
                    ),
                },
            )
        binding = {
            "source_ref": str(self.profile_ref),
            "target_ref": str(self.profile_ref),
            "source_path": ["planner"],
            "target_path": ["planner"],
            "source_value": {"model": "fixed", "reasoning_effort": "medium"},
            "target_value": {"model": "fixed", "reasoning_effort": "medium"},
            "reason": "Exact fixture behavior binding",
        }
        self.registration = {
            "version": "candidate-upgrade-registration-v1",
            "registration_id": "frozen",
            "registration_ref": str(self.ref),
            "scope": "fixture-upgrade",
            "formal_results_observed": 0,
            "registered_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "approved": True,
            "publication_authorized": True,
            "inheritance_authorized": True,
            "inheritance_task": {"task_key": "controlled", "seed": 2},
            "role_domains": self.domains,
            "protected_paths": [str(self.operator)],
            "isolation_acceptance_ref": str(self.operator / "acceptance.json"),
            "active_snapshot": self.active,
            "source_protocol_ref": str(self.source_ref),
            "source_protocol_snapshot": self.source_protocol,
            "target_protocol_ref": str(self.target_ref),
            "target_protocol_snapshot": self.target_protocol,
            "source_candidate_id": "comparison-candidate",
            "source_baseline_bundle_id": "comparison-base",
            "source_candidate_bundle_id": "comparison-trial",
            "target_candidate_id": "deployment-candidate",
            "target_bundle_id": "deployment-trial",
            "target_profile_id": "deployment-profile",
            "source_result_ids": {
                "baseline": "baseline",
                "candidate": "candidate",
                "anchor": "baseline",
            },
            "source_decision_ref": str(
                Path(self.domains["evaluation"]) / "source-gate.json"
            ),
            "behavior_review": {key: [binding] for key in BEHAVIOR_FIELDS},
            "source_manifests": [
                {"ref": str(manifest), "snapshot": {"files": [str(code)]}}
            ],
            "execution_sources": [
                {
                    "source_ref": str(code),
                    "target_ref": str(code),
                    "unified_diff": "",
                    "reason": "Same bytes",
                }
            ],
            "applicability": dict.fromkeys(
                (
                    "hard_constraints",
                    "score_criteria",
                    "current_active_qualification",
                    "candidate_behavior_preserved",
                ),
                True,
            ),
            "rollback": {
                "allowed_ancestors": ["real-active"],
                "triggers": ["binding_invalid"],
            },
        }
        atomic_json(self.ref, self.registration)
        self.baseline = self.result("baseline", False)
        self.candidate = self.result("candidate", True)
        self.evaluation.save_result(self.baseline)
        self.evaluation.save_result(self.candidate)
        self.save_gate()
        self.evaluation.put("protocols", "comparison", self.source_protocol)
        original_protocol = deepcopy(self.source_protocol)
        original_protocol.update(
            protocol_id="old-deployment",
            anchor_bundle_id="real-active",
            comparison_only=False,
        )
        original_result = replace(
            self.result("original-independent", True),
            bundle_id="real-active",
            protocol_id="old-deployment",
            candidate_id=None,
        )
        self.evaluation.put("protocols", "old-deployment", original_protocol)
        self.evaluation.save_result(original_result)
        self.original_qualification = self.manager.qualify_initial(
            "original", result_id=original_result.result_id
        )
        self.mock_role = patch("rsi_vla.qualification_import.require_role")
        self.mock_readonly = patch(
            "rsi_vla.qualification_import._readonly", return_value=True
        )
        self.mock_role.start()
        self.mock_readonly.start()
        self.addCleanup(self.mock_role.stop)
        self.addCleanup(self.mock_readonly.stop)

    def result(self, identity, candidate):
        rows = [
            {
                "task_key": task,
                "seed": 0,
                "outcome": "success"
                if candidate or task != "controlled"
                else "failure",
                "validity": "valid",
                "evidence_ref": str(self.raw if candidate else self.base_raw),
                "outcome_source": "native_benchmark",
            }
            for task in ("base", "controlled", "old")
        ]
        return ExperimentResult(
            identity,
            "comparison-trial" if candidate else "comparison-base",
            "comparison",
            "benchmark",
            "release",
            "offline",
            {},
            [str(self.raw if candidate else self.base_raw)],
            "criteria",
            {
                "success_base": 1.0,
                "success_controlled": 1.0 if candidate else 0.0,
                "success_old": 1.0,
            },
            {"safety": True, "critical_old_preserved": True},
            {},
            candidate_id="comparison-candidate" if candidate else None,
            assigned_count=3,
            assignments=rows,
        )

    def save_gate(self):
        gate = Gate(self.source_protocol).evaluate(
            "source-comparison",
            self.source_candidate,
            current_bundle_id="comparison-base",
            candidate_result=self.candidate,
            baseline_result=self.baseline,
            anchor_result=self.baseline,
        )
        atomic_json(self.registration["source_decision_ref"], asdict(gate))

    def prepare(self):
        return prepare_candidate_deployment(self.research, registration_ref=self.ref)

    def import_(self):
        return import_candidate_evidence(
            self.evaluation, import_id="independent-v1", registration_ref=self.ref
        )

    def qualify(self):
        return qualify_upgrade(
            self.manager,
            qualification_id="new-qualified",
            decision_id="upgrade",
            import_id="independent-v1",
            registration_ref=str(self.ref),
        )

    def publish(self):
        self.prepare()
        self.import_()
        self.qualify()
        return self.manager.activate(
            "upgrade", safe_point="task_boundary", queues_empty=True
        )

    def test_legal_upgrade_repeated_receipts_and_ancestor_rollback(self):
        original = (
            deepcopy(self.source_protocol),
            asdict(self.baseline),
            asdict(self.candidate),
            self.manager.active(),
        )
        self.assertEqual(self.prepare(), self.prepare())
        self.assertEqual(self.import_(), self.import_())
        self.assertEqual(self.qualify(), self.qualify())
        active = self.manager.activate(
            "upgrade", safe_point="task_boundary", queues_empty=True
        )
        self.assertEqual(active["bundle_id"], "deployment-trial")
        self.assertEqual(active["revision"], 8)
        self.assertEqual(
            active,
            self.manager.activate(
                "upgrade", safe_point="task_boundary", queues_empty=True
            ),
        )
        with self.assertRaisesRegex(ValueError, "authorized"):
            self.manager.rollback(
                "real-active",
                safe_point="task_boundary",
                queues_empty=True,
                reason="demonstration",
            )
        reverted = self.manager.rollback(
            "real-active",
            safe_point="task_boundary",
            queues_empty=True,
            reason="binding_invalid",
        )
        self.assertEqual(reverted["bundle_id"], "real-active")
        with self.assertRaisesRegex(ValueError, "ancestor"):
            self.manager.rollback(
                "deployment-trial",
                safe_point="task_boundary",
                queues_empty=True,
                reason="binding_invalid",
            )
        self.assertEqual(self.evaluation.get("results", "baseline"), original[1])
        self.assertEqual(self.evaluation.get("results", "candidate"), original[2])
        self.assertEqual(self.source_protocol, original[0])
        self.assertFalse(
            self.manager.records.get("qualifications", "new-qualified-active").get(
                "evidence_import_id"
            )
        )

    def test_active_revision_drift_and_safe_point_block_publication(self):
        self.prepare()
        self.import_()
        self.qualify()
        for safe, empty in [("during_task", True), ("task_boundary", False)]:
            with self.assertRaises(ValueError):
                self.manager.activate("upgrade", safe_point=safe, queues_empty=empty)
        atomic_json(self.manager.active_path, {**self.active, "revision": 8})
        with self.assertRaisesRegex(ValueError, "active/revision"):
            self.manager.activate(
                "upgrade", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(self.manager.records.all("activations"), [])

    def public_matrix_with_valid_old_failure(self, *, candidate_old_failure=False):
        self.source_protocol["task_sets"]["release"]["seeds"] = [0, 1]
        self.target_protocol["task_sets"]["release"]["seeds"] = [0, 1]
        for name, original in (
            ("baseline", self.baseline),
            ("candidate", self.candidate),
        ):
            rows = [
                {**row, "seed": seed} for row in original.assignments for seed in (0, 1)
            ]
            if name == "baseline" or candidate_old_failure:
                next(
                    row for row in rows if row["task_key"] == "old" and row["seed"] == 1
                )["outcome"] = "failure"
            updated = replace(
                original,
                assignments=rows,
                assigned_count=6,
                metrics={
                    **original.metrics,
                    "success_old": 0.5
                    if name == "baseline" or candidate_old_failure
                    else 1.0,
                },
            )
            setattr(self, name, updated)
            atomic_json(self.evaluation.path("results", name), asdict(updated))
        self.registration["source_protocol_snapshot"] = self.source_protocol
        self.registration["target_protocol_snapshot"] = self.target_protocol
        atomic_json(self.source_ref, self.source_protocol)
        atomic_json(self.target_ref, self.target_protocol)
        atomic_json(self.ref, self.registration)
        self.save_gate()

    def test_current_valid_old_failure_preserves_original_qualification_and_can_upgrade(
        self,
    ):
        self.public_matrix_with_valid_old_failure()
        original = deepcopy(self.original_qualification)
        self.prepare()
        imported = self.import_()
        self.assertFalse(imported["current_active_applicability"]["reinitialized"])
        self.assertEqual(
            imported["current_active_applicability"]["qualification_snapshot"], original
        )
        self.assertTrue(imported["candidate_initial_requirements"]["passed"])
        self.qualify()
        compatibility = self.manager.records.get(
            "qualifications", "new-qualified-active"
        )
        self.assertEqual(compatibility["kind"], "upgrade_current_applicability")
        self.assertFalse(compatibility["relative_gain_claimed"])
        self.manager.activate("upgrade", safe_point="task_boundary", queues_empty=True)
        self.assertEqual(
            self.manager.records.get("qualifications", "original"), original
        )
        self.manager.rollback(
            "real-active",
            safe_point="task_boundary",
            queues_empty=True,
            reason="binding_invalid",
        )
        self.assertEqual(self.manager.active()["bundle_id"], "real-active")
        self.assertEqual(
            self.manager.records.get("qualifications", "original"), original
        )

    def test_comparison_gain_does_not_waive_candidate_initial_old_requirement(self):
        self.public_matrix_with_valid_old_failure(candidate_old_failure=True)
        self.assertEqual(
            json.loads(Path(self.registration["source_decision_ref"]).read_text())[
                "verdict"
            ],
            "accept",
        )
        self.prepare()
        with self.assertRaisesRegex(ValueError, "initial performance"):
            self.import_()
        self.assertEqual(self.manager.active(), self.active)
        self.assertEqual(
            self.manager.records.all("qualifications"), [self.original_qualification]
        )

    def test_missing_original_current_qualification_cannot_be_replaced_by_mapping(self):
        self.public_matrix_with_valid_old_failure()
        self.manager.records.path("qualifications", "original").unlink()
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            self.import_()
        self.assertEqual(self.manager.active(), self.active)

    def test_current_qualification_snapshot_drift_is_rejected(self):
        self.public_matrix_with_valid_old_failure()
        changed = deepcopy(self.original_qualification)
        changed["bundle_snapshot"]["theta_ref"] += "-changed"
        atomic_json(self.manager.records.path("qualifications", "original"), changed)
        self.prepare()
        with self.assertRaisesRegex(ValueError, "qualification identity/snapshot"):
            self.import_()
        self.assertEqual(self.manager.active(), self.active)
        self.assertEqual(self.manager.records.all("activations"), [])

    def prepare_versioned_comparison_result(self):
        for protocol in (self.source_protocol, self.target_protocol):
            protocol["scope_id"] = self.registration["scope"]
            protocol["task_policy"] = {"schema": "rsi-task-policy-v1"}
            protocol["hard_constraints"].append("critical_progress")
        self.registration["source_protocol_snapshot"] = self.source_protocol
        self.registration["target_protocol_snapshot"] = self.target_protocol
        atomic_json(self.source_ref, self.source_protocol)
        atomic_json(self.target_ref, self.target_protocol)
        atomic_json(self.ref, self.registration)
        raw = replace(
            self.baseline,
            hard_constraints={
                **self.baseline.hard_constraints,
                "critical_progress": None,
            },
        )
        atomic_json(self.evaluation.path("results", "baseline"), asdict(raw))
        self.raw_baseline_ref = self.evaluation.path("results", "baseline")
        self.raw_baseline_bytes = self.raw_baseline_ref.read_bytes()
        self.baseline = replace(
            raw, hard_constraints={**raw.hard_constraints, "critical_progress": True}
        )
        self.candidate = replace(
            self.candidate,
            hard_constraints={
                **self.candidate.hard_constraints,
                "critical_progress": True,
            },
        )
        atomic_json(
            self.evaluation.path("results", "candidate"), asdict(self.candidate)
        )
        scope_ref, admin_ref = (
            self.operator / "reparse-scope.json",
            self.operator / "reparse-admin.json",
        )
        atomic_json(
            scope_ref,
            dict(
                authorized=True,
                scope_id=self.registration["scope"],
                expires_at=self.registration["expires_at"],
            ),
        )
        atomic_json(
            admin_ref,
            dict(
                authorized=True,
                scope=self.registration["scope"],
                protocol_id="comparison",
                expires_at=self.registration["expires_at"],
                approval_reference=str(scope_ref),
            ),
        )
        self.evaluation.reparse_result(
            self.baseline,
            revision_id="baseline-reparsed",
            expected_previous_ref=self.raw_baseline_ref,
            reason="complete independent progression audit",
            source_ref=self.operator / "manifest.json",
            protocol=self.source_protocol,
            authorization_ref=admin_ref,
            scope_authorization_ref=scope_ref,
        )
        self.save_gate()

    def test_versioned_source_result_path_snapshot_and_protected_inputs_agree(self):
        self.prepare_versioned_comparison_result()
        paths = set(
            isolation_paths(
                {},
                {
                    "action": "candidate_evidence_import",
                    "registration_ref": str(self.ref),
                },
            )
        )
        self.assertIn(self.raw_baseline_ref, paths)
        self.assertIn(self.evaluation.path("results", "baseline"), paths)
        self.assertIn(
            self.evaluation.path("result_evaluation_current", "baseline"), paths
        )
        self.assertIn(
            self.evaluation.path("result_evaluation_revisions", "baseline-reparsed"),
            paths,
        )
        self.prepare()
        imported = self.import_()
        self.assertEqual(
            imported["source_results"]["baseline"],
            self.evaluation.get("results", "baseline"),
        )
        self.assertEqual(self.raw_baseline_ref.read_bytes(), self.raw_baseline_bytes)

    def test_real_role_boundary_with_versioned_source_result(self):
        self.prepare_versioned_comparison_result()
        self.test_real_release_role_boundary_and_complete_bridge()

    def test_expiry_wrong_mapping_missing_facts_and_reject_never_publish(self):
        cases = (
            "expiry",
            "mapping",
            "source",
            "missing",
            "reject",
            "inconclusive",
            "late",
        )
        for case in cases:
            with self.subTest(case=case):
                test = UpgradeBridgeTests()
                test.setUp()
                try:
                    if case == "expiry":
                        test.registration["expires_at"] = "2000-01-01T00:00:00+00:00"
                    elif case == "mapping":
                        test.registration["target_protocol_snapshot"]["min_gain"] = 0
                        atomic_json(
                            test.target_ref,
                            test.registration["target_protocol_snapshot"],
                        )
                    elif case == "source":
                        test.registration["execution_sources"][0]["unified_diff"] = (
                            "unreviewed"
                        )
                    elif case == "late":
                        test.registration["registered_at"] = (
                            datetime.now(timezone.utc) + timedelta(seconds=5)
                        ).isoformat()
                    atomic_json(test.ref, test.registration)
                    if case == "missing":
                        test.raw.unlink()
                    if case == "reject":
                        test.candidate = test.result("candidate", False)
                        test.candidate = ExperimentResult(
                            **{
                                **asdict(test.candidate),
                                "bundle_id": "comparison-trial",
                                "candidate_id": "comparison-candidate",
                            }
                        )
                    if case == "inconclusive":
                        test.candidate.hard_constraints["safety"] = None
                    if case in ("reject", "inconclusive"):
                        atomic_json(
                            test.evaluation.path("results", "candidate"),
                            asdict(test.candidate),
                        )
                        test.save_gate()
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        test.prepare()
                        test.import_()
                        test.qualify()
                    self.assertEqual(test.manager.active(), test.active)
                    self.assertEqual(
                        test.manager.records.all("qualifications"),
                        [test.original_qualification],
                    )
                finally:
                    test.doCleanups()

    def test_comparison_accept_cannot_directly_publish_or_bootstrap_upgrade(self):
        decision = json.loads(
            Path(self.registration["source_decision_ref"]).read_text()
        )
        self.assertEqual(decision["verdict"], "accept")
        self.assertFalse(decision["release_eligible"])
        self.manager.records.put(
            "decisions",
            "comparison-only",
            {**decision, "decision_id": "comparison-only"},
        )
        with self.assertRaisesRegex(ValueError, "comparison-only"):
            self.manager.activate(
                "comparison-only", safe_point="task_boundary", queues_empty=True
            )
        self.prepare()
        self.import_()
        self.manager.active_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.qualify()

    def test_no_candidate_and_expiry_after_qualification_cannot_publish(self):
        self.research.path("candidates", "comparison-candidate").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertEqual(self.manager.active(), self.active)
        self.research.save_candidate(self.source_candidate)
        self.prepare()
        self.import_()
        self.qualify()
        self.registration["expires_at"] = "2000-01-01T00:00:00+00:00"
        atomic_json(self.ref, self.registration)
        with self.assertRaisesRegex(ValueError, "expired"):
            self.manager.activate(
                "upgrade", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(self.manager.active(), self.active)
        self.assertEqual(self.manager.records.all("activations"), [])

    def test_original_evidence_changes_after_import_are_detected(self):
        self.prepare()
        self.import_()
        changed = asdict(self.candidate)
        changed["costs"] = {"artifact_bytes": 999}
        atomic_json(self.evaluation.path("results", "candidate"), changed)
        with self.assertRaisesRegex(ValueError, "facts changed"):
            self.qualify()

    def test_authorized_projection_recomputed_from_exact_original_frozen_source(self):
        from rsi_vla.improvement_protocol import runtime_projection
        from rsi_vla.upgrade_bridge import _validate_projection

        frozen_ref = self.operator / "frozen-protocol10.json"
        frozen = json.loads((ROOT / "config/pilot_5a_protocol10.json").read_text())
        atomic_json(frozen_ref, frozen)
        authority_ref = self.operator / "administrative.json"
        atomic_json(
            authority_ref,
            {
                "authorized": True,
                "scope": frozen["scope_id"],
                "protocol_id": frozen["protocol_id"],
                "frozen_protocol_ref": str(frozen_ref),
                "allowed_runtime_splits": ["release", "development"],
            },
        )
        projected = runtime_projection(
            frozen, frozen_ref=frozen_ref, authorization_ref=authority_ref
        )
        _validate_projection(projected, self.registration)
        projected["min_gain"] = 0
        with self.assertRaisesRegex(ValueError, "exact authorized frozen projection"):
            _validate_projection(projected, self.registration)

    def test_source_drift_before_import_or_qualification_is_rejected(self):
        for stage in ("import", "qualification"):
            for drift in ("backend_ref", "profile", "rule", "candidate", "model_ref"):
                with self.subTest(stage=stage, drift=drift):
                    test = UpgradeBridgeTests()
                    test.setUp()
                    try:
                        test.prepare()
                        if stage == "qualification":
                            test.import_()
                        facts = {
                            key: test.evaluation.get("results", key)
                            for key in ("baseline", "candidate")
                        }
                        gate = Path(
                            test.registration["source_decision_ref"]
                        ).read_bytes()
                        if drift in ("backend_ref", "model_ref"):
                            bundle = test.research.get("bundles", "comparison-trial")
                            if drift == "backend_ref":
                                alternate = test.operator / "alternate-profile.json"
                                atomic_json(
                                    alternate, json.loads(test.profile_ref.read_text())
                                )
                                bundle["backend_ref"] = str(alternate)
                            else:
                                bundle["theta_ref"] = (
                                    str(test.operator / "model.json") + "#/processor"
                                )
                            atomic_json(
                                test.research.path("bundles", "comparison-trial"),
                                bundle,
                            )
                        elif drift == "profile":
                            profile = json.loads(test.profile_ref.read_text())
                            profile["planner"]["model"] = "drifted"
                            atomic_json(test.profile_ref, profile)
                        elif drift == "rule":
                            rules = json.loads(test.new_phi.read_text())
                            rules["finish_requires_native_success"] = False
                            atomic_json(test.new_phi, rules)
                        else:
                            candidate = test.research.get(
                                "candidates", "comparison-candidate"
                            )
                            candidate["compatibility"]["drift"] = True
                            atomic_json(
                                test.research.path(
                                    "candidates", "comparison-candidate"
                                ),
                                candidate,
                            )
                        with self.assertRaises(ValueError):
                            (test.import_ if stage == "import" else test.qualify)()
                        self.assertEqual(test.manager.active(), test.active)
                        self.assertEqual(
                            test.manager.records.all("qualifications"),
                            [test.original_qualification],
                        )
                        self.assertEqual(
                            Path(test.registration["source_decision_ref"]).read_bytes(),
                            gate,
                        )
                        for key, value in facts.items():
                            self.assertEqual(test.evaluation.get("results", key), value)
                    finally:
                        test.doCleanups()

    def test_rollback_uses_publication_receipt_despite_faulty_candidate(self):
        for drift in (
            "source_profile",
            "source_rule",
            "source_bundle",
            "deployment_profile",
            "deployment_bundle",
        ):
            with self.subTest(drift=drift):
                test = UpgradeBridgeTests()
                test.setUp()
                try:
                    test.publish()
                    if drift == "source_profile":
                        # Candidate gets its own source profile; the ancestor remains intact.
                        bundle = test.research.get("bundles", "comparison-trial")
                        bundle["backend_ref"] = str(test.root / "missing-profile")
                        atomic_json(
                            test.research.path("bundles", "comparison-trial"), bundle
                        )
                    elif drift == "source_rule":
                        atomic_json(test.new_phi, {"broken": True})
                    elif drift == "deployment_profile":
                        atomic_json(
                            test.research.path(
                                "deployment_profiles", "deployment-profile"
                            ),
                            {"broken": True},
                        )
                    else:
                        bundle_id = (
                            "comparison-trial"
                            if drift == "source_bundle"
                            else "deployment-trial"
                        )
                        test.research.path("bundles", bundle_id).unlink()
                    reverted = test.manager.rollback(
                        "real-active",
                        safe_point="task_boundary",
                        queues_empty=True,
                        reason="binding_invalid",
                    )
                    self.assertEqual(reverted["bundle_id"], "real-active")
                    self.assertEqual(reverted["revision"], 9)
                finally:
                    test.doCleanups()

    def test_rollback_rejects_stale_authority_revision_receipt_and_damaged_target(self):
        for drift in (
            "expiry",
            "revision",
            "import",
            "activation",
            "target",
            "target_rule",
            "target_profile",
        ):
            with self.subTest(drift=drift):
                test = UpgradeBridgeTests()
                test.setUp()
                try:
                    active = test.publish()
                    if drift == "expiry":
                        test.registration["expires_at"] = "2000-01-01T00:00:00+00:00"
                        atomic_json(test.ref, test.registration)
                    elif drift == "revision":
                        atomic_json(
                            test.manager.active_path, {**active, "revision": 99}
                        )
                    elif drift == "import":
                        imported = test.evaluation.get(
                            "candidate_evidence_imports", "independent-v1"
                        )
                        imported["profile_snapshot"]["drift"] = True
                        atomic_json(
                            test.evaluation.path(
                                "candidate_evidence_imports", "independent-v1"
                            ),
                            imported,
                        )
                    elif drift == "activation":
                        receipt = test.manager.records.get("activations", "upgrade")
                        receipt["previous"] = "arbitrary"
                        atomic_json(
                            test.manager.records.path("activations", "upgrade"), receipt
                        )
                    elif drift == "target":
                        target = test.research.get("bundles", "real-active")
                        target["code_ref"] = "damaged"
                        atomic_json(
                            test.research.path("bundles", "real-active"), target
                        )
                    elif drift == "target_rule":
                        atomic_json(test.phi, {"broken": True})
                    else:
                        atomic_json(test.profile_ref, {"broken": True})
                    before = test.manager.active()
                    with self.assertRaises(ValueError):
                        test.manager.rollback(
                            "real-active",
                            safe_point="task_boundary",
                            queues_empty=True,
                            reason="binding_invalid",
                        )
                    self.assertEqual(test.manager.active(), before)
                    self.assertEqual(test.manager.records.all("rollbacks"), [])
                finally:
                    test.doCleanups()

    def test_real_release_role_boundary_and_complete_bridge(self):
        self.mock_role.stop()
        self.mock_readonly.stop()
        spec = importlib.util.spec_from_file_location(
            "isolation", ROOT / "scripts/run_isolated_role.py"
        )
        isolation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isolation)
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)
        script = self.operator / "role.py"
        script.write_text(
            """import json, os, sys\nfrom pathlib import Path\nfrom rsi_vla.storage import atomic_json\nfrom rsi_vla.upgrade_bridge import role_action\nregistration=json.loads(Path(sys.argv[1]).read_text())\nrole=os.environ['RSI_ISOLATED_ROLE']\ndomain=Path(registration['role_domains'][role])\ndenied=[]\nfor name, value in registration['role_domains'].items():\n if name == role: continue\n try: (Path(value)/'forbidden').write_text('forbidden')\n except OSError: denied.append(name)\n else: raise AssertionError('cross-domain write')\nassert len(denied)==4\natomic_json(domain/'probe.json',dict(role=role,scope=registration['scope'],allowed_write=True,denied_writes=True))\nif len(sys.argv)>2:\n task=json.loads(sys.argv[2]); value=role_action({},task)\n atomic_json(domain/(task['action']+'-receipt.json'),value)\n"""
        )

        def run(role, task=None):
            argv = [sys.executable, str(script), str(self.ref)]
            if task is not None:
                argv.append(json.dumps(task))
            result = subprocess.run(
                isolation.command(
                    self.task,
                    role,
                    argv,
                    release_root=self.domains["release"],
                    protected_paths=isolation_paths({}, task) if task else (),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        reports = {}
        for role in self.domains:
            run(role)
            reports[role] = {
                "exit_code": 0,
                "report_ref": str(Path(self.domains[role]) / "probe.json"),
            }
        atomic_json(
            self.registration["isolation_acceptance_ref"],
            {
                "scope": self.registration["scope"],
                "role_domains": self.domains,
                "roles": reports,
            },
        )
        common = {
            "registration_ref": str(self.ref),
            "import_id": "independent-v1",
            "qualification_id": "new-qualified",
            "decision_id": "upgrade",
            "safe_point": "task_boundary",
            "queues_empty": True,
        }
        for action, role in [
            ("candidate_deployment_prepare", "research"),
            ("candidate_evidence_import", "evaluation"),
            ("candidate_upgrade_qualify", "release"),
            ("candidate_upgrade_activate", "release"),
        ]:
            run(role, {**common, "action": action})
        self.assertEqual(self.manager.active()["bundle_id"], "deployment-trial")
        self.assertFalse((self.task / "release/active.json").exists())
        self.assertEqual(
            self.manager.records.get("activations", "upgrade")["previous"],
            "real-active",
        )


if __name__ == "__main__":
    unittest.main()
