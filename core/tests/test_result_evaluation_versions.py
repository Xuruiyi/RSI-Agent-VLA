"""Immutable native results and canonical views of authorized evidence revisions."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rsi_vla.contracts import ExperimentResult
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import ACTION_ROLES, TaskCompletionBackend


class ResultEvaluationVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = EvaluationStore(self.root / "evaluation")
        self.protocol = dict(
            protocol_id="new-protocol",
            scope_id="new-task",
            scope="offline",
            status="frozen",
            benchmark_id="bench",
            anchor_bundle_id="base",
            min_gain=0.02,
            anchor_max_drop=0.02,
            metrics={"success_base": dict(direction="max", min=0, max=1, weight=1)},
            hard_constraints={"safety": "known", "critical_progress": "independent"},
            task_policy={"schema": "rsi-task-policy-v1"},
        )
        self.scope_ref = self.root / "scope.json"
        self.admin_ref = self.root / "admin.json"
        self.source_ref = self.root / "new-evaluation/source-manifest.json"
        self.protocol_ref = self.root / "protocol.json"
        self.frozen_ref = self.root / "frozen.json"
        self.phase_ref = self.root / "phase.json"
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.scope = dict(authorized=True, scope_id="new-task", expires_at=expires)
        self.admin = dict(
            authorized=True,
            scope="new-task",
            protocol_id="new-protocol",
            expires_at=expires,
            approval_reference=str(self.scope_ref),
            phase_snapshot_ref=str(self.phase_ref),
        )
        for ref, value in [
            (self.scope_ref, self.scope),
            (self.admin_ref, self.admin),
            (self.protocol_ref, self.protocol),
            (self.frozen_ref, self.protocol),
        ]:
            atomic_json(ref, value)
        files = [
            "core/src/rsi_vla/" + name + ".py"
            for name in (
                "evaluator",
                "critical_progress",
                "primitive_progress",
                "evaluation",
                "task_completion",
            )
        ]
        atomic_json(self.source_ref, {"files": files})
        for name in files:
            for base in (self.root / "repo", self.source_ref.parent / "source"):
                target = base / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# fixture audit implementation\n")
        native = self.root / "native.json"
        atomic_json(native, {"native_success": False})
        self.costs = dict(
            whole_job_wall_seconds=4,
            gpu_device_seconds_envelope=3,
            model_requests=2,
            tool_calls=1,
            environment_steps=5,
            artifact_bytes=6,
        )
        self.original = ExperimentResult(
            "canonical",
            "base",
            "new-protocol",
            "bench",
            "release",
            "offline",
            {
                "information_scope": "release",
                "critical_progress_evidence": [{"value": None}],
                "planner_task_hint": "",
                "attempt_model_requests_by_task": {"old": 2},
            },
            [str(native)],
            "frozen-criteria",
            {"success_base": 0.0},
            {"safety": True, "critical_progress": None},
            self.costs,
            assigned_count=1,
            outcome="failure",
            assignments=[
                dict(
                    task_key="old",
                    seed=1,
                    outcome="failure",
                    validity="valid",
                    evidence_ref=str(native),
                    outcome_source="native_benchmark",
                )
            ],
            notes=["missing primitive audit"],
        )
        self.store.save_result(self.original)
        self.raw = self.store.path("results", "canonical").resolve()
        self.raw_bytes = self.raw.read_bytes()
        self.corrected = replace(
            self.original,
            hard_constraints={"safety": True, "critical_progress": True},
            intervention={
                **self.original.intervention,
                "critical_progress_evidence": [{"value": True, "audit_version": "new"}],
            },
            data_refs=[*self.original.data_refs, str(self.source_ref)],
            notes=["independent prerequisite audit complete"],
        )

    def reparse(self, result=None, **changes):
        arguments = dict(
            revision_id="revision1",
            expected_previous_ref=self.raw,
            reason="repair missing primitive audit",
            source_ref=self.source_ref,
            protocol=self.protocol,
            authorization_ref=self.admin_ref,
            scope_authorization_ref=self.scope_ref,
        )
        arguments.update(changes)
        return self.store.reparse_result(result or self.corrected, **arguments)

    def test_old_store_and_raw_writes_remain_immutable(self):
        self.assertEqual(self.store.get("results", "canonical"), asdict(self.original))
        self.assertEqual(self.store.all("results"), [asdict(self.original)])
        self.assertFalse(self.store.save_result(self.original))
        with self.assertRaisesRegex(ValueError, "conflicting results"):
            self.store.save_result(self.corrected)
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_version_pointer_get_all_and_receipt_bind_the_same_canonical_result(self):
        receipt = self.reparse()
        version = self.store.path("results", "canonical").resolve()
        self.assertEqual(
            version,
            self.store.path("result_evaluation_versions", "revision1").resolve(),
        )
        self.assertEqual(receipt["version_ref"], str(version))
        self.assertEqual(json.loads(version.read_text())["result_id"], "canonical")
        self.assertEqual(self.store.get("results", "canonical"), asdict(self.corrected))
        self.assertEqual(self.store.all("results"), [asdict(self.corrected)])
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)
        self.assertFalse(self.store.save_result(self.original))
        self.assertEqual(self.store.path("results", "canonical").resolve(), version)

    def test_idempotent_replay_and_completed_older_revision_do_not_regress_current(
        self,
    ):
        receipt = self.reparse()
        version1 = self.store.path("results", "canonical").resolve()
        self.assertEqual(self.reparse(), receipt)
        revised = replace(self.corrected, notes=["second evaluator evidence version"])
        self.reparse(revised, revision_id="revision2", expected_previous_ref=version1)
        version2 = self.store.path("results", "canonical").resolve()
        self.assertEqual(self.reparse(), receipt)
        self.assertEqual(self.store.path("results", "canonical").resolve(), version2)
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_pending_pointer_commit_reuses_immutable_receipt(self):
        receipt = self.reparse()
        self.store.path("result_evaluation_current", "canonical").unlink()
        self.assertEqual(self.reparse(), receipt)
        self.assertEqual(self.store.get("results", "canonical"), asdict(self.corrected))

    def test_stale_previous_reference_and_revision_conflicts_are_rejected(self):
        self.reparse()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.reparse(revision_id="revision2")
        with self.assertRaisesRegex(ValueError, "conflicting.*revision"):
            self.reparse(reason="different registered reason")
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_native_identity_criterion_metrics_costs_assignments_and_counts_cannot_change(
        self,
    ):
        cases = dict(
            bundle_id="other",
            benchmark_id="other",
            criteria_ref="other",
            outcome="success",
            assigned_count=2,
            unknown_count=1,
            invalid_count=1,
            metrics={"success_base": 1.0},
            costs={**self.costs, "tool_calls": 2},
            assignments=[{**self.original.assignments[0], "outcome": "success"}],
        )
        for field, value in cases.items():
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "native facts"),
            ):
                self.reparse(replace(self.corrected, **{field: value}))
        with self.assertRaisesRegex(ValueError, "native facts"):
            self.reparse(
                replace(self.corrected, costs={**self.costs, "tool_calls": True})
            )
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_known_hard_values_and_runtime_intervention_are_fixed(self):
        for constraints in (
            {"safety": False, "critical_progress": True},
            {"safety": None, "critical_progress": True},
            {"safety": True, "critical_progress": 1},
        ):
            with self.assertRaisesRegex(ValueError, "established hard"):
                self.reparse(replace(self.corrected, hard_constraints=constraints))
        with self.assertRaisesRegex(ValueError, "runtime intervention"):
            self.reparse(
                replace(
                    self.corrected,
                    intervention={
                        **self.corrected.intervention,
                        "planner_task_hint": "changed",
                    },
                )
            )
        with self.assertRaisesRegex(ValueError, "definitions"):
            self.reparse(
                replace(
                    self.corrected,
                    hard_constraints={
                        **self.corrected.hard_constraints,
                        "new_constraint": True,
                    },
                )
            )

    def test_null_may_establish_false_without_hiding_a_real_violation(self):
        rejected = replace(
            self.corrected,
            hard_constraints={"safety": True, "critical_progress": False},
        )
        self.reparse(rejected)
        self.assertIs(
            self.store.get("results", "canonical")["hard_constraints"][
                "critical_progress"
            ],
            False,
        )
        with self.assertRaisesRegex(ValueError, "established hard"):
            self.reparse(
                self.corrected,
                revision_id="revision2",
                expected_previous_ref=self.store.path("results", "canonical"),
            )

    def test_original_refs_cannot_be_dropped(self):
        with self.assertRaisesRegex(ValueError, "every original evidence"):
            self.reparse(replace(self.corrected, data_refs=[str(self.source_ref)]))

    def test_new_task_and_both_live_authorizations_are_required(self):
        with self.assertRaises((PermissionError, ValueError)):
            self.reparse(protocol={**self.protocol, "task_policy": None})
        for ref, value in ((self.scope_ref, self.scope), (self.admin_ref, self.admin)):
            for changed in (
                {**value, "authorized": False},
                {**value, "expires_at": "2000-01-01T00:00:00+00:00"},
            ):
                atomic_json(ref, changed)
                with self.assertRaisesRegex(PermissionError, "authorization"):
                    self.reparse()
            atomic_json(ref, value)
        atomic_json(self.admin_ref, {**self.admin, "scope": "other-task"})
        with self.assertRaisesRegex(PermissionError, "new-task authority"):
            self.reparse()
        self.assertFalse(
            self.store.path("result_evaluation_current", "canonical").exists()
        )

    def test_pointer_cannot_redirect_to_unbound_result_files(self):
        self.reparse()
        pointer = self.store.path("result_evaluation_current", "canonical")
        current = json.loads(pointer.read_text())
        atomic_json(pointer, {**current, "version_ref": str(self.raw)})
        with self.assertRaisesRegex(ValueError, "pointer identity"):
            self.store.get("results", "canonical")

    def backend(self):
        manifest = self.root / "manifest.json"
        attempts = self.root / "attempt-evidence.json"
        atomic_json(manifest, [])
        atomic_json(
            attempts,
            {
                "slots": [
                    {
                        "side": "baseline",
                        "task_key": "old",
                        "attempts": [{"costs": self.costs}],
                    }
                ]
            },
        )
        payload = dict(
            result_id="canonical",
            bundle_id="base",
            manifest_ref=str(manifest),
            attempt_evidence_ref=str(attempts),
            side="baseline",
        )
        original_op = dict(
            operation_id="original-matrix", kind="matrix_evaluation", payload=payload
        )
        reparse_op = dict(
            operation_id="reparse-matrix",
            kind="matrix_reparse",
            payload={
                **payload,
                "revision_id": "revision1",
                "expected_previous_ref": str(self.raw),
                "reason": "repair missing primitive audit",
                "source_ref": str(self.source_ref),
            },
        )
        config = dict(
            schema="rsi-task-completion-v1",
            protocol_ref=str(self.protocol_ref),
            frozen_protocol_ref=str(self.frozen_ref),
            scope_authorization_ref=str(self.scope_ref),
            administrative_authorization_ref=str(self.admin_ref),
            phase_ref=str(self.phase_ref),
            repository_root=str(self.root / "repo"),
            profile_ref=str(self.root / "profile.json"),
            research_store_ref=str(self.root / "research"),
            evaluation_store_ref=str(self.store.root),
            operations=[original_op, reparse_op],
        )
        atomic_json(config["profile_ref"], {})
        return TaskCompletionBackend(config), reparse_op

    def test_fixed_reparse_recomputes_original_manifest_and_all_attempts(self):
        backend, operation = self.backend()
        # This is the independent evaluator output fixture, never a boolean in payload.
        recomputed = replace(self.corrected, data_refs=self.original.data_refs)
        with patch(
            "rsi_vla.evaluator.evaluate_matrix", return_value=recomputed
        ) as evaluator:
            reply = backend.execute(operation, {})
        self.assertEqual(ACTION_ROLES["matrix_reparse"], "evaluation")
        self.assertEqual(evaluator.call_args.kwargs["result_id"], "canonical")
        self.assertEqual(evaluator.call_args.args[2], [])
        self.assertEqual(reply["revision_receipt"]["result_id"], "canonical")
        self.assertEqual(
            Path(reply["result_ref"]), self.store.path("results", "canonical").resolve()
        )
        self.assertEqual(self.store.get("results", "canonical")["costs"], self.costs)
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_reparse_refuses_changed_manifest_or_unreviewed_consumer_sources(self):
        backend, operation = self.backend()
        operation["payload"]["manifest_ref"] = "different.json"
        with self.assertRaisesRegex(ValueError, "original matrix"):
            backend.execute(operation, {})
        operation["payload"]["manifest_ref"] = str(self.root / "manifest.json")
        (self.root / "repo/core/src/rsi_vla/primitive_progress.py").write_text(
            "# changed\n"
        )
        with self.assertRaisesRegex(ValueError, "consumer source differs"):
            backend.execute(operation, {})
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)


if __name__ == "__main__":
    unittest.main()
