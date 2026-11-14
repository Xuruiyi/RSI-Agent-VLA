"""Only absent derived audit references may be repaired; native facts stay immutable."""

import json
from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest.mock import patch

import test_evaluator as evaluator_fixture
import test_result_evaluation_versions as version_fixture
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.evaluator import evaluate_matrix
from rsi_vla.storage import atomic_json


class SavedAuditTests(unittest.TestCase):
    def setUp(self):
        evaluator_fixture.EvaluatorTests.setUp(self)
        self.manifest = evaluator_fixture.EvaluatorTests.native(self)
        self.runtime_root = Path(self.manifest[0]["result_path"]).parent
        self.identity = dict(
            run_id="r1",
            episode_id="base-s3",
            bundle_id="initial-medium",
            protocol_id=self.protocol["protocol_id"],
        )
        atomic_json(self.runtime_root / "runtime_context.json", self.identity)
        self.report = dict(
            identity=self.identity,
            checks={},
            passed=True,
            costs={"model_requests": 2},
            notes=[],
            effective_rules={},
        )

    def evaluate(self, **kwargs):
        with patch(
            "rsi_vla.evaluator.inspect_attempt", return_value=self.report
        ) as inspect:
            result = evaluate_matrix(
                self.protocol,
                self.profile,
                self.manifest,
                result_id="fixture",
                bundle_id="initial-medium",
                **kwargs,
            )
            inspect.assert_called_once()
        return result

    def test_default_never_emits_a_phantom_independent_audit_reference(self):
        result = self.evaluate()
        phantom = self.runtime_root / "independent-audit.json"
        self.assertFalse(phantom.exists())
        self.assertNotIn(str(phantom), result.data_refs)

    def test_explicit_audit_root_saves_the_actual_inspector_report(self):
        output = self.runtime_root / "evaluation/result_audits/revision"
        result = self.evaluate(audit_output_root=output)
        saved = output / "r1/independent-audit.json"
        self.assertEqual(json.loads(saved.read_text()), self.report)
        self.assertIn(str(saved.resolve()), result.data_refs)
        self.assertNotIn(
            str(self.runtime_root / "independent-audit.json"), result.data_refs
        )

    def test_an_existing_audit_identity_cannot_be_overwritten(self):
        output = self.runtime_root / "evaluation/result_audits/revision"
        saved = output / "r1/independent-audit.json"
        atomic_json(
            saved, {**self.report, "identity": {**self.identity, "run_id": "other"}}
        )
        before = saved.read_bytes()
        with self.assertRaisesRegex(ValueError, "conflicting independent audit"):
            self.evaluate(audit_output_root=output)
        self.assertEqual(saved.read_bytes(), before)


class AuditReferenceRepairTests(unittest.TestCase):
    def setUp(self):
        version_fixture.ResultEvaluationVersionTests.setUp(self)
        self.store = EvaluationStore(self.root / "repair-evaluation")
        self.runtime_root = self.root / "runtime/r"
        self.identity = dict(
            run_id="r", episode_id="e", bundle_id="base", protocol_id="new-protocol"
        )
        self.context_ref = self.runtime_root / "runtime_context.json"
        self.native_ref = self.runtime_root / "independent_result.json"
        self.phantom = self.runtime_root / "independent-audit.json"
        atomic_json(self.context_ref, self.identity)
        atomic_json(self.native_ref, {**self.identity, "benchmark_success": False})
        self.original = replace(
            self.corrected,
            data_refs=[str(self.native_ref), str(self.phantom), str(self.source_ref)],
            assignments=[
                {**self.corrected.assignments[0], "evidence_ref": str(self.native_ref)}
            ],
        )
        self.store.save_result(self.original)
        self.raw = self.store.path("results", "canonical").resolve()
        self.raw_bytes = self.raw.read_bytes()
        self.audit_ref = (
            self.store.root / "result_audits/revision1/r/independent-audit.json"
        )
        self.report = dict(
            identity=self.identity,
            checks={"identity": True},
            passed=True,
            costs=self.costs,
            notes=[],
            outcome="failure",
        )
        atomic_json(self.audit_ref, self.report)
        self.repair = dict(
            missing_ref=str(self.phantom),
            replacement_ref=str(self.audit_ref.resolve()),
            runtime_context_ref=str(self.context_ref.resolve()),
            identity=self.identity,
        )
        self.corrected = replace(
            self.original,
            data_refs=[
                str(self.native_ref),
                str(self.audit_ref.resolve()),
                str(self.source_ref),
            ],
        )

    def reparse(self, result=None, **changes):
        arguments = dict(derived_ref_repairs=[self.repair])
        arguments.update(changes)
        return version_fixture.ResultEvaluationVersionTests.reparse(
            self, result, **arguments
        )

    def test_absent_derived_audit_repair_retains_raw_previous_version_and_receipt(self):
        old_receipt = self.store.reparse_result(
            self.original,
            revision_id="legacy-v6",
            expected_previous_ref=self.raw,
            reason="legacy evaluation fixture",
            source_ref=self.source_ref,
            protocol=self.protocol,
            authorization_ref=self.admin_ref,
            scope_authorization_ref=self.scope_ref,
        )
        previous = self.store.path("results", "canonical").resolve()
        previous_bytes = previous.read_bytes()
        receipt_ref = self.store.path("result_evaluation_revisions", "legacy-v6")
        receipt_bytes = receipt_ref.read_bytes()
        receipt = self.reparse(expected_previous_ref=previous)
        self.assertEqual(receipt["derived_ref_repairs"], [self.repair])
        self.assertEqual(receipt["previous_ref"], old_receipt["version_ref"])
        self.assertEqual(self.store.get("results", "canonical"), asdict(self.corrected))
        self.assertEqual(previous.read_bytes(), previous_bytes)
        self.assertEqual(receipt_ref.read_bytes(), receipt_bytes)
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)
        self.assertFalse(self.phantom.exists())
        self.assertEqual(self.reparse(expected_previous_ref=previous), receipt)

    def test_missing_native_reference_cannot_be_hidden_by_derived_repair(self):
        self.native_ref.unlink()
        with self.assertRaisesRegex(ValueError, "missing raw/native"):
            self.reparse()
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_wrong_runtime_identity_is_refused(self):
        atomic_json(
            self.audit_ref,
            {**self.report, "identity": {**self.identity, "run_id": "other"}},
        )
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            self.reparse()
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_existing_or_native_references_cannot_be_substituted(self):
        self.phantom.write_text("a pre-existing audit must remain referenced")
        with self.assertRaisesRegex(ValueError, "only a missing derived"):
            self.reparse()
        self.phantom.unlink()
        with self.assertRaisesRegex(ValueError, "only a missing derived"):
            self.reparse(
                derived_ref_repairs=[
                    {**self.repair, "missing_ref": str(self.native_ref)}
                ]
            )

    def test_existing_reference_deletion_is_refused(self):
        with self.assertRaisesRegex(ValueError, "preserve every original"):
            self.reparse(
                replace(
                    self.corrected,
                    data_refs=[str(self.audit_ref.resolve()), str(self.source_ref)],
                )
            )

    def test_replacement_outside_evaluation_store_is_refused(self):
        outside = self.root / "outside/independent-audit.json"
        atomic_json(outside, self.report)
        with self.assertRaisesRegex(ValueError, "only a missing derived"):
            self.reparse(
                derived_ref_repairs=[{**self.repair, "replacement_ref": str(outside)}]
            )

    def test_fixed_handler_computes_and_records_mapping_without_payload_booleans(self):
        backend, operation = version_fixture.ResultEvaluationVersionTests.backend(self)
        self.audit_ref.unlink()

        def compute(protocol, profile, manifest, *, audit_output_root, **identity):
            self.assertEqual(
                audit_output_root, self.store.root / "result_audits/revision1"
            )
            fresh = audit_output_root / "r/independent-audit.json"
            atomic_json(fresh, self.report)
            return replace(
                self.corrected, data_refs=[str(self.native_ref), str(fresh.resolve())]
            )

        with patch(
            "rsi_vla.evaluator.evaluate_matrix", side_effect=compute
        ) as evaluate:
            reply = backend.execute(operation, {})
            replay = backend.execute(operation, {})
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(replay, reply)
        self.assertEqual(
            reply["revision_receipt"]["derived_ref_repairs"], [self.repair]
        )
        self.assertEqual(json.loads(self.audit_ref.read_text()), self.report)
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)
        self.assertEqual(self.store.get("results", "canonical"), reply["result"])
        self.assertNotIn(str(self.phantom), reply["result"]["data_refs"])


if __name__ == "__main__":
    unittest.main()
