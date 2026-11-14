"""Only same-ID evidence completion may advance an inconclusive logical Gate."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.contracts import Candidate, ExperimentResult, RunBundle
from rsi_vla.evaluation import EvaluationStore, Gate
from rsi_vla.gate_versions import decision_inputs
from rsi_vla.storage import ResearchStore, atomic_json, initialize
from rsi_vla.task_completion import ACTION_ROLES, TaskCompletionBackend
import test_upgrade_bridge


class GateReconsiderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = EvaluationStore(self.root / 'evaluation')
        self.research = ResearchStore(self.root / 'research')
        self.protocol = dict(protocol_id='p', scope_id='scope', status='frozen', scope='offline', benchmark_id='b',
            anchor_bundle_id='base', metrics={'success_base': {'direction': 'max', 'min': 0, 'max': 1, 'weight': 1}},
            hard_constraints={'safety': 'fixed', 'critical_progress': 'independent geometry'}, min_gain=.02,
            anchor_max_drop=.02, comparison_only=True, task_policy={'schema': 'rsi-task-policy-v1'})
        self.protocol_ref = self.root / 'protocol.json'
        self.source_ref = self.root / 'evaluation-source.json'
        self.scope_ref, self.auth_ref = self.root / 'scope.json', self.root / 'authority.json'
        self.phase_ref = self.root / 'phase.json'
        expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        for ref, value in ((self.protocol_ref, self.protocol), (self.source_ref, {'files': ['independent fixture']}),
            (self.scope_ref, {'authorized': True, 'scope_id': 'scope', 'expires_at': expiry}),
            (self.auth_ref, {'authorized': True, 'scope': 'scope', 'protocol_id': 'p', 'expires_at': expiry,
                            'approval_reference': str(self.scope_ref), 'phase_snapshot_ref': str(self.phase_ref)})):
            atomic_json(ref, value)
        bundle = RunBundle('base', 'theta', 'processor', 'phi', 'rho', 'psi', 'memory', 'p', 'backend', 'code')
        initialize(self.research, bundle, psi={}, memory={'provenance': 'fixture', 'legal_information': []})
        self.research.save_bundle(replace(bundle, bundle_id='trial', parent_bundle_id='base', phi_ref='new-phi'))
        self.candidate = Candidate('candidate', 'base', 'trial', 'p', 'b', {'phi_ref': 'new-phi'}, ['new-phi'], [], {}, {}, 'offline')
        self.research.save_candidate(self.candidate)
        native = self.root / 'native.json'
        atomic_json(native, {'native': 'unchanged'})
        self.baseline = ExperimentResult('baseline', 'base', 'p', 'b', 'release', 'offline', {}, [str(native)],
            'fixed-criterion', {'success_base': 0}, {'safety': True, 'critical_progress': True}, {'model_requests': 4},
            assigned_count=1, outcome='failure')
        self.original = replace(self.baseline, result_id='candidate-result', bundle_id='trial', candidate_id='candidate',
            metrics={'success_base': 1}, hard_constraints={'safety': True, 'critical_progress': None}, outcome='success')
        self.store.save_result(self.baseline)
        self.store.save_result(self.original)
        self.raw_result = self.store.path('results', self.original.result_id)
        self.result_ids = {'candidate': 'candidate-result', 'baseline': 'baseline', 'anchor': 'baseline'}
        self.previous_results = {side: str(self.store.path('results', key).resolve()) for side, key in self.result_ids.items()}
        self.original_gate = Gate(self.protocol).evaluate('logical-gate', self.candidate, current_bundle_id='base',
            candidate_result=self.original, baseline_result=self.baseline, anchor_result=self.baseline)
        self.assertEqual(self.original_gate.verdict, 'inconclusive')
        self.store.put('decisions', 'logical-gate', self.original_gate)
        self.raw_gate = self.store.path('decisions', 'logical-gate')
        self.raw_gate_bytes, self.raw_result_bytes = self.raw_gate.read_bytes(), self.raw_result.read_bytes()

    def reparse(self, value=True):
        result = replace(self.original, hard_constraints={'safety': True, 'critical_progress': value},
                         data_refs=self.original.data_refs + [str(self.source_ref)])
        return self.store.reparse_result(result, revision_id='result-v9', expected_previous_ref=self.raw_result,
            reason='complete independent primitive audit', source_ref=self.source_ref, protocol=self.protocol,
            authorization_ref=self.auth_ref, scope_authorization_ref=self.scope_ref)

    def arguments(self, **changes):
        args = dict(decision_id='logical-gate', candidate=self.candidate, current_bundle_id='base', result_ids=self.result_ids,
            previous_result_refs=self.previous_results, revision_id='gate-v2', expected_previous_ref=self.raw_gate,
            reason='independent evidence completion only', source_ref=self.source_ref, protocol=self.protocol,
            frozen_protocol_ref=self.protocol_ref, authorization_ref=self.auth_ref, scope_authorization_ref=self.scope_ref)
        args.update(changes)
        return args

    def reconsider(self, **changes):
        return self.store.reconsider_gate(**self.arguments(**changes))

    def test_successful_reconsideration_keeps_raw_identity_scores_and_current_entry(self):
        self.reparse()
        receipt = self.reconsider()
        current = self.store.get('decisions', 'logical-gate')
        self.assertEqual(current['verdict'], 'accept')
        self.assertFalse(current['release_eligible'])
        for key in ('decision_id', 'candidate_id', 'base_bundle_id', 'bundle_id', 'protocol_id', 'benchmark_id', 'scope', 'scores', 'result_refs'):
            self.assertEqual(current[key], asdict(self.original_gate)[key])
        self.assertEqual(self.store.path('decisions', 'logical-gate'), Path(receipt['version_ref']))
        self.assertEqual(self.store.all('decisions'), [current])
        self.assertEqual(self.raw_gate.read_bytes(), self.raw_gate_bytes)
        self.assertEqual(self.raw_result.read_bytes(), self.raw_result_bytes)
        self.assertFalse(self.store.put('decisions', 'logical-gate', self.original_gate))
        with self.assertRaisesRegex(ValueError, 'conflicting decisions'):
            self.store.put('decisions', 'logical-gate', current)
        self.assertEqual(self.reconsider(), receipt)

    def test_existing_baseline_revision_alone_is_not_new_gate_evidence(self):
        revised = replace(self.baseline, notes=['earlier baseline v8'])
        self.store.reparse_result(revised, revision_id='baseline-v8',
            expected_previous_ref=self.store.path('results', 'baseline'), reason='earlier evidence',
            source_ref=self.source_ref, protocol=self.protocol, authorization_ref=self.auth_ref,
            scope_authorization_ref=self.scope_ref)
        previous = {**self.previous_results, 'baseline': str(self.store.path('results', 'baseline')),
                    'anchor': str(self.store.path('results', 'baseline'))}
        with self.assertRaisesRegex(ValueError, 'new same-ID'):
            self.reconsider(previous_result_refs=previous)
        self.assertEqual(self.raw_gate.read_bytes(), self.raw_gate_bytes)

    def test_reject_and_accept_are_never_reopened(self):
        for verdict in ('accept', 'reject'):
            terminal = {**asdict(self.original_gate), 'verdict': verdict}
            atomic_json(self.raw_gate, terminal)
            with self.assertRaisesRegex(ValueError, 'inconclusive'):
                self.reconsider()
        atomic_json(self.raw_gate, asdict(self.original_gate))
        self.reparse()
        self.reconsider()
        current = self.store.path('decisions', 'logical-gate')
        with self.assertRaisesRegex(ValueError, 'inconclusive'):
            self.reconsider(revision_id='gate-v3', expected_previous_ref=current)

    def test_false_new_constraint_legitimately_rejects_without_rewriting_raw(self):
        self.reparse(False)
        self.reconsider()
        self.assertEqual(self.store.get('decisions', 'logical-gate')['verdict'], 'reject')
        self.assertEqual(self.raw_gate.read_bytes(), self.raw_gate_bytes)

    def test_changed_protocol_candidate_results_or_forged_native_costs_rejected(self):
        self.reparse()
        for field, value in (('min_gain', 0), ('hard_constraints', {'safety': 'different'})):
            with self.assertRaises(PermissionError):
                self.reconsider(protocol={**self.protocol, field: value})
        with self.assertRaisesRegex(ValueError, 'result IDs'):
            self.reconsider(result_ids={**self.result_ids, 'candidate': 'baseline'})
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            self.reconsider(candidate=replace(self.candidate, candidate_id='another'))
        current = self.store.path('results', self.original.result_id)
        data = json.loads(current.read_text())
        atomic_json(current, {**data, 'costs': {'model_requests': 1}})
        with self.assertRaisesRegex(ValueError, 'native facts or costs'):
            self.reconsider()

    def test_expired_authority_and_reusing_older_inputs_for_an_inconclusive_version_are_rejected(self):
        auth = json.loads(self.auth_ref.read_text())
        atomic_json(self.auth_ref, {**auth, 'expires_at': '2000-01-01T00:00:00+00:00'})
        with self.assertRaises(PermissionError):
            self.reconsider()
        atomic_json(self.auth_ref, auth)
        noted = replace(self.original, notes=['new independent evidence still incomplete'])
        self.store.reparse_result(noted, revision_id='result-v9', expected_previous_ref=self.raw_result,
            reason='new independent evidence remains unknown', source_ref=self.source_ref,
            protocol=self.protocol, authorization_ref=self.auth_ref, scope_authorization_ref=self.scope_ref)
        self.reconsider()
        current = self.store.path('decisions', 'logical-gate')
        self.assertEqual(self.store.get('decisions', 'logical-gate')['verdict'], 'inconclusive')
        with self.assertRaisesRegex(ValueError, 'immutable receipt inputs'):
            self.reconsider(revision_id='gate-v3', expected_previous_ref=current)

    def test_current_pointer_and_receipt_tampering_are_detected(self):
        self.reparse()
        receipt = self.reconsider()
        pointer = self.store.path('gate_evaluation_current', 'logical-gate')
        before = json.loads(pointer.read_text())
        atomic_json(pointer, {**before, 'decision_id': 'another'})
        with self.assertRaisesRegex(ValueError, 'pointer'):
            self.store.get('decisions', 'logical-gate')
        atomic_json(pointer, before)
        receipt_ref = self.store.path('gate_evaluation_revisions', 'gate-v2')
        atomic_json(receipt_ref, {**receipt, 'raw_decision_ref': 'different'})
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.store.path('decisions', 'logical-gate')

    def test_pending_pointer_commit_can_resume_and_does_not_reopen_terminal(self):
        self.reparse()
        receipt = self.reconsider()
        self.store.path('gate_evaluation_current', 'logical-gate').unlink()
        self.assertEqual(self.reconsider(), receipt)
        self.assertEqual(self.store.get('decisions', 'logical-gate')['verdict'], 'accept')
        self.assertEqual(self.raw_gate.read_bytes(), self.raw_gate_bytes)

    def test_task_role_consumes_current_gate_without_new_dynamic_action(self):
        self.assertEqual(ACTION_ROLES['gate_reconsider'], 'evaluation')
        self.reparse()
        payload = self.arguments()
        for key in ('candidate', 'protocol', 'frozen_protocol_ref', 'authorization_ref', 'scope_authorization_ref'):
            payload.pop(key)
        payload['candidate_id'] = 'candidate'
        payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
        operation = {'operation_id': 'reconsider', 'kind': 'gate_reconsider', 'payload': payload}
        config = dict(schema='rsi-task-completion-v1', protocol_ref=str(self.protocol_ref), frozen_protocol_ref=str(self.protocol_ref),
            research_store_ref=str(self.research.root), evaluation_store_ref=str(self.store.root), operations=[operation],
            phase_ref=str(self.phase_ref), administrative_authorization_ref=str(self.auth_ref), scope_authorization_ref=str(self.scope_ref))
        backend = TaskCompletionBackend(config)
        with patch.dict('os.environ', {'RSI_ISOLATED_ROLE': 'research'}):
            with self.assertRaises(PermissionError):
                backend.execute(operation, {})
        with patch.dict('os.environ', {'RSI_ISOLATED_ROLE': 'evaluation'}):
            result = backend.execute(operation, {})
        self.assertEqual(result['decision']['verdict'], 'accept')
        self.assertEqual(result['decision']['decision_id'], 'logical-gate')
        self.assertEqual(result['decision_ref'], str(self.store.path('decisions', 'logical-gate')))


class UpgradeGateVersionTests(unittest.TestCase):
    def test_upgrade_reads_frozen_canonical_current_and_keeps_resolution_receipt(self):
        fixture = test_upgrade_bridge.UpgradeBridgeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare_versioned_comparison_result()
        original_raw_gate = Path(fixture.registration['source_decision_ref'])
        fixture.registration['source_decision_ref'] = str(fixture.evaluation.root / 'decisions/source-comparison.json')
        atomic_json(fixture.ref, fixture.registration)
        unknown = replace(fixture.candidate, hard_constraints={**fixture.candidate.hard_constraints, 'critical_progress': None})
        atomic_json(fixture.evaluation.path('results', 'candidate'), asdict(unknown))
        old_gate = Gate(fixture.source_protocol).evaluate('source-comparison', fixture.source_candidate,
            current_bundle_id='comparison-base', candidate_result=unknown,
            baseline_result=fixture.baseline, anchor_result=fixture.baseline)
        self.assertEqual(old_gate.verdict, 'inconclusive')
        fixture.evaluation.put('decisions', old_gate.decision_id, old_gate)
        canonical = fixture.evaluation.path('decisions', old_gate.decision_id)
        raw_bytes = canonical.read_bytes()
        previous = {side: str(fixture.evaluation.path('results', key))
                    for side, key in fixture.registration['source_result_ids'].items()}
        fixture.evaluation.reparse_result(fixture.candidate, revision_id='candidate-v9',
            expected_previous_ref=fixture.evaluation.path('results', 'candidate'), reason='completed candidate audit',
            source_ref=fixture.operator / 'manifest.json', protocol=fixture.source_protocol,
            authorization_ref=fixture.operator / 'reparse-admin.json', scope_authorization_ref=fixture.operator / 'reparse-scope.json')
        receipt = fixture.evaluation.reconsider_gate(decision_id=old_gate.decision_id,
            candidate=fixture.source_candidate, current_bundle_id='comparison-base', result_ids=fixture.registration['source_result_ids'],
            previous_result_refs=previous, revision_id='gate-v2', expected_previous_ref=canonical,
            reason='same comparison with independent candidate audit', source_ref=fixture.operator / 'manifest.json',
            protocol=fixture.source_protocol, frozen_protocol_ref=fixture.source_ref,
            authorization_ref=fixture.operator / 'reparse-admin.json', scope_authorization_ref=fixture.operator / 'reparse-scope.json')
        fixture.prepare()
        imported = fixture.import_()
        resolution = imported['source_gate_resolution']
        self.assertEqual(resolution['canonical_ref'], fixture.registration['source_decision_ref'])
        self.assertEqual(resolution['version_ref'], receipt['version_ref'])
        self.assertEqual(resolution['receipt_snapshot'], receipt)
        paths = decision_inputs(fixture.evaluation, old_gate.decision_id)
        from rsi_vla.upgrade_bridge import isolation_paths
        protected = set(isolation_paths({}, {'action': 'candidate_evidence_import', 'registration_ref': str(fixture.ref)}))
        self.assertTrue(paths <= protected)
        self.assertEqual(canonical.read_bytes(), raw_bytes)
        self.assertTrue(original_raw_gate.is_file())
        fixture.qualify()
        self.assertEqual(fixture.manager.active(), fixture.active)
