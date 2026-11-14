"""Release-only derivation preserves development identities and runtime semantics."""

import copy
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import test_research_tools
from rsi_vla.release_preparation import derive_release_candidate
from rsi_vla.storage import atomic_json
from rsi_vla.trial_reparse import reparse_trial


class ReleaseDerivationTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    opportunity = test_research_tools.ResearchToolsTest.opportunity

    def test_derivation_and_reject_behavior_or_split_changes(self):
        profile = self.root / 'backend.json'
        atomic_json(profile, {'profile_id': 'development', 'control': {'action_chunk': 5}, 'collection': {'enabled': True}})
        self.base = replace(self.base, backend_ref=str(profile))
        self.tools.base = self.base
        atomic_json(self.store.path('bundles', self.base.bundle_id), asdict(self.base))
        self.tools.domain_policy['rule_operators'] = ['planner_segment_handoff_hint']
        self.opportunity()
        self.tools.plan_rule_experiment('hint', opportunity_id='opportunity', bundle_id='hint-trial',
                                       planner_segment_handoff_hint=True, criterion_ref='criterion-v1',
                                       distinguishing_prediction='fixture', cost_limit={'offline_calls': 1})
        trial = self.tools.prepare_rule_trial('hint')
        identity = {'run_id': 'hint-original-run', 'episode_id': 'hint-episode',
                    'bundle_id': trial['bundle_id'], 'protocol_id': self.protocol['protocol_id']}
        result = replace(self.result('hint-result', trial['bundle_id']), intervention={
            'information_scope': 'development', 'planner_segment_handoff_hint': True,
            'identity': identity})
        self.tools.record_experiment_result('hint', result)
        original_files = {p: p.read_bytes() for kind in ('development_results', 'candidates', 'bundles')
                          for p in (self.store.root / kind).glob('*.json')}
        checks = dict(native_success=True, technical_valid=True, runtime_intervention_loaded=True,
                      critical_progress=None, collection_complete=True, resource_accounting_complete=True)
        self.tools._save('trial_collections', 'hint', {'result': asdict(result), 'checks': checks})
        self.tools._save('research_cycles', 'hint-cycle', {
            'result_id': result.result_id, 'action': 'stop', 'candidate': None,
            'screening_checks': checks, 'criterion': 'development-native-valid-loaded-progress-data-v1'})
        output = self.root / 'raw'
        atomic_json(output / 'runtime_context.json', identity)
        atomic_json(output / 'dataset_manifest.json', {'episodes': [
            {'path': str(output / 'synthetic-episode'), 'persisted_steps': 1}]})
        self.tools._save('trial_dispatches', 'hint-dispatch', {'experiment_id': 'hint', 'output': str(output)})
        phase = self.root / 'phase.json'
        dimensions = dict(wall_seconds=10000, storage_bytes=10000000, model_requests=100,
                          tool_calls=100, environment_steps=1000, simulation_episodes=10, gpu_seconds=10000)
        atomic_json(phase, {'limits': dimensions, 'used': dict.fromkeys(dimensions, 0),
                           'remaining': dimensions, 'unresolved_costs': []})
        permit = self.root / 'permit.json'
        atomic_json(permit, {'experiment_id': 'hint', 'job_id': 'job', 'run_id': identity['run_id'],
                            'output_dir': str(output), 'phase_snapshot_ref': str(phase)})
        for name in ('whole_job_wall_seconds', 'artifact_bytes'):
            self.job.state['limits'][name] = 10000000
            self.job.state['used'][name] = 0
        self.job.state['status'] = 'stopped'
        self.job._save()
        revised_checks = {**checks, 'critical_progress': True}
        revised = replace(result, result_id='hint-v2-bound-result', intervention={
            **result.intervention, 'runtime_loaded': True, 'screening_checks': revised_checks})
        with patch('rsi_vla.trial_reparse._collect', return_value=(revised, revised_checks)), \
             patch('rpent.flywheel.episode.validate_episode', return_value=None):
            cycle = reparse_trial(self.tools, 'hint', output=output, permit_path=permit,
                                 revision_id='v2', audit_root=self.root / 'audit-v2')
        self.assertEqual(cycle['action'], 'candidate_for_independent_evaluation')
        self.assertEqual(self.job.state['status'], 'stopped')
        source = cycle['candidate']
        source_cycle_id = 'hint-v2'
        old = copy.deepcopy(self.protocol)
        for k in ('tasks', 'asset_binding', 'metrics', 'hard_constraints', 'cost_weights', 'initial_requirements', 'recovery', 'outcomes', 'state_evaluation', 'draft_comparison'):
            old[k] = {} if k != 'tasks' else []
        old.update(runtime_config={}, min_gain=.02, anchor_max_drop=.02, budget={'per_attempt_proposal': {'model_requests': 20}})
        old['task_sets']['release']['authorized'] = False
        oldpath = self.root / 'old-protocol.json'
        atomic_json(oldpath, old)
        new = copy.deepcopy(old)
        new.update(protocol_id='release-only', anchor_bundle_id='release-base', release_authorized=True, allowed_runtime_splits=['release'],
                   development_derivation={'source_protocol_id': source['protocol_id'], 'source_candidate_id': source['candidate_id'], 'source_protocol_ref': str(oldpath)})
        for split in ('development', 'release', 'sealed'):
            new['task_sets'][split]['authorized'] = split == 'release'
        current = json.loads(profile.read_text())
        args = dict(base_id='release-base', trial_id='release-trial', candidate_id='release-candidate', backend_ref=profile, code_ref='common-source-v2')
        submission_ref = Path(source['compatibility']['cost_snapshot_ref'])
        submission_bytes = submission_ref.read_bytes()
        source_bytes = self.store.path('candidates', source['candidate_id']).read_bytes()
        submission = json.loads(submission_bytes)
        admission = {'phase_snapshot_ref': submission['phase_snapshot_ref'],
                     'phase': copy.deepcopy(submission['phase']),
                     'purpose': 'release admission at derivation'}
        admission['phase']['used']['wall_seconds'] += 5
        admission['phase']['used']['model_requests'] += 2
        admission_ref = self.root / 'release-admission.json'
        atomic_json(admission_ref, admission)
        args['release_admission_snapshot_ref'] = admission_ref
        job_before = self.store.path('jobs', 'job').read_bytes()
        phase_before = phase.read_bytes()
        for label, mutate in (
                ('different authoritative phase', lambda a: a.update(phase_snapshot_ref='other-ledger.json')),
                ('unresolved actual costs', lambda a: a['phase'].update(unresolved_costs=['missing usage'])),
                ('predates source', lambda a: a['phase']['used'].update(wall_seconds=0))):
            bad_admission = copy.deepcopy(admission)
            mutate(bad_admission)
            atomic_json(admission_ref, bad_admission)
            with self.subTest(admission=label), self.assertRaisesRegex(ValueError, label):
                derive_release_candidate(self.store, source_cycle_id, new, current, **args)
        atomic_json(admission_ref, admission)
        wrong_submission = copy.deepcopy(submission)
        wrong_submission['phase']['used']['model_requests'] += 1
        atomic_json(submission_ref, wrong_submission)
        with self.assertRaisesRegex(ValueError, 'differs from Candidate costs'):
            derive_release_candidate(self.store, source_cycle_id, new, current, **args)
        submission_ref.write_bytes(submission_bytes)
        bad = copy.deepcopy(current)
        bad['control']['action_chunk'] = 2
        with self.assertRaisesRegex(ValueError, 'new development required'):
            derive_release_candidate(self.store, source_cycle_id, new, bad, **args)
        bad_protocol = copy.deepcopy(new)
        bad_protocol['task_sets']['release']['seeds'] = [1]
        with self.assertRaisesRegex(ValueError, 'assignment split'):
            derive_release_candidate(self.store, source_cycle_id, bad_protocol, current, **args)
        candidate, derivation = derive_release_candidate(self.store, source_cycle_id, new, current, **args)
        self.assertEqual(candidate['development_results'], ['hint-v2-bound-result'])
        self.assertEqual(self.store.get('development_results', 'hint-result')['protocol_id'], self.protocol['protocol_id'])
        self.assertEqual(candidate['changes'], source['changes'])
        self.assertFalse(derivation.get('development_results_relabelled', False))
        self.assertEqual(derivation['new_development_runs'], 0)
        self.assertEqual(derivation['source_development_results'], candidate['compatibility']['source_development_results'])
        self.assertEqual(derivation['source_development_results'][0]['result_id'], 'hint-v2-bound-result')
        self.assertEqual(derivation['source_development_results'][0]['identity'], identity)
        self.assertEqual(candidate['costs'], source['costs'])
        self.assertEqual(candidate['compatibility']['cost_snapshot_ref'], str(submission_ref))
        provenance = derivation['cost_provenance']
        self.assertEqual(provenance, candidate['compatibility']['cost_provenance'])
        self.assertEqual(provenance['costs_basis'], 'source_candidate_at_submission')
        self.assertEqual(provenance['source_candidate_ref'], str(self.store.path('candidates', source['candidate_id']).resolve()))
        self.assertEqual(provenance['source_cost_snapshot_ref'], str(submission_ref))
        self.assertEqual(provenance['release_admission_snapshot_ref'], str(admission_ref))
        self.assertEqual(provenance['inherited_costs'], source['costs'])
        self.assertEqual(provenance['cost_booking'], 'historical_reuse_no_charge')
        self.assertEqual(provenance['later_costs_basis'], 'authoritative_phase_ledger')
        self.assertNotEqual(candidate['costs']['model_requests'], admission['phase']['used']['model_requests'])
        self.assertEqual(self.store.path('jobs', 'job').read_bytes(), job_before)
        self.assertEqual(phase.read_bytes(), phase_before)
        self.assertEqual(submission_ref.read_bytes(), submission_bytes)
        self.assertEqual(self.store.path('candidates', source['candidate_id']).read_bytes(), source_bytes)
        # Later actual charges remain in the phase ledger, outside both frozen snapshots.
        later_phase = json.loads(phase.read_text())
        later_phase['used']['model_requests'] += 3
        later_phase['release_trial_charges'] = [{'run_id': 'later-release', 'costs': {'model_requests': 3}}]
        atomic_json(phase, later_phase)
        later_phase_bytes = phase.read_bytes()
        self.assertEqual(candidate['compatibility']['cost_scope'], 'authoritative_phase_history_at_submission')
        self.assertEqual(derivation['source_development_results'][0]['protocol_id'], self.protocol['protocol_id'])
        self.assertTrue(all(p.read_bytes() == value for p, value in original_files.items()))
        self.assertEqual((candidate, derivation), derive_release_candidate(
            self.store, source_cycle_id, new, current, **args))

        self.assertEqual(phase.read_bytes(), later_phase_bytes)
        self.assertEqual(self.store.path('jobs', 'job').read_bytes(), job_before)
        self.assertEqual(submission_ref.read_bytes(), submission_bytes)
        self.assertEqual(json.loads(admission_ref.read_text()), admission)
