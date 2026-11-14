"""Revision -> finding -> Candidate/stop with a stopped original job."""

import copy
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import test_research_tools
from rsi_vla.storage import atomic_json
from rsi_vla.trial_reparse import reparse_trial


class TrialReparseTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    opportunity = test_research_tools.ResearchToolsTest.opportunity
    prepare = test_research_tools.ResearchToolsTest.prepare

    def fixture(self):
        trial = self.prepare()
        original_result = replace(self.result('original', chunk_steps=2),
                                  criteria_ref='criterion-v1')
        self.tools.record_experiment_result('experiment', original_result)
        checks = {'native_success': True, 'technical_valid': True, 'runtime_intervention_loaded': True,
                  'critical_progress': None, 'collection_complete': True, 'resource_accounting_complete': True}
        self.tools._save('trial_collections', 'experiment', {'result': asdict(original_result), 'checks': checks})
        self.tools._save('research_cycles', 'experiment-cycle', {'result_id': 'original', 'action': 'stop',
                         'candidate': None, 'screening_checks': checks, 'criterion': 'development-native-valid-loaded-progress-data-v1'})
        out = self.root / 'raw'
        out.mkdir()
        atomic_json(out / 'runtime_context.json', {'run_id': 'actual-run', 'episode_id': 'e', 'bundle_id': trial['bundle_id'], 'protocol_id': self.protocol['protocol_id']})
        atomic_json(out / 'dataset_manifest.json', {'episodes': [{'path': str(out / 'synthetic-episode'), 'persisted_steps': 1}]})
        (out / 'original.txt').write_text('original immutable output')
        self.tools._save('trial_dispatches', 'experiment-dispatch', {'experiment_id': 'experiment', 'output': str(out)})
        phase = self.root / 'phase.json'
        dimensions = dict(wall_seconds=10000, storage_bytes=10000000, model_requests=100, tool_calls=100,
                          environment_steps=1000, simulation_episodes=10, gpu_seconds=10000)
        used = dict.fromkeys(dimensions, 0)
        used['model_requests'] = 11  # Original Research + older failed run, already charged.
        atomic_json(phase, {'limits': dimensions, 'used': used,
                           'remaining': {k: v-used[k] for k, v in dimensions.items()}, 'unresolved_costs': [],
                           'research_charges': [{'request_id': 'original-Research', 'costs': {'model_requests': 1}}],
                           'development_trial_charges': [{'run_id': 'old-failure', 'costs': {'model_requests': 10}}]})
        permit = self.root / 'permit.json'
        atomic_json(permit, {'experiment_id': 'experiment', 'job_id': 'job', 'run_id': 'actual-run', 'output_dir': str(out), 'phase_snapshot_ref': str(phase)})
        for name in ('whole_job_wall_seconds', 'artifact_bytes'):
            self.job.state['limits'][name] = 10000000
            self.job.state['used'][name] = 0
        self.job.state['status'] = 'stopped'
        self.job._save()
        return out, permit, phase, original_result, checks

    def test_revised_candidate_stop_identity_and_single_charge(self):
        for progress in (True, None, False):
            with self.subTest(progress=progress):
                # Each subcase is an independent synthetic store.
                self.setUp()
                out, permit, phase, original, checks = self.fixture()
                old = {str(p): p.read_bytes() for kind in ('research_findings', 'trial_collections', 'research_cycles', 'trial_dispatches', 'research_experiments', 'rule_trials') for p in (self.store.root / kind).glob('*.json')}
                raw = {str(p): p.read_bytes() for p in out.rglob('*') if p.is_file()}
                checks = {**checks, 'critical_progress': progress}
                result = replace(original, result_id='experiment-audit-v2-bound-result',
                                 intervention={**original.intervention, 'runtime_loaded': True, 'screening_checks': checks})
                with patch('rsi_vla.trial_reparse._collect', return_value=(result, checks)) as collector, \
                     patch('rpent.flywheel.episode.validate_episode', return_value=None), \
                     patch('rsi_vla.research_cycle.execute_runtime', side_effect=AssertionError('no Runtime')), \
                     patch('rsi_vla.research_cycle.prepare_runtime', side_effect=AssertionError('no new plan')):
                    args = dict(output=out, permit_path=permit, revision_id='audit-v2', audit_root=self.root / 'audit')
                    first = reparse_trial(self.tools, 'experiment', **args)
                    charges = copy.deepcopy(self.job.state['charges'])
                    phase_bytes = phase.read_bytes()
                    second = reparse_trial(self.tools, 'experiment', **args)
                    self.assertEqual(first, second)
                    self.assertEqual(collector.call_count, 1)
                    self.assertEqual(charges, self.job.state['charges'])
                    self.assertEqual(phase_bytes, phase.read_bytes())
                    self.assertEqual(self.job.state['status'], 'stopped')
                    self.assertEqual(len([c for c in charges if c['reason']=='experiment experiment']), 1)
                    self.assertTrue(all(Path(p).read_bytes() == b for p, b in {**old, **raw}.items()))
                    self.assertEqual(first['action'], 'candidate_for_independent_evaluation' if progress is True else 'stop')
                    if progress is True:
                        self.assertEqual(first['candidate']['costs']['model_requests'], 11)
                        self.assertEqual(first['candidate']['compatibility']['finding_revision_id'], 'experiment-audit-v2')
                    else:
                        self.assertIsNone(first['candidate'])
                    dispatch = self.store.get('trial_dispatches', 'experiment-dispatch')
                    atomic_json(self.store.path('trial_dispatches', 'experiment-dispatch'), {**dispatch, 'output': str(self.root / 'different')})
                    with self.assertRaises(ValueError):
                        reparse_trial(self.tools, 'experiment', **args)

    def revised_collector(self, original, checks):
        def collect(tools, experiment_id, output, *, revision_id, derived_root, **kwargs):
            # Synthetic audit content is distinct for every revision.
            atomic_json(derived_root / 'independent-audit.json', {'revision_id': revision_id})
            atomic_json(derived_root / 'critical-progress.json', {'revision_id': revision_id,
                                                               'value': checks['critical_progress']})
            current_checks = dict(checks)
            return replace(original, result_id=experiment_id + '-' + revision_id + '-bound-result',
                           intervention={**original.intervention, 'runtime_loaded': True,
                                         'screening_checks': current_checks}), current_checks
        return collect

    def test_revision_roots_and_snapshots_are_immutable_and_separate(self):
        out, permit, phase, original, checks = self.fixture()
        checks['critical_progress'] = True
        args = dict(output=out, permit_path=permit, revision_id='v2', audit_root=self.root / 'audit-v2')
        with patch('rsi_vla.trial_reparse._collect', side_effect=self.revised_collector(original, checks)) as collector, \
             patch('rpent.flywheel.episode.validate_episode', return_value=None):
            first = reparse_trial(self.tools, 'experiment', **args)
            preserved = {p: p.read_bytes() for p in args['audit_root'].rglob('*') if p.is_file()}
            old_candidate = self.store.path('candidates', first['candidate']['candidate_id']).read_bytes()
            job_bytes, phase_bytes = self.store.path('jobs', 'job').read_bytes(), phase.read_bytes()
            for conflicting_root in (args['audit_root'], args['audit_root'] / 'nested'):
                with self.subTest(root=conflicting_root), self.assertRaisesRegex(ValueError, 'another revision'):
                    reparse_trial(self.tools, 'experiment', **{**args, 'revision_id': 'v3', 'audit_root': conflicting_root})
            with self.assertRaisesRegex(ValueError, 'conflicting reparse revision input'):
                reparse_trial(self.tools, 'experiment', **{**args, 'audit_root': self.root / 'elsewhere'})
            self.assertEqual(collector.call_count, 1)
            self.assertEqual(job_bytes, self.store.path('jobs', 'job').read_bytes())
            self.assertEqual(phase_bytes, phase.read_bytes())
            second = reparse_trial(self.tools, 'experiment', **{**args, 'revision_id': 'v3', 'audit_root': self.root / 'audit-v3'})
            self.assertNotEqual(first['result_id'], second['result_id'])
            self.assertNotEqual(first['candidate']['candidate_id'], second['candidate']['candidate_id'])
            self.assertTrue(all(p.read_bytes() == value for p, value in preserved.items()))
            self.assertEqual(old_candidate, self.store.path('candidates', first['candidate']['candidate_id']).read_bytes())
            phase_after = json.loads(phase.read_text())
            self.assertEqual(phase_after['used']['model_requests'], 11)
            self.assertEqual(len(phase_after['offline_revision_charges']), 2)
            self.assertEqual(len([c for c in self.job.state['charges'] if c['reason'] == 'experiment experiment']), 1)
            current_job, current_phase = self.store.path('jobs', 'job').read_bytes(), phase.read_bytes()
            self.assertEqual(first, reparse_trial(self.tools, 'experiment', **args))
            self.assertEqual(current_job, self.store.path('jobs', 'job').read_bytes())
            self.assertEqual(current_phase, phase.read_bytes())
            self.assertTrue(all(p.read_bytes() == value for p, value in preserved.items()))

    def test_reject_raw_overlap_and_unbound_existing_snapshot_before_collection(self):
        out, permit, phase, original, checks = self.fixture()
        args = dict(output=out, permit_path=permit, revision_id='v2')
        snapshot_root = self.root / 'other-audit'
        snapshot_root.mkdir()
        snapshot = snapshot_root / 'candidate-cost-snapshot.json'
        snapshot.write_text('original cost snapshot')
        phase_bytes = phase.read_bytes()
        with patch('rsi_vla.trial_reparse._collect') as collector:
            for root in (out, out / 'audit', out.parent, snapshot_root):
                with self.subTest(root=root), self.assertRaises(ValueError):
                    reparse_trial(self.tools, 'experiment', **args, audit_root=root)
            collector.assert_not_called()
        self.assertEqual(snapshot.read_text(), 'original cost snapshot')
        self.assertEqual(phase_bytes, phase.read_bytes())
        self.assertFalse(self.store.path('trial_reparse_inputs', 'experiment-v2').exists())

    def test_phase_cost_dimensions_include_history_and_retain_unknown_usage(self):
        out, permit, phase, original, checks = self.fixture()
        checks['critical_progress'] = True
        historic = json.loads(phase.read_text())
        historic['used'].update(wall_seconds=123, gpu_seconds=99, storage_bytes=1024,
                                tool_calls=8, simulation_episodes=3, environment_steps=42)
        historic['cost_observations'] = [{'run_id': 'old-failed-run', 'full_token_totals': None,
                                         'observed_input_tokens': 17, 'missing_request_usage': [2]}]
        atomic_json(phase, historic)
        with patch('rsi_vla.trial_reparse._collect', side_effect=self.revised_collector(original, checks)), \
             patch('rpent.flywheel.episode.validate_episode', return_value=None):
            cycle = reparse_trial(self.tools, 'experiment', output=out, permit_path=permit,
                                 revision_id='v2', audit_root=self.root / 'audit')
        candidate = cycle['candidate']
        snapshot = json.loads(Path(candidate['compatibility']['cost_snapshot_ref']).read_text())
        mapping = candidate['compatibility']['cost_dimension_mapping']
        self.assertEqual(candidate['costs'], {mapping.get(k, k): v for k, v in snapshot['phase']['used'].items()})
        self.assertEqual(candidate['costs']['model_requests'], 11)
        self.assertEqual(candidate['costs']['gpu_device_seconds_envelope'], 99)
        self.assertGreater(candidate['costs']['whole_job_wall_seconds'], 123)
        self.assertGreater(candidate['costs']['artifact_bytes'], 1024)
        self.assertEqual(candidate['costs']['simulation_episodes'], 3)
        self.assertEqual(candidate['compatibility']['cost_scope'], 'authoritative_phase_history_at_submission')
        self.assertEqual(snapshot['phase']['cost_observations'], historic['cost_observations'])
        self.assertNotIn('input_tokens', candidate['costs'])
        self.assertNotIn('llm_input_tokens', candidate['costs'])

    def test_every_failed_or_unknown_screening_condition_stops(self):
        for name in ('native_success', 'technical_valid', 'runtime_intervention_loaded',
                     'critical_progress', 'collection_complete'):
            for value in (False, None):
                with self.subTest(check=name, value=value):
                    self.setUp()
                    out, permit, phase, original, checks = self.fixture()
                    checks.update(critical_progress=True)
                    checks[name] = value
                    with patch('rsi_vla.trial_reparse._collect', side_effect=self.revised_collector(original, checks)), \
                         patch('rpent.flywheel.episode.validate_episode', return_value=None):
                        cycle = reparse_trial(self.tools, 'experiment', output=out, permit_path=permit,
                                             revision_id='v2', audit_root=self.root / 'audit')
                    self.assertEqual(cycle['action'], 'stop')
                    self.assertIsNone(cycle['candidate'])
                    self.assertIs(cycle['screening_checks'][name], value)

    def test_unresolved_phase_costs_stop_even_if_original_screening_passed(self):
        out, permit, phase, original, checks = self.fixture()
        checks['critical_progress'] = True
        value = json.loads(phase.read_text())
        value['unresolved_costs'] = [{'run_id': 'old-failure', 'missing': ['tool_calls']}]
        atomic_json(phase, value)
        with patch('rsi_vla.trial_reparse._collect', side_effect=self.revised_collector(original, checks)), \
             patch('rpent.flywheel.episode.validate_episode', return_value=None):
            cycle = reparse_trial(self.tools, 'experiment', output=out, permit_path=permit,
                                 revision_id='v2', audit_root=self.root / 'audit')
        self.assertEqual(cycle['action'], 'stop')
        self.assertIsNone(cycle['candidate'])
        self.assertFalse(cycle['screening_checks']['resource_accounting_complete'])
        self.assertTrue(self.store.get('research_cycles', 'experiment-cycle')['screening_checks']['resource_accounting_complete'])
