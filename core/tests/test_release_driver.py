"""GPU denial and terminal task failures preserve the fixed matrix and no retries."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('release_driver', ROOT / 'scripts/run_release_comparison.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ReleaseDriverTest(unittest.TestCase):
    def test_gpu_threshold_and_missing_memory(self):
        with patch.object(driver.subprocess, 'run', side_effect=[SimpleNamespace(stdout='0, a\n1, b\n'), SimpleNamespace(stdout='a, 10, 12289\nb, 20, 12288\n')]):
            self.assertEqual(driver.gpu_admission()['selected'], 1)
        with patch.object(driver.subprocess, 'run', side_effect=[SimpleNamespace(stdout='0, a\n1, b\n'), SimpleNamespace(stdout='a, 10, N/A\nb, 20, 12289\n')]):
            self.assertIsNone(driver.gpu_admission()['selected'])

    def fixture(self, root):
        slot = dict(model_requests=20, tool_calls=60, simulation_episodes=1, environment_steps=2000, wall_seconds=900, gpu_seconds=900, storage_bytes=1073741824)
        phase = root / 'phase.json'
        caps = {k: v*9 for k, v in slot.items()}
        atomic_json(phase, {'limits': caps, 'used': dict.fromkeys(caps, 0), 'remaining': caps, 'unresolved_costs': []})
        rows = [{'side': side, 'task_key': str(i), 'bundle_id': side, 'seed': 3, 'run_id': side+str(i), 'episode_id': str(i), 'output': str(root / 'outputs' / (side+str(i))), 'status': 'unstarted', 'outcome': 'unknown', 'validity': 'valid'} for side in ('initial', 'candidate') for i in ('base_salad', 'position_swap_salad', 'old_milk', 'old_drawer')]
        profile = root / 'profile.json'
        atomic_json(profile, {})
        plan = {'phase_snapshot_ref': str(phase), 'slot_envelope': slot, 'dynamic_envelope': {k: v*8 for k, v in slot.items()}, 'slots': rows, 'artifacts': str(root), 'protocol': 'fixture', 'profile': str(profile), 'source_manifest_ref': 'fixture', 'candidate_id': 'c', 'source_candidate_id': 'source'}
        path = root / 'plan.json'
        atomic_json(path, plan)
        atomic_json(root / 'phase-before.json', json.loads(phase.read_text()))
        return path

    def test_wrong_assignment_or_reused_output_is_rejected_before_admission(self):
        for wrong in ('task', 'run', 'output'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp))
                plan = driver.read(path)
                field = {'task': 'task_key', 'run': 'run_id', 'output': 'output'}[wrong]
                plan['slots'][1][field] = plan['slots'][0][field]
                atomic_json(path, plan)
                with patch.object(driver, 'gpu_admission') as gpu, patch.object(driver, 'execute_runtime') as launch:
                    with self.assertRaisesRegex(ValueError, 'fixed tasks'):
                        driver.execute_matrix(path)
                    gpu.assert_not_called()
                    launch.assert_not_called()

    def test_inheritance_reserve_is_required_without_consuming_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            plan = driver.read(path)
            phase = driver.read(root / 'phase.json')
            driver.check_budget(plan, phase, remaining_slots=8)
            phase['remaining']['simulation_episodes'] = 8
            with self.assertRaisesRegex(ValueError, 'inheritance reserve'):
                driver.check_budget(plan, phase, remaining_slots=8)
            phase['remaining']['simulation_episodes'] = 9
            phase['remaining']['tool_calls'] = 539
            with self.assertRaisesRegex(ValueError, 'tool_calls'):
                driver.check_budget(plan, phase, remaining_slots=8)

    def test_campaign_offline_charges_have_independent_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            plan = driver.read(path)
            plan['campaign_id'] = 'new-release'
            driver.book_offline(plan, root, 'final-matrix', 2.5)
            driver.book_offline(plan, root, 'final-matrix', 2.5)
            phase = driver.read(root / 'phase.json')
            self.assertEqual(phase['used']['wall_seconds'], 2.5)
            self.assertEqual(phase['release_offline_charges'][0]['key'], 'new-release:final-matrix')

    def test_no_gpu_never_issues_permit_or_launches_and_restart_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            with patch.object(driver, 'gpu_admission', return_value={'selected': None}), patch.object(driver, 'evaluate', return_value={'passed': False}), patch.object(driver, 'execute_runtime') as launch:
                driver.execute_matrix(path)
                launch.assert_not_called()
                self.assertFalse(list(root.rglob('permit.json')))
                self.assertEqual([s['outcome'] for s in driver.read(path)['slots']], ['unknown']*8)
                with self.assertRaisesRegex(ValueError, 'do not restore or retry'):
                    driver.execute_matrix(path)

    def test_valid_failure_continues_initial_four_then_failed_qualification_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            def prepare(repo, artifacts, **kw):
                return {'blockers': [] if 'authorization_path' in kw else ['permit missing'], 'context': {'limits': {'wall_seconds': 900}, 'protocol_id': 'p', 'run_id': kw['run_id'], 'episode_id': kw['task_key'], 'bundle_id': kw['bundle_id'], 'output_dir': str(kw['output_dir']), 'collection': {}, 'code_ref': 'v', 'theta_ref': 't', 'processor_ref': 'p', 'memory_ref': 'm'}}
            def launch(plan):
                out = Path(plan['context']['output_dir']);out.mkdir(parents=True)
                atomic_json(out / 'runtime_context.json', plan['context'])
            def audit(output, **kwargs):
                c = driver.read(Path(output) / 'runtime_context.json')
                return {'identity': {k: c[k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}, 'passed': True, 'outcome': 'failure', 'costs': {'whole_job_wall_seconds': 1, 'gpu_device_seconds_envelope': 1, 'model_requests': 1, 'tool_calls': 1, 'environment_steps': 30, 'artifact_bytes': 1}}
            with patch.object(driver, 'gpu_admission', return_value={'selected': 0}), patch.object(driver, 'prepare_runtime', side_effect=prepare), patch.object(driver, 'execute_runtime', side_effect=launch) as executor, patch.object(driver, 'inspect_attempt', side_effect=audit), patch.object(driver, 'inspect_progress', return_value={'value': True}), patch.object(driver, 'evaluate', return_value={'passed': False, 'reason': 'base failure'}):
                driver.execute_matrix(path)
                self.assertEqual(executor.call_count, 4)
                rows = driver.read(path)['slots']
                self.assertEqual([r['outcome'] for r in rows], ['failure']*4+['unknown']*4)
                self.assertEqual([r['validity'] for r in rows], ['valid']*8)
                self.assertEqual(len(driver.read(root / 'phase.json')['release_trial_charges']), 4)


    def test_full_initial_one_still_runs_all_candidates_and_technical_stop_does_not(self):
        for technical_stop, progress_value, expected_starts in ((False, True, 8), (True, True, 1), (False, None, 1)):
            with self.subTest(technical_stop=technical_stop), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = self.fixture(root)
                def prepare(repo, artifacts, **kw):
                    return {'blockers': [], 'context': {'limits': {'wall_seconds': 900}, 'protocol_id': 'p', 'run_id': kw['run_id'], 'episode_id': kw['task_key'], 'bundle_id': kw['bundle_id'], 'output_dir': str(kw['output_dir']), 'collection': {}, 'code_ref': 'v', 'theta_ref': 't', 'processor_ref': 'p', 'memory_ref': 'm'}}
                def launch(plan):
                    out = Path(plan['context']['output_dir']);out.mkdir(parents=True)
                    atomic_json(out / 'runtime_context.json', plan['context'])
                def audit(output, **kwargs):
                    c = driver.read(Path(output) / 'runtime_context.json')
                    return {'identity': {k: c[k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}, 'passed': not technical_stop, 'outcome': 'unknown' if technical_stop else 'success', 'costs': {'whole_job_wall_seconds': 1, 'gpu_device_seconds_envelope': 1, 'model_requests': 1, 'tool_calls': 1, 'environment_steps': 30, 'artifact_bytes': 1}}
                with patch.object(driver, 'gpu_admission', return_value={'selected': 0}), patch.object(driver, 'prepare_runtime', side_effect=prepare), patch.object(driver, 'execute_runtime', side_effect=launch) as executor, patch.object(driver, 'inspect_attempt', side_effect=audit), patch.object(driver, 'inspect_progress', return_value={'value': progress_value}), patch.object(driver, 'evaluate', return_value={'passed': True, 'baseline_J': 1}) as evaluation:
                    driver.execute_matrix(path)
                self.assertEqual(executor.call_count, expected_starts)
                rows = driver.read(path)['slots']
                self.assertEqual(sum(r['status'] == 'unstarted' for r in rows), 8-expected_starts)
                self.assertEqual(driver.read(root / 'execution-record.json')['runtime_start_attempts'], expected_starts)
                self.assertEqual(evaluation.call_count, 1 if expected_starts == 1 else 2)
                self.assertGreaterEqual(driver.read(root / 'phase.json')['remaining']['simulation_episodes'], 1)



    def test_missing_startup_evidence_retains_measured_wall_and_unknown_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            def prepare(repo, artifacts, **kw):
                return {'blockers': [], 'context': {'limits': {'wall_seconds': 900}, 'protocol_id': 'p', 'run_id': kw['run_id'], 'episode_id': kw['task_key'], 'bundle_id': kw['bundle_id'], 'output_dir': str(kw['output_dir']), 'collection': {}, 'code_ref': 'v', 'theta_ref': 't', 'processor_ref': 'p', 'memory_ref': 'm'}}
            def launch(plan):
                out = Path(plan['context']['output_dir']);out.mkdir(parents=True)
                atomic_json(out / 'runtime_context.json', plan['context'])
            with patch.object(driver, 'gpu_admission', return_value={'selected': 0}), patch.object(driver, 'prepare_runtime', side_effect=prepare), patch.object(driver, 'execute_runtime', side_effect=launch), patch.object(driver, 'inspect_attempt', return_value={'identity': {'run_id': 'initial0'}, 'passed': False, 'outcome': 'unknown', 'costs': {}, 'notes': []}), patch.object(driver, 'inspect_progress', return_value={'value': None}), patch.object(driver, 'evaluate', return_value={'passed': False}):
                driver.execute_matrix(path)
            phase = driver.read(root / 'phase.json')
            costs = phase['release_trial_charges'][0]['costs']
            self.assertGreater(costs['wall_seconds'], 0)
            self.assertEqual(costs['gpu_seconds'], costs['wall_seconds'])
            self.assertNotIn('model_requests', costs)
            self.assertTrue(phase['unresolved_costs'])
            self.assertEqual(driver.read(root / 'execution-record.json')['runtime_start_attempts'], 1)



class Release08DriverTest(unittest.TestCase):
    def fixture(self, root):
        path = ReleaseDriverTest().fixture(root)
        plan = driver.read(path)
        protocol = root / 'protocol.json'
        atomic_json(protocol, {'protocol_id': 'rsi-libero-pro-5a-release08-four-seed3',
                    'anchor_bundle_id': 'initial-medium-5a-release08',
                    'authorization_scope': driver.RELEASE08_SCOPE, 'comparison_only': True})
        atomic_json(plan['profile'], {'profile_id': 'matched-5a-release08-20261113'})
        for slot in plan['slots'][:4]:
            slot['bundle_id'] = 'initial-medium-5a-release08'
        plan.update(protocol=str(protocol), comparison_only=True, campaign_id='test-release08',
                    max_runtime_start_attempts=10, max_technical_recoveries=2, max_recoveries_per_slot=1,
                    inheritance_reserve=dict(plan['slot_envelope']),
                    offline_reserve={'wall_seconds': 600, 'storage_bytes': 268435456},
                    dynamic_envelope={k: v*10 for k, v in plan['slot_envelope'].items()})
        phase = driver.read(root / 'phase.json')
        phase['limits'] = {k: v*11 + plan['offline_reserve'].get(k, 0) for k, v in plan['slot_envelope'].items()}
        phase['remaining'] = dict(phase['limits'])
        atomic_json(root / 'phase.json', phase)
        atomic_json(root / 'phase-before.json', phase)
        atomic_json(path, plan)
        return path

    def run_fake(self, path, failed_runs=(), shared_fault=False, cleanup=True, missing_cost=False,
                 resume=False, progress=True, evaluation_error=False):
        def prepare(repo, artifacts, **kw):
            return {'blockers': [], 'context': {'limits': {'wall_seconds': 900}, 'protocol_id': 'p',
                    'run_id': kw['run_id'], 'episode_id': kw['run_id'] + '-episode',
                    'bundle_id': kw['bundle_id'], 'output_dir': str(kw['output_dir']), 'collection': {},
                    'code_ref': 'v', 'theta_ref': 't', 'processor_ref': 'p', 'memory_ref': 'm'}}
        def launch(prepared):
            context = prepared['context']
            out = Path(context['output_dir']); out.mkdir(parents=True)
            identity = {k: context[k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
            atomic_json(out / 'runtime_context.json', context)
            atomic_json(out / 'supervisor.json', {'identity': identity, 'cleanup_confirmed': cleanup})
            atomic_json(out / 'request_gate.json', {'identity': identity, 'cleanup_complete': cleanup, 'active_requests_at_close': 0})
        def audit(output, **kwargs):
            context = driver.read(Path(output) / 'runtime_context.json')
            failed = context['run_id'] in failed_runs
            costs = {'whole_job_wall_seconds': 1, 'gpu_device_seconds_envelope': 1, 'model_requests': 1,
                     'tool_calls': 1, 'environment_steps': 30, 'artifact_bytes': 1}
            if missing_cost and failed:
                del costs['model_requests']
            return {'identity': {k: context[k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')},
                    'passed': not failed, 'checks': {**dict.fromkeys(('identity', 'runtime_source_snapshot', 'startup_input_identity', 'actual_model_configuration', 'model_reference', 'vla_actual_loading_and_prediction', 'model_transport', 'tool_budget', 'request_budget', 'step_budget', 'wall_gpu_storage_budget', 'all_tools_use_counted_dispatcher'), True), 'model_reference': not (failed and shared_fault), 'execution_health': not failed},
                    'outcome': 'unknown' if failed else 'failure', 'costs': costs}
        with patch.object(driver, 'release08_gpu_admission', return_value={'selected': 0}), \
                patch.object(driver, 'prepare_runtime', side_effect=prepare), \
                patch.object(driver, 'execute_runtime', side_effect=launch) as executor, \
                patch.object(driver, 'inspect_attempt', side_effect=audit), \
                patch.object(driver, 'inspect_progress', return_value={'value': progress}), \
                patch.object(driver, 'evaluate', side_effect=evaluation_error if callable(evaluation_error) else ValueError('synthetic evaluator unavailable') if evaluation_error else None, return_value={'passed': True, 'baseline_J': 1}):
            driver.execute_matrix(path, resume=resume)
        return executor.call_count, driver.read(path)

    def test_progress_evaluation_pause_retains_charge_and_resume_starts_only_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            starts, plan = self.run_fake(path, progress=None)
            self.assertEqual(starts, 1)
            self.assertEqual(plan['status'], 'evaluation_paused')
            first = plan['slots'][0]['attempts'][0]
            costs = dict(first['costs'])
            phase = driver.read(plan['phase_snapshot_ref'])
            self.assertEqual(phase['used']['simulation_episodes'], 1)
            self.assertEqual(len(phase['release_trial_charges']), 1)
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 7)
            self.assertEqual(final['slots'][0]['attempts'][0]['costs'], costs)
            self.assertEqual(final['slots'][0]['attempts'][0]['run_id'], first['run_id'])
            phase = driver.read(plan['phase_snapshot_ref'])
            self.assertEqual(phase['used']['simulation_episodes'], 8)
            self.assertEqual(len(phase['release_trial_charges']), 8)
            with patch.object(driver, 'execute_runtime') as launch:
                with self.assertRaisesRegex(ValueError, 'closed campaigns'):
                    driver.execute_matrix(path, resume=True)
                launch.assert_not_called()

    def test_initial_evaluator_exception_saves_four_completed_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            starts, plan = self.run_fake(path, evaluation_error=True)
            self.assertEqual(starts, 4)
            self.assertEqual(plan['status'], 'evaluation_paused')
            self.assertTrue(all(s.get('selected_attempt') for s in plan['slots'][:4]))
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 4)
            self.assertEqual(final['status'], 'completed_dynamic_paused')
            self.assertEqual(driver.read(path.parent / 'execution-record.json')['runtime_start_attempts'], 8)

    def test_unknown_submission_never_relaunches_or_recharges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            plan['slots'][0]['attempts'][0]['status'] = 'started'
            (path.parent / 'slots' / plan['slots'][0]['attempts'][0]['run_id'] / 'runtime-returned.json').unlink()
            atomic_json(path, plan)
            phase = driver.read(plan['phase_snapshot_ref'])
            with patch.object(driver, 'release08_gpu_admission') as gpu, patch.object(driver, 'execute_runtime') as launch:
                with self.assertRaisesRegex(ValueError, 'submission status unknown'):
                    driver.execute_matrix(path, resume=True)
                gpu.assert_not_called()
                launch.assert_not_called()
            self.assertEqual(driver.read(plan['phase_snapshot_ref']), phase)

    def test_execution_lock_blocks_duplicate_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            with path.with_suffix('.execution.lock').open('a') as lock:
                driver.fcntl.flock(lock, driver.fcntl.LOCK_EX | driver.fcntl.LOCK_NB)
                with patch.object(driver, 'execute_runtime') as launch:
                    with self.assertRaisesRegex(ValueError, 'already active'):
                        driver.execute_matrix(path, resume=True)
                    launch.assert_not_called()

    def test_final_evaluator_pause_resumes_evaluation_without_new_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            def evaluation(plan, root, final=False):
                if final:
                    raise ValueError('synthetic final evaluator unavailable')
                return {'passed': True}
            starts, plan = self.run_fake(path, evaluation_error=evaluation)
            self.assertEqual(starts, 8)
            self.assertEqual(plan['status'], 'evaluation_paused')
            self.assertTrue(plan['final_evaluation_pending'])
            before = driver.read(plan['phase_snapshot_ref'])['release_trial_charges']
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 0)
            self.assertEqual(final['status'], 'completed_dynamic_paused')
            self.assertEqual(driver.read(plan['phase_snapshot_ref'])['release_trial_charges'], before)

    def test_wrong_returned_receipt_identity_pauses_without_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            attempt = plan['slots'][0]['attempts'][0]
            receipt_path = path.parent / 'slots' / attempt['run_id'] / 'runtime-returned.json'
            receipt = driver.read(receipt_path)
            receipt['identity']['run_id'] = 'another-run'
            atomic_json(receipt_path, receipt)
            starts, paused = self.run_fake(path, resume=True)
            self.assertEqual(starts, 0)
            self.assertEqual(paused['status'], 'evaluation_paused')
            self.assertIn('receipt identity', paused['stop_reason'])
            self.assertEqual(len(driver.read(plan['phase_snapshot_ref'])['release_trial_charges']), 1)

    def test_existing_charge_survives_checkpoint_write_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            attempt = plan['slots'][0]['attempts'][0]
            original = attempt.pop('costs')
            atomic_json(path, plan)
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 7)
            self.assertEqual(final['slots'][0]['attempts'][0]['costs'], original)
            self.assertEqual(len(driver.read(plan['phase_snapshot_ref'])['release_trial_charges']), 8)

    def test_returned_receipt_reconciles_interrupted_started_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            plan['slots'][0]['attempts'][0]['status'] = 'started'
            atomic_json(path, plan)
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 7)
            self.assertEqual(final['status'], 'completed_dynamic_paused')
            self.assertEqual(len(driver.read(plan['phase_snapshot_ref'])['release_trial_charges']), 8)

    def test_final_evaluation_versions_preserve_earlier_outputs_and_execution_binding(self):
        from dataclasses import make_dataclass
        Decision = make_dataclass('Decision', [('verdict', str)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = self.fixture(root); plan = driver.read(path)
            old_gate = {'decision': {'verdict': 'reject'}, 'historical': True}
            atomic_json(root / 'independent-gate.json', old_gate)
            candidate = SimpleNamespace(base_bundle_id='initial', bundle_id='new', candidate_id='c')
            with patch.object(driver, 'ResearchStore') as store, \
                    patch.object(driver, 'Candidate', return_value=candidate), \
                    patch.object(driver, 'select_comparison_attempts'), \
                    patch.object(driver, 'evaluate_matrix', return_value=SimpleNamespace(result_id='r', intervention={})), \
                    patch.object(driver, 'inspect_initial_comparison_result', return_value={'passed': True}), \
                    patch.object(driver, 'inspect_initial_result', return_value={'passed': False}), \
                    patch.object(driver, 'EvaluationStore') as evaluations, \
                    patch.object(driver, 'Gate') as gate, patch.object(driver, 'score', return_value=1):
                store.return_value.get.return_value = {}
                gate.return_value.evaluate.return_value = Decision('reject')
                driver.evaluate(plan, root, final=True)
                first = root / 'evaluation-revisions/read-default-sdk-closure-v2/independent-gate.json'
                saved = first.read_bytes()
                plan['evaluation_version'] = 'v3-offline-example'
                driver.evaluate(plan, root, final=True)
            self.assertEqual(driver.read(root / 'independent-gate.json'), old_gate)
            self.assertEqual(first.read_bytes(), saved)
            current = driver.read(root / 'current-evaluation.json')
            self.assertEqual(current['version'], 'v3-offline-example')
            self.assertEqual(current['execution_source_manifest_ref'], plan['source_manifest_ref'])
            self.assertEqual(evaluations.call_args.args[0], root / 'evaluation-revisions/v3-offline-example/evaluation')

    def test_completed_attempt_survives_interruption_before_slot_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            plan['slots'][0]['attempts'][0].update(status='completed', validity='valid', outcome='failure',
                                                  critical_progress=True, fault_scope=None)
            atomic_json(path, plan)
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 7)
            self.assertEqual(len(final['slots'][0]['attempts']), 1)
            self.assertEqual(final['slots'][0]['selected_attempt']['status'], 'completed')

    def test_shared_fault_checkpoint_cannot_trigger_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, plan = self.run_fake(path, progress=None)
            plan['slots'][0]['attempts'][0].update(status='technical_stop', validity='invalid',
                                                  fault_scope='unresolved_or_shared')
            atomic_json(path, plan)
            starts, final = self.run_fake(path, resume=True)
            self.assertEqual(starts, 0)
            self.assertEqual(final['status'], 'stopped_dynamic_paused')
            self.assertEqual(len(final['slots'][0]['attempts']), 1)

    def authorize_cleaned_baseline(self, path):
        plan = driver.read(path)
        phase = driver.read(plan['phase_snapshot_ref'])
        phase['limits']['storage_bytes'] += 756257057
        phase['used']['storage_bytes'] += 756257057
        phase['derived_storage_charges'] = [{'key': 'release08-temporary-historical-baseline', 'storage_bytes': 756257057}]
        atomic_json(plan['phase_snapshot_ref'], phase)
        exception = {'authorized': True, 'authorization_scope': driver.RELEASE08_SCOPE,
                     'exception_kind': 'cleaned_temporary_historical_baseline', 'actual_bytes': 756257057,
                     'phase_snapshot_ref': plan['phase_snapshot_ref'],
                     'phase_charge_key': 'release08-temporary-historical-baseline',
                     'all_costs_retained': True, 'temporary_baseline_removed': True,
                     'dynamic_continuation_authorized': True}
        reference = path.parent / 'budget-exception.json'
        atomic_json(reference, exception)
        plan.update(budget_exception_ref=str(reference), offline_actual_storage_bytes=756257057,
                    status='stopped_dynamic_paused', evidence_root=str(path.parent))
        atomic_json(path, plan)
        atomic_json(path.parent / 'execution-record.json', {'runtime_start_attempts': 0, 'status': 'stopped_dynamic_paused'})
        atomic_json(path.parent / 'offline-budget-stop.json', {'stop_new_dynamic_starts': True})
        return plan, phase, exception

    def test_exact_cleaned_baseline_exception_allows_first_matrix_without_refund(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            plan, phase, exception = self.authorize_cleaned_baseline(path)
            starts, final = self.run_fake(path)
            self.assertEqual(starts, 8)
            self.assertEqual(final['status'], 'completed_dynamic_paused')
            self.assertGreaterEqual(driver.read(plan['phase_snapshot_ref'])['used']['storage_bytes'], 756257057)
            self.assertTrue(driver.read(path.parent / 'offline-budget-stop.json')['stop_new_dynamic_starts'])
            with patch.object(driver, 'release08_gpu_admission') as gpu:
                with self.assertRaisesRegex(ValueError, 'cannot restore|already entered execution'):
                    driver.execute_matrix(path)
                gpu.assert_not_called()

    def test_exception_only_excludes_baseline_and_counts_all_other_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            plan, phase, exception = self.authorize_cleaned_baseline(path)
            plan['offline_actual_storage_bytes'] += 17
            phase['release_derived_storage_charges'] = [{'root': str(path.parent), 'storage_bytes': 19}]
            before = json.dumps(phase, sort_keys=True)
            remaining = driver.offline_remaining(plan, phase)
            self.assertEqual(remaining['storage_bytes'], 268435456 - 36)
            self.assertEqual(json.dumps(phase, sort_keys=True), before)
            plan['offline_actual_storage_bytes'] += 268435456
            with self.assertRaisesRegex(ValueError, 'offline reserve exceeded'):
                driver.offline_remaining(plan, phase)

    def test_exception_rejects_wrong_fields_unpaid_costs_or_started_attempts(self):
        wrong_values = {'authorized': False, 'authorization_scope': 'old', 'exception_kind': 'generic',
                        'actual_bytes': 756257058, 'phase_snapshot_ref': '/tmp/wrong-phase.json',
                        'phase_charge_key': 'unrelated-storage', 'all_costs_retained': False,
                        'temporary_baseline_removed': False, 'dynamic_continuation_authorized': False}
        for field, value in wrong_values.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp)); plan, phase, exception = self.authorize_cleaned_baseline(path)
                exception[field] = value; atomic_json(plan['budget_exception_ref'], exception)
                with patch.object(driver, 'release08_gpu_admission') as gpu:
                    with self.assertRaisesRegex(ValueError, 'invalid cleaned'):
                        driver.execute_matrix(path)
                    gpu.assert_not_called()
        for invalid in ('unpaid', 'duplicate_charge', 'wrong_charge_bytes', 'external_underreported', 'started_record', 'started_slot', 'started_attempt'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp)); plan, phase, exception = self.authorize_cleaned_baseline(path)
                if invalid == 'unpaid':
                    phase['used']['storage_bytes'] = 0; atomic_json(plan['phase_snapshot_ref'], phase)
                elif invalid == 'duplicate_charge':
                    phase['derived_storage_charges'] *= 2; atomic_json(plan['phase_snapshot_ref'], phase)
                elif invalid == 'wrong_charge_bytes':
                    phase['derived_storage_charges'][0]['storage_bytes'] += 1; atomic_json(plan['phase_snapshot_ref'], phase)
                elif invalid == 'external_underreported':
                    plan['offline_actual_storage_bytes'] -= 1; atomic_json(path, plan)
                elif invalid == 'started_record':
                    atomic_json(path.parent / 'execution-record.json', {'runtime_start_attempts': 1})
                elif invalid == 'started_slot':
                    plan['slots'][0]['status'] = 'started'; atomic_json(path, plan)
                else:
                    plan['slots'][0]['attempts'] = [{'status': 'started'}]; atomic_json(path, plan)
                with patch.object(driver, 'release08_gpu_admission') as gpu:
                    with self.assertRaisesRegex(ValueError, 'invalid cleaned|cannot restore|already entered execution'):
                        driver.execute_matrix(path)
                    gpu.assert_not_called()

    def test_valid_failure_and_baseline_one_complete_eight_without_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            starts, plan = self.run_fake(path)
            self.assertEqual(starts, 8)
            self.assertEqual(plan['status'], 'completed_dynamic_paused')
            self.assertTrue(all(s['outcome'] == 'failure' and len(s['attempts']) == 1 for s in plan['slots']))
            self.assertEqual(driver.read(path.parent / 'execution-record.json')['technical_recoveries'], 0)

    def test_two_local_technical_recoveries_preserve_costs_and_select_valid_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            plan = driver.read(path)
            failures = [plan['slots'][0]['run_id'], plan['slots'][1]['run_id']]
            starts, plan = self.run_fake(path, failures)
            self.assertEqual(starts, 10)
            self.assertEqual(plan['status'], 'completed_dynamic_paused')
            for slot in plan['slots'][:2]:
                self.assertEqual(len(slot['attempts']), 2)
                self.assertEqual(slot['attempts'][0]['validity'], 'invalid')
                self.assertEqual(slot['selected_attempt']['run_id'], slot['attempts'][1]['run_id'])
                permits = [driver.read(a['permit_ref']) for a in slot['attempts']]
                self.assertNotEqual(permits[0]['output_dir'], permits[1]['output_dir'])
                self.assertEqual(permits[1]['recovery_of_run_id'], permits[0]['run_id'])
            phase = driver.read(path.parent / 'phase.json')
            self.assertEqual(len(phase['release_trial_charges']), 10)
            self.assertEqual(phase['used']['simulation_episodes'], 10)
            self.assertGreaterEqual(phase['remaining']['simulation_episodes'], 1)

    def test_localized_slot_fault_continues_other_slots_after_single_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            run = driver.read(path)['slots'][0]['run_id']
            starts, plan = self.run_fake(path, [run, run+'-recovery1'])
            self.assertEqual(starts, 9)
            self.assertEqual(len(plan['slots'][0]['attempts']), 2)
            self.assertTrue(plan['slots'][0]['localized_failure_retained'])
            self.assertEqual(sum(s['status'] == 'unstarted' for s in plan['slots']), 0)

    def test_global_recovery_exhaustion_retains_local_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp)); before = driver.read(path)
            failures = [s['run_id'] for s in before['slots'][:3]]
            starts, plan = self.run_fake(path, failures)
            self.assertEqual(starts, 10)
            self.assertTrue(plan['slots'][2]['localized_failure_retained'])
            self.assertEqual(len(plan['slots'][2]['attempts']), 1)
            self.assertEqual(sum(s['status'] == 'unstarted' for s in plan['slots']), 0)
            eligibility = list(path.parent.glob('slots/*/recovery-eligibility.json'))
            self.assertTrue(any(driver.read(p)['eligible'] for p in eligibility))

    def test_unselected_confirmed_violation_reaches_gate_despite_missing_slots(self):
        fixture_spec = importlib.util.spec_from_file_location('gate_fixture', ROOT / 'core/tests/test_gate_tristate.py')
        fixture_module = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture_module)
        fixture = fixture_module.GateTristateTests(); fixture.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempts = [{'run_id': name, 'episode_id': name+'-episode', 'output': str(root/name),
                         'costs': dict.fromkeys(driver.COST_FIELDS.values(), 2)} for name in ('early', 'later')]
            slot = {'side': 'candidate', 'task_key': 'old_milk', 'seed': 3,
                    'bundle_id': fixture.candidate.bundle_id, 'attempts': attempts}
            plan = {'slots': [slot]}
            def matrix(protocol, profile, manifest, **kwargs):
                return SimpleNamespace(assignments=[{'task_key': 'old_milk', 'seed': 3,
                       'validity': 'invalid', 'outcome': 'unknown'}], hard_constraints={'safety': False})
            def audit(output, **kwargs):
                name = Path(output).name
                return {'passed': False, 'checks': {'identity': True},
                        'confirmed_violations': ['safety'] if name == 'later' else [],
                        'identity': {'run_id': name, 'episode_id': name+'-episode',
                            'bundle_id': slot['bundle_id'], 'protocol_id': fixture.p['protocol_id']}}
            with patch.object(driver, 'evaluate_matrix', side_effect=matrix), \
                    patch.object(driver, 'inspect_attempt', side_effect=audit), \
                    patch.object(driver, 'inspect_progress', return_value={'value': None}):
                driver.select_comparison_attempts(plan, root, fixture.p, {})
            self.assertIsNone(slot['final_selected_attempt'])
            self.assertEqual(plan['final_attempt_violations'][0]['run_id'], 'later')
            candidate = fixture.result(fixture.candidate.bundle_id, incomplete=True)
            driver.apply_comparison_attempt_evidence(plan, 'candidate', candidate)
            self.assertEqual(candidate.costs['model_requests'], 4)
            self.assertIs(candidate.hard_constraints['safety'], False)
            self.assertEqual(fixture.gate(candidate).verdict, 'reject')

    def test_unselected_progress_false_rejects_and_unknown_stays_inconclusive(self):
        fixture_spec = importlib.util.spec_from_file_location('gate_fixture', ROOT / 'core/tests/test_gate_tristate.py')
        fixture_module = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture_module)
        fixture = fixture_module.GateTristateTests(); fixture.setUp()
        for progress_value, verdict in ((False, 'reject'), (None, 'inconclusive')):
            with self.subTest(progress=progress_value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                attempt = {'run_id': 'candidate-attempt', 'episode_id': 'candidate-episode',
                           'output': str(root / 'attempt'), 'costs': dict.fromkeys(driver.COST_FIELDS.values(), 1)}
                slot = {'side': 'candidate', 'task_key': 'old_milk', 'seed': 3,
                        'bundle_id': fixture.candidate.bundle_id, 'attempts': [attempt]}
                plan = {'slots': [slot]}
                identity = {'run_id': attempt['run_id'], 'episode_id': attempt['episode_id'],
                            'bundle_id': slot['bundle_id'], 'protocol_id': fixture.p['protocol_id']}
                evaluated = SimpleNamespace(assignments=[{'task_key': 'old_milk', 'seed': 3,
                             'validity': 'valid', 'outcome': 'failure'}],
                             hard_constraints={'critical_progress': progress_value})
                with patch.object(driver, 'evaluate_matrix', return_value=evaluated), \
                        patch.object(driver, 'inspect_attempt', return_value={'passed': True,
                            'checks': {'identity': True}, 'identity': identity, 'confirmed_violations': []}), \
                        patch.object(driver, 'inspect_progress', return_value={'value': progress_value}):
                    driver.select_comparison_attempts(plan, root, fixture.p, {})
                self.assertIsNone(slot['final_selected_attempt'])
                self.assertEqual([v['constraint'] for v in plan['final_attempt_violations']],
                                 ['critical_progress'] if progress_value is False else [])
                candidate = fixture.result(fixture.candidate.bundle_id, incomplete=True)
                driver.apply_comparison_attempt_evidence(plan, 'candidate', candidate)
                self.assertEqual(fixture.gate(candidate).verdict, verdict)

    def test_comparison_check_and_qualification_are_saved_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = self.fixture(root); plan = driver.read(path)
            candidate = SimpleNamespace(base_bundle_id='initial', bundle_id='new', candidate_id='c')
            comparison = {'passed': True, 'reason': 'technically comparable'}
            qualification = {'passed': False, 'reason': 'old capability failure', 'failure_kind': 'capability'}
            with patch.object(driver, 'ResearchStore') as store, \
                    patch.object(driver, 'Candidate', return_value=candidate), \
                    patch.object(driver, 'select_comparison_attempts'), \
                    patch.object(driver, 'evaluate_matrix', return_value=SimpleNamespace(result_id='r', intervention={})), \
                    patch.object(driver, 'inspect_initial_comparison_result', return_value=comparison), \
                    patch.object(driver, 'inspect_initial_result', return_value=qualification):
                store.return_value.get.return_value = {}
                returned = driver.evaluate(plan, root)
            self.assertEqual(returned, comparison)
            self.assertEqual(driver.read(root / 'evaluation-revisions/read-default-sdk-closure-v2/initial-comparison-check.json'), comparison)
            self.assertEqual(driver.read(root / 'evaluation-revisions/read-default-sdk-closure-v2/initial-qualification-check.json'), qualification)

    def test_final_attempt_selection_rechecks_earliest_failure_without_recharging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempts = [{'run_id': 'early', 'episode_id': 'e1', 'output': str(root/'early'), 'costs': {'model_requests': 3}},
                        {'run_id': 'later', 'episode_id': 'e2', 'output': str(root/'later'), 'costs': {'model_requests': 5}}]
            slot = {'task_key': 'old_milk', 'seed': 3, 'bundle_id': 'initial', 'attempts': attempts,
                    'selected_attempt': attempts[1]}
            plan = {'slots': [slot]}
            def matrix(protocol, profile, manifest, **kwargs):
                return SimpleNamespace(assignments=[{'task_key': 'old_milk', 'seed': 3, 'validity': 'valid',
                       'outcome': 'failure' if manifest[0]['run_id'] == 'early' else 'success'}])
            with patch.object(driver, 'evaluate_matrix', side_effect=matrix) as evaluator, \
                    patch.object(driver, 'inspect_attempt', return_value={'passed': True, 'identity': {}}), \
                    patch.object(driver, 'inspect_progress', return_value={'value': True}), \
                    patch.object(driver, '_phase_charge') as charge:
                driver.select_comparison_attempts(plan, root, {}, {})
            self.assertEqual(evaluator.call_count, 2)
            self.assertEqual(slot['final_selected_attempt']['run_id'], 'early')
            self.assertEqual(slot['final_selected_attempt']['outcome'], 'failure')
            self.assertEqual([a['costs']['model_requests'] for a in attempts], [3, 5])
            charge.assert_not_called()

    def test_shared_cleanup_and_unknown_cost_faults_never_recover(self):
        for options in ({'shared_fault': True}, {'cleanup': False}, {'missing_cost': True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp))
                run = driver.read(path)['slots'][0]['run_id']
                starts, plan = self.run_fake(path, [run], **options)
                self.assertEqual(starts, 1)
                self.assertEqual(plan['status'], 'stopped_dynamic_paused')

    def test_budget_reserves_two_recoveries_and_remaining_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp)); plan = driver.read(path)
            phase = driver.read(path.parent / 'phase.json')
            driver.check_budget(plan, phase, remaining_slots=8, remaining_recoveries=2)
            phase['remaining']['wall_seconds'] -= 1
            with self.assertRaisesRegex(ValueError, 'inheritance reserve'):
                driver.check_budget(plan, phase, remaining_slots=8, remaining_recoveries=2)

    def test_wrong_scope_and_reverse_binding_rejected_before_gpu(self):
        for mutation in ('scope', 'comparison'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp)); plan = driver.read(path)
                if mutation == 'scope':
                    protocol = driver.read(plan['protocol']); protocol['authorization_scope'] = 'historical'
                    atomic_json(plan['protocol'], protocol)
                else:
                    plan['comparison_only'] = False; atomic_json(path, plan)
                with patch.object(driver, 'release08_gpu_admission') as gpu:
                    with self.assertRaisesRegex(ValueError, 'scoped protocol|comparison-only plan'):
                        driver.execute_matrix(path)
                    gpu.assert_not_called()

    def test_budget_stop_and_external_offline_cost_refuse_before_admission(self):
        for stop in ('file', 'status', 'storage'):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp)); plan = driver.read(path)
                if stop == 'file':
                    atomic_json(path.parent / 'offline-budget-stop.json', {'stop_new_dynamic_starts': True})
                elif stop == 'status':
                    plan['status'] = 'stopped_dynamic_paused'
                else:
                    plan['offline_actual_storage_bytes'] = 268435457
                atomic_json(path, plan)
                with patch.object(driver, 'release08_gpu_admission') as gpu, patch.object(driver, 'prepare_runtime') as prepare:
                    with self.assertRaisesRegex(ValueError, 'cannot restore|offline reserve exceeded'):
                        driver.execute_matrix(path)
                    gpu.assert_not_called()
                    prepare.assert_not_called()

    def test_gpu_wait_uses_cumulative_limit_and_offline_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = self.fixture(root); plan = driver.read(path)
            plan['evidence_root'] = str(root)
            now = [0.0]
            def sleep(seconds):
                self.assertLessEqual(seconds, 30)
                now[0] += seconds
            with patch.object(driver, 'gpu_admission', return_value={'selected': None}) as query, \
                    patch.object(driver.time, 'perf_counter', side_effect=lambda: now[0]), \
                    patch.object(driver.time, 'sleep', side_effect=sleep):
                admission = driver.release08_gpu_admission(plan, root, 'first')
                self.assertEqual(admission['cumulative_wait_seconds'], 300)
                self.assertEqual(query.call_count, 11)
                driver.release08_gpu_admission(plan, root, 'second')
                self.assertEqual(query.call_count, 12)
            self.assertEqual(driver.read(root / 'phase.json')['used']['wall_seconds'], 300)


def test_launcher_postprocessing_failure_reaches_independent_consumer(tmp_path, monkeypatch):
    from tests.unit_tests.robots.libero.test_execution_result_chain import test_tool_events_reach_final_result_and_consumers
    from rsi_vla.adapters.runtime_acceptance import inspect_attempt
    test_tool_events_reach_final_result_and_consumers(tmp_path, monkeypatch, 'success', 'success', 'completed')
    run = tmp_path / 'run'
    raw = (run / 'result.json').read_bytes()
    atomic_json(run / 'launcher_failure.json', {'error': 'postprocessing failed'})
    report = inspect_attempt(run, require_vla=False)
    assert report['checks']['launcher_wrapper_completion'] is False
    assert report['passed'] is False
    assert (run / 'result.json').read_bytes() == raw
