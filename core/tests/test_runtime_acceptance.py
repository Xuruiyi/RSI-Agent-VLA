"""SDK/L0 accounting and native persistence regressions, without a Runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.adapters.runtime_acceptance import inspect_attempt

from rsi_vla.adapters.runtime_acceptance import audit_native_persistence, audit_sdk_tools


class ToolAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(run_id='run', episode_id='episode', bundle_id='bundle', protocol_id='protocol')

    def stream(self, events):
        (self.root/'codex_fixture.stream.jsonl').write_text(
            ''.join(json.dumps(event)+'\n' for event in events))

    def sdk(self, item_id='a', method='item/started', **updates):
        item = dict(id=item_id, type='mcpToolCall', server='rpent', tool='vla_execute',
                    status='inProgress', started_at_ms=1000)
        item.update(updates)
        return dict(method=method, payload=dict(item=item))

    def span(self, tool='vla_execute', sequence=1, start_ms=1001, status='completed'):
        return [dict(self.identity, kind='tool_call', tool=tool, sequence=sequence,
                     utc_ns=start_ms*1_000_000),
                dict(self.identity, kind='tool_end', tool=tool, sequence=sequence+1,
                     utc_ns=(start_ms+5)*1_000_000, status=status)]

    def test_actual_release_shape_keeps_two_started_and_one_completed(self):
        # The real stream starts view_env_state and vla_execute but delivers
        # only the former completion before the SDK's generic local 429 error.
        self.stream([self.sdk('view', tool='view_env_state', started_at_ms=1000),
                     self.sdk('view', 'item/completed', tool='view_env_state',
                              started_at_ms=1000, completed_at_ms=1010, status='completed'),
                     self.sdk('vla', started_at_ms=1020)])
        audit = audit_sdk_tools(self.root, self.identity,
                               self.span('view_env_state') + self.span(sequence=3, start_ms=1021))
        self.assertEqual(audit['issues'], [])
        self.assertEqual(audit['counted_tool_calls'], 2)
        self.assertEqual(sum(tool['sdk_started'] for tool in audit['tools']), 2)
        self.assertEqual(sum(tool['sdk_completed'] for tool in audit['tools']), 1)
        self.assertEqual(audit['sdk_terminal_missing'], ['vla'])
        self.assertEqual(audit['tools'][1]['l0_terminal_sequence'], 4)
        self.assertFalse(audit['tools'][1]['sdk_completed'])

    def test_failed_and_incomplete_sdk_items_still_count_once(self):
        for status in ('failed', 'incomplete'):
            with self.subTest(status=status):
                self.stream([self.sdk(), self.sdk(method='item/completed', status=status,
                                                  completed_at_ms=1010)])
                audit = audit_sdk_tools(self.root, self.identity, self.span())
                self.assertEqual(audit['counted_tool_calls'], 1)
                self.assertTrue(any(status in issue for issue in audit['issues']))

    def test_started_external_failed_tool_is_counted(self):
        self.stream([self.sdk(type='commandExecution', server=None, status='failed')])
        audit = audit_sdk_tools(self.root, self.identity, [])
        self.assertEqual(audit['counted_tool_calls'], 1)
        self.assertEqual(len(audit['outside_tools']), 1)
        self.assertTrue(audit['issues'])

    def test_missing_l0_or_sdk_coverage_cannot_pass(self):
        self.stream([self.sdk()])
        for events in ([], self.span()+self.span('view_env_state', sequence=3, start_ms=1021)):
            with self.subTest(events=events):
                self.assertTrue(audit_sdk_tools(self.root, self.identity, events)['issues'])

    def test_wrong_identity_or_ambiguous_same_tool_cannot_be_associated(self):
        self.stream([self.sdk()])
        wrong = [dict(event, bundle_id='other') for event in self.span()]
        self.assertTrue(audit_sdk_tools(self.root, self.identity, wrong)['issues'])
        ambiguous = self.span()+self.span(sequence=3, start_ms=1021)
        self.assertTrue(audit_sdk_tools(self.root, self.identity, ambiguous)['issues'])

    def test_sequential_calls_to_same_tool_can_be_reconciled(self):
        self.stream([self.sdk(), self.sdk(method='item/completed', completed_at_ms=1010, status='completed'),
                     self.sdk('b', started_at_ms=1020)])
        audit = audit_sdk_tools(self.root, self.identity, self.span()+self.span(sequence=3, start_ms=1021))
        self.assertEqual(audit['issues'], [])
        self.assertEqual(audit['counted_tool_calls'], 2)

    def test_ambiguous_association_keeps_union_and_known_call_lower_bound(self):
        self.stream([self.sdk(), self.sdk('b')])
        audit = audit_sdk_tools(self.root, self.identity, self.span()+self.span(sequence=3, start_ms=1021))
        self.assertTrue(audit['issues'])
        self.assertEqual(audit['counted_tool_calls'], 4)
        self.assertEqual(audit['minimum_tool_calls'], 2)

    def test_uncovered_different_tool_or_disjoint_time_retains_two_known_calls(self):
        self.stream([self.sdk()])
        for events in (self.span('view_env_state'), self.span(start_ms=900)):
            with self.subTest(events=events):
                audit = audit_sdk_tools(self.root, self.identity, events)
                self.assertTrue(audit['issues'])
                self.assertEqual(audit['counted_tool_calls'], 2)
                self.assertEqual(audit['minimum_tool_calls'], 2)

    def test_explicit_failed_event_counts_without_completed_event(self):
        self.stream([self.sdk(), self.sdk(method='item/failed')])
        audit = audit_sdk_tools(self.root, self.identity, self.span())
        self.assertEqual(audit['counted_tool_calls'], 1)
        self.assertTrue(any('failed' in issue for issue in audit['issues']))

    def test_missing_other_artifacts_does_not_erase_external_tool_violation(self):
        (self.root/'runtime_context.json').write_text(json.dumps(self.identity))
        self.stream([self.sdk(type='commandExecution', server=None, status='failed')])
        audit = inspect_attempt(self.root)
        self.assertFalse(audit['passed'])
        self.assertIn('safety', audit['confirmed_violations'])
        self.assertFalse(audit['checks']['required_artifacts_complete'])

    def test_l0_failed_tool_remains_unresolved_and_counted(self):
        self.stream([self.sdk()])
        audit = audit_sdk_tools(self.root, self.identity, self.span(status='failed'))
        self.assertEqual(audit['counted_tool_calls'], 1)
        self.assertTrue(audit['issues'])

    def test_local_rejected_terminal_does_not_fabricate_admission_or_double_count(self):
        self.stream([self.sdk(), self.sdk(method='item/completed', status='failed', completed_at_ms=1010)])
        rejected = dict(self.identity, kind='tool_end', tool='vla_execute', sequence=3,
                        utc_ns=1005*1_000_000, status='rejected')
        audit = audit_sdk_tools(self.root, self.identity, [rejected])
        self.assertEqual(audit['issues'], [])
        self.assertEqual(audit['counted_tool_calls'], 1)
        self.assertEqual(audit['l0_tool_calls'], 0)
        self.assertEqual(audit['l0_rejected_tool_calls'], 1)
        self.assertFalse(audit['tools'][0]['l0_admitted'])
        self.assertIsNone(audit['tools'][0]['l0_start_sequence'])
        self.assertEqual(audit['tools'][0]['status'], 'failed')

    def test_rejected_sdk_attempt_above_call_cap_remains_resource_violation(self):
        context = dict(self.identity, limits=dict(environment_steps=20, tool_calls=1))
        self.stream([self.sdk(), self.sdk(method='item/completed', status='completed', completed_at_ms=1010),
                     self.sdk('rejected', started_at_ms=1020)])
        events = self.span()+[dict(self.identity, kind='tool_end', tool='vla_execute',
                                 sequence=3, utc_ns=1025*1_000_000, status='rejected')]
        (self.root/'action_events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
        (self.root/'environment_service').mkdir()
        (self.root/'environment_service/action_events.jsonl').write_text('')
        (self.root/'model_requests.jsonl').write_text('')
        documents = {
            'runtime_context.json': context,
            'result.json': dict(self.identity),
            'runtime_costs.json': dict(self.identity, tool_calls=1),
            'launcher_costs.json': dict(self.identity),
            'supervisor.json': dict(self.identity),
            'request_gate.json': dict(self.identity),
            'model_load.json': dict(self.identity),
            'environment_service/runtime_costs.json': dict(self.identity, construction_complete=True,
                reset_calls=1, reset_completed=1, setup_steps_confirmed=5, policy_steps_confirmed=5,
                service_closed=True, pending_action=False, environment_steps_unknown=0,
                environment_steps_confirmed=10),
        }
        def read_fixture(path):
            return documents[str(Path(path).relative_to(self.root))]
        with patch('rsi_vla.adapters.runtime_acceptance.read', side_effect=read_fixture):
            report = inspect_attempt(self.root)
        self.assertEqual(report['tool_audit']['issues'], [])
        self.assertEqual(report['tool_audit']['counted_tool_calls'], 2)
        self.assertIs(report['checks']['tool_budget'], False)
        self.assertIn('resource_limits', report['confirmed_violations'])

    def test_missing_model_config_is_unknown_but_observed_wrong_path_violates(self):
        context = dict(self.identity, limits=dict(environment_steps=20, tool_calls=10,
            model_requests=10, wall_seconds=100, gpu_seconds=100, storage_bytes=100000))
        documents = {
            'runtime_context.json': context,
            'result.json': dict(self.identity, benchmark_success=True, terminated=True),
            'environment_service/runtime_costs.json': dict(self.identity,
                construction_complete=True, reset_calls=1, reset_completed=1, setup_steps_confirmed=5,
                policy_steps_confirmed=5, service_closed=True, pending_action=False,
                environment_steps_unknown=0, environment_steps_confirmed=10, reset_wall_seconds=1),
            'runtime_costs.json': dict(self.identity, tool_calls=0),
            'launcher_costs.json': dict(self.identity, whole_job_wall_seconds=10,
                gpu_device_seconds_envelope=10, training_seconds=0, manual_seconds=0),
            'supervisor.json': dict(self.identity, status='completed', exit_code=0, cleanup_confirmed=True),
            'request_gate.json': dict(self.identity, requests_forwarded=1,
                hard_request_cap_enforced=True, active_requests_at_close=0),
            'model_load.json': dict(self.identity, loaded=True, embodiment='libero', action_chunk=5,
                prediction_count=1, last_actions_finite=True, last_action_shape=[5, 7],
                model_path='/model', norm_stats_path='/norm'),
            'launch_plan.json': dict(context=context, environment=dict(PI05_CHECKPOINT_PATH='/model',
                                                                      PI05_NORM_STATS_PATH='/norm')),
        }
        (self.root/'environment_service').mkdir()
        (self.root/'action_events.jsonl').write_text('')
        (self.root/'environment_service/action_events.jsonl').write_text('')
        (self.root/'model_requests.jsonl').write_text(json.dumps(dict(kind='request_admitted',
            request_number=1, model='gpt-6.1-sol', reasoning_effort='medium')))
        def read_fixture(path):
            return documents[str(Path(path).relative_to(self.root))]
        for mismatch in (False, True):
            documents['model_load.json']['norm_stats_path'] = '/other' if mismatch else '/norm'
            with patch('rsi_vla.adapters.runtime_acceptance.read', side_effect=read_fixture):
                report = inspect_attempt(self.root)
            self.assertIs(report['checks']['model_reference'], False if mismatch else None)
            self.assertEqual('compatibility' in report['confirmed_violations'], mismatch)

        # An ambiguous pairing keeps costs unknown, rather than charging the
        # unmatched union as if it were four definitely distinct calls.
        documents['model_load.json']['norm_stats_path'] = '/norm'
        documents['runtime_costs.json']['tool_calls'] = 2
        context['limits']['tool_calls'] = 2
        self.stream([self.sdk(), self.sdk('b')])
        (self.root/'action_events.jsonl').write_text(''.join(json.dumps(event)+'\n'
            for event in self.span()+self.span(sequence=3, start_ms=1021)))
        with patch('rsi_vla.adapters.runtime_acceptance.read', side_effect=read_fixture):
            report = inspect_attempt(self.root)
        self.assertNotIn('tool_calls', report['costs'])
        self.assertEqual(report['costs']['tool_calls_lower_bound'], 2)
        self.assertEqual(report['costs']['tool_calls_upper_bound'], 4)
        self.assertIsNone(report['checks']['tool_budget'])
        self.assertNotIn('resource_limits', report['confirmed_violations'])

        # Consume the producer's actual ambiguous-cost report in an isolated
        # phase ledger. Known dimensions still charge; absent tool_calls stays
        # unresolved and never enters the additive used vector as None.
        from rsi_vla.research_cycle import _phase_charge
        dimensions = dict.fromkeys(('wall_seconds', 'gpu_seconds', 'model_requests',
            'tool_calls', 'environment_steps', 'storage_bytes', 'simulation_episodes'), 1000000)
        phase_path = self.root/'phase.json'
        phase_path.write_text(json.dumps(dict(limits=dimensions,
            used=dict.fromkeys(dimensions, 0), remaining=dimensions.copy(), unresolved_costs=[])))
        permit = dict(phase_snapshot_ref=str(phase_path), permit_ref='offline-permit')
        _phase_charge(permit, 'ambiguous-run', report['costs'], self.root)
        phase = json.loads(phase_path.read_text())
        self.assertEqual(phase['used']['model_requests'], 1)
        self.assertEqual(phase['used']['wall_seconds'], 10)
        self.assertEqual(phase['used']['environment_steps'], 10)
        self.assertEqual(phase['used']['simulation_episodes'], 1)
        self.assertEqual(phase['used']['tool_calls'], 0)
        self.assertEqual(phase['unresolved_costs'], [dict(run_id='ambiguous-run',
            missing=['tool_calls'], output_ref=str(self.root))])
        self.assertNotIn('tool_calls', phase['development_trial_charges'][0]['costs'])

    def test_required_native_state_and_files_are_checked(self):
        marker = dict(identity=self.identity, sequence=7, step=1, reason='native_success')
        (self.root/'native_completion.json').write_text(json.dumps(marker))
        artifacts = ['agentview_policy.png', 'agentview.png', 'wrist.png']
        for name in artifacts:
            (self.root/name).mkdir()
            (self.root/name/'01.png').write_bytes(b'saved-observation')
        state = dict(step_idx=1, terminated=True, state={'robot0_eef_pos': [0, 0, 0]}, artifacts=artifacts)
        (self.root/'states.json').write_text(json.dumps(dict(steps=[state])))
        events = [dict(self.identity, kind='native_completion', reason='native_success', step=1, sequence=7)]
        self.assertEqual(audit_native_persistence(self.root, self.identity, events)['issues'], [])
        (self.root/'wrist.png/01.png').unlink()
        self.assertTrue(audit_native_persistence(self.root, self.identity, events)['issues'])
        marker['identity'] = dict(self.identity, run_id='other')
        (self.root/'native_completion.json').write_text(json.dumps(marker))
        self.assertTrue(audit_native_persistence(self.root, self.identity, events)['issues'])


if __name__ == '__main__':
    unittest.main()
