"""Non-actuating finish claims retain independent native outcomes and prerequisites."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rsi_vla.critical_progress import audit_finish, inspect_progress
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class FinishAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='finish-audit-', dir='/tmp/rsi-followup-main')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
        for name in ('rpent/tools/common.py', 'rpent/tools/toolkit.py',
                     'robots/libero/toolkit.py', 'robots/libero/robot_spec.py',
                     'rpent/planner/codex.py', 'robots/libero/result.py'):
            target = self.root/'code_snapshot/RPent'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/'RPent'/name, target)
        self.events = [dict(**self.identity, kind='run_start', sequence=0,
                            effective_rules={'finish_requires_native_success': False}),
                       dict(**self.identity, kind='tool_call', tool='finish', sequence=1, utc_ns=2000000),
                       dict(**self.identity, kind='tool_end', tool='finish', sequence=2, utc_ns=3000000, status='completed')]
        self.args = {'status': 'failure', 'summary': 'native still false'}
        start = dict(id='f', type='mcpToolCall', server='rpent', tool='finish',
                     arguments=self.args, status='inProgress', started_at_ms=1)
        end = {**start, 'status': 'completed', 'completed_at_ms': 4,
               'result': {'content': [{'type': 'text', 'text': json.dumps({'_finish': True, **self.args})}]}}
        self.sdk = [{'method': 'item/started', 'payload': {'item': start}},
                    {'method': 'item/completed', 'payload': {'item': end}}]
        atomic_json(self.root/'runtime_context.json', self.identity)
        atomic_json(self.root/'launch_plan.json', {'context': self.identity, 'command': ['python', '--am-backend', 'pi05']})
        self.raw = {**self.identity, 'planner_finish': {'_finish': True, **self.args},
                    'benchmark_success': False, 'attempt_closed': True,
                    'attempt_outcome_source': {'native_success': 'closed native worker'}}
        self.service = {'identity': self.identity, 'service_closed': True,
                        'pending_action': False, 'environment_steps_unknown': 0, 'last_native_success': False}

    def check(self):
        atomic_json(self.root/'result.json', self.raw)
        atomic_json(self.root/'environment_service/runtime_costs.json', self.service)
        (self.root/'codex_test.stream.jsonl').write_text('\n'.join(map(json.dumps, self.sdk)))
        return audit_finish(self.root, self.identity, self.events)

    def test_complete_claim_is_not_environment_success(self):
        self.assertIs(self.check()['value'], True)
        self.assertIs(self.raw['benchmark_success'], False)
        self.raw['planner_finish']['status'] = 'success'
        self.sdk[0]['payload']['item']['arguments'] = {'status': 'success', 'summary': self.args['summary']}
        self.sdk[1]['payload']['item']['arguments'] = copy.deepcopy(self.sdk[0]['payload']['item']['arguments'])
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = json.dumps({'_finish': True, 'status': 'success', 'summary': self.args['summary']})
        self.assertIs(self.check()['value'], True)
        self.assertIs(self.raw['benchmark_success'], False)

    def test_missing_terminal_wrong_owner_or_unclosed_native_remains_unknown(self):
        original = copy.deepcopy(self.sdk)
        self.sdk.pop()
        self.assertIsNone(self.check()['value'])
        self.sdk = original
        self.service['identity'] = {**self.identity, 'run_id': 'wrong'}
        self.assertIsNone(self.check()['value'])
        self.service['identity'] = self.identity
        self.service['pending_action'] = True
        self.assertIsNone(self.check()['value'])

    def test_source_override_or_changed_pure_return_remains_unknown(self):
        source = self.root/'code_snapshot/RPent/rpent/tools/common.py'
        source.write_text(source.read_text().replace('return ToolResult(data={"_finish": True, "status": status, "summary": summary})', 'raise RuntimeError("changed")'))
        self.assertIsNone(self.check()['value'])

    def test_contradictory_return_or_actions_after_finish_fail(self):
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = json.dumps({'_finish': True, 'status': 'stuck', 'summary': self.args['summary']})
        self.assertIs(self.check()['value'], False)
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = json.dumps({'_finish': True, **self.args})
        self.events.append(dict(**self.identity, sequence=3, kind='action_send'))
        self.assertIs(self.check()['value'], False)

    def test_finish_does_not_replace_vla_prerequisites(self):
        self.check()
        (self.root/'action_events.jsonl').write_text('\n'.join(map(json.dumps, self.events)))
        atomic_json(self.root/'states.json', {'steps': []})
        report = inspect_progress(self.root, self.identity)
        self.assertIsNot(report['value'], True)
        self.assertIs(report['finish']['value'], True)
        self.assertIsNone(report['checks']['supported_actual_tools'])

    def enable_guard(self, status, native):
        self.events[0]['effective_rules']['finish_requires_native_success'] = True
        args = {'status': status, 'summary': 'observed native outcome'}
        self.raw['planner_finish'] = {'_finish': True, **args}
        self.raw['benchmark_success'] = native
        self.service['last_native_success'] = native
        for event in self.sdk:
            event['payload']['item']['arguments'] = copy.deepcopy(args)
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = json.dumps({'_finish': True, **args})

    def test_guard_accepts_independently_confirmed_native_success(self):
        self.enable_guard('success', True)
        report = self.check()
        self.assertIs(report['value'], True)
        self.assertIs(report['checks']['guarded_common_dispatch'], True)
        self.assertIs(report['checks']['guarded_success_matches_native'], True)

    def test_guard_permits_honest_unsuccessful_finish(self):
        for status in ('failure', 'stuck'):
            with self.subTest(status=status):
                self.enable_guard(status, False)
                self.assertIs(self.check()['value'], True)

    def test_guard_does_not_accept_success_against_native_failure(self):
        self.enable_guard('success', False)
        report = self.check()
        self.assertIs(report['value'], False)
        self.assertIs(report['checks']['guarded_success_matches_native'], False)

    def test_guard_missing_or_conflicting_native_evidence_stays_unestablished(self):
        self.enable_guard('success', True)
        self.service['last_native_success'] = None
        self.assertIsNone(self.check()['value'])
        self.service['last_native_success'] = True
        self.service['identity']['run_id'] = 'wrong'
        self.assertIsNone(self.check()['value'])

    def test_guard_preserves_return_and_post_finish_action_checks(self):
        self.enable_guard('success', True)
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = json.dumps({'error': 'finish refused'})
        self.assertIs(self.check()['value'], False)
        self.enable_guard('success', True)
        self.events.append(dict(**self.identity, sequence=3, kind='action_send'))
        self.assertIs(self.check()['value'], False)
