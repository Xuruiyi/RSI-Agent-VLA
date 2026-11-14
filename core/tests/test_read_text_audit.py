"""SDK read and immutable revision regressions; CPU fakes, no models or Runtime."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from rsi_vla.read_text_audit import audit_read, READ_PATH, READ_SOURCES, REVIEWED_REVISION
from rsi_vla.storage import atomic_json


class ActualReadAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
        repo = Path(__file__).resolve().parents[2] / 'RPent'
        self.sources = {name: (repo / name).read_bytes() for name in READ_SOURCES}
        for name, content in self.sources.items():
            p = self.root / 'source-origin' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        frozen = self.root / 'code_snapshot/RPent/robots/libero'
        frozen.mkdir(parents=True)
        for name in ('robot_spec.py', 'toolkit.py'):
            (frozen / name).write_bytes((repo / 'robots/libero' / name).read_bytes())
        atomic_json(self.root / 'runtime_context.json', self.identity)
        atomic_json(self.root / 'runtime_costs.json', {'tool_calls': 1})
        atomic_json(self.root / 'launch_plan.json', {'cwd': str(repo), 'environment': {'RPENT_REPO_ROOT': str(repo)}})
        atomic_json(self.root / 'before.json', {'RPent_HEAD_and_origin': REVIEWED_REVISION + '\n', 'RPent': '## main\n'})
        self.provenance = dict(identity=self.identity, run_root=str(self.root), repository=str(repo),
                               revision=REVIEWED_REVISION, git_state_before_ref=str(self.root / 'before.json'), files=list(READ_SOURCES))
        self.ref = self.root / 'provenance.json'
        atomic_json(self.ref, self.provenance)
        text = '---\nname: stop-that-shit\n---\nMeet the task requirements.\n'
        args = {'path': READ_PATH, 'max_chars': 12000}
        start = dict(id='call1', type='mcpToolCall', server='rpent', tool='read_text_file', arguments=args,
                     status='inProgress', started_at_ms=1)
        end = {**start, 'status': 'completed', 'completed_at_ms': 4,
               'result': {'content': [{'type': 'text', 'text': json.dumps({'path': READ_PATH, 'size': len(text), 'content': text})}]}}
        self.sdk = [{'method': 'item/started', 'payload': {'item': start}}, {'method': 'item/completed', 'payload': {'item': end}}]
        self.calls = [dict(self.identity, sequence=0, kind='tool_call', tool='read_text_file', utc_ns=2000000)]
        self.ends = [dict(self.identity, sequence=1, kind='tool_end', tool='read_text_file', utc_ns=3000000, status='completed')]

    def check(self, source=True):
        (self.root / 'codex_test.stream.jsonl').write_text('\n'.join(json.dumps(e) for e in self.sdk))
        (self.root / 'action_events.jsonl').write_text('\n'.join(json.dumps(e) for e in self.calls + self.ends))
        with patch('rsi_vla.read_text_audit.subprocess.run', side_effect=lambda args, **kw: SimpleNamespace(stdout=self.sources[args[-1].split(':', 1)[1]])):
            return audit_read(self.root, self.identity, self.calls, self.ends, source_ref=self.ref if source else None)

    def test_complete_actual_read_and_missing_source(self):
        self.assertIs(self.check()['value'], True)
        self.assertEqual(self.check()['source_basis']['kind'], 'posthoc_git_origin_not_runtime_frozen')
        self.assertIsNone(self.check(source=False)['value'])

    def test_wrong_identity_server_origin_and_uncovered_path(self):
        original = copy.deepcopy(self.sdk)
        self.sdk[1]['payload']['item']['server'] = 'external'
        self.assertIs(self.check()['value'], False)
        self.sdk = original
        self.provenance['identity'] = {**self.identity, 'run_id': 'other'}
        atomic_json(self.ref, self.provenance)
        self.assertIs(self.check()['value'], False)
        self.provenance['identity'] = self.identity
        self.provenance['repository'] = '/another/source'
        atomic_json(self.ref, self.provenance)
        self.assertIs(self.check()['value'], False)
        self.provenance['repository'] = json.loads((self.root / 'launch_plan.json').read_text())['cwd']
        atomic_json(self.ref, self.provenance)
        self.sdk[1]['payload']['item']['arguments'] = {'path': '/private/other', 'max_chars': 12000}
        self.assertIs(self.check()['value'], False)

    def test_truncation_missing_terminal_and_dirty_source(self):
        original = copy.deepcopy(self.sdk)
        self.sdk[1]['payload']['item']['result']['content'][0]['text'] = 'truncated invalid json'
        self.assertIs(self.check()['value'], False)
        self.sdk.pop()
        self.assertIs(self.check()['value'], False)
        self.sdk = original
        atomic_json(self.root / 'before.json', {'RPent_HEAD_and_origin': REVIEWED_REVISION + '\n', 'RPent': '## main\n M rpent/tools/common.py\n'})
        self.assertIsNone(self.check()['value'])
