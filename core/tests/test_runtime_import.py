"""Actual journal/result files, including identity conflicts and unknown gaps."""
import json
import tempfile
import unittest
from pathlib import Path

from rsi_vla.adapters.rpent import read_trajectory


class RuntimeImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
        self.states = self.root / 'states.json'
        self.states.write_text(json.dumps({'steps': []}))
        self.transcript = self.root / 'transcript.json'
        self.transcript.write_text(json.dumps({'runtime': {'identity': self.identity}}))
        self.journal = self.root / 'action_events.jsonl'
        self.journal.write_text(json.dumps({**self.identity, 'schema_version': '1.0', 'sequence': 0,
                                           'kind': 'action_send', 'sent': [[1]*7], 'execution_mask': None}) + '\n')
        self.result = self.root / 'result.json'
        self.result.write_text(json.dumps({**self.identity, 'schema_version': '1.0', 'robot': 'libero',
                                          'terminated': False, 'truncated': False, 'benchmark_success': None}))

    def read(self):
        return read_trajectory(self.states, self.transcript, run_id='r', episode_id='e', bundle_id='b',
                               protocol_id='p', action_events_path=self.journal, independent_result_path=self.result)

    def test_sent_without_feedback_stays_unknown_and_bound_identity_is_retained(self):
        events = self.read()
        self.assertEqual(events[0].sent, [[1]*7])
        self.assertIsNone(events[0].execution_mask)
        self.assertEqual(events[-1].outcome, 'unknown')
        self.assertEqual(events[-1].flags['import_binding']['meaning'], 'matched runtime identity')

    def test_result_episode_conflict_is_rejected(self):
        record = json.loads(self.result.read_text())
        record['episode_id'] = 'other'
        self.result.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.read()

    def test_journal_gap_or_protocol_conflict_is_rejected(self):
        record = json.loads(self.journal.read_text())
        record['sequence'] = 2
        self.journal.write_text(json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'sequence gap'):
            self.read()
        record['sequence'] = 0
        record['protocol_id'] = 'other'
        self.journal.write_text(json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.read()


if __name__ == '__main__':
    unittest.main()
