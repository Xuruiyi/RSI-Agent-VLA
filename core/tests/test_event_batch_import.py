"""Batch event import preserves conflicts and resumes committed prefixes."""

from dataclasses import replace
from unittest.mock import patch
import unittest

import test_core
from rsi_vla.contracts import ExecutionEvent
from rsi_vla.evidence import EventLedger


class BatchEventImportTests(unittest.TestCase):
    setUp = test_core.CoreTest.setUp
    result = test_core.CoreTest.result

    def test_batch_reads_old_facts_once_and_preserves_idempotency(self):
        ledger = EventLedger(self.store)
        bundle = self.bundle.bundle_id
        facts = [ExecutionEvent('batch-'+str(i), 'batch-run', 'batch-episode', i, bundle,
                                'observation', 'fixture') for i in range(80)]
        with patch.object(self.store, 'all', wraps=self.store.all) as read:
            self.assertEqual(ledger.append_many(facts), [True]*80)
            self.assertEqual(read.call_count, 1)
        self.assertEqual(ledger.append_many(facts), [False]*80)
        with self.assertRaisesRegex(ValueError, 'sequence'):
            ledger.append_many([replace(facts[0], event_id='different-identity')])
        with self.assertRaisesRegex(ValueError, 'conflicting events'):
            ledger.append_many([replace(facts[0], feedback='different-fact')])

    def test_partial_prefix_can_resume_and_other_ledger_writes_remain_visible(self):
        ledger = EventLedger(self.store)
        bundle = self.bundle.bundle_id
        first = ExecutionEvent('prefix', 'prefix-run', 'prefix-episode', 0, bundle, 'observation', 'fixture')
        with self.assertRaisesRegex(ValueError, 'sequence'):
            ledger.append_many([first, replace(first, event_id='conflict')])
        self.assertEqual(ledger.append_many([first, replace(first, event_id='second', sequence=1)]), [False, True])
        EventLedger(self.store).append(replace(first, event_id='third', sequence=2))
        with self.assertRaisesRegex(ValueError, 'sequence'):
            ledger.append_many([replace(first, event_id='new-third', sequence=2)])
