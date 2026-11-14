"""Real bounded CPU processes and isolated ledgers for accounting regressions."""

import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

from rsi_vla.process_accounting import StorageMeter, settle_receipt, accounted_storage_bytes
from rsi_vla.storage import atomic_json
from rsi_vla.adapters.runtime_supervisor import supervise


class ProcessAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cleanup_releases_occupancy_but_preserves_cumulative_cost_and_hardlink_dedup(self):
        meter = StorageMeter([self.root, self.root / 'nested'])
        path = self.root / 'data'
        path.write_bytes(b'x' * 1234)
        os.link(path, self.root / 'alias')
        first = meter.sample()
        self.assertEqual(first['cumulative_new_bytes_lower_bound'], 1234)
        path.unlink()
        (self.root / 'alias').unlink()
        final = meter.sample()
        self.assertEqual(final['current_new_bytes'], 0)
        self.assertEqual(final['cumulative_new_bytes_lower_bound'], 1234)
        self.assertEqual(final['sampled_peak_new_bytes'], 1234)
        self.assertIsNone(final['exact_cumulative_bytes'])

    def test_scratch_growth_stops_process_even_when_output_is_small(self):
        scratch = self.root / 'scratch'
        scratch.mkdir()
        out = self.root / 'out'
        context = dict(dynamic_authorized=True, authorization_ref='isolated-offline-fixture',
            output_dir=str(out), storage_write_roots=[str(out), str(scratch)],
            limits=dict(wall_seconds=2, storage_bytes=16000))
        context_path = self.root / 'context.json'
        atomic_json(context_path, context)
        result = supervise([sys.executable, '-c',
            "from pathlib import Path; import time; Path(" + repr(str(scratch / 'cache')) + ").write_bytes(b'x'*20000); time.sleep(20)"],
            context_path=context_path, poll_interval_s=.01, terminate_grace_s=.1)
        self.assertEqual(result['stop_reason'], 'storage_bytes')
        self.assertTrue(result['cleanup_confirmed'])
        self.assertGreaterEqual(result['storage_accounting']['cumulative_new_bytes_lower_bound'], 20000)
        self.assertTrue((out / 'storage-progress.json').exists())

    def test_late_evaluation_files_add_to_retained_scratch_cost(self):
        out = self.root / 'output'
        out.mkdir()
        atomic_json(out / 'supervisor.json', dict(storage_bytes=5000, storage_output_bytes_at_sample=0))
        before = accounted_storage_bytes(out)
        (out / 'evaluation.json').write_bytes(b'x' * 200)
        self.assertEqual(accounted_storage_bytes(out), before + 200)

    def test_actual_cli_records_full_child_and_refuses_duplicate_launch(self):
        phase = self.root / 'phase-cli.json'
        atomic_json(phase, dict(limits=dict(wall_seconds=20, storage_bytes=1000000),
            used=dict(wall_seconds=0, storage_bytes=0), remaining=dict(wall_seconds=20, storage_bytes=1000000)))
        repo = Path(__file__).resolve().parents[2]
        output = self.root / 'metered'
        command = [sys.executable, str(repo / 'scripts/run_metered_process.py'),
            '--scope', 'isolated-test', '--key', 'once', '--output', str(output),
            '--phase', str(phase), '--wall-seconds', '3', '--storage-bytes', '200000']
        first = subprocess.run(command + ['--', sys.executable, '-c',
            'import time; time.sleep(.08); print("metered child")'], capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        receipt = json.loads((output / 'once-receipt.json').read_text())
        self.assertGreaterEqual(receipt['wall_seconds'], .08)
        self.assertTrue(receipt['cleanup_confirmed'])
        before = phase.read_bytes()
        duplicate = subprocess.run(command + ['--', sys.executable, '-c', 'raise RuntimeError("must not start")'],
                                   capture_output=True, text=True)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn('operation already recorded', duplicate.stderr)
        self.assertEqual(phase.read_bytes(), before)
        reconciled = subprocess.run(command + ['--reconcile'], capture_output=True, text=True)
        self.assertEqual(reconciled.returncode, 0, reconciled.stderr)
        self.assertEqual(phase.read_bytes(), before)

    def test_receipt_survives_evaluator_failure_and_reconciles_without_duplicate_cost(self):
        phase = self.root / 'phase.json'
        atomic_json(phase, dict(limits=dict(wall_seconds=10, storage_bytes=1000),
            used=dict(wall_seconds=0, storage_bytes=0), remaining=dict(wall_seconds=10, storage_bytes=1000)))
        receipt = self.root / 'receipt.json'
        atomic_json(receipt, dict(scope='test', key='process', status='completed', wall_seconds=.25,
            storage=dict(cumulative_new_bytes_lower_bound=100)))
        first = settle_receipt(phase, receipt)
        before = phase.read_bytes()
        try:
            raise ValueError('isolated evaluator unavailable')
        except ValueError:
            pass
        self.assertEqual(settle_receipt(phase, receipt), first)
        self.assertEqual(phase.read_bytes(), before)
        self.assertEqual(json.loads(before)['used'], dict(wall_seconds=.25, storage_bytes=100))
        atomic_json(receipt, dict(scope='test', key='uncertain', status='started', wall_seconds=None))
        with self.assertRaisesRegex(ValueError, 'submission status unknown'):
            settle_receipt(phase, receipt)


if __name__ == '__main__':
    unittest.main()
