"""Bounded CPU child-process checks; no model, simulator or GPU is loaded."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.adapters.runtime_supervisor import _live_group, supervise


class RuntimeSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def context(self, **limits):
        context = {
            "schema_version": "1.0", "run_id": "r", "episode_id": "e", "bundle_id": "b",
            "protocol_id": "p", "profile_ref": "profile.json", "output_dir": str(self.root / "out"),
            "dynamic_authorized": True, "authorization_ref": "offline-fake-child-test",
            "gpu_count": 1, "limits": {"wall_seconds": 3, "storage_bytes": 100000, **limits},
        }
        path = self.root / 'context.json'
        path.write_text(json.dumps(context))
        return path

    def run_child(self, code, **limits):
        return supervise([sys.executable, '-c', code], context_path=self.context(**limits),
                         poll_interval_s=.01, terminate_grace_s=.1)

    def test_absent_group_needs_no_proc_scan(self):
        with patch("rsi_vla.adapters.runtime_supervisor.os.killpg", side_effect=ProcessLookupError), \
                patch("rsi_vla.adapters.runtime_supervisor.Path.glob", side_effect=AssertionError("unnecessary scan")):
            self.assertEqual(_live_group(1234), [])

    def test_existing_group_still_checks_membership(self):
        with patch("rsi_vla.adapters.runtime_supervisor.os.killpg") as exists, \
                patch("rsi_vla.adapters.runtime_supervisor.Path.glob", return_value=[]) as scan:
            self.assertEqual(_live_group(1234), [])
        exists.assert_called_once_with(1234, 0)
        scan.assert_called_once()

    def test_natural_exit_records_identity_costs_and_keeps_artifacts(self):
        result = self.run_child('print("offline child")')
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(result['run_id'], 'r')
        self.assertGreater(result['wall_seconds'], 0)
        self.assertIsNone(result['gpu_device_seconds'])
        self.assertEqual(result['gpu_device_seconds_envelope'], result['wall_seconds'])
        self.assertIn('offline child', (self.root / 'out' / 'backend.log').read_text())

    def test_wall_limit_stops_group_and_does_not_call_it_task_failure(self):
        result = self.run_child('import time; time.sleep(30)', wall_seconds=.08)
        self.assertEqual(result['stop_reason'], 'wall_seconds')
        self.assertEqual(result['status'], 'interrupted')
        self.assertTrue(result['cleanup_confirmed'])
        self.assertLess(result['wall_seconds'], 2)

    def test_storage_stop_reports_sampled_overshoot_without_deleting(self):
        result = self.run_child('import sys,time; sys.stdout.write("x"*20000); sys.stdout.flush(); time.sleep(30)', storage_bytes=10000)
        self.assertEqual(result['stop_reason'], 'storage_bytes')
        self.assertGreater(result['storage_overshoot_bytes'], 0)
        self.assertTrue((self.root / 'out' / 'backend.log').exists())

    def test_gpu_envelope_stops_allocated_job_without_fake_utilization(self):
        result = self.run_child('import time; time.sleep(30)', gpu_seconds=.08)
        self.assertEqual(result['stop_reason'], 'gpu_seconds_envelope')
        self.assertIsNone(result['gpu_device_seconds'])

    def test_orphaned_child_is_drained_before_job_returns(self):
        result = self.run_child('import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])')
        self.assertEqual(result['stop_reason'], 'orphaned_backend_processes')
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(result['remaining_process_ids'], [])

    def test_no_authorization_does_not_start_a_child(self):
        path = self.context()
        context = json.loads(path.read_text())
        context['dynamic_authorized'] = False
        path.write_text(json.dumps(context))
        marker = self.root / 'started'
        with self.assertRaises(PermissionError):
            supervise([sys.executable, '-c', f'open({str(marker)!r},"w").close()'], context_path=path)
        self.assertFalse(marker.exists())

    def test_failed_spawn_retains_costs_and_technical_error(self):
        result = supervise(['/nonexistent/rsi-offline-fixture'], context_path=self.context(),
                           poll_interval_s=.01, terminate_grace_s=.1)
        self.assertEqual(result['stop_reason'], 'backend_startup_error')
        self.assertIn('FileNotFoundError', result['error'])
        self.assertTrue((self.root / 'out' / 'supervisor.json').exists())
        self.assertIsNone(result['exit_code'])

    def test_complete_wall_deadline_reserves_delayed_graceful_cleanup(self):
        code = ('import signal,time,sys\n'
                'def stop(*_):\n'
                ' print("cleanup started",flush=True)\n'
                ' time.sleep(.05)\n'
                ' print("cleanup completed",flush=True)\n'
                ' sys.exit(0)\n'
                'signal.signal(signal.SIGTERM,stop)\n'
                'time.sleep(30)\n')
        result = self.run_child(code, wall_seconds=.5)
        self.assertEqual(result['stop_reason'], 'wall_seconds')
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue(result['cleanup_confirmed'])
        self.assertLess(result['wall_seconds'], .5)
        self.assertIn('cleanup completed', (self.root / 'out' / 'backend.log').read_text())


if __name__ == '__main__':
    unittest.main()
