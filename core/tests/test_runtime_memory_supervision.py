# Copyright 2026 RSI-Agent-VLA Authors.
"""Real CPU process-tree RSS stops and cleanup; no model/simulator/GPU is loaded."""

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from rsi_vla.adapters.runtime_supervisor import supervise
import test_runtime_supervisor as supervisor_fixtures

MIB = 1024**2


def process_identity(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "uid": path.stat().st_uid,
            "state": fields[0],
            "start_ticks": int(fields[19]),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(pid):
    identity = process_identity(pid)
    return identity is not None and identity["state"] not in {"Z", "X"}


def cleanup_recorded_child(record_ref):
    """Drain only this fixture's recorded same-UID PID with its original birth."""
    if not record_ref.exists():
        return
    original = json.loads(record_ref.read_text())
    current = process_identity(original["pid"])
    if (
        current
        and current["uid"] == original["uid"] == os.getuid()
        and current["start_ticks"] == original["start_ticks"]
        and current["state"] not in {"Z", "X"}
    ):
        try:
            os.kill(original["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


class RuntimeMemorySupervisionTests(unittest.TestCase):
    # Reuse the existing authorization/output/context fixture without inheriting
    # its tests, which are selected independently when validating defaults.
    setUp = supervisor_fixtures.RuntimeSupervisorTests.setUp
    context = supervisor_fixtures.RuntimeSupervisorTests.context

    def memory_context(self, limit):
        path = self.context()
        value = json.loads(path.read_text())
        value.update(memory_limit_bytes=limit, gpu_count=0)
        path.write_text(json.dumps(value))
        return path

    def supervise_child(self, code, *, memory_limit):
        return supervise(
            [sys.executable, "-c", code],
            context_path=self.memory_context(memory_limit),
            poll_interval_s=0.01,
            terminate_grace_s=0.1,
        )

    def allocating_child(self, *, size_mib, detached=False, parent_seconds=30):
        record = self.root / "allocated-child.json"
        self.addCleanup(cleanup_recorded_child, record)
        child = (
            "import json,os,time; from pathlib import Path; "
            'pid=os.getpid(); fields=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split(); '
            f"Path({str(record)!r}).write_text(json.dumps(dict(pid=pid,uid=os.getuid(),start_ticks=int(fields[19])))); "
            f"buffer=bytearray({size_mib}*1024**2); time.sleep(30)"
        )
        code = (
            "import subprocess,sys,time; "
            f'subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session={detached!r}); '
            f"time.sleep({parent_seconds!r})"
        )
        return code, record

    def assert_memory_evidence(self, report, limit):
        self.assertEqual(report["memory_limit_bytes"], limit)
        accounting = report["memory_accounting"]
        self.assertGreater(accounting["sampled_peak_rss_bytes"], 0)
        self.assertGreaterEqual(accounting["current_rss_bytes"], 0)
        self.assertIsNone(accounting["exact_peak_rss_bytes"])
        self.assertIn("shared", accounting["basis"])
        self.assertIn("polling", accounting["gap"])
        self.assertIn("no kernel cgroup quota", report["memory_enforcement"])
        return accounting

    def test_real_allocating_subtree_stops_at_sampled_rss_limit(self):
        limit = 48 * MIB
        code, record = self.allocating_child(size_mib=64)
        result = self.supervise_child(code, memory_limit=limit)
        accounting = self.assert_memory_evidence(result, limit)
        self.assertGreaterEqual(accounting["sampled_peak_rss_bytes"], limit)
        self.assertEqual(result["stop_reason"], "memory_bytes")
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["remaining_process_ids"], [])
        self.assertTrue(record.exists())
        self.assertFalse(live(json.loads(record.read_text())["pid"]))

    def test_small_real_process_exits_naturally_below_limit(self):
        limit = 128 * MIB
        result = self.supervise_child(
            "import time; buffer=bytearray(1024**2); time.sleep(.25)",
            memory_limit=limit,
        )
        accounting = self.assert_memory_evidence(result, limit)
        self.assertLess(accounting["sampled_peak_rss_bytes"], limit)
        self.assertIsNone(result["stop_reason"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["cleanup_confirmed"])

    def test_observed_setsid_child_counts_and_is_drained_after_parent_exit(self):
        # Keep the parent alive long enough to observe the PPid chain before
        # adoption; process-group membership alone cannot clean this child.
        code, record = self.allocating_child(
            size_mib=32, detached=True, parent_seconds=0.3
        )
        result = self.supervise_child(code, memory_limit=128 * MIB)
        accounting = self.assert_memory_evidence(result, 128 * MIB)
        self.assertGreaterEqual(accounting["sampled_peak_rss_bytes"], 32 * MIB)
        self.assertEqual(result["stop_reason"], "orphaned_backend_processes")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["remaining_process_ids"], [])
        self.assertTrue(record.exists())
        self.assertFalse(live(json.loads(record.read_text())["pid"]))

    def external_process(self, context_ref):
        ready = self.root / "external-ready"
        code = (
            "import time; from pathlib import Path; "
            f"buffer=bytearray(64*1024**2); Path({str(ready)!r}).touch(); time.sleep(30)"
        )
        env = dict(os.environ, RSI_RUNTIME_CONTEXT=str(context_ref))
        env.pop("RSI_TRAINING_JOB_ROOT", None)
        process = subprocess.Popen(
            [sys.executable, "-c", code], env=env, start_new_session=True
        )

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

        self.addCleanup(cleanup)
        deadline = time.monotonic() + 2
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "external fixture did not become ready")
        return process

    def test_unrelated_different_tag_memory_is_not_counted_or_terminated(self):
        external = self.external_process(self.root / "different-context.json")
        limit = 32 * MIB
        result = self.supervise_child(
            "import time; buffer=bytearray(1024**2); time.sleep(.25)",
            memory_limit=limit,
        )
        accounting = self.assert_memory_evidence(result, limit)
        self.assertLess(accounting["sampled_peak_rss_bytes"], limit)
        self.assertIsNone(result["stop_reason"])
        self.assertTrue(result["cleanup_confirmed"])
        self.assertIsNone(external.poll())

    def test_same_tag_without_observed_ancestry_remains_unknown_and_unharmed(self):
        limit = 256 * MIB
        context = self.memory_context(limit)
        external = self.external_process(context)
        result = supervise(
            [sys.executable, "-c", "import time; time.sleep(.25)"],
            context_path=context,
            poll_interval_s=0.01,
            terminate_grace_s=0.1,
        )
        accounting = self.assert_memory_evidence(result, limit)
        self.assertFalse(result["cleanup_confirmed"])
        self.assertIn(external.pid, result["remaining_process_ids"])
        row = next(
            item
            for item in accounting["current_processes"]
            if item["pid"] == external.pid
        )
        self.assertFalse(row["ancestor_observed"])
        self.assertIsNone(external.poll())

    def test_invalid_memory_limits_reject_before_spawning(self):
        marker = self.root / "must-not-start"
        for limit in (0, -1, True, 1.5, "1048576"):
            with self.subTest(memory_limit=limit):
                with self.assertRaisesRegex(ValueError, "memory_limit_bytes"):
                    supervise(
                        [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).touch()",
                        ],
                        context_path=self.memory_context(limit),
                    )
                self.assertFalse(marker.exists())
