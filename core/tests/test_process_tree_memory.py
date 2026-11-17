# Copyright 2026 RSI-Agent-VLA Authors.
"""Read actual CPU process-tree RSS, including an owned separate session."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.process_accounting import ProcessTreeMemoryMeter


class ProcessTreeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = self.root / "context.json"
        self.context.write_text("{}")

    def spawn(self, code, *, context=None):
        env = {**os.environ, "RSI_RUNTIME_CONTEXT": str(context or self.context)}
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def cleanup():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

        self.addCleanup(cleanup)
        return process

    def test_actual_detached_child_counted_other_marker_excluded_and_peaks_labeled(
        self,
    ):
        child_ref = self.root / "child.json"
        child_code = "import time; data=bytearray(32*1024*1024); time.sleep(30)"
        leader_code = f"""
import json,signal,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True)
def stop(*_):
 child.terminate();child.wait(timeout=2);sys.exit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(child_ref)!r}).write_text(json.dumps({{'pid':child.pid}}))
time.sleep(30)
"""
        leader = self.spawn(leader_code)
        unrelated = self.spawn(
            "import time; data=bytearray(1024*1024); time.sleep(30)",
            context=self.root / "other-job.json",
        )
        meter = ProcessTreeMemoryMeter(leader.pid, context_ref=self.context)
        deadline = time.monotonic() + 3
        record = None
        while time.monotonic() < deadline:
            record = meter.sample()
            if child_ref.exists() and record["current_rss_bytes"] > 32 * 1024**2:
                break
            time.sleep(0.02)
        self.assertIsNotNone(record)
        child_pid = json.loads(child_ref.read_text())["pid"]
        by_pid = {item["pid"]: item for item in record["current_processes"]}
        self.assertIn(leader.pid, by_pid)
        self.assertIn(child_pid, by_pid)
        self.assertTrue(by_pid[leader.pid]["ancestor_observed"])
        self.assertTrue(by_pid[child_pid]["ancestor_observed"])
        self.assertNotEqual(by_pid[child_pid]["pgid"], leader.pid)
        self.assertNotIn(unrelated.pid, by_pid)
        self.assertGreater(record["current_rss_bytes"], 32 * 1024**2)
        self.assertLess(record["current_rss_bytes"], 256 * 1024**2)
        self.assertIsNone(record["exact_peak_rss_bytes"])
        self.assertGreaterEqual(
            record["sampled_peak_rss_bytes"], record["current_rss_bytes"]
        )
        self.assertIsNone(unrelated.poll())
        # After a real ancestor observation, a later parent change preserves
        # proof only for the same PID birth. Keep the real child owned here.
        read_process = meter._process

        def reparented(pid):
            value = read_process(pid)
            return {**value, "ppid": 1} if pid == child_pid else value

        with patch.object(meter, "_process", side_effect=reparented):
            after = meter.sample()
        self.assertTrue(
            next(
                item for item in after["current_processes"] if item["pid"] == child_pid
            )["ancestor_observed"]
        )

    def test_unrelated_process_with_exact_current_tag_is_attributed_without_signaling(
        self,
    ):
        leader = self.spawn("import time; time.sleep(30)")
        tagged = self.spawn("import time; data=bytearray(8*1024*1024); time.sleep(30)")
        meter = ProcessTreeMemoryMeter(leader.pid, context_ref=self.context)
        record = meter.sample()
        self.assertIn(tagged.pid, [item["pid"] for item in record["current_processes"]])
        self.assertFalse(
            next(
                item
                for item in record["current_processes"]
                if item["pid"] == tagged.pid
            )["ancestor_observed"]
        )
        self.assertIsNone(tagged.poll())

    def test_reused_pid_birth_without_current_marker_is_not_the_old_job(self):
        leader = self.spawn("import time; time.sleep(30)")
        meter = ProcessTreeMemoryMeter(leader.pid, context_ref=self.context)
        read_process = meter._process
        read_markers = meter._markers

        def replaced(pid):
            value = read_process(pid)
            return (
                {**value, "start_ticks": value["start_ticks"] + 1, "ppid": 1}
                if pid == leader.pid
                else value
            )

        def markers(pid):
            return (
                {"RSI_RUNTIME_CONTEXT": str(self.root / "other-job.json")}
                if pid == leader.pid
                else read_markers(pid)
            )

        with (
            patch.object(meter, "_process", side_effect=replaced),
            patch.object(meter, "_markers", side_effect=markers),
        ):
            record = meter.sample()
        self.assertNotIn(
            leader.pid, [item["pid"] for item in record["current_processes"]]
        )
        self.assertIsNone(leader.poll())
        # Reuse of a known PID with a current tag is observable but has no
        # inherited ancestry proof for the new birth identity.
        with patch.object(meter, "_process", side_effect=replaced):
            current_tag = meter.sample()
        row = next(
            item
            for item in current_tag["current_processes"]
            if item["pid"] == leader.pid
        )
        self.assertFalse(row["ancestor_observed"])

    def test_foreign_uid_is_not_accepted_as_the_owned_leader(self):
        fake = {"uid": os.getuid() + 1, "start_ticks": 123}
        with patch.object(ProcessTreeMemoryMeter, "_process", return_value=fake):
            with self.assertRaisesRegex(PermissionError, "another user"):
                ProcessTreeMemoryMeter(123, context_ref=self.context)


if __name__ == "__main__":
    unittest.main()
