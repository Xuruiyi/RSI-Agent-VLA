# Copyright 2026 RSI-Agent-VLA Authors.
"""Concurrent Host registration writes must each finish with one valid JSON file."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rsi_vla import storage


class AtomicHostWritesTests(unittest.TestCase):
    def test_overlapping_registration_writers_do_not_erase_another_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "host.json"
            barrier = threading.Barrier(2)
            replace = storage.os.replace

            def concurrent_replace(source, destination):
                barrier.wait(timeout=3)
                replace(source, destination)

            with patch.object(storage.os, "replace", side_effect=concurrent_replace):
                with ThreadPoolExecutor(max_workers=2) as workers:
                    jobs = [
                        workers.submit(storage.atomic_json, target, {"registration": i})
                        for i in (1, 2)
                    ]
                    for job in jobs:
                        job.result(timeout=4)
            self.assertIn(
                json.loads(target.read_text()),
                [{"registration": 1}, {"registration": 2}],
            )
            self.assertEqual(list(Path(directory).iterdir()), [target])
