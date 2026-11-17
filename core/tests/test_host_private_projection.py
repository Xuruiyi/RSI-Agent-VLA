# Copyright 2026 RSI-Agent-VLA Authors.
"""Raw Host and new-goal evaluator returns cannot bypass the public task projection."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi_vla.protocol10_roles import ROLES, _sandbox_command, role_isolation
from rsi_vla.storage import atomic_json


class HostPrivateProjectionTests(unittest.TestCase):
    def test_real_roles_read_public_task_and_costs_without_private_host_payload(self):
        with tempfile.TemporaryDirectory(
            prefix="host-private-projection-"
        ) as directory:
            root = Path(directory)
            for role, relative in ROLES.items():
                (root / relative).mkdir(parents=True, exist_ok=True)
                (root / "scratch" / role).mkdir(parents=True)
            Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)
            host = root / "operator/midterm-host"
            raw = host / "jobs/evaluate/inputs.json"
            log = raw.with_name("backend.log")
            progress = root / "operator/continuation/coordinator/progress.json"
            goal = root / "state/evaluation/new-goals/new-protocol"
            private_protocol = goal / "protocol.json"
            private_receipt = goal / "evaluation-materialization.json"
            private_paths = [raw, log, progress, private_protocol, private_receipt]
            for path in private_paths:
                atomic_json(path, {"private": "SYNTHETIC-HIDDEN-ANSWER"})
            public = raw.with_name("public-inputs.json")
            completion = raw.with_name("completion.json")
            atomic_json(public, {"references": {"task": "public-task.json"}})
            atomic_json(
                completion, {"actual": {"wall_seconds": 0.1}, "cleanup_confirmed": True}
            )
            task = goal / "materialization.json"
            atomic_json(
                task,
                {"schema": "rsi-materialized-goal-v1", "task": {"task_key": "new"}},
            )
            config = {
                "isolation_task_root": str(root),
                "operations": [
                    {
                        "kind": "goal_materialization",
                        "payload": {
                            "output_root": str(goal.parent),
                            "new_protocol_id": "new-protocol",
                        },
                    }
                ],
            }
            program = f"""
from pathlib import Path
import json
for ref in {list(map(str, private_paths))!r}:
 try: value=Path(ref).read_text()
 except OSError: continue
 assert 'SYNTHETIC-HIDDEN-ANSWER' not in value, ref
assert json.loads(Path({str(public)!r}).read_text())['references']['task']=='public-task.json'
assert json.loads(Path({str(completion)!r}).read_text())['actual']['wall_seconds']==.1
assert json.loads(Path({str(task)!r}).read_text())['task']['task_key']=='new'
try: Path({str(raw)!r}).write_text('forged')
except OSError: pass
else: raise AssertionError('raw Host provenance writable')
"""
            for role in ("research", "runtime"):
                with self.subTest(role=role):
                    result = subprocess.run(
                        _sandbox_command(
                            root,
                            role,
                            [sys.executable, "-c", program],
                            **role_isolation(config, role),
                        ),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
