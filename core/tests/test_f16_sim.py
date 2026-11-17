# Copyright 2026 RSI-Agent-VLA Authors.
"""CPU F16 compositions use real Host, native receipts and role namespaces.

Model, simulator and trainer compatibility remains outside these synthetic scenes.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import test_goal_host
import test_meta_exam_host
from rsi_vla.midterm_host import MidtermHost, read
from rsi_vla.protocol10_roles import _sandbox_command, role_isolation


class GoalBoundaryRecoveryTests(unittest.TestCase):
    setUp = test_goal_host.GoalHostTests.setUp
    host = test_goal_host.GoalHostTests.host
    end_job = test_goal_host.GoalHostTests.end_job

    def test_materialized_goal_restart_costs_and_real_consumer_boundaries(self):
        self.assertEqual(self.host().advance()["status"], "running")
        host, completion = self.end_job("produce")
        original_completion = completion.read_bytes()
        receipt_ref = Path(self.config["continuation_root"]) / "receipts/produce.json"
        original_receipt = receipt_ref.read_bytes()
        costs = host.ledger.read()["used"]
        native = read(receipt_ref)["value"]
        public = Path(native["materialization_ref"])
        materialization = read(public)
        self.assertEqual(materialization["task"]["task_key"], "joint-packing")
        private = Path(native["protocol_ref"])
        self.assertIn("F12-PRIVATE-EXAM-LABEL", private.read_text())
        self.assertNotIn("F12-PRIVATE-EXAM-LABEL", public.read_text())

        # A reconstructed production Host consumes the original result. It does
        # not re-materialize the task or replace the settled producer receipt.
        restarted = MidtermHost(self.host_ref)
        self.assertEqual(restarted.advance()["status"], "running")
        host, _ = self.end_job("consume")
        self.assertEqual(host.advance()["status"], "completed")
        settled = host.ledger.read()["used"]
        self.assertGreaterEqual(settled["wall_seconds"], costs["wall_seconds"])
        self.assertEqual(completion.read_bytes(), original_completion)
        self.assertEqual(receipt_ref.read_bytes(), original_receipt)
        for _ in range(2):
            self.assertEqual(self.host().advance()["status"], "completed")
        self.assertEqual(self.host().ledger.read()["used"], settled)
        self.assertEqual(settled["model_requests"], 0)
        self.assertEqual(settled["gpu_seconds"], 0)
        prepared = read(
            Path(self.config["continuation_root"]) / "receipts/consume.json"
        )["value"]
        self.assertEqual(prepared["task_binding"]["materialization_ref"], str(public))
        self.assertEqual(prepared["environment_initialization"], "unmeasured")

        # Use the same generated task, raw Host receipt and private protocol,
        # rather than a second fixture which merely resembles their structure.
        release = self.root / "release"
        release.mkdir(exist_ok=True)
        program = f"""
import json
from pathlib import Path
assert json.loads(Path({str(public)!r}).read_text())['task']['task_key']=='joint-packing'
for ref in {[str(private), str(receipt_ref)]!r}:
 try: value=Path(ref).read_text()
 except OSError: continue
 assert 'F12-PRIVATE-EXAM-LABEL' not in value, ref
try: Path({str(release / "forged-active.json")!r}).write_text('forged')
except OSError: pass
else: raise AssertionError('release domain writable')
"""
        for role in ("research", "runtime"):
            with self.subTest(role=role):
                result = subprocess.run(
                    _sandbox_command(
                        self.root,
                        role,
                        [sys.executable, "-c", program],
                        **role_isolation(self.core, role),
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class MetaInheritanceBoundaryTests(unittest.TestCase):
    setUp = test_meta_exam_host.MetaExamHostTests.setUp
    prepare = test_meta_exam_host.MetaExamHostTests.prepare
    finish = test_meta_exam_host.MetaExamHostTests.finish

    def test_scored_policy_inheritance_keeps_same_job_and_seals_feedback(self):
        core = self.prepare()
        host = self.finish()
        score = self.evaluations.get("meta_exam_scores", "exam")
        self.assertEqual(score["verdict"], "accept", score)
        self.assertEqual(
            score["metrics"]["endpoint"], {"psi0": 0.25, "psi1": 1.0, "psi0_late": 0.25}
        )
        original_job = self.store.get("jobs", "psi0-exam-process")
        original_selections = self.store.all("meta_exam_selections")
        original_costs = host.ledger.read()["used"]
        self.assertEqual(MidtermHost(self.host_ref).advance()["status"], "completed")
        self.assertEqual(host.ledger.read()["used"], original_costs)
        self.assertEqual(self.store.all("meta_exam_selections"), original_selections)

        hidden = [
            self.problem_ref,
            self.evaluations.root / "meta_exam_scores/exam.json",
            Path(self.domains["release"])
            / "research_policy_qualifications/exam-policy.json",
        ]
        program = f"""
from pathlib import Path
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.release import ReleaseManager
from rsi_vla.storage import ResearchStore
from rsi_vla.version_binding import start_research_job
s=ResearchStore({str(self.store.root)!r})
m=ReleaseManager({self.domains["release"]!r},s,EvaluationStore({self.domains["evaluation"]!r}))
j=start_research_job(s,'f16-new-job',bundle_id='base',limits={{'offline_calls':2}},release_manager=m,scope='offline')
assert j.state['bundle_id']=={self.changed.bundle_id!r}
assert s.get('bundles',j.state['bundle_id'])['psi_ref']=={self.changed.psi_ref!r}
old=start_research_job(s,'psi0-exam-process',bundle_id={self.changed.bundle_id!r},limits={self.limits!r},release_manager=m,scope='offline')
assert old.state=={original_job!r}
for ref in {list(map(str, hidden))!r}:
 try: Path(ref).read_text()
 except OSError: pass
 else: raise AssertionError('sealed meta material readable: '+ref)
try: Path({str(Path(self.domains["release"]) / "forged-active.json")!r}).write_text('forged')
except OSError: pass
else: raise AssertionError('research can publish')
"""
        result = subprocess.run(
            _sandbox_command(
                self.root,
                "research",
                [sys.executable, "-c", program],
                **role_isolation(core, "research"),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.store.get("jobs", "psi0-exam-process"), original_job)
        self.assertEqual(
            self.store.get("jobs", "f16-new-job")["bundle_id"], self.changed.bundle_id
        )
        self.assertEqual(host.ledger.read()["used"]["model_requests"], 0)
        self.assertEqual(host.ledger.read()["used"]["gpu_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
