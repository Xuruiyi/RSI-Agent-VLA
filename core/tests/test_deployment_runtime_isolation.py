"""Deployment readers protect existing runtime facts without closing their output."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.protocol10_roles import (
    _sandbox_command,
    deployment_runtime_protected_paths,
    run_role,
)
from rsi_vla.storage import atomic_json
import test_upgrade_bridge as upgrade_fixture


class DeploymentRuntimeIsolationTests(unittest.TestCase):
    def setUp(self):
        self.f = upgrade_fixture.UpgradeBridgeTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        self.current = Path(f.domains["runtime"]) / "inheritance"
        self.extra = Path(f.domains["runtime"]) / "development"
        atomic_json(self.extra / "fact.json", {"native": "fixture-development"})
        self.slot = dict(run_id="inheritance", split="deployment", output_ref=str(self.current))
        self.config = dict(
            schema="rsi-task-completion-v1",
            offline_fixture=True,
            isolation_task_root=str(f.task),
            protocol_ref=str(f.target_ref),
            release_root=f.domains["release"],
            research_store_ref=f.domains["research"],
            evaluation_store_ref=f.domains["evaluation"],
            runtime_slots=[
                dict(run_id="baseline", output_ref=str(f.base_raw.parent)),
                dict(run_id="candidate", output_ref=str(f.raw.parent)),
                dict(run_id="development", output_ref=str(self.extra)),
                dict(run_id="missing", output_ref=str(Path(f.domains["runtime"]) / "missing")),
                self.slot,
            ],
        )
        self.task = dict(action="execute", operation=dict(
            operation_id="inheritance-runtime", kind="runtime", dynamic=True,
            payload=dict(run_id="inheritance")), inputs={})

    def test_only_deployment_protects_existing_other_slots_and_upgrade_inputs(self):
        paths = set(deployment_runtime_protected_paths(self.config, self.task))
        self.assertTrue({self.f.base_raw.parent, self.f.raw.parent, self.extra}.issubset(paths))
        self.assertIn(self.f.source_ref, paths)
        self.assertIn(self.f.target_ref, paths)
        self.assertNotIn(self.current, paths)
        self.assertNotIn(Path(self.f.domains["runtime"]), paths)
        self.assertNotIn(Path(self.f.domains["runtime"]) / "missing", paths)
        self.slot["split"] = "development"
        self.assertEqual(deployment_runtime_protected_paths(self.config, self.task), ())

    def test_run_role_passes_readonly_directories_to_runtime_boundary(self):
        captured = []

        def returned(argv, **kwargs):
            captured.extend(argv)
            ref = Path(argv[argv.index("--request") + 1])
            request = json.loads(ref.read_text())
            atomic_json(request["receipt_ref"], dict(request=request, role="runtime", value={"status": "completed"}))
            return SimpleNamespace(returncode=0, communicate=lambda: ("", ""))

        with patch("rsi_vla.protocol10_roles.subprocess.Popen", side_effect=returned):
            result = run_role(self.config, self.task)
        self.assertEqual(result["role"], "runtime")
        mounts = [captured[index + 1] for index, value in enumerate(captured) if value == "--ro-bind"]
        self.assertIn(str(self.f.base_raw.parent), mounts)
        self.assertIn(str(self.f.raw.parent), mounts)
        self.assertIn(str(self.extra), mounts)
        self.assertNotIn(str(self.current), mounts)
        self.assertNotIn("--dev-bind", captured)

    def test_real_namespace_reads_upgrade_qualification_preserves_old_facts_and_writes_current(self):
        f = self.f
        # The existing fixture creates real five-role acceptance probes and an
        # actual independently qualified upgrade through the production mounts.
        f.test_real_release_role_boundary_and_complete_bridge()
        original = {path: path.read_bytes() for path in [f.base_raw, f.raw, self.extra / "fact.json"]}
        script = f.operator / "deployment-read-probe.py"
        script.write_text('''import json, sys
from pathlib import Path
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.release import ReleaseManager
from rsi_vla.storage import ResearchStore, atomic_json
config=json.loads(Path(sys.argv[1]).read_text())
domains=config['domains']
manager=ReleaseManager(domains['release'], ResearchStore(domains['research']), EvaluationStore(domains['evaluation']))
bundle, active=manager.deployment_bundle(scope='offline', safe_point='task_boundary')
assert bundle['bundle_id']=='deployment-trial'
assert active['bundle_id']==bundle['bundle_id']
denied=[]
for raw in config['old_facts']:
 path=Path(raw)
 assert path.read_bytes()
 for target in [path, path.parent/'forbidden-replacement.json']:
  try:
   with target.open('a') as stream: stream.write('forbidden')
  except OSError: denied.append(str(target))
  else: raise AssertionError('old runtime evidence remained writable: '+str(target))
current=Path(config['current_output'])
current.mkdir()
atomic_json(current/'cpu-probe.json', dict(bundle_id=bundle['bundle_id'], denied=denied, current_writable=True))
''')
        probe_ref = f.operator / "deployment-read-probe.json"
        atomic_json(probe_ref, dict(domains=f.domains, old_facts=[str(path) for path in original], current_output=str(self.current)))
        returned = subprocess.run(
            _sandbox_command(f.task, "runtime", [sys.executable, str(script), str(probe_ref)],
                gpu=False, protected_paths=deployment_runtime_protected_paths(self.config, self.task)),
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(returned.returncode, 0, returned.stdout + returned.stderr)
        probe = json.loads((self.current / "cpu-probe.json").read_text())
        self.assertEqual(probe["bundle_id"], "deployment-trial")
        self.assertEqual(len(probe["denied"]), 6)
        self.assertTrue(probe["current_writable"])
        for path, expected in original.items():
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse((path.parent / "forbidden-replacement.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
