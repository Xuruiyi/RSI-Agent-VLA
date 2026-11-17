# ruff: noqa: UP017 - Python 3.10 support
"""Release crash recovery and independent safety facts in real role namespaces."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import test_core
from rsi_vla import initialize
from rsi_vla.contracts import Candidate, ExperimentResult, RunBundle
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.release import ReleaseManager
from rsi_vla.storage import ResearchStore, atomic_json

ROOT = Path(__file__).resolve().parents[2]


class ReconciliationTests(unittest.TestCase):
    setUp = test_core.CoreTest.setUp
    result = test_core.CoreTest.result

    def decide(self):
        for result in (
            self.result(),
            self.result("baseline", "initial", 0.5),
            self.result("anchor", "initial", 0.5),
        ):
            self.evals.save_result(result)
        return self.release.evaluate(
            "accept",
            "candidate",
            candidate_result="candidate-result",
            baseline_result="baseline",
            anchor_result="anchor",
        )

    def test_record_before_pointer_crash_reconciles_original_operation_and_duplicates(
        self,
    ):
        self.decide()
        before = self.release.active()
        with (
            patch(
                "rsi_vla.release.atomic_json",
                side_effect=OSError("synthetic pointer crash"),
            ),
            self.assertRaisesRegex(OSError, "pointer crash"),
        ):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(self.release.active(), before)
        self.assertEqual(len(self.release.records.all("activations")), 1)
        self.assertFalse(self.release.records.all("activation_commits"))
        after = self.release.activate(
            "accept", safe_point="task_boundary", queues_empty=True
        )
        self.assertEqual(after["revision"], before["revision"] + 1)
        self.assertEqual(
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            ),
            after,
        )
        self.assertEqual(len(self.release.records.all("activations")), 1)
        self.assertEqual(len(self.release.records.all("activation_commits")), 1)

    def test_changed_independent_result_revokes_qualification(self):
        self.decide()
        path = self.evals.path("results", "candidate-result")
        data = json.loads(path.read_text())
        data["hard_constraints"]["safety"] = False
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(self.release.active()["bundle_id"], "initial")

    def test_new_task_protocol_cannot_activate_with_only_caller_booleans(self):
        self.decide()
        protocol = self.evals.get("protocols", "offline-test")
        atomic_json(
            self.evals.path("protocols", "offline-test"),
            {**protocol, "task_policy": {"schema": "rsi-task-policy-v1"}},
        )
        with self.assertRaisesRegex(ValueError, "independent safety registration"):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )
        self.assertEqual(self.release.active()["bundle_id"], "initial")

    def test_pending_record_does_not_allow_stale_revision_or_cross_scope(self):
        self.decide()
        before = self.release.active()
        with self.assertRaisesRegex(ValueError, "scope"):
            self.release.activate(
                "accept",
                safe_point="task_boundary",
                queues_empty=True,
                scope="hardware",
            )
        atomic_json(
            self.release.active_path, {**before, "revision": before["revision"] + 1}
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            self.release.activate(
                "accept", safe_point="task_boundary", queues_empty=True
            )


ROLE_SCRIPT = r"""import json, os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.release import ReleaseManager
from rsi_vla.runtime import execution_binding, resolve_reference
from rsi_vla.storage import ResearchStore, atomic_json
from unittest.mock import patch
config=json.loads(Path(sys.argv[1]).read_text()); action=sys.argv[2]
domains=config['role_domains']; role=os.environ['RSI_ISOLATED_ROLE']; domain=Path(domains[role])
manager=ReleaseManager(domains['release'], ResearchStore(domains['research']), EvaluationStore(domains['evaluation']))
if action=='probe':
 denied=[]
 for name,path in domains.items():
  if name==role: continue
  try: (Path(path)/'forbidden').write_text('bad')
  except OSError: denied.append(name)
  else: raise AssertionError('cross-role write succeeded')
 atomic_json(domain/'probe.json',dict(role=role,scope=config['scope'],allowed_write=True,denied_writes=len(denied)==4))
elif action.startswith('observe'):
 active=manager.active(); now=datetime.now(timezone.utc)
 common=dict(scope=config['scope'],active_snapshot=active,observed_at=now.isoformat(),expires_at=(now+timedelta(minutes=20)).isoformat(),evidence_level='synthetic_offline',producer_role=role,evidence_refs=['synthetic-engineering-no-devices'])
 if role=='operator':
  value=dict(common,schema='rsi-release-safety-observation-v1',operations=['initialize','activate','inherit','rollback'],safe_point='task_boundary',queues_empty=True,pending_actions=[],unresolved_costs=[],cleanup_complete=True,physical_state='known_safe')
  if action=='observe-unsafe': value['physical_state']='unknown'
  if action=='observe-cost': value['unresolved_costs']=['unknown job charge']
  atomic_json(config['safety_ref'],value)
 else:
  bundle=manager.research.get('bundles',active['bundle_id'])
  binding=execution_binding(manager.research,bundle,resolve_reference(bundle['backend_ref']))
  if action=='observe-mismatch': binding['processor']='wrong-processor'
  atomic_json(config['loaded_ref'],dict(common,schema='rsi-service-loading-observation-v1',service_stopped=action=='observe-stopped',loaded_binding=binding))
elif action=='register': manager.register_safety(sys.argv[1])
elif action=='initialize': manager.establish_initial('initial-qualified',safe_point='task_boundary',queues_empty=True)
elif action=='decide': manager.evaluate('accept','candidate',candidate_result='candidate-result',baseline_result='baseline',anchor_result='anchor')
elif action=='crash':
 with patch('rsi_vla.release.atomic_json',side_effect=OSError('synthetic crash')):
  try: manager.activate('accept',safe_point='task_boundary',queues_empty=True)
  except OSError: pass
  else: raise AssertionError('did not crash')
elif action=='activate': manager.activate('accept',safe_point='task_boundary',queues_empty=True)
elif action=='rollback': manager.rollback('initial',safe_point='task_boundary',queues_empty=True,reason='synthetic regression')
elif action=='inherit':
 bundle,active=manager.deployment_bundle(scope='offline',safe_point='task_boundary')
 atomic_json(domain/'new-binding.json',dict(bundle=bundle,active=active))
elif action=='nonancestor': manager.rollback('unused',safe_point='task_boundary',queues_empty=True,reason='synthetic')
else: raise ValueError(action)
"""


class ProtectedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="f13-release-", dir=ROOT / "RPent"
        )
        self.addCleanup(self.tmp.cleanup)
        self.task = Path(self.tmp.name)
        self.domains = {
            name: str(self.task / relative)
            for name, relative in {
                "operator": "operator",
                "research": "state/research",
                "evaluation": "state/evaluation",
                "release": "release",
                "runtime": "runtime",
            }.items()
        }
        for path in self.domains.values():
            Path(path).mkdir(parents=True)
        self.store = ResearchStore(self.domains["research"])
        self.evals = EvaluationStore(self.domains["evaluation"])
        self.manager = ReleaseManager(self.domains["release"], self.store, self.evals)
        self.operator = Path(self.domains["operator"])
        for name, value in (
            ("profile", {}),
            ("theta", {"model": "synthetic"}),
            ("phi", {"rules": []}),
        ):
            atomic_json(self.operator / (name + ".json"), value)
        bundle = RunBundle(
            "initial",
            str(self.operator / "theta.json"),
            "processor-v0",
            str(self.operator / "phi.json"),
            "rho-v0",
            "psi-v0",
            "memory-v0",
            "offline-test",
            str(self.operator / "profile.json"),
            "code-v0",
        )
        initialize(
            self.store,
            bundle,
            psi={"learned": False},
            memory={"provenance": "external_prior", "legal_information": ["public"]},
        )
        target = replace(
            bundle,
            bundle_id="candidate-bundle",
            parent_bundle_id="initial",
            code_ref="code-v1",
        )
        self.store.save_bundle(target)
        self.store.save_candidate(
            Candidate(
                "candidate",
                "initial",
                "candidate-bundle",
                "offline-test",
                "fixture-v1",
                {"code_ref": "code-v1"},
                ["code-v1"],
                [],
                {},
                {},
                "offline",
            )
        )
        self.evals.put("protocols", "offline-test", test_core.offline_protocol())

        def result(key, bundle, success):
            return ExperimentResult(
                key,
                bundle,
                "offline-test",
                "fixture-v1",
                "release",
                "offline",
                {},
                ["synthetic-evidence"],
                "synthetic-criteria",
                {"success": success},
                {"safety": True, "compatibility": True},
                {"seconds": 1},
                "candidate" if bundle == "candidate-bundle" else None,
                assigned_count=10,
            )

        for item in (
            result("initial-result", "initial", 0.5),
            result("baseline", "initial", 0.5),
            result("anchor", "initial", 0.5),
            result("candidate-result", "candidate-bundle", 0.9),
        ):
            self.evals.save_result(item)
        self.manager.bootstrap("initial")
        self.manager.qualify_initial("initial-qualified", result_id="initial-result")
        self.ref = self.operator / "registration.json"
        config = {
            "schema": "rsi-release-safety-registration-v1",
            "authorized": True,
            "scope": "f13-synthetic",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "role_domains": self.domains,
            "protected_paths": [str(self.operator)],
            "isolation_acceptance_ref": str(self.operator / "acceptance.json"),
            "safety_ref": str(self.operator / "safety.json"),
            "loaded_ref": str(Path(self.domains["runtime"]) / "loaded.json"),
        }
        atomic_json(self.ref, config)
        self.script = self.operator / "role.py"
        self.script.write_text(ROLE_SCRIPT)
        spec = importlib.util.spec_from_file_location(
            "release_isolation", ROOT / "scripts/run_isolated_role.py"
        )
        self.isolation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.isolation)
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)
        reports = {}
        for role in self.domains:
            self.run_role(role, "probe")
            reports[role] = {
                "exit_code": 0,
                "report_ref": str(Path(self.domains[role]) / "probe.json"),
            }
        atomic_json(
            config["isolation_acceptance_ref"],
            {"scope": config["scope"], "role_domains": self.domains, "roles": reports},
        )
        self.run_role("operator", "observe")
        self.run_role("runtime", "observe-stopped")
        self.run_role("release", "register")
        self.run_role("release", "initialize")
        self.refresh()
        self.run_role("release", "decide")

    def run_role(self, role, action, *, ok=True):
        protect = []
        loaded = Path(self.domains["runtime"]) / "loaded.json"
        if role == "runtime" and action == "inherit":
            protect = [loaded]
        command = self.isolation.command(
            self.task,
            role,
            [sys.executable, str(self.script), str(self.ref), action],
            protected_paths=protect,
        )
        returned = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=20
        )
        if ok:
            self.assertEqual(returned.returncode, 0, returned.stdout + returned.stderr)
        else:
            self.assertNotEqual(
                returned.returncode, 0, returned.stdout + returned.stderr
            )
        return returned

    def refresh(self, loading="observe"):
        self.run_role("operator", "observe")
        self.run_role("runtime", loading)

    def test_real_roles_crash_reconcile_inherit_and_rollback(self):
        self.run_role("runtime", "inherit")
        old = json.loads(
            (Path(self.domains["runtime"]) / "new-binding.json").read_text()
        )
        self.run_role("release", "crash")
        self.assertEqual(self.manager.active()["bundle_id"], "initial")
        self.run_role("release", "activate")
        self.run_role("release", "activate")
        self.assertEqual(self.manager.active()["bundle_id"], "candidate-bundle")
        # Stale loading/safety evidence blocks the affected new task.
        self.run_role("runtime", "inherit", ok=False)
        self.refresh("observe-mismatch")
        bad = self.run_role("runtime", "inherit", ok=False)
        self.assertIn("loading differs", bad.stderr)
        self.refresh()
        self.run_role("runtime", "inherit")
        new = json.loads(
            (Path(self.domains["runtime"]) / "new-binding.json").read_text()
        )
        self.assertEqual(new["bundle"]["code_ref"], "code-v1")
        self.assertEqual(old["bundle"]["code_ref"], "code-v0")
        self.run_role("release", "nonancestor", ok=False)
        self.run_role("operator", "observe-unsafe")
        self.run_role("release", "rollback", ok=False)
        self.refresh()
        self.run_role("release", "rollback")
        self.assertEqual(self.manager.active()["bundle_id"], "initial")
        self.assertEqual(len(self.manager.records.all("activations")), 1)
        self.assertEqual(len(self.manager.records.all("rollbacks")), 1)

    def test_forged_safety_and_unknown_cost_cannot_activate(self):
        before = self.manager.active()
        self.run_role("operator", "observe-cost")
        self.run_role("release", "activate", ok=False)
        self.assertEqual(self.manager.active(), before)
        self.refresh()
        config = json.loads(self.ref.read_text())
        forged = Path(self.domains["research"]) / "forged.json"
        atomic_json(forged, json.loads(Path(config["safety_ref"]).read_text()))
        config["safety_ref"] = str(forged)
        # Even a matching JSON cannot substitute a different producer/domain.
        atomic_json(self.ref, config)
        returned = self.run_role("release", "activate", ok=False)
        self.assertIn("registration changed", returned.stderr)
        self.assertEqual(self.manager.active(), before)


if __name__ == "__main__":
    unittest.main()
