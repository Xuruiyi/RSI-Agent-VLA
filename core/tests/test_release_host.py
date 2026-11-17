# ruff: noqa: UP017 - Python 3.10 support
"""One real Host graph evaluates, publishes, binds a new Runtime and rolls back."""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import test_meta_exam_host
from rsi_vla.__main__ import main as core_main
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.midterm_host import build_configuration, read
from rsi_vla.release import ReleaseManager
from rsi_vla.rule_updates import write_rule_trial
from rsi_vla.storage import ResearchStore, atomic_json

ROOT = Path(__file__).resolve().parents[2]


class ReleaseHostTests(unittest.TestCase):
    finish = test_meta_exam_host.MetaExamHostTests.finish

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f13-host-", dir=ROOT / "RPent")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.operator = self.root / "operator"
        self.operator.mkdir()
        self.domains = {
            name: str(self.root / relative)
            for name, relative in {
                "operator": "operator",
                "research": "state/research",
                "evaluation": "state/evaluation",
                "release": "release",
                "runtime": "runtime",
            }.items()
        }
        for ref in self.domains.values():
            Path(ref).mkdir(parents=True, exist_ok=True)
        p = read(ROOT / "config/pilot_v2.json")
        p.update(
            scope="offline",
            scope_id="offline",
            synthetic_engineering=True,
            task_policy={"schema": "rsi-task-policy-v1"},
            hard_constraints=["safety", "compatibility"],
            initial_requirements={
                "metric_minima": {"success_base": 1, "success_old": 1},
                "required_old_tasks": ["old_milk", "old_drawer"],
            },
            cost_weights={},
        )
        p, profile = self.configure_fixture(
            p, read(ROOT / "config/run_profile_v2.json")
        )
        self.protocol = p
        self.protocol_ref = self.operator / "protocol.json"
        atomic_json(self.protocol_ref, p)
        self.profile_ref = self.operator / "profile.json"
        atomic_json(self.profile_ref, profile)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(
                [
                    "init",
                    "--artifacts",
                    str(self.root / "state"),
                    "--profile",
                    str(self.profile_ref),
                    "--protocol",
                    str(self.protocol_ref),
                ]
            )
        self.store = ResearchStore(self.domains["research"])
        self.evals = EvaluationStore(self.domains["evaluation"])
        self.manager = ReleaseManager(self.domains["release"], self.store, self.evals)
        self.evals.put("protocols", p["protocol_id"], p)
        base = RunBundle(**self.store.get("bundles", "initial-medium"))
        self.prepare_candidate(base)
        self.manager.bootstrap(base.bundle_id)
        self.problem_ref = Path(self.domains["evaluation"]) / "problem.json"
        groups = {row["task_key"]: row["group"] for row in p["tasks"]}
        tasks = [
            {
                "task_key": key,
                "seed": seed,
                "sequence": [0, 1, 0, 0, 0] if groups[key] == "controlled" else [1] * 5,
                "expected_observation": 1,
            }
            for key in p["task_sets"]["release"]["task_keys"]
            for seed in p["task_sets"]["release"]["seeds"]
        ]
        atomic_json(
            self.problem_ref,
            self.problem_document(p, tasks),
        )
        self.scope_ref = self.operator / "scope.json"
        self.phase_ref = self.operator / "phase.json"
        self.registration_ref = self.operator / "publication.json"
        expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        registration = {
            "schema": "rsi-release-safety-registration-v1",
            "authorized": True,
            "scope": "offline",
            "service_mode": "no_service",
            "expires_at": expiry,
            "role_domains": self.domains,
            "protected_paths": [str(self.registration_ref)],
            "isolation_acceptance_ref": str(self.operator / "acceptance.json"),
            "safety_ref": str(self.operator / "safety.json"),
            "loaded_ref": str(Path(self.domains["runtime"]) / "loaded.json"),
            "phase_ref": str(self.phase_ref),
            "matrices": [
                {
                    "result_id": identity,
                    "bundle_id": bundle,
                    "candidate_id": candidate,
                    "problem_ref": str(self.problem_ref),
                    "evaluation_kind": getattr(
                        self, "matrix_evaluation_kind", "sequence"
                    ),
                }
                for identity, bundle, candidate in [
                    ("initial", "initial-medium", None),
                    ("baseline", "initial-medium", None),
                    ("anchor", "initial-medium", None),
                    ("candidate-result", "changed", "candidate"),
                ]
            ],
        }
        atomic_json(self.registration_ref, registration)
        ops = []

        def add(identity, kind, **payload):
            ops.append(
                {
                    "operation_id": identity,
                    "kind": kind,
                    "stage": "release",
                    "dynamic": False,
                    "payload": {
                        "registration_ref": str(self.registration_ref),
                        **payload,
                    },
                }
            )

        def matrix(identity, bundle, candidate=None):
            add(
                identity + "-matrix",
                "release_matrix_evaluation",
                result_id=identity,
                bundle_id=bundle,
                candidate_id=candidate,
            )

        def observe(label):
            add(label + "-loaded", "release_loaded")
            add(label + "-safe", "release_observe")

        matrix("initial", "initial-medium")
        observe("bootstrap")
        add(
            "initialize",
            "release_initialize",
            qualification_id="initial-qualified",
            result_id="initial",
        )
        observe("before-activation")
        matrix("baseline", "initial-medium")
        matrix("anchor", "initial-medium")
        matrix("candidate-result", "changed", "candidate")
        add(
            "decide",
            "release_evaluate",
            decision_id="accept",
            candidate_id="candidate",
            candidate_result="candidate-result",
            baseline_result="baseline",
            anchor_result="anchor",
        )
        add("activate", "release_activate", decision_id="accept", scope="offline")
        observe("after-activation")
        argv = [
            "--root",
            str(ROOT),
            "--artifacts",
            str(self.root / "state"),
            "--protocol",
            str(self.protocol_ref),
            "--deployment-release",
            self.domains["release"],
            "--run",
            "new-runtime",
            "--task-key",
            "position_swap_salad",
            "--seed",
            "3",
            "--output",
            str(Path(self.domains["runtime"]) / "new-runtime"),
        ]
        add("inherit", "release_inherit", runtime_argv=argv)
        add(
            "rollback",
            "release_rollback",
            target_bundle_id="initial-medium",
            reason="synthetic regression",
        )
        observe("after-rollback")
        add(
            "inherit-rollback",
            "release_inherit",
            runtime_argv=[
                "restored-runtime" if arg == "new-runtime" else arg for arg in argv
            ],
        )
        identities = [op["operation_id"] for op in ops]
        nodes = {
            key: {"operation_id": key, "transitions": [{"next": nxt}]}
            for key, nxt in zip(identities, identities[1:] + ["done"])
        }
        nodes["done"] = {"terminal": "release_complete"}
        caps = {
            "wall_seconds": 5000,
            "storage_bytes": 2 * 1024**3,
            "current_disk_bytes": 2 * 1024**3,
            "model_requests": 0,
            "gpu_seconds": 0,
            "concurrent_gpu_devices": 0,
        }
        atomic_json(
            self.scope_ref,
            {
                "schema": "rsi-midterm-scope-v1",
                "scope_id": "offline",
                "status": "active",
                "authorized": True,
                "dynamic_authorized": False,
                "cpu_authorized": True,
                "cpu_operation_ids": identities,
                "owner_uid": os.getuid(),
                "write_root": str(self.root),
                "storage_write_roots": [str(self.root)],
                "outer_limits": caps,
                "expires_at": expiry,
            },
        )
        limits = {k: v for k, v in caps.items() if k != "concurrent_gpu_devices"}
        atomic_json(
            self.phase_ref,
            {
                "limits": limits,
                "used": dict.fromkeys(limits, 0),
                "remaining": limits,
                "unresolved_costs": [],
            },
        )
        admin_ref = self.operator / "admin.json"
        atomic_json(
            admin_ref,
            {
                "authorized": True,
                "scope": "offline",
                "protocol_id": p["protocol_id"],
                "phase_snapshot_ref": str(self.phase_ref),
                "approval_reference": str(self.scope_ref),
                "expires_at": expiry,
            },
        )
        core = {
            "schema": "rsi-task-completion-v1",
            "offline_fixture": True,
            "repository_root": str(ROOT),
            "isolation_task_root": str(self.root),
            "scope_authorization_ref": str(self.scope_ref),
            "administrative_authorization_ref": str(admin_ref),
            "phase_ref": str(self.phase_ref),
            "protocol_ref": str(self.protocol_ref),
            "frozen_protocol_ref": str(self.protocol_ref),
            "profile_ref": str(self.profile_ref),
            "base_bundle_id": "initial-medium",
            "research_store_ref": self.domains["research"],
            "evaluation_store_ref": self.domains["evaluation"],
            "release_root": self.domains["release"],
            "research_requests": [],
            "operations": ops,
            "continuation": {
                "schema": "rsi-task-continuation-v1",
                "start": identities[0],
                "nodes": nodes,
            },
            "role_isolation": {"hidden_paths": [str(self.problem_ref)]},
        }
        self.core_ref = self.operator / "core.json"
        self.host_ref = self.operator / "host.json"
        atomic_json(self.core_ref, core)
        host = build_configuration(
            self.core_ref,
            self.host_ref,
            self.operator / "hub.json",
            python=sys.executable,
            poll_seconds=0.02,
        )
        atomic_json(self.host_ref, host)
        atomic_json(host["inputs_ref"], {})
        spec = importlib.util.spec_from_file_location(
            "release_host_isolation", ROOT / "scripts/run_isolated_role.py"
        )
        iso = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(iso)
        script = self.operator / "probe.py"
        script.write_text(
            'import json,os,sys\nfrom pathlib import Path\nfrom rsi_vla.storage import atomic_json\nc=json.loads(Path(sys.argv[1]).read_text());r=os.environ["RSI_ISOLATED_ROLE"];d=c["role_domains"];denied=[]\nfor n,p in d.items():\n if n==r:continue\n try:(Path(p)/"forbidden").write_text("bad")\n except OSError:denied.append(n)\n else:raise AssertionError("cross-domain write")\natomic_json(Path(d[r])/"probe.json",dict(role=r,scope=c["scope"],allowed_write=True,denied_writes=len(denied)==4))\n'
        )
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)
        reports = {}
        for role in self.domains:
            result = subprocess.run(
                iso.command(
                    self.root,
                    role,
                    [sys.executable, str(script), str(self.registration_ref)],
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reports[role] = {
                "exit_code": 0,
                "report_ref": str(Path(self.domains[role]) / "probe.json"),
            }
        atomic_json(
            registration["isolation_acceptance_ref"],
            {"scope": "offline", "role_domains": self.domains, "roles": reports},
        )

    def configure_fixture(self, protocol, profile):
        """Prepare the owning protocol/profile before any immutable Store write."""
        return self.configure_protocol(protocol), profile

    def problem_document(self, protocol, tasks):
        """Register the independent task document consumed by each full matrix."""
        return {
            "schema": "rsi-synthetic-sequence-problem-v1",
            "information_scope": "release",
            "tasks": tasks,
        }

    def configure_protocol(self, protocol):
        """Allow integration fixtures to register their owning task contract first."""
        return protocol

    def prepare_candidate(self, base):
        """Prepare the concrete candidate consumed by the common Host graph."""
        p = self.protocol
        target, _diff = write_rule_trial(
            self.store,
            base,
            bundle_id="changed-rule",
            artifact_id="changed",
            chunk_steps=2,
        )
        candidate_profile = read(self.profile_ref)
        candidate_profile["control"]["executed_prefix_steps"] = 2
        candidate_profile_ref = self.operator / "candidate-profile.json"
        atomic_json(candidate_profile_ref, candidate_profile)
        target = replace(
            target, bundle_id="changed", backend_ref=str(candidate_profile_ref)
        )
        self.store.save_bundle(target)
        self.store.save_candidate(
            Candidate(
                "candidate",
                base.bundle_id,
                target.bundle_id,
                p["protocol_id"],
                p["benchmark_id"],
                {
                    "phi_ref": target.phi_ref,
                    "diff_ref": target.diff_ref,
                    "backend_ref": target.backend_ref,
                },
                [target.phi_ref, target.diff_ref],
                [],
                {},
                {},
                "offline",
            )
        )

    def test_automatic_matrix_qualification_activation_runtime_binding_and_rollback(
        self,
    ):
        host = self.finish()
        receipts = host.coordinator().status()["receipts"]
        self.assertEqual(receipts["decide"]["value"]["verdict"], "accept")
        self.assertEqual(
            receipts["inherit"]["value"]["plan"]["context"]["bundle_id"], "changed"
        )
        self.assertEqual(
            receipts["inherit"]["value"]["selection"]["mode"], "deployment"
        )
        self.assertEqual(
            receipts["inherit-rollback"]["value"]["plan"]["context"]["bundle_id"],
            "initial-medium",
        )
        self.assertEqual(self.manager.active()["bundle_id"], "initial-medium")
        self.assertEqual(len(self.manager.records.all("activations")), 1)
        self.assertEqual(len(self.manager.records.all("rollbacks")), 1)
        self.assertEqual(host.ledger.read()["used"]["model_requests"], 0)
        self.assertEqual(host.ledger.read()["used"]["gpu_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
