# ruff: noqa: UP017 - Python 3.10 support
"""Actual Host -> isolated inherited program -> developed candidates -> meta score."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.effective_psi import freeze_effective_psi
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.meta_exam import build_operations
from rsi_vla.midterm_host import MidtermHost, build_configuration, read
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json, initialize
from test_objective_evaluation import fixture

ROOT = Path(__file__).resolve().parents[2]


class MetaExamHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f14-meta-", dir=ROOT / "RPent")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.domains = {
            name: str(self.root / path)
            for name, path in {
                "operator": "operator",
                "research": "state/research",
                "evaluation": "state/evaluation",
                "release": "release",
                "runtime": "runtime",
            }.items()
        }
        for ref in self.domains.values():
            Path(ref).mkdir(parents=True)
        self.operator = Path(self.domains["operator"])
        self.store = ResearchStore(self.domains["research"])
        self.evaluations = EvaluationStore(self.domains["evaluation"])
        (self.evaluations.root / "results").mkdir()
        p, _, _ = fixture()
        p.update(
            scope_id="offline",
            synthetic_engineering=True,
            task_policy={"schema": "rsi-task-policy-v1"},
        )
        p["metrics"] = {
            "success_controlled": {
                "direction": "max",
                "min": 0,
                "max": 1,
                "weight": 0.75,
            },
            "success_old": {"direction": "max", "min": 0, "max": 1, "weight": 0.25},
        }
        self.protocol = p
        self.protocol_ref = self.operator / "protocol.json"
        atomic_json(self.protocol_ref, p)
        profile = self.operator / "profile.json"
        atomic_json(profile, {"control": {"action_chunk": 5}})
        self.base = RunBundle(
            "base",
            "theta0",
            "processor",
            str(profile) + "#/control",
            "rho",
            "psi0",
            "memory",
            p["protocol_id"],
            str(profile),
            "code",
        )
        source = self.store.root / "psi0.py"
        source.write_text(
            'def select_action(context):\n    return "inspect_evidence"\n'
        )
        initialize(
            self.store,
            self.base,
            psi={
                "research_rule_plugin": {
                    "schema": "research-rule-select-action-v1",
                    "source_ref": str(source),
                    "source": source.read_text(),
                }
            },
            memory={
                "provenance": "synthetic-development",
                "legal_information": ["public development sequence"],
            },
        )
        self.domain_policy = {
            "domain": "synthetic",
            "allowed_goal_kinds": ["manipulation"],
            "policy_ref": "synthetic-domain",
            "historical_development_bundle_ids": ["base"],
        }
        self.domain_ref = self.operator / "domain.json"
        atomic_json(self.domain_ref, self.domain_policy)
        self.limits = {
            "whole_job_wall_seconds": 60,
            "artifact_bytes": 10_000_000,
            "offline_calls": 20,
        }
        job = ResearchJob(self.store, "generation", "base", self.limits)
        tools = ResearchTools(self.store, job, p, domain_policy=self.domain_policy)
        config = {
            "isolation_task_root": str(self.root),
            "offline_fixture": True,
            "role_isolation": {"hidden_paths": [self.domains["evaluation"]]},
        }
        tools.prepare_code_candidate(
            "psi-revision",
            base_source_ref=source,
            source='def select_action(context):\n    return "bounded_experiment" if context["evidence_available"] else "defer"\n',
            declared_paths=["research_rule.py"],
            inputs={"evidence_available": True, "budget_available": True},
            config=config,
            timeout_seconds=3,
        )
        generated = tools.execute_code_candidate("psi-revision", config=config)
        self.assertTrue(generated["technical_valid"], generated)
        self.changed = RunBundle(
            **self.store.get("bundles", generated["candidate"]["bundle_id"])
        )
        self.store.put(
            "memory",
            "late-memory",
            {
                "provenance": "synthetic-development",
                "legal_information": [
                    "public development sequence",
                    "later legal fact",
                ],
            },
        )
        self.late = replace(self.base, bundle_id="late-base", memory_ref="late-memory")
        self.store.save_bundle(self.late)
        self.arms = {}
        for name, bundle, snapshot_id in (
            ("psi0", self.base, "psi0-frozen"),
            ("psi1", self.changed, "psi1-frozen"),
        ):
            policy_job = ResearchJob(
                self.store, name + "-freeze", bundle.bundle_id, self.limits
            )
            policy_tools = ResearchTools(
                self.store, policy_job, p, domain_policy=self.domain_policy
            )
            plugin = self.store.get("psi", bundle.psi_ref)["research_rule_plugin"]
            freeze_effective_psi(
                policy_tools,
                snapshot_id,
                program_refs=[plugin["source_ref"]],
                prompt_rules=["submit after development"],
                retrieval_rules=["development only"],
                dependencies={"model": "none", "tools": "bounded-rule"},
                development_costs=generated["costs"]
                if name == "psi1"
                else {"whole_job_wall_seconds": 0},
                development_families=["development"],
                reserved_families=["heldout"],
                development_mechanisms=["public-sequence"],
                reserved_mechanisms=["hidden-sequence"],
                origin_kind="program",
                origin_ref=plugin["source_ref"],
                parent_snapshot_id="psi0-frozen" if name == "psi1" else None,
            )
            self.arms[name] = {
                "policy_snapshot_id": snapshot_id,
                "policy_snapshot_ref": str(
                    self.store.path("effective_psi", snapshot_id).resolve()
                ),
                "facts_ref": str(
                    self.store.path("fact_memory_snapshots", "psi0-frozen").resolve()
                ),
            }
        latefacts = {
            **self.store.get("fact_memory_snapshots", "psi0-frozen"),
            "memory_ref": "late-memory",
            "memory": self.store.get("memory", "late-memory"),
        }
        self.store.put("fact_memory_snapshots", "late-facts", latefacts)
        self.arms["psi0_late"] = {
            **self.arms["psi0"],
            "facts_ref": str(
                self.store.path("fact_memory_snapshots", "late-facts").resolve()
            ),
        }
        self.store.save_development_result(
            ExperimentResult(
                "shared-development",
                "base",
                p["protocol_id"],
                p["benchmark_id"],
                "development",
                "offline",
                {"information_scope": "development"},
                ["synthetic-public-facts"],
                p["protocol_id"],
                {},
                {},
                {},
                outcome="success",
            )
        )
        self.problem_ref = Path(self.domains["evaluation"]) / "heldout.json"
        tasks = [
            {
                "task_key": key,
                "seed": seed,
                "sequence": [0, 1, 0, 0, 0] if key == "target" else [1] * 5,
                "expected_observation": 1,
            }
            for key in ["target", "old"]
            for seed in [0, 1]
        ]
        atomic_json(
            self.problem_ref,
            {
                "schema": "rsi-synthetic-sequence-problem-v1",
                "information_scope": "meta_exam",
                "tasks": tasks,
            },
        )
        self.dev_ref = self.operator / "development.json"
        atomic_json(
            self.dev_ref,
            {
                "schema": "rsi-synthetic-sequence-problem-v1",
                "information_scope": "development",
                "tasks": [
                    {
                        "task_key": "public-development",
                        "seed": 0,
                        "sequence": [1] * 5,
                        "expected_observation": 1,
                    }
                ],
            },
        )
        self.registration_ref = self.operator / "exam.json"
        self.registration = {
            "schema": "rsi-meta-exam-v1",
            "exam_id": "exam",
            "scope": "offline",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "mode": "endpoint",
            "arms": self.arms,
            "problems": [
                {
                    "problem_id": "sequence",
                    "family": "heldout",
                    "mechanism": "hidden-sequence",
                    "protocol_ref": str(self.protocol_ref),
                    "problem_kind": "synthetic_rule_choice",
                    "problem_ref": str(self.problem_ref),
                }
            ],
            "protocol_snapshots": {"sequence": p},
            "repetitions": [0],
            "budget_points": [20],
            "budget_dimension": "wall_seconds",
            "common": {
                "theta_ref": "theta0",
                "phi_ref": self.base.phi_ref,
                "non_test_policy_ref": "rho",
                "model_ref": "none",
                "tools_ref": "bounded-rule",
            },
            "shared_facts_ref": self.arms["psi0"]["facts_ref"],
            "missing_score": 0,
            "minimum_gain": 0.1,
            "retention_max_drop": 0,
            "policy_generation": {
                "generation_id": "psi-revision",
                "costs": generated["costs"],
                "independent_generations": 1,
            },
            "role_domains": self.domains,
            "protected_paths": [str(self.operator)],
            "isolation_acceptance_ref": str(self.operator / "acceptance.json"),
            "slots": [],
        }
        self.scope_ref = self.operator / "scope.json"
        self.phase_ref = self.operator / "phase.json"
        self.admin_ref = self.operator / "admin.json"
        self.core_ref = self.operator / "core.json"
        self.host_ref = self.operator / "host.json"
        spec = importlib.util.spec_from_file_location(
            "meta_isolation", ROOT / "scripts/run_isolated_role.py"
        )
        self.isolation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.isolation)

    def prepare(self, mode="endpoint"):
        points = [20] if mode == "endpoint" else [20, 40]
        self.registration.update(mode=mode, budget_points=points)
        for arm, bundle in (
            ("psi0", self.base),
            ("psi1", self.changed),
            ("psi0_late", self.late),
        ):
            prior = []
            for point in points:
                identity = f"{arm}-{point}"
                request = {
                    "request_id": identity,
                    "job_id": arm + "-exam-process",
                    "base_bundle_id": bundle.bundle_id,
                    "domain_policy_ref": str(self.domain_ref),
                    "result_ids": ["shared-development", *prior],
                    "job_limits": self.limits,
                }
                experiments = {
                    action: {
                        "experiment_id": identity + "-experiment",
                        "bundle_id": identity + "-bundle",
                        "chunk_steps": steps,
                        "criterion_ref": self.protocol["protocol_id"],
                        "distinguishing_prediction": "shorter chunk identifies the target sequence element",
                        "cost_limit": {"offline_calls": 1},
                        "evidence_level": "offline",
                    }
                    for action, steps in (
                        ("inspect_evidence", 4),
                        ("bounded_experiment", 2),
                    )
                }
                self.registration["slots"].append(
                    {
                        "slot_id": identity,
                        "problem_id": "sequence",
                        "repeat": 0,
                        "arm": arm,
                        "budget_point": point,
                        "research_request": request,
                        "action_experiments": experiments,
                        "development_problem_ref": str(self.dev_ref),
                        "development_value": "select a useful chunk using legal development evidence",
                        "hypotheses": [
                            {
                                "explanation": "chunk boundary misses target",
                                "prediction": "shorter chunk changes observation",
                                "distinguishing_test": "public sequence",
                            },
                            {
                                "explanation": "observation uninformative",
                                "prediction": "chunk change does not help",
                                "distinguishing_test": "public sequence",
                            },
                        ],
                        "candidate_id": identity + "-candidate",
                        "evaluation_operations": [],
                        "result_id": identity + "-exam-result",
                    }
                )
                prior.append(identity + "-development")
        atomic_json(self.registration_ref, self.registration)
        generated = build_operations(self.registration_ref)
        caps = {
            "wall_seconds": 4000,
            "storage_bytes": 2 * 1024**3,
            "current_disk_bytes": 2 * 1024**3,
            "model_requests": 0,
            "gpu_seconds": 0,
            "concurrent_gpu_devices": 0,
        }
        expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        atomic_json(
            self.scope_ref,
            {
                "schema": "rsi-midterm-scope-v1",
                "scope_id": "offline",
                "status": "active",
                "authorized": True,
                "dynamic_authorized": False,
                "cpu_authorized": True,
                "cpu_operation_ids": [
                    o["operation_id"] for o in generated["operations"]
                ],
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
        atomic_json(
            self.admin_ref,
            {
                "authorized": True,
                "dynamic_authorized": False,
                "scope": "offline",
                "protocol_id": self.protocol["protocol_id"],
                "expires_at": expiry,
                "phase_snapshot_ref": str(self.phase_ref),
                "approval_reference": str(self.scope_ref),
            },
        )
        core = {
            "schema": "rsi-task-completion-v1",
            "offline_fixture": True,
            "repository_root": str(ROOT),
            "isolation_task_root": str(self.root),
            "scope_authorization_ref": str(self.scope_ref),
            "administrative_authorization_ref": str(self.admin_ref),
            "phase_ref": str(self.phase_ref),
            "protocol_ref": str(self.protocol_ref),
            "frozen_protocol_ref": str(self.protocol_ref),
            "base_bundle_id": "base",
            "research_store_ref": self.domains["research"],
            "evaluation_store_ref": self.domains["evaluation"],
            "release_root": self.domains["release"],
            "profile_ref": str(self.operator / "profile.json"),
            "role_isolation": {
                "hidden_paths": [
                    str(self.problem_ref),
                    str(Path(self.domains["evaluation"]) / "results"),
                ]
            },
            **generated,
        }
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
        probe = self.operator / "probe.py"
        probe.write_text(
            'import json,os,sys\nfrom pathlib import Path\nfrom rsi_vla.storage import atomic_json\nc=json.loads(Path(sys.argv[1]).read_text());r=os.environ["RSI_ISOLATED_ROLE"];d=c["role_domains"];denied=[]\nfor n,p in d.items():\n if n==r:continue\n try:(Path(p)/"forbidden").write_text("bad")\n except OSError:denied.append(n)\n else:raise AssertionError("cross-domain write")\natomic_json(Path(d[r])/"probe.json",dict(role=r,scope=c["scope"],allowed_write=True,denied_writes=len(denied)==4))\n'
        )
        reports = {}
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)
        for role in self.domains:
            result = subprocess.run(
                self.isolation.command(
                    self.root,
                    role,
                    [sys.executable, str(probe), str(self.registration_ref)],
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
            self.registration["isolation_acceptance_ref"],
            {"scope": "offline", "role_domains": self.domains, "roles": reports},
        )
        return core

    def finish(self):
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline:
            host = MidtermHost(self.host_ref)
            state = host.advance()
            if state["status"] == "completed":
                return host
            if state["status"] in {"unknown", "blocked", "failed"}:
                self.fail(
                    repr(state)
                    + "\n"
                    + repr(host.coordinator().status().get("pending_receipt"))
                    + "\n"
                    + "\n".join(
                        p.read_text()[-6000:]
                        for p in host.root.glob("jobs/*/backend.log")
                    )
                )
            time.sleep(0.03)
        self.fail("bounded meta Host did not complete: " + repr(state))

    def test_policy_revision_freeze_real_host_exam_and_new_job_inheritance(self):
        self.prepare()
        host = self.finish()
        score = self.evaluations.get("meta_exam_scores", "exam")
        self.assertEqual(score["verdict"], "accept", score)
        self.assertEqual(
            score["metrics"]["endpoint"], {"psi0": 0.25, "psi1": 1.0, "psi0_late": 0.25}
        )
        self.assertIsNone(score["metrics"]["auc"])
        self.assertEqual(len(self.store.all("meta_exam_selections")), 3)
        for arm, expected in (
            ("psi0", "inspect_evidence"),
            ("psi1", "bounded_experiment"),
            ("psi0_late", "inspect_evidence"),
        ):
            selected = self.store.get("meta_exam_selections", arm + "-20")
            self.assertEqual(selected["action"], expected)
            self.assertTrue(selected["candidate_snapshot"]["development_results"])
            self.assertTrue(selected["feedback"]["receipt"]["wall_seconds"] > 0)
        self.assertEqual(host.ledger.read()["used"]["model_requests"], 0)
        self.assertEqual(host.ledger.read()["used"]["gpu_seconds"], 0)
        self.assertEqual(len(host.coordinator().status()["receipts"]), 6)
        self.assertEqual(
            read(Path(self.domains["release"]) / "active_research_policy.json")[
                "bundle_id"
            ],
            self.changed.bundle_id,
        )
        # Enter a fresh real Research role, with sealed exam/qualifications hidden.
        script = self.operator / "inherit.py"
        script.write_text(
            'import json,sys\nfrom pathlib import Path\nfrom rsi_vla.evaluation import EvaluationStore\nfrom rsi_vla.release import ReleaseManager\nfrom rsi_vla.storage import ResearchStore,atomic_json\nfrom rsi_vla.version_binding import start_research_job\nc=json.loads(Path(sys.argv[1]).read_text());s=ResearchStore(c["research_store_ref"]);m=ReleaseManager(c["release_root"],s,EvaluationStore(c["evaluation_store_ref"]))\nj=start_research_job(s,"post-exam",bundle_id="base",limits={"offline_calls":2},release_manager=m,scope="offline")\nassert j.state["bundle_id"]=="psi-revision-bundle"\nold=start_research_job(s,"psi0-exam-process",bundle_id="psi-revision-bundle",limits={"whole_job_wall_seconds":60,"artifact_bytes":10000000,"offline_calls":20},release_manager=m,scope="offline")\nassert old.state["bundle_id"]=="base"\nfor ref in [Path(c["evaluation_store_ref"])/"meta_exam_scores",Path(c["release_root"])/"research_policy_qualifications"]:\n try:(ref/("exam.json" if ref.name=="meta_exam_scores" else "exam-policy.json")).read_text()\n except (PermissionError,FileNotFoundError):pass\n else:raise AssertionError("sealed scores readable")\natomic_json(s.root/"inheritance-proof.json",j.state)\n'
        )
        hidden = [
            str(Path(self.domains["evaluation"]) / "meta_exam_scores"),
            str(Path(self.domains["release"]) / "research_policy_qualifications"),
        ]
        result = subprocess.run(
            self.isolation.command(
                self.root,
                "research",
                [sys.executable, str(script), str(self.core_ref)],
                hidden_paths=hidden,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registered_milestones_keep_same_process_job_and_fixed_candidates(self):
        self.prepare("milestones")
        host = self.finish()
        score = self.evaluations.get("meta_exam_scores", "exam")
        self.assertEqual(score["verdict"], "accept", score)
        self.assertEqual(score["metrics"]["auc"]["psi1"], 1.0)
        self.assertEqual(len(self.store.all("meta_exam_selections")), 6)
        self.assertEqual(
            len(self.store.get("jobs", "psi1-exam-process")["charges"]) > 3, True
        )
        self.assertEqual(len(host.coordinator().status()["receipts"]), 9)


if __name__ == "__main__":
    unittest.main()
