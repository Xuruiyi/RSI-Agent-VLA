# Copyright 2026 RSI-Agent-VLA contributors.

"""Offline Q16 raw-result consumers and actual ReleaseManager transitions."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rsi_vla.__main__ import main as core_main
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.midterm_deployment import (
    activate,
    decide,
    evaluate_qualification,
    inheritance_plan,
    initialize,
)
from rsi_vla.release import ReleaseManager
from rsi_vla.runtime import execute_runtime, execution_binding
from rsi_vla.storage import ResearchStore, atomic_json

from robots.libero import execution_arms

ROOT = Path(execution_arms.__file__).resolve().parents[3]


class DirectDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix=".p5-offline-",
            dir=ROOT / "RPent"
            if os.environ.get("RPENT_RUN_NAMESPACE_TEST") == "1"
            else None,
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.domains = {
            k: str(self.root / v)
            for k, v in {
                "operator": "operator",
                "research": "state/research",
                "evaluation": "state/evaluation",
                "release": "release",
                "runtime": "runtime",
            }.items()
        }
        for path in self.domains.values():
            Path(path).mkdir(parents=True)
        self.operator = Path(self.domains["operator"])
        self.reg_ref = self.operator / "direct-Q.json"
        now = datetime.now(timezone.utc)
        self.registered = now - timedelta(minutes=2)
        self.scope_ref = self.operator / "scope.json"
        atomic_json(
            self.scope_ref,
            {
                "schema": "rsi-midterm-scope-v1",
                "scope_id": "new-task",
                "authorized": True,
                "dynamic_authorized": True,
                "owner_uid": os.getuid(),
                "status": "active",
                "write_root": str(self.root),
                "accepted_at": (now - timedelta(hours=1)).isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
            },
        )
        self.protocol = json.loads((ROOT / "config/pilot_v2.json").read_text())
        self.protocol.update(
            protocol_id="new-Q16",
            scope_id="new-task",
            comparison_only=False,
            dynamic_authorized=True,
        )
        self.protocol["initial_requirements"] = {
            "metric_minima": {"success_base": 1, "success_old": 1},
            "required_old_tasks": ["old_milk", "old_drawer"],
        }
        self.protocol["hard_constraints"]["initial_capabilities"] = (
            "base and both critical old tasks must all succeed"
        )
        self.protocol["task_sets"]["release"]["seeds"] = [16, 17]
        self.protocol["task_sets"]["development"]["seeds"] = [18]
        self.protocol["budget"]["per_attempt_proposal"].update(
            max_episode_steps=400, tool_calls=512
        )
        self.protocol_ref = self.operator / "Q-protocol.json"
        atomic_json(self.protocol_ref, self.protocol)
        profile = json.loads((ROOT / "config/run_profile_v2.json").read_text())
        profile["dynamic_authorized"] = True
        profile.pop("collection", None)
        profile.pop("auxiliary_models", None)
        profile["control"]["max_episode_steps"] = 400
        profile["planner"]["backend"] = "flash"
        profile["execution_arm"] = {
            "schema": "libero-execution-arm-v1",
            "arm": "D",
            "max_chunks": 80,
        }
        profile["common_execution"] = {
            "schema": "libero-common-execution-v1",
            "observation_cadence": "prediction",
            "allowed_motion_tools": ["vla_execute"],
            "action_prefix_steps": 5,
        }
        profile_ref = self.operator / "profile.json"
        atomic_json(profile_ref, profile)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(
                [
                    "init",
                    "--artifacts",
                    str(self.state),
                    "--profile",
                    str(profile_ref),
                    "--protocol",
                    str(self.protocol_ref),
                ]
            )
        self.research = ResearchStore(self.domains["research"])
        self.evaluation = EvaluationStore(self.domains["evaluation"])
        self.base = self.research.get("bundles", self.protocol["anchor_bundle_id"])
        self.profile = json.loads(Path(self.base["backend_ref"]).read_text())
        theta = self.operator / "theta1.json"
        atomic_json(theta, {"checkpoint": "new-checkpoint"})
        self.trial = {
            **self.base,
            "bundle_id": "Q-candidate",
            "theta_ref": str(theta),
            "parent_bundle_id": self.base["bundle_id"],
        }
        self.research.save_bundle(RunBundle(**self.trial))
        self.candidate = Candidate(
            "Q-candidate-id",
            self.base["bundle_id"],
            "Q-candidate",
            self.protocol["protocol_id"],
            self.protocol["benchmark_id"],
            {"theta_ref": str(theta)},
            [str(theta)],
            [],
            {},
            {},
            "simulation",
        )
        self.research.save_candidate(self.candidate)
        self.selection_ref = self.operator / "candidate-selection.json"
        source_gate = Path(self.domains["evaluation"]) / "P4-gate.json"
        atomic_json(
            source_gate,
            {
                "verdict": "accept",
                "protocol_id": "P4-comparison",
                "release_eligible": False,
                "bundle_id": "Q-candidate",
            },
        )
        selected_at = now - timedelta(minutes=1)
        self.selection = {
            "registration_ref": str(self.reg_ref),
            "selection_rule": "fixed P4 eligibility then utility/cost ranking",
            "candidate_id": self.candidate.candidate_id,
            "execution_binding": execution_binding(
                self.research, self.trial, self.profile
            ),
            "source_result_refs": [str(source_gate)],
            "theta_effect_admission_ref": str(source_gate),
            "selected_at": selected_at.isoformat(),
        }
        atomic_json(self.selection_ref, self.selection)
        manifests = {}
        for side, bundle in (("baseline", self.base), ("candidate", self.trial)):
            rows = []
            for task in self.protocol["tasks"]:
                for seed in (16, 17):
                    run = f"{side}-{task['task_key']}-{seed}"
                    out = Path(self.domains["runtime"]) / run
                    identity = {
                        "run_id": run,
                        "episode_id": task["task_key"] + f"-s{seed}",
                        "bundle_id": bundle["bundle_id"],
                        "protocol_id": self.protocol["protocol_id"],
                    }
                    binding = execution_binding(self.research, bundle, self.profile)
                    atomic_json(
                        out / "runtime_context.json",
                        {
                            **identity,
                            "execution_binding": binding,
                            "execution_arm": self.profile["execution_arm"],
                            "common_execution": self.profile["common_execution"],
                            "limits": {"environment_steps": 400},
                        },
                    )
                    (out / "action_events.jsonl").write_text(
                        json.dumps(
                            {
                                **identity,
                                "sequence": 0,
                                "kind": "run_start",
                                "utc_ns": int(
                                    (now + timedelta(seconds=5)).timestamp() * 1e9
                                ),
                            }
                        )
                        + "\n"
                    )
                    success = side == "candidate" or task["group"] != "controlled"
                    native = {
                        **identity,
                        "task": {
                            "suite": task["suite"],
                            "task": task["task_id"],
                            "seed": seed,
                        },
                        "planner": self.profile["planner"],
                        "benchmark_success": success,
                        "terminated": success,
                        "truncated": not success,
                        "native_result_source": {
                            "terminated": "native fixture",
                            "truncated": "native fixture",
                        },
                    }
                    atomic_json(out / "result.json", native)
                    rows.append(
                        {
                            **{k: identity[k] for k in ("run_id", "episode_id")},
                            "task_key": task["task_key"],
                            "seed": seed,
                            "result_path": str(out / "result.json"),
                        }
                    )
            manifests[side] = rows
        self.registration = {
            "version": "rsi-midterm-direct-deployment-v1",
            "authorized": True,
            "scope": "new-task",
            "scope_authorization_ref": str(self.scope_ref),
            "task_root": str(self.root),
            "artifacts_root": str(self.state),
            "role_domains": self.domains,
            "registration_ref": str(self.reg_ref),
            "registered_at": self.registered.isoformat(),
            "expires_at": (now + timedelta(minutes=30)).isoformat(),
            "repository_root": str(ROOT),
            "python_executable": sys.executable,
            "protocol_ref": str(self.protocol_ref),
            "protocol_snapshot": self.protocol,
            "controller_selection_rules": {
                "baseline": "theta0 legal development utility/cost ranking",
                "candidate": self.selection["selection_rule"],
            },
            "p4_protocol_id": "P4-comparison",
            "baseline": {
                "execution_binding": execution_binding(
                    self.research, self.base, self.profile
                ),
                "result_id": "Q-baseline-result",
                "manifest": manifests["baseline"],
            },
            "candidate": {
                "selection_ref": str(self.selection_ref),
                "result_id": "Q-candidate-result",
                "manifest": manifests["candidate"],
            },
            "initial_qualification_id": "new-initial",
            "decision_id": "new-upgrade",
            "protected_paths": [str(self.scope_ref), str(Path(__file__).parent)],
            "isolation_acceptance_ref": str(self.operator / "isolation.json"),
            "inheritance": {
                "run_id": "new-inherited-task",
                "task_key": "position_swap_salad",
                "seed": 18,
                "output_dir": str(Path(self.domains["runtime"]) / "inherit"),
                "cuda_device": 0,
                "authorization_ref": str(self.operator / "inherit-permit.json"),
            },
        }
        atomic_json(self.reg_ref, self.registration)
        self.manager = ReleaseManager(
            self.domains["release"], self.research, self.evaluation
        )
        self.role = patch("rsi_vla.midterm_deployment.require_role")
        self.role.start()
        self.addCleanup(self.role.stop)
        self.readonly = patch("rsi_vla.midterm_deployment._readonly", return_value=True)
        self.readonly.start()
        self.addCleanup(self.readonly.stop)
        self.science_verify = patch(
            "rsi_vla.midterm_science.verify_scientific_admission",
            return_value=json.loads(source_gate.read_text()),
        )
        self.science_verify.start()
        self.addCleanup(self.science_verify.stop)
        keys = [
            "execution_health",
            "launcher_wrapper_completion",
            "complete_usage",
            "wall_gpu_storage_budget",
            "identity",
            "model_reference",
            "vla_actual_loading_and_prediction",
            "tool_budget",
            "request_budget",
            "step_budget",
            "all_tools_use_counted_dispatcher",
            "no_mid_attempt_reset",
            "clean_completion",
        ]
        report = {
            "passed": True,
            "checks": dict.fromkeys(keys, True),
            "costs": {"model_requests": 0},
            "notes": [],
            "confirmed_violations": [],
        }
        self.audit = patch("rsi_vla.evaluator.inspect_attempt", return_value=report)
        self.audit.start()
        self.addCleanup(self.audit.stop)
        self.progress = patch(
            "rsi_vla.evaluator.inspect_progress",
            return_value={"value": True, "evidence_refs": []},
        )
        self.progress.start()
        self.addCleanup(self.progress.stop)

    def evaluate(self):
        evaluate_qualification(self.reg_ref, "baseline")
        evaluate_qualification(self.reg_ref, "candidate")

    def test_fresh_qualification_precedes_real_activation_then_upgrade_and_selection(
        self,
    ):
        self.evaluate()
        self.assertFalse(self.manager.active_path.exists())
        initial = initialize(self.reg_ref, queues_empty=True)
        self.assertEqual(initial["active"]["activation"], "qualified_initial")
        self.assertEqual(initial["active"]["revision"], 1)
        self.assertEqual(initialize(self.reg_ref, queues_empty=True), initial)
        decision = decide(self.reg_ref)
        self.assertEqual(decision["decision"]["verdict"], "accept")
        self.assertEqual(self.manager.active(), initial["active"])
        current = activate(self.reg_ref, queues_empty=True)["active"]
        self.assertEqual(current["revision"], 2)
        plan = inheritance_plan(self.reg_ref)
        self.assertEqual(plan["context"]["bundle_id"], "Q-candidate")
        self.assertEqual(plan["context"]["release_binding"], current)
        self.assertEqual(
            plan["context"]["deployment_release_root"], self.domains["release"]
        )
        self.assertEqual(plan["selection"]["mode"], "deployment")
        self.assertFalse(Path(plan["context"]["output_dir"]).exists())

    def test_candidate_base_failure_cannot_be_accepted_even_with_net_utility_gain(self):
        row = next(
            r
            for r in self.registration["candidate"]["manifest"]
            if r["task_key"] == "base_salad"
        )
        path = Path(row["result_path"])
        v = json.loads(path.read_text())
        v.update(benchmark_success=False, terminated=False, truncated=True)
        atomic_json(path, v)
        self.evaluate()
        initialize(self.reg_ref, queues_empty=True)
        current = self.manager.active()
        result = decide(self.reg_ref)
        self.assertEqual(result["decision"]["verdict"], "reject")
        self.assertEqual(
            activate(self.reg_ref, queues_empty=True)["status"], "kept_current"
        )
        self.assertEqual(self.manager.active(), current)

    def test_failed_initial_old_task_keeps_domain_uninitialized(self):
        row = next(
            r
            for r in self.registration["baseline"]["manifest"]
            if r["task_key"] == "old_milk"
        )
        path = Path(row["result_path"])
        v = json.loads(path.read_text())
        v.update(benchmark_success=False, terminated=False, truncated=True)
        atomic_json(path, v)
        evaluate_qualification(self.reg_ref, "baseline")
        self.assertEqual(
            initialize(self.reg_ref, queues_empty=True)["status"],
            "no_qualified_initial",
        )
        self.assertFalse(self.manager.active_path.exists())
        self.assertFalse(self.manager.records.all("qualifications"))
        self.assertEqual(decide(self.reg_ref)["status"], "blocked_no_qualified_initial")
        self.assertEqual(
            activate(self.reg_ref, queues_empty=True)["status"],
            "blocked_no_qualified_initial",
        )
        self.assertEqual(
            inheritance_plan(self.reg_ref)["status"], "blocked_no_qualified_initial"
        )

    def test_live_active_is_rechecked_before_any_runtime_launch(self):
        self.evaluate()
        initialize(self.reg_ref, queues_empty=True)
        plan = inheritance_plan(self.reg_ref)
        permit = {
            k: plan["context"][k]
            for k in ("run_id", "bundle_id", "protocol_id", "gpu_devices")
        }
        permit.update(
            authorized=True,
            task_key="position_swap_salad",
            seed=18,
            simulation_episodes=1,
            limits=plan["context"]["limits"],
            scope="new-task",
            approval_reference=str(self.scope_ref),
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        )
        ref = self.operator / "inherit-permit.json"
        atomic_json(ref, permit)
        plan = inheritance_plan(self.reg_ref)
        self.assertFalse(plan["blockers"])
        atomic_json(self.manager.active_path, {**self.manager.active(), "revision": 99})
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            execute_runtime(plan)
        self.assertFalse(Path(plan["context"]["output_dir"]).exists())

    def test_rule_and_seed_changes_or_missing_theta1_are_rejected(self):
        self.selection["selection_rule"] = "choose after observing Q"
        atomic_json(self.selection_ref, self.selection)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            evaluate_qualification(self.reg_ref, "candidate")
        self.registration["protocol_snapshot"]["task_sets"]["release"]["seeds"] = [
            17,
            18,
        ]
        atomic_json(self.protocol_ref, self.registration["protocol_snapshot"])
        atomic_json(self.reg_ref, self.registration)
        with self.assertRaisesRegex(ValueError, "fixed denominator"):
            evaluate_qualification(self.reg_ref, "baseline")

    def test_inconclusive_candidate_keeps_the_real_initial_version(self):
        path = Path(self.registration["candidate"]["manifest"][0]["result_path"])
        native = json.loads(path.read_text())
        native.update(benchmark_success=None, terminated=False, truncated=False)
        atomic_json(path, native)
        self.evaluate()
        initial = initialize(self.reg_ref, queues_empty=True)["active"]
        self.assertEqual(decide(self.reg_ref)["decision"]["verdict"], "inconclusive")
        self.assertEqual(
            activate(self.reg_ref, queues_empty=True)["status"], "kept_current"
        )
        self.assertEqual(self.manager.active(), initial)

    def test_distinct_bundle_name_cannot_substitute_for_theta1(self):
        self.selection["execution_binding"]["bundle"]["theta_ref"] = self.base[
            "theta_ref"
        ]
        atomic_json(self.selection_ref, self.selection)
        with self.assertRaisesRegex(ValueError, "distinct admitted theta1"):
            evaluate_qualification(self.reg_ref, "candidate")

    def test_backdated_registration_field_does_not_admit_prior_Q_outputs(self):
        row = self.registration["baseline"]["manifest"][0]
        path = Path(row["result_path"]).parent / "action_events.jsonl"
        event = json.loads(path.read_text())
        event["utc_ns"] = int(
            (datetime.now(timezone.utc) - timedelta(seconds=1)).timestamp() * 1e9
        )
        path.write_text(json.dumps(event) + "\n")
        with self.assertRaisesRegex(ValueError, "precede the frozen controller"):
            evaluate_qualification(self.reg_ref, "baseline")

    def test_missing_Q_run_preserves_eight_slots_and_is_inconclusive(self):
        row = self.registration["candidate"]["manifest"][0]
        (Path(row["result_path"]).parent / "runtime_context.json").unlink()
        self.evaluate()
        initialize(self.reg_ref, queues_empty=True)
        candidate = self.evaluation.get("results", "Q-candidate-result")
        self.assertEqual(
            (candidate["assigned_count"], candidate["unknown_count"]), (8, 1)
        )
        self.assertIsNone(candidate["hard_constraints"]["initial_capabilities"])
        self.assertEqual(decide(self.reg_ref)["decision"]["verdict"], "inconclusive")

    def namespace_flow(self):
        """Actual role mounts; only model/physics facts remain CPU fixtures."""
        self.attach_actual_scientific_fixture()
        with patch("rsi_vla.midterm_science._readonly", return_value=True):
            self.evaluate()
        spec = importlib.util.spec_from_file_location(
            "_direct_role_runner", ROOT / "scripts/run_isolated_role.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "core/src") + os.pathsep + str(ROOT / "RPent"),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        roles = {}
        probe_script = """
import json,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text());role=sys.argv[2]
domains=config['role_domains'];denied=[]
for name,path in domains.items():
 if name==role:continue
 try:(Path(path)/'forbidden-write').write_text('forbidden')
 except OSError:denied.append(name)
report={'scope':config['scope'],'role':role,'allowed_write':True,'denied_writes':len(denied)==4}
(Path(domains[role])/'role-probe.json').write_text(json.dumps(report))
assert report['denied_writes']
"""
        for role in self.domains:
            command = runner.command(
                self.root,
                role,
                [sys.executable, "-c", probe_script, str(self.reg_ref), role],
                protected_paths=self.registration["protected_paths"],
            )
            result = subprocess.run(
                command, capture_output=True, text=True, env=env, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            roles[role] = {
                "exit_code": result.returncode,
                "report_ref": str(Path(self.domains[role]) / "role-probe.json"),
            }
        atomic_json(
            self.registration["isolation_acceptance_ref"],
            {
                "scope": self.registration["scope"],
                "role_domains": self.domains,
                "roles": roles,
            },
        )
        mirror = str(Path(__file__).parent)
        script = (
            "import sys; from pathlib import Path; import rsi_vla; rsi_vla.__path__.insert(0,"
            + repr(mirror)
            + '); from unittest.mock import patch; patch("rsi_vla.critical_progress.inspect_progress",return_value={"value":True,"evidence_refs":[],"offline_fixture":True}).start(); import rsi_vla.runtime as runtime; runtime.DYNAMIC_LOCK=Path("/tmp/offline-p5.lock"); from rsi_vla.midterm_deployment import main; main()'
        )
        attempt_registration = json.loads(
            json.dumps(self.scientific_fixture.registration)
        )
        attempt_registration.update(
            role_domains=self.domains,
            protected_paths=self.registration["protected_paths"],
            isolation_acceptance_ref=self.registration["isolation_acceptance_ref"],
            evaluation_store_ref=self.domains["evaluation"],
        )
        attempt_row = attempt_registration["cells"]["theta1-D"]["manifest"][0]
        attempt_row["attempt_receipt_ref"] = str(
            Path(self.domains["evaluation"])
            / "attempts"
            / (attempt_row["run_id"] + ".json")
        )
        attempt_registration_ref = self.operator / "scientific-attempt-role-check.json"
        atomic_json(attempt_registration_ref, attempt_registration)
        attempt_script = script.replace(
            "from rsi_vla.midterm_deployment import main",
            "from rsi_vla.midterm_science import main",
        )
        attempt_args = [
            sys.executable,
            "-c",
            attempt_script,
            "--registration",
            str(attempt_registration_ref),
            "--attempt-cell",
            "theta1-D",
            "--attempt-run-id",
            attempt_row["run_id"],
        ]
        attempt_command = runner.command(
            self.root,
            "evaluation",
            attempt_args,
            protected_paths=self.registration["protected_paths"],
        )
        attempt_result = subprocess.run(
            attempt_command, capture_output=True, text=True, env=env, timeout=30
        )
        self.assertEqual(attempt_result.returncode, 0, attempt_result.stderr)
        self.assertEqual(
            json.loads(Path(attempt_row["attempt_receipt_ref"]).read_text())[
                "validity"
            ],
            "valid",
        )
        wrong_role = runner.command(
            self.root,
            "release",
            attempt_args,
            protected_paths=self.registration["protected_paths"],
        )
        wrong_result = subprocess.run(
            wrong_role, capture_output=True, text=True, env=env, timeout=30
        )
        self.assertNotEqual(wrong_result.returncode, 0)
        self.assertIn("role write boundary", wrong_result.stderr)
        results = []
        for operation in ("initialize", "decide", "activate", "inherit-plan"):
            role = "runtime" if operation == "inherit-plan" else "release"
            command = runner.command(
                self.root,
                role,
                [
                    sys.executable,
                    "-c",
                    script,
                    "--registration",
                    str(self.reg_ref),
                    "--operation",
                    operation,
                    "--queues-empty",
                ],
                protected_paths=self.registration["protected_paths"],
            )
            result = subprocess.run(
                command, capture_output=True, text=True, env=env, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0]["active"]["activation"], "qualified_initial")
        self.assertEqual(results[1]["decision"]["verdict"], "accept")
        self.assertEqual(results[2]["active"]["revision"], 2)
        self.assertEqual(results[3]["selection"]["selected_bundle_id"], "Q-candidate")
        self.assertFalse(Path(results[3]["context"]["output_dir"]).exists())
        return {
            "role_probes": 5,
            "fixed_role_operations": 5,
            "independent_attempt_operations": 1,
            "wrong_role_attempt_rejected": True,
            "shared_dynamic_lock_modified": False,
            "runtime_started": False,
        }

    def attach_actual_scientific_fixture(self):
        from rsi_vla.midterm_science import write_admission
        from test_midterm_science import ScienceFixture

        science = ScienceFixture()
        science.fixture_parent = self.root
        science.setUp()
        self.scientific_fixture = science
        self.addCleanup(science.doCleanups)
        scope = json.loads(self.scope_ref.read_text())
        scope["scope_id"] = "science-task"
        atomic_json(self.scope_ref, scope)
        self.registration["scope"] = "science-task"
        self.protocol["scope_id"] = "science-task"
        self.registration["p4_protocol_id"] = science.protocol["protocol_id"]
        science.registration["scope_authorization_ref"] = str(self.scope_ref)
        for kind in ("training", "load"):
            ref = science.training[kind + "_permit_ref"]
            permit = json.loads(Path(ref).read_text())
            permit["scope_ref"] = str(self.scope_ref)
            atomic_json(ref, permit)
        atomic_json(science.reg_ref, science.registration)
        for cell in science.registration["cells"].values():
            for row in cell["manifest"]:
                context = json.loads(
                    (
                        Path(row["result_path"]).parent / "runtime_context.json"
                    ).read_text()
                )
                permit_ref = context["authorization_ref"]
                permit = json.loads(Path(permit_ref).read_text())
                permit["approval_reference"] = str(self.scope_ref)
                atomic_json(permit_ref, permit)
        science.refresh_attempts()
        with patch("rsi_vla.midterm_science.require_role"):
            ref, _ = write_admission(science.reg_ref, "theta-effect")
        copied = Path(self.domains["evaluation"]) / "scientific-admission.json"
        copied.write_bytes(Path(ref).read_bytes())
        self.base.update(theta_ref=science.theta0, processor_ref=science.processor0)
        self.trial.update(theta_ref=science.theta1, processor_ref=science.processor1)
        atomic_json(self.research.path("bundles", self.base["bundle_id"]), self.base)
        atomic_json(self.research.path("bundles", self.trial["bundle_id"]), self.trial)
        candidate = asdict(self.candidate)
        candidate["changes"] = {
            "theta_ref": science.theta1,
            "processor_ref": science.processor1,
        }
        atomic_json(
            self.research.path("candidates", candidate["candidate_id"]), candidate
        )
        self.registration["baseline"]["execution_binding"] = execution_binding(
            self.research, self.base, self.profile
        )
        self.selection["execution_binding"] = execution_binding(
            self.research, self.trial, self.profile
        )
        self.selection["theta_effect_admission_ref"] = str(copied)
        self.selection["source_result_refs"] = [str(copied)]
        atomic_json(self.selection_ref, self.selection)
        self.registration["protected_paths"] += [str(science.root)]
        self.registration["protocol_snapshot"] = self.protocol
        atomic_json(self.protocol_ref, self.protocol)
        atomic_json(self.reg_ref, self.registration)
        for side, binding in [
            ("baseline", self.registration["baseline"]["execution_binding"]),
            ("candidate", self.selection["execution_binding"]),
        ]:
            for row in self.registration[side]["manifest"]:
                path = Path(row["result_path"]).parent / "runtime_context.json"
                context = json.loads(path.read_text())
                context["execution_binding"] = binding
                atomic_json(path, context)
        self.science_verify.stop()

    def test_actual_scientific_receipt_is_consumed_and_point_gate_is_refused(self):
        self.science_verify.stop()
        with patch("rsi_vla.midterm_science._readonly", return_value=True):
            with self.assertRaisesRegex(PermissionError, "point Gate"):
                evaluate_qualification(self.reg_ref, "candidate")
        self.attach_actual_scientific_fixture()
        with patch("rsi_vla.midterm_science._readonly", return_value=True):
            self.evaluate()
            initialize(self.reg_ref, queues_empty=True)
            self.assertEqual(decide(self.reg_ref)["decision"]["verdict"], "accept")

    def test_actual_scientific_inconclusive_preserves_qualified_baseline_without_candidate_Q(
        self,
    ):
        from rsi_vla.midterm_science import write_admission
        from test_midterm_science import ScientificAdmissionTests

        self.attach_actual_scientific_fixture()
        science = self.scientific_fixture
        ScientificAdmissionTests.set_controlled(science, "theta1-D", 1)
        with patch("rsi_vla.midterm_science.require_role"):
            receipt_ref, receipt = write_admission(
                science.reg_ref, "theta-inconclusive"
            )
        self.assertEqual(receipt["verdict"], "inconclusive")
        copied = Path(self.domains["evaluation"]) / "scientific-inconclusive.json"
        copied.write_bytes(Path(receipt_ref).read_bytes())
        self.selection["theta_effect_admission_ref"] = str(copied)
        self.selection["source_result_refs"] = [str(copied)]
        atomic_json(self.selection_ref, self.selection)
        for row in self.registration["candidate"]["manifest"]:
            import shutil

            shutil.rmtree(Path(row["result_path"]).parent)
        with patch("rsi_vla.midterm_science._readonly", return_value=True):
            evaluate_qualification(self.reg_ref, "baseline")
            before = initialize(self.reg_ref, queues_empty=True)["active"]
            refused = decide(self.reg_ref)
            self.assertEqual(refused["decision"]["verdict"], "inconclusive")
            self.assertTrue(refused["scientific_admission_blocked"])
            self.assertFalse(refused["candidate_runtime_started"])
            self.assertEqual(
                activate(self.reg_ref, queues_empty=True)["status"], "kept_current"
            )
            self.assertEqual(self.manager.active(), before)
            self.assertEqual(
                inheritance_plan(self.reg_ref)["selection"]["selected_bundle_id"],
                self.base["bundle_id"],
            )

    def test_known_required_old_failure_overrides_another_unknown(self):
        rows = self.registration["candidate"]["manifest"]
        unknown = Path(rows[0]["result_path"])
        native = json.loads(unknown.read_text())
        native.update(benchmark_success=None, terminated=False, truncated=False)
        atomic_json(unknown, native)
        old = next(row for row in rows if row["task_key"] == "old_drawer")
        path = Path(old["result_path"])
        native = json.loads(path.read_text())
        native.update(benchmark_success=False, terminated=False, truncated=True)
        atomic_json(path, native)
        self.evaluate()
        initialize(self.reg_ref, queues_empty=True)
        candidate = self.evaluation.get("results", "Q-candidate-result")
        self.assertFalse(candidate["hard_constraints"]["initial_capabilities"])
        self.assertEqual(decide(self.reg_ref)["decision"]["verdict"], "reject")


if __name__ == "__main__":
    unittest.main()
