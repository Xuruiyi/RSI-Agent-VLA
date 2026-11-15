"""CPU business flow through the production broker, entry point and bwrap roles."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import test_protocol10_integration as integration
from rsi_vla.protocol10_backend import Protocol10Backend, read
from rsi_vla.protocol10_coordinator import Coordinator
from rsi_vla.protocol10_roles import run_operator
from rsi_vla.storage import atomic_json


def fixture_from_config(config):
    f = integration.Protocol10IntegrationTests()
    f.config = config
    f.protocol = read(config["protocol_ref"])
    f.protocol_ref = Path(config["protocol_ref"])
    f.root = Path(config["isolation_task_root"])
    f.artifacts = Path(config["artifacts_root"])
    f.phase_ref = Path(config["phase_ref"])
    f.slots = config["runtime_slots"]
    from rsi_vla.storage import ResearchStore

    f.store = ResearchStore(config["research"]["store_ref"])
    f.launches, f.research_launches = [], 0
    for key in (
        "ceiling",
        "candidate_failure",
        "evaluation_fault",
        "missing_cost",
        "cleanup_unknown",
        "incomplete_candidate_evidence",
        "invalid_response",
    ):
        setattr(f, key, False)
    f.evaluation_fault = bool(
        config.get("cpu_evaluation_failure_ref")
        and Path(config["cpu_evaluation_failure_ref"]).exists()
    )
    f.technical_fault_run = None
    f.recovery_resolved = {}
    return f


def boundary_factory(config):
    f = fixture_from_config(config)
    f._write_runtime_permits = lambda side: None
    role = os.environ.get("RSI_ISOLATED_ROLE")
    role_root = Path(config["isolation_task_root"])
    role_domains = {
        "runtime": role_root / "runtime",
        "research": role_root / "state/research",
        "evaluation": role_root / "state/evaluation",
    }
    if role in role_domains:
        denied = []
        for name, path in {"operator": role_root / "operator", **role_domains}.items():
            if name == role:
                continue
            try:
                (path / "forbidden").write_text("forbidden")
            except OSError:
                denied.append(name)
            else:
                raise AssertionError("role crossed another write domain")
        if role == "research":
            assert os.environ["CODEX_HOME"] == "/tmp/codex"
            Path("/tmp/codex/cpu-sdk-journal.json").write_text("offline SDK scratch")
            for name in ("auth.json", "config.toml"):
                path = Path("/tmp/codex") / name
                if path.exists():
                    assert path.is_symlink()
        atomic_json(
            role_domains[role] / "role-write-boundaries.json",
            {"role": role, "denied": denied, "scratch": os.environ["TMPDIR"]},
        )
    runtime = f.runtime_execute

    def execute(plan):
        assert role == "runtime"
        runtime(plan)
        denied = []
        root = Path(config["isolation_task_root"])
        for target in (
            root / "operator/forbidden",
            root / "state/research/forbidden",
            root / "state/evaluation/forbidden",
            Path(config["phase_ref"]),
        ):
            try:
                with target.open("a") as stream:
                    stream.write("forbidden")
            except OSError:
                denied.append(str(target))
            else:
                raise AssertionError("Runtime crossed role write boundary")
        atomic_json(
            Path(plan["context"]["output_dir"]) / "write-boundaries.json",
            {"denied": denied},
        )

    from rsi_vla import evaluator

    evaluator.inspect_attempt = f.audit
    evaluator.inspect_progress = f.progress
    return {
        "runtime_execute": execute,
        "audit": f.audit,
        "progress_inspector": f.progress,
        "research_execute": f.research_execute,
        "trial_collector": f.collect,
    }


def driver(config_ref):
    config = read(config_ref)
    f = fixture_from_config(config)
    root = f.root / "operator/protocol10"
    f.backend = Protocol10Backend(root, config)
    f.coordinator = Coordinator(root / "coordinator", f.backend.registration, f.backend)
    first = f.backend.observe(f.coordinator.status(), {})["next_operation"]
    f.coordinator.advance(f.inputs())
    # A lost operator acknowledgement resolves from the role receipt, with new time inputs.
    f.backend.records.path("runtime_returns", first["payload"]["run_id"]).unlink()
    f.backend.records.path("receipts", first["operation_id"]).unlink()
    retry_inputs = f.inputs()
    retry_inputs["now"] = "2026-11-13T01:00:00Z"
    f.backend.execute(first, retry_inputs)
    f.coordinator.advance(f.inputs())
    before_evaluation = read(f.phase_ref)
    failure_path = Path(config["cpu_evaluation_failure_ref"])
    failure_path.write_text("offline evaluation parser failure")
    state = f.coordinator.advance(f.inputs())
    assert state["status"] == "waiting", state
    failure_path.unlink()
    state = f.coordinator.advance(f.inputs())
    assert state["status"] == "ready", state
    assert read(f.phase_ref) == before_evaluation
    for _ in range(100):
        if f.store.path("bundles", "candidate-protocol10-01").exists():
            f._write_runtime_permits("candidate")
            f._write_runtime_permits("development")
        state = f.coordinator.advance(f.inputs())
        if state["status"] == "completed":
            break
        if state["status"] == "waiting":
            raise AssertionError(str(state.get("pending_receipt") or state["wait"]))
    assert state["terminal"] == "comparison_complete", state
    before = read(f.phase_ref)
    f.coordinator.advance(f.inputs())
    f.coordinator.advance(f.inputs())
    assert read(f.phase_ref) == before
    atomic_json(
        f.root / "operator/driver-result.json", {"state": state, "phase": before}
    )


class RoleBusinessTests(unittest.TestCase):
    def test_actual_isolated_roles_complete_business_and_resume_without_recharge(self):
        f = integration.Protocol10IntegrationTests()
        evidence = os.environ.get("RSI_ROLE_EVIDENCE_ROOT")
        if evidence:
            import tempfile

            make_temp = tempfile.TemporaryDirectory

            def retained_temp():
                value = make_temp(prefix="protocol10-roles-", dir=evidence)
                value._finalizer.detach()
                value.cleanup = lambda: None
                return value

            with patch(
                "test_protocol10_integration.tempfile.TemporaryDirectory", retained_temp
            ):
                f.setUp()
        else:
            f.setUp()
        self.addCleanup(f.doCleanups)
        for patcher in f.patches:
            patcher.stop()
        root = f.root
        f.config["isolation_task_root"] = str(root)
        f.config["cpu_evaluation_failure_ref"] = str(
            root / "operator/cpu-evaluation-failure"
        )
        f.config["research"]["output_ref"] = str(f.store.root / "output")
        permit = read(f.config["research"]["permit_ref"])
        permit["output_dir"] = f.config["research"]["output_ref"]
        atomic_json(f.config["research"]["permit_ref"], permit)
        for slot in f.slots:
            slot["output_ref"] = str(root / "runtime" / slot["run_id"])
            slot["scratch_ref"] = str(root / "scratch/runtime")
            slot["permit_ref"] = str(
                root / "operator/permits" / (slot["run_id"] + ".json")
            )
        f._write_runtime_permits("baseline")
        config_ref = root / "configuration.json"
        atomic_json(config_ref, f.config)
        # Create immutable registration before isolated readers consume it.
        Protocol10Backend(root / "operator/protocol10", f.config)
        # unittest discovery changes only this process's sys.path. The isolated
        # driver and role factories need the original checkout in PYTHONPATH.
        tests_root = Path(__file__).resolve().parent
        pythonpath = os.pathsep.join(
            [
                str(tests_root),
                str(tests_root.parent / "src"),
                *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []),
            ]
        )
        with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
            status = run_operator(
                f.config,
                [
                    sys.executable,
                    "-m",
                    "test_protocol10_roles",
                    "--driver",
                    str(config_ref),
                ],
                boundary_factory="test_protocol10_roles:boundary_factory",
            )
        self.assertEqual(status, 0)
        result = read(root / "operator/driver-result.json")
        self.assertEqual(result["phase"]["used"]["simulation_episodes"], 18)
        self.assertEqual(len(result["phase"]["research_charges"]), 1)
        self.assertEqual(len(result["phase"]["protocol10_trial_charges"]), 16)
        self.assertEqual(len(result["phase"]["development_trial_charges"]), 2)
        self.assertEqual(
            result["state"]["receipts"]["gate"]["decision"]["verdict"], "accept"
        )
        for role, path in [
            ("runtime", "runtime"),
            ("research", "state/research"),
            ("evaluation", "state/evaluation"),
        ]:
            receipts = [read(p) for p in (root / path / "role_receipts").glob("*.json")]
            self.assertTrue(receipts, role)
            self.assertTrue(all(r["role"] == role for r in receipts))
            errors = [r for r in receipts if r.get("error")]
            self.assertEqual(len(errors), 1 if role == "evaluation" else 0)
        self.assertEqual(
            len(list((root / "runtime").glob("*/write-boundaries.json"))), 18
        )
        # The actual top-level CLI reaches operator isolation before rejecting fixtures.
        import subprocess

        returned = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "scripts/run_protocol10.py"),
                "status",
                "--root",
                str(root / "operator/protocol10"),
                "--configuration",
                str(config_ref),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(returned.returncode, 2)
        self.assertIn("offline fixtures require injected", returned.stdout)
        atomic_json(
            root / "operator/top-level-cli.json",
            {"returncode": returned.returncode, "stdout": returned.stdout},
        )
        if evidence:
            print("OS role evidence: " + str(root))


if __name__ == "__main__":
    if "--driver" in sys.argv:
        driver(sys.argv[-1])
    else:
        unittest.main()
