# Copyright 2026 RSI-Agent-VLA Authors.
"""Real namespace acceptance for hidden inputs and registered CPU candidates."""

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.protocol10_roles import (
    _sandbox_command,
    role_configuration,
    role_isolation,
    run_role,
)
from rsi_vla.storage import atomic_json


def projection_probe(config):
    """Assert real mount contents before the unmodified backend is initialized."""
    assert "private_configuration_note" not in config
    for ref in config["isolation_probe_refs"]:
        try:
            content = Path(ref).read_text()
        except OSError:
            pass
        else:
            assert "F04-HIDDEN-LABEL" not in content, ref
            assert "F04-SEALED-LABEL" not in content, ref
    public = json.loads(Path(config["protocol_ref"]).read_text())
    assert "private_label" not in public
    assert "F04-SEALED-LABEL" not in json.dumps(public)
    if config.get("projection_runtime_preparation"):
        from rsi_vla.runtime import prepare_runtime

        preparation = config["projection_runtime_preparation"]
        plan = prepare_runtime(
            config["repository_root"], preparation["artifacts_ref"],
            protocol_path=config["protocol_ref"], run_id="public-preparation",
            bundle_id="base", task_key="position_swap_salad", seed=1,
            split="development", output_dir=Path(config["isolation_task_root"]) / "runtime/public-only",
        )
        assert plan["context"]["task"] == {"suite": "libero_object_swap", "task": 2, "seed": 1}
        assert plan["context"]["protocol_id"] == public["protocol_id"]
        assert "sealed" not in public["task_sets"]
        assert "F04-SEALED-LABEL" not in json.dumps(plan)
        atomic_json(Path(config["research_store_ref"]) / "public-runtime-plan.json", plan)
    atomic_json(Path(config["research_store_ref"]) / "public-probe.json",
                {"protocol_id": public["protocol_id"], "hidden_denied": True})
    return {}


class F04RoleBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f04-role-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("operator", "state/research", "state/evaluation", "release", "runtime"):
            (self.root / name).mkdir(parents=True)
        self.hidden = self.root / "hidden"
        self.hidden.mkdir()
        self.secret = self.hidden / "labels.json"
        self.secret.write_text("F04-HIDDEN-LABEL")
        self.public = self.hidden / "public.json"
        self.public.write_text("public-input")
        self.alias = self.root / "label-alias"
        self.alias.symlink_to(self.secret)
        self.config = {"isolation_task_root": str(self.root), "offline_fixture": True,
                       "role_isolation": {"hidden_paths": [str(self.hidden)]}}
        Path("/tmp/rsi-agent-vla-dynamic.lock").touch(exist_ok=True)

    def run_probe(self, role, script, **kwargs):
        with patch.dict(os.environ, {"F04_INHERITED_SECRET": "F04-HIDDEN-LABEL"}):
            returned = subprocess.run(
                _sandbox_command(self.root, role, [sys.executable, "-c", script], **kwargs),
                capture_output=True, text=True, timeout=30, check=False,
            )
        self.assertEqual(returned.returncode, 0, returned.stdout + returned.stderr)
        return json.loads(returned.stdout)

    def test_runtime_cannot_read_hidden_absolute_parent_symlink_receipts_or_other_scratch(self):
        shared = self.root / "state/evaluation/role_receipts/hidden.json"
        atomic_json(shared, {"hidden": "F04-HIDDEN-LABEL"})
        other = self.root / "scratch/evaluation/log.txt"
        other.parent.mkdir(parents=True)
        other.write_text("F04-HIDDEN-LABEL")
        requests = self.root / "operator/role_requests"
        atomic_json(requests / "current.json", {"public": True})
        atomic_json(requests / "other.json", {"hidden": "F04-HIDDEN-LABEL"})
        paths = [self.secret, self.hidden / "../hidden/labels.json", self.alias,
                 shared, other, requests / "other.json"]
        script = f'''
import json, os
from pathlib import Path
for value in {list(map(str, paths))!r}:
 try: content = Path(value).read_text()
 except OSError: pass
 else: assert 'F04-HIDDEN-LABEL' not in content, value
assert json.loads(Path({str(requests / "current.json")!r}).read_text()) == {{'public': True}}
assert Path({str(self.public)!r}).read_text() == 'public-input'
assert 'F04_INHERITED_SECRET' not in os.environ
assert 'RSI_ROLE_BROKER' not in os.environ
for value in {str(self.root / "release/forbidden")!r}, {str(self.hidden / "forbidden")!r}:
 try: Path(value).write_text('forbidden')
 except OSError: pass
 else: raise AssertionError(value)
Path({str(self.root / "runtime/allowed")!r}).write_text('allowed')
print(json.dumps({{'hidden_denied': {len(paths)}}}))
'''
        policy = role_isolation(self.config, "runtime")
        policy["read_grants"] = [str(self.public)]
        result = self.run_probe("runtime", script, request_ref=requests / "current.json", **policy)
        self.assertEqual(result["hidden_denied"], 6)
        self.assertFalse((self.hidden / "forbidden").exists())

    def test_trusted_evaluation_and_release_can_read_hidden_and_write_their_domain(self):
        for role, domain in (("evaluation", "state/evaluation"), ("release", "release")):
            with self.subTest(role=role):
                result = self.run_probe(role, f'''
import json
from pathlib import Path
assert Path({str(self.secret)!r}).read_text() == 'F04-HIDDEN-LABEL'
Path({str(self.root / domain / 'trusted-output')!r}).write_text('ok')
print(json.dumps({{'trusted': True}}))
''', **role_isolation(self.config, role))
                self.assertTrue(result["trusted"])

    def projection_configuration(self):
        from rsi_vla.contracts import RunBundle
        from rsi_vla.storage import ResearchStore, initialize

        protocol = {
            "protocol_id": "f04-public", "scope_id": "f04-scope", "scope": "offline",
            "status": "frozen", "benchmark_id": "synthetic", "anchor_bundle_id": "base",
            "metrics": {"success": {"direction": "max", "min": 0, "max": 1, "weight": 1}},
            "hard_constraints": ["resource_limits"], "min_gain": 0, "anchor_max_drop": 0,
            "task_policy": {"schema": "rsi-task-policy-v1"},
            "budget": {"runtime_per_start": {"model_requests": 0, "gpu_seconds": 0}},
        }
        private = {**protocol, "private_label": "F04-HIDDEN-LABEL"}
        private_ref = self.hidden / "protocol.json"
        public_ref = self.root / "public-protocol.json"
        atomic_json(private_ref, private)
        atomic_json(public_ref, protocol)
        scope_ref = self.root / "scope.json"
        admin_ref = self.root / "admin.json"
        atomic_json(scope_ref, {"authorized": True, "scope_id": "f04-scope"})
        atomic_json(admin_ref, {"authorized": True, "scope": "f04-scope", "protocol_id": "f04-public"})
        store_ref = self.root / "state/research/store"
        bundle = RunBundle("base", "theta", "processor", "phi", "rho", "psi", "memory",
                           "f04-public", "backend", "code")
        initialize(ResearchStore(store_ref), bundle, psi={},
                   memory={"provenance": "synthetic", "legal_information": "public"})
        domain_ref = self.root / "domain.json"
        atomic_json(domain_ref, {"domain": "synthetic"})
        operation = {"operation_id": "public-registration", "kind": "development_register",
                     "dynamic": False, "payload": {"request_id": "request",
                     "source_experiment_id": "experiment", "experiment_id": "experiment",
                     "output_ref": str(self.root / "runtime/public-input")}}
        private_config_ref = self.hidden / "configuration.json"
        old_receipt = self.root / "state/research/role_receipts/old-private.json"
        atomic_json(old_receipt, {"private": "F04-HIDDEN-LABEL"})
        config = {
            **self.config, "schema": "rsi-task-completion-v1", "base_bundle_id": "base",
            "protocol_ref": str(private_ref), "frozen_protocol_ref": str(private_ref),
            "public_protocol_ref": str(public_ref), "public_frozen_protocol_ref": str(public_ref),
            "private_configuration_ref": str(private_config_ref),
            "public_configuration_ref": str(self.root / "public-configuration.json"),
            "scope_authorization_ref": str(scope_ref), "administrative_authorization_ref": str(admin_ref),
            "research_store_ref": str(store_ref), "evaluation_store_ref": str(self.root / "state/evaluation"),
            "repository_root": str(Path(__file__).resolve().parents[2]),
            "research_requests": [{"request_id": "request", "job_id": "job", "job_limits": {"offline_calls": 1},
                                   "domain_policy_ref": str(domain_ref)}],
            "operations": [operation], "runtime_slots": [],
            "isolation_probe_refs": [str(private_ref), str(self.secret), str(private_config_ref), str(old_receipt)],
            "private_configuration_note": "F04-HIDDEN-LABEL",
        }
        public_config = {key: value for key, value in config.items() if key not in {
            "public_protocol_ref", "public_frozen_protocol_ref", "private_configuration_ref",
            "public_configuration_ref", "private_configuration_note"}}
        public_config.update(protocol_ref=str(public_ref), frozen_protocol_ref=str(public_ref))
        atomic_json(config["public_configuration_ref"], public_config)
        atomic_json(private_config_ref, config)
        return config, operation

    def test_actual_task_backend_consumes_public_protocol_without_private_request_or_receipt_leak(self):
        config, operation = self.projection_configuration()
        result = run_role(config, {"action": "execute", "operation": operation, "inputs": {}},
                          boundary_factory="test_f04_role_boundaries:projection_probe")
        self.assertNotIn("error", result, result)
        self.assertEqual(result["value"], {"registered": True})
        self.assertNotIn("F04-HIDDEN-LABEL", json.dumps(result))
        stored = json.loads((Path(config["research_store_ref"]) / "trial_dispatches/experiment-dispatch.json").read_text())
        self.assertEqual(stored["protocol_id"], "f04-public")
        self.assertTrue(json.loads((Path(config["research_store_ref"]) / "public-probe.json").read_text())["hidden_denied"])
        # Trusted evaluation initializes the same actual backend from the full
        # source configuration and retains the independent hidden label.
        probe = f'''
import json
from pathlib import Path
from rsi_vla.task_completion import TaskCompletionBackend
config=json.loads(Path({config["private_configuration_ref"]!r}).read_text())
backend=TaskCompletionBackend(config)
assert backend.protocol['private_label']=='F04-HIDDEN-LABEL'
print(json.dumps({{'trusted': True}}))
'''
        self.assertTrue(self.run_probe("evaluation", probe, **role_isolation(config, "evaluation"))["trusted"])

    def runtime_task_projection(self):
        config, operation = self.projection_configuration()
        repository = Path(config["repository_root"])
        template = json.loads((repository / "config/pilot_v2.json").read_text())
        public = json.loads(Path(config["public_protocol_ref"]).read_text())
        public["runtime_config"] = template["runtime_config"]
        public["budget"]["per_attempt_proposal"] = template["budget"]["per_attempt_proposal"]
        task = next(row for row in template["tasks"] if row["task_key"] == "position_swap_salad")
        public["tasks"] = [task]
        public["task_sets"] = {
            "development": {"task_keys": ["position_swap_salad"], "seeds": [1]},
            "release": {"task_keys": ["position_swap_salad"], "seeds": [3]},
        }
        private = {**public, "private_label": "F04-HIDDEN-LABEL",
                   "tasks": [*template["tasks"], {"task_key": "sealed-only", "label": "F04-SEALED-LABEL"}],
                   "task_sets": {**template["task_sets"], "sealed": {"sentinel": "F04-SEALED-LABEL"}}}
        atomic_json(config["protocol_ref"], private)
        atomic_json(config["public_protocol_ref"], public)
        return config, operation, public

    def test_public_task_subset_prepares_runtime_inside_actual_role(self):
        from rsi_vla.__main__ import main as core_main

        config, operation, public = self.runtime_task_projection()
        artifacts = self.root / "state/research/runtime-assets"
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(["init", "--artifacts", str(artifacts), "--profile",
                       str(Path(config["repository_root"]) / "config/run_profile_v2.json"),
                       "--protocol", config["public_protocol_ref"]])
        preparation = {"artifacts_ref": str(artifacts)}
        config["projection_runtime_preparation"] = preparation
        public_config = json.loads(Path(config["public_configuration_ref"]).read_text())
        public_config["projection_runtime_preparation"] = preparation
        atomic_json(config["public_configuration_ref"], public_config)
        atomic_json(config["private_configuration_ref"], config)
        result = run_role(config, {"action": "execute", "operation": operation, "inputs": {}},
                          boundary_factory="test_f04_role_boundaries:projection_probe")
        self.assertNotIn("error", result, result)
        self.assertEqual(result["value"], {"registered": True})
        plan = json.loads((Path(config["research_store_ref"]) / "public-runtime-plan.json").read_text())
        self.assertEqual(plan["context"]["protocol_id"], public["protocol_id"])
        self.assertFalse(plan["context"]["dynamic_authorized"])
        self.assertFalse((self.root / "runtime/public-only").exists())
        self.assertNotIn("F04-SEALED-LABEL", json.dumps(result))
        self.assertNotIn("F04-HIDDEN-LABEL", json.dumps(result))

    def test_public_task_projection_rejects_changed_actions_budget_policy_and_order(self):
        config, _, public = self.runtime_task_projection()
        changes = [
            lambda value: value["tasks"][0].update(task_id=999),
            lambda value: value["tasks"][0].pop("suite"),
            lambda value: value["task_sets"]["development"].update(seeds=[2, 1]),
            lambda value: value["budget"]["per_attempt_proposal"].update(tool_calls=999),
            lambda value: value["task_policy"].update(schema="changed-policy"),
            lambda value: value["tasks"][0].update(task_id=True),
            lambda value: value["budget"]["runtime_per_start"].update(model_requests=False),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                value = json.loads(json.dumps(public))
                change(value)
                atomic_json(config["public_protocol_ref"], value)
                with self.assertRaisesRegex(ValueError, "identity, policy or fields"):
                    role_configuration(config, "research")

    def test_public_projection_rejects_protocol_or_registered_control_drift(self):
        config, _ = self.projection_configuration()
        incomplete = {key: value for key, value in config.items() if key != "public_protocol_ref"}
        with self.assertRaisesRegex(ValueError, "registered public protocol references"):
            role_configuration(incomplete, "research")
        public_ref = Path(config["public_protocol_ref"])
        public = json.loads(public_ref.read_text())
        public["scope_id"] = "another-scope"
        atomic_json(public_ref, public)
        with self.assertRaisesRegex(ValueError, "identity"):
            role_configuration(config, "research")
        config, _ = self.projection_configuration()
        public_config_ref = Path(config["public_configuration_ref"])
        public_config = json.loads(public_config_ref.read_text())
        public_config["operations"] = []
        atomic_json(public_config_ref, public_config)
        with self.assertRaisesRegex(ValueError, "registered role control"):
            role_configuration(config, "research")

    def candidate_entry(self, source):
        path = self.root / "state/research/candidate.py"
        path.write_text(source)
        self.config["code_candidate_executions"] = [{
            "execution_id": "cpu-check", "source_ref": str(path),
            "input_refs": [str(self.public)], "timeout_seconds": 10,
        }]
        # Public input is explicitly separate from hidden candidate material.
        self.config["role_isolation"]["hidden_paths"] = [str(self.secret)]
        return {"action": "code_candidate_execute", "execution_id": "cpu-check"}

    def test_registered_candidate_runs_without_hidden_devices_network_release_or_environment(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        task = self.candidate_entry(f'''
import argparse, json, os, socket
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--input', action='append');a=p.parse_args()
assert Path(a.input[0]).read_text() == 'public-input'
for value in {str(self.secret)!r}, {str(self.alias)!r}, {str(self.root / 'release')!r}:
 assert not Path(value).exists(), value
assert not list(Path('/dev').glob('nvidia*'))
assert not Path('/dev/dri').exists()
assert 'F04_INHERITED_SECRET' not in os.environ
assert 'RSI_ROLE_BROKER' not in os.environ
assert 'OPENAI_API_KEY' not in os.environ
try: socket.create_connection(('127.0.0.1', {port}), timeout=.2)
except OSError: pass
else: raise AssertionError('network escaped')
try: Path({str(self.root / 'release/forbidden')!r}).write_text('forbidden')
except OSError: pass
else: raise AssertionError('release write escaped')
Path(a.output, 'result.json').write_text(json.dumps({{'public_input': True, 'isolated': True}}))
''')
        with patch.dict(os.environ, {"F04_INHERITED_SECRET": "F04-HIDDEN-LABEL", "OPENAI_API_KEY": "unused-test-key"}):
            result = run_role(self.config, task)
        self.assertNotIn("error", result, result)
        self.assertEqual(result["value"]["returncode"], 0)
        self.assertTrue(json.loads((Path(result["output_ref"]) / "result.json").read_text())["isolated"])
        self.assertEqual(run_role(self.config, task), result)
        self.assertFalse((self.root / "release/forbidden").exists())

    def test_candidate_rejects_host_arguments_hidden_grants_and_gpu(self):
        task = self.candidate_entry("pass")
        with self.assertRaisesRegex(ValueError, "only a registered"):
            run_role(self.config, {**task, "argv": ["/bin/sh"]})
        self.config["code_candidate_executions"][0]["read_grants"] = [str(self.hidden)]
        with self.assertRaisesRegex(ValueError, "protected domain"):
            run_role(self.config, task)
        output = self.root / "state/research/code_candidates/gpu/output"
        output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "parent mount"):
            _sandbox_command(self.root, "runtime", ["/bin/true"],
                             hidden_paths=[self.secret], read_grants=[self.hidden])
        for path in ("/proc", "/dev", "/sys"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "namespaces"):
                _sandbox_command(self.root, "runtime", ["/bin/true"], read_grants=[path])
        with self.assertRaisesRegex(ValueError, "system dependency"):
            _sandbox_command(self.root, "candidate", ["/bin/true"], write_root=output,
                             hidden_paths=["/usr/bin/true"])
        with self.assertRaisesRegex(ValueError, "GPU"):
            _sandbox_command(self.root, "candidate", ["/bin/true"], gpu=True, write_root=output)

    def test_candidate_timeout_kills_descendants_and_prevents_resubmission(self):
        task = self.candidate_entry("""
import argparse, subprocess, sys, time
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--input', action='append');a=p.parse_args()
subprocess.Popen([sys.executable, '-c', "import pathlib,time;time.sleep(.6);pathlib.Path('escaped-child').write_text('escaped')"])
time.sleep(10)
""")
        self.config["code_candidate_executions"][0]["timeout_seconds"] = .2
        result = run_role(self.config, task)
        self.assertIn("timeout", result["error"])
        time.sleep(.7)
        self.assertFalse((Path(result["output_ref"]) / "escaped-child").exists())
        self.assertEqual(run_role(self.config, task), result)

    def test_candidate_rejects_symlink_output_into_release(self):
        task = self.candidate_entry("pass")
        domain = self.root / "state/research/code_candidates/cpu-check"
        domain.mkdir(parents=True)
        (domain / "output").symlink_to(self.root / "release", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            run_role(self.config, task)


if __name__ == "__main__":
    unittest.main(verbosity=2)
