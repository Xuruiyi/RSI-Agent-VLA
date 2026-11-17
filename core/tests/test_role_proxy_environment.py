# Copyright 2026 RSI-Agent-VLA Authors.
"""Proxy allowlisting crosses real role mounts without any network request."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PROXIES = {
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "socks5://127.0.0.1:9",
    "NO_PROXY": "127.0.0.1,localhost,fixture.invalid",
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "all_proxy": "socks5://127.0.0.1:9",
    "no_proxy": "127.0.0.1,localhost,fixture.invalid",
}


class RoleProxyEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="role-proxy-", dir=ROOT / "RPent"
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (
            "operator",
            "runtime",
            "state/research",
            "state/evaluation",
            "release",
        ):
            (self.root / name).mkdir(parents=True)
        self.credentials = self.root / "operator/empty-codex-home"
        self.credentials.mkdir()
        lock = Path("/tmp/rsi-agent-vla-dynamic.lock")
        if not lock.exists():
            lock.touch()
        spec = importlib.util.spec_from_file_location(
            "role_proxy_isolation", ROOT / "scripts/run_isolated_role.py"
        )
        self.isolation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.isolation)
        self.program = (
            "import json,os; "
            f"print(json.dumps(dict(proxies={{k:os.environ.get(k) for k in {list(PROXIES)!r}}},"
            "extra_present='RSI_PROXY_UNRELATED_SECRET' in os.environ)))"
        )

    def command(self, role):
        kwargs = {}
        if role == "candidate":
            output = self.root / "state/research/code_candidates/proxy-probe"
            output.mkdir(parents=True, exist_ok=True)
            kwargs = {
                "write_root": output,
                "read_grants": [Path(sys.executable).resolve().parents[1]],
            }
        with patch.dict(
            os.environ,
            {
                **PROXIES,
                "CODEX_HOME": str(self.credentials),
                "RSI_PROXY_UNRELATED_SECRET": "must-not-cross-role",
            },
        ):
            return self.isolation.command(
                self.root,
                role,
                [str(Path(sys.executable).resolve()), "-c", self.program],
                **kwargs,
            )

    def test_static_allowlist_preserves_clear_environment_and_network_roles(self):
        for role in (
            "runtime",
            "research",
            "operator",
            "evaluation",
            "release",
            "candidate",
        ):
            with self.subTest(role=role):
                command = self.command(role)
                self.assertEqual(command[:2], ["/usr/bin/env", "-i"])
                env = {
                    command[index + 1]: command[index + 2]
                    for index, token in enumerate(command)
                    if token == "--setenv"
                }
                self.assertNotIn("RSI_PROXY_UNRELATED_SECRET", env)
                for key, value in PROXIES.items():
                    if role in {"runtime", "research"}:
                        self.assertEqual(env[key], value)
                    else:
                        self.assertNotIn(key, env)
                if role == "candidate":
                    self.assertIn("--unshare-all", command)
                elif role in {"runtime", "research"}:
                    self.assertNotIn("--unshare-net", command)
                else:
                    self.assertIn("--unshare-net", command)

    def run_role(self, role):
        returned = subprocess.run(
            self.command(role), check=False, capture_output=True, text=True, timeout=15
        )
        self.assertEqual(returned.returncode, 0, returned.stderr)
        value = json.loads(returned.stdout)
        self.assertFalse(value["extra_present"])
        return value["proxies"]

    def test_real_runtime_and_research_read_only_dummy_proxy_values(self):
        for role in ("runtime", "research"):
            with self.subTest(role=role):
                self.assertEqual(self.run_role(role), PROXIES)

    def test_real_candidate_and_other_roles_do_not_inherit_proxy_or_extra_environment(
        self,
    ):
        for role in ("candidate", "operator", "evaluation", "release"):
            with self.subTest(role=role):
                self.assertEqual(self.run_role(role), dict.fromkeys(PROXIES))


if __name__ == "__main__":
    unittest.main()
