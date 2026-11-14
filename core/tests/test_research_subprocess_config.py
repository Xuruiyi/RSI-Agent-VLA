"""Research SDK overrides retain the shared provider and budget configuration."""

from dataclasses import fields
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openai_codex import CodexConfig
from rpent.planner.codex import build_codex_config
from rsi_vla.bounded_runner import _research_codex_config


class ResearchSubprocessConfigTests(unittest.TestCase):
    def test_appended_flags_preserve_config_and_do_not_mutate_builder_result(self):
        shared = CodexConfig(
            codex_bin="/fixture/codex",
            cwd="/fixture/original",
            config_overrides=(
                'model_provider="fixture-budget"',
                "features.plugins=true",
            ),
            env={"RSI_REQUEST_GATE_URL": "http://127.0.0.1:19"},
            client_name="fixture",
        )
        original = shared.config_overrides
        with patch(
            "rpent.planner.codex.build_codex_config", return_value=shared
        ) as builder:
            config = _research_codex_config(Path("/fixture/research"))
        builder.assert_called_once_with(
            mcp_url=None, base_url=None, api_key=None, cwd="/fixture/research"
        )
        self.assertIsNot(config, shared)
        self.assertEqual(shared.config_overrides, original)
        self.assertEqual(
            config.config_overrides,
            original + ("features.plugins=false", "features.remote_plugin=false"),
        )
        for field in fields(shared):
            if field.name != "config_overrides":
                self.assertEqual(
                    getattr(config, field.name), getattr(shared, field.name)
                )

    def test_real_builder_keeps_budget_provider_and_shared_planner_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user_config = root / "config.toml"
            user_config.write_text(
                '[features]\nplugins=true\nremote_plugin=true\n[mcp_servers.extra]\ncommand="fixture-server"\n'
            )
            original = user_config.read_bytes()
            with patch.dict(
                os.environ,
                {
                    "CODEX_HOME": str(root),
                    "RSI_REQUEST_GATE_URL": "http://127.0.0.1:19",
                    "CODEX_MODEL_CONTEXT_WINDOW": "32000",
                    "CODEX_AUTO_COMPACT_TOKEN_LIMIT": "24000",
                },
            ):
                shared = build_codex_config(
                    mcp_url=None, base_url=None, api_key=None, cwd=str(root)
                )
                research = _research_codex_config(root)
                after = build_codex_config(
                    mcp_url=None, base_url=None, api_key=None, cwd=str(root)
                )
            self.assertEqual(after.config_overrides, shared.config_overrides)
            self.assertEqual(research.config_overrides[:-2], shared.config_overrides)
            self.assertEqual(
                research.config_overrides[-2:],
                ("features.plugins=false", "features.remote_plugin=false"),
            )
            for required in (
                'model_provider="rsi_budget"',
                'model_providers.rsi_budget.base_url="http://127.0.0.1:19"',
                "model_providers.rsi_budget.request_max_retries=0",
                "model_providers.rsi_budget.stream_max_retries=0",
                "features.apps=false",
                "features.shell_tool=false",
                'web_search="disabled"',
                "mcp_servers.extra.enabled=false",
                "model_context_window=32000",
                "model_auto_compact_token_limit=24000",
                "project_doc_max_bytes=0",
            ):
                self.assertIn(required, research.config_overrides)
            self.assertFalse(
                any(
                    value.startswith("mcp_servers.rpent.url=")
                    for value in research.config_overrides
                )
            )
            self.assertNotIn("features.plugins=false", shared.config_overrides)
            self.assertNotIn("features.remote_plugin=false", shared.config_overrides)
            self.assertEqual(user_config.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
