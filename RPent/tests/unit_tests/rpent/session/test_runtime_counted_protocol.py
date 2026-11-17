# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual model options, completed attempts and v2 finite rule parsing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.planner.codex import _Recorder
from rpent.robots.components.pi05_vla_server import (
    PI05_EMBODIMENTS,
    Pi05VLAFacade,
    build_model_cfg,
)
from rpent.session.runtime import RuntimeJournal


def test_actual_journal_freezes_rules_and_rejects_changed_launch_input(tmp_path):
    rules_ref = tmp_path / "rules.json"
    rules = {
        "schema": "rsi-runtime-rules-v2",
        "chunk_steps": 5,
        "baseline_chunk_steps": 5,
        "finish_requires_native_success": True,
    }
    rules_ref.write_text(json.dumps(rules))
    context = {
        "run_id": "bound-run",
        "episode_id": "episode",
        "bundle_id": "bundle",
        "protocol_id": "protocol",
        "output_dir": str(tmp_path / "first"),
        "runtime_rules_ref": str(rules_ref),
        "execution_binding": {"references": {"phi_ref": rules}},
    }
    journal = RuntimeJournal(context)
    rules_ref.write_text(json.dumps({**rules, "chunk_steps": 2}))
    assert journal.chunk_steps == 5
    assert journal.finish_requires_native_success is True
    with pytest.raises(ValueError, match="frozen job binding"):
        RuntimeJournal({**context, "output_dir": str(tmp_path / "second")})


def test_interrupted_model_turn_is_not_normal_attempt_close():
    record = _Recorder(
        max_turns=20, dashboard_events=SimpleNamespace(emit=lambda *_: None)
    )
    record.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "interrupted"}}}
    )
    assert record.stats()["attempt_closed"] is False
    record.observe(
        {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}}
    )
    assert record.stats()["attempt_closed"] is True


def test_v2_native_finish_guard_and_actual_loaded_model_metadata(tmp_path, monkeypatch):
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v2",
                "chunk_steps": 5,
                "baseline_chunk_steps": 5,
                "finish_requires_native_success": True,
            }
        )
    )
    context = {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "offline-fixture",
        "output_dir": str(tmp_path),
        "runtime_rules_ref": str(rules),
    }
    file = tmp_path / "context.json"
    file.write_text(json.dumps(context))
    monkeypatch.setenv("RSI_RUNTIME_CONTEXT", str(file))
    journal = RuntimeJournal(context)
    assert journal.finish_requires_native_success is True
    server = Pi05VLAFacade.__new__(Pi05VLAFacade)
    server._embodiment = "libero"
    server._action_chunk = 5
    server._model = SimpleNamespace(
        predict_action_batch=lambda *_args, **_kwargs: (np.zeros((1, 5, 7)), {})
    )
    cfg = build_model_cfg("/offline-checkpoint", PI05_EMBODIMENTS["libero"])
    server._record_runtime_model_load(cfg, "/offline-checkpoint", "/offline-norm")
    actions = server.predict({"states": np.zeros((1, 8))})
    saved = json.loads((tmp_path / "model_load.json").read_text())
    assert saved["prediction_count"] == 1 and saved["last_action_shape"] == [1, 5, 7]
    assert saved["last_actions_finite"] is True and actions.shape == (1, 5, 7)
