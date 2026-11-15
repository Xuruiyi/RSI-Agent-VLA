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

"""Offline execution facts, identity and enforced admission boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.tools import LiberoPrimitives
from rpent.session.runtime import (
    RuntimeBudgetExceeded,
    RuntimeJournal,
    planner_admission,
)


def _context(tmp_path, **limits):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile.json",
        "output_dir": str(tmp_path),
        "parent_decision_id": "d",
        "limits": {"environment_steps": 3, "tool_calls": 2, **limits},
        "planner": {
            "backend": "codex",
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
        },
    }


def _records(journal):
    return [json.loads(line) for line in journal.events_path.read_text().splitlines()]


def test_disconnected_rpc_never_claims_execution_and_does_not_refund(tmp_path):
    journal = RuntimeJournal(_context(tmp_path, environment_steps=1))
    calls = []

    def lost():
        calls.append(1)
        raise ConnectionError("feedback lost after send")

    with pytest.raises(ConnectionError):
        journal.action(lost, [0] * 7, source="scripted")
    assert _records(journal)[-1]["execution_mask"] is None
    assert journal.costs()["environment_steps_unknown"] == 1
    with pytest.raises(RuntimeBudgetExceeded):
        journal.action(lost, [0] * 7, source="scripted")
    assert calls == [1]
    assert journal.costs()["environment_steps_reserved"] == 1


def test_tools_and_chunks_reserve_total_budget_before_dispatch(tmp_path):
    journal = RuntimeJournal(_context(tmp_path))
    journal.admit_tool("read_image")
    journal.admit_tool("finish")
    with pytest.raises(RuntimeBudgetExceeded):
        journal.admit_tool("read_image")
    with pytest.raises(RuntimeBudgetExceeded):
        journal.action(
            lambda: pytest.fail("must not dispatch"),
            np.zeros((4, 7)),
            source="vla",
            chunk=True,
        )
    assert journal.costs()["tool_calls"] == 2
    assert journal.costs()["environment_steps_reserved"] == 0


def test_rule_consumes_prefix_and_preserves_full_proposal_and_feedback(tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v1",
                "chunk_steps": 1,
                "baseline_chunk_steps": 3,
            }
        )
    )
    context = _context(tmp_path)
    context["runtime_rules_ref"] = str(rules)
    journal = RuntimeJournal(context)
    sent = []
    obs = {"states": np.zeros(8), "task_descriptions": "task"}

    def chunk(actions, **_kwargs):
        sent.append(actions.copy())
        return obs, np.zeros(1), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), {}

    primitives = LiberoPrimitives(
        env=SimpleNamespace(
            chunk_step=chunk,
            rsi_chunk_step=chunk,
            return_all_frames=False,
            terminated=False,
        ),
        model=SimpleNamespace(
            predict=lambda *_args, **_kwargs: np.arange(21).reshape(3, 7)
        ),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        runtime_journal=journal,
    )
    primitives.set_obs(obs)
    primitives._vlm_chunk("task")
    records = _records(journal)
    proposal = next(row for row in records if row["kind"] == "action_proposal")
    feedback = next(row for row in records if row["kind"] == "action_feedback")
    assert len(proposal["proposal"]) == 3 and sent[0].shape == (1, 7)
    assert feedback["execution_mask"] == [True, False, False]
    assert feedback["applied"] is None
    assert all(
        row["bundle_id"] == "b" and row["parent_decision_id"] == "d" for row in records
    )
    assert [row["sequence"] for row in records] == list(range(len(records)))
    journal.finalize()
    costs = json.loads(journal.costs_path.read_text())
    assert costs["input_tokens"] is None and costs["gpu_device_seconds"] is None
    assert costs["environment_steps_confirmed"] == 1


@pytest.mark.parametrize("cap,baseline", [(True, 5), (0, 5), (6, 5), (2, True)])
def test_rule_bounds_are_not_inferred_from_bool_or_invalid_caps(
    tmp_path, cap, baseline
):
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v1",
                "chunk_steps": cap,
                "baseline_chunk_steps": baseline,
            }
        )
    )
    context = _context(tmp_path)
    context["runtime_rules_ref"] = str(rules)
    with pytest.raises(ValueError):
        RuntimeJournal(context)


def test_finite_token_budget_refuses_codex_before_backend_construction(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, input_tokens=100, output_tokens=20)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    monkeypatch.setenv("RSI_RUNTIME_CONTEXT", str(path))
    with pytest.raises(RuntimeBudgetExceeded, match="no request dispatched"):
        planner_admission("codex", "gpt-6.1-sol", "medium")
    with pytest.raises(ValueError, match="actual planner"):
        planner_admission("codex", "gpt-6.1-sol", "low")


def test_unobserved_backend_usage_is_not_zero_cost(tmp_path):
    journal = RuntimeJournal(_context(tmp_path))
    journal.record_planner_stats({"usage_observed": False, "total_input_tokens": 0})
    assert journal.costs()["input_tokens"] is None
    journal.record_planner_stats(
        {
            "total_input_tokens": 50,
            "total_cached_input_tokens": 40,
            "total_output_tokens": 2,
        }
    )
    assert journal.costs()["input_tokens"] == 50
    assert journal.costs()["cached_input_tokens"] == 40


def test_existing_run_journal_is_preserved(tmp_path):
    journal = RuntimeJournal(_context(tmp_path))
    old = journal.events_path.read_bytes()
    with pytest.raises(FileExistsError):
        RuntimeJournal(_context(tmp_path))
    assert journal.events_path.read_bytes() == old
