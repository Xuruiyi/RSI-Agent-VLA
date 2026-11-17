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

"""Actual L1 writes under fixed/feedback policies, without changing control media."""

import json

import numpy as np
import pytest

from rpent.session import EnvState
from rpent.session.media import configure_media_policy
from rpent.session.runtime import RuntimeJournal


def _runtime(root, *, feedback=False, scope="development", arm="D", limit=100000):
    policy = {"type": "fixed", "retain_l0": "all", "media": "all_available"}
    if feedback:
        policy = {
            "type": "rsi-relation-feedback-v1",
            "values": {
                "media": {
                    "all_available": {"count": 1, "total": 0.2},
                    "one_per_event": {"count": 1, "total": 0.8},
                }
            },
        }
    context = {
        "run_id": "run",
        "episode_id": "episode",
        "bundle_id": "bundle",
        "protocol_id": "protocol",
        "output_dir": str(root),
        "information_scope": scope,
        "execution_arm": {"arm": arm},
        "planner": {"backend": "flash"},
        "execution_binding": {
            "bundle": {"rho_ref": "rho"},
            "records": {"rho_ref": {"rhoA": policy}},
        },
        "l1_media_policy": {
            "schema": "rsi-l1-media-policy-v1",
            "rho_ref": "rho",
            "optional_artifacts": ["agentview_world_high.npz", "wrist_world_high.npz"],
            "max_persisted_bytes": limit,
        },
    }
    journal = RuntimeJournal(context)
    state = EnvState(root)
    configure_media_policy(state, journal)
    return state, journal


def _save(state):
    with state.record_step(state={}):
        state.save("agentview_world_high.npz", np.ones((4, 4, 3)))
        state.save("wrist_world_high.npz", np.zeros((4, 4, 3)))
        state.save("agentview_policy.png", np.zeros((4, 4, 3), np.uint8))
        state.save("audit.json", {"result": "unknown"})


def test_feedback_changes_actual_bytes_and_keeps_control_and_l0(tmp_path):
    fixed, fixed_journal = _runtime(tmp_path / "fixed")
    selected, selected_journal = _runtime(tmp_path / "selected", feedback=True)
    for state, journal in [(fixed, fixed_journal), (selected, selected_journal)]:
        journal.emit("action_proposal", proposal=[1], missing=["applied_action"])
        _save(state)
    assert (fixed._output_dir / "wrist_world_high.npz/00.npz").is_file()
    assert not (selected._output_dir / "wrist_world_high.npz/00.npz").exists()
    for state in [fixed, selected]:
        assert state.load("audit.json") == {"result": "unknown"}
        assert state.load("agentview_policy.png").shape == (4, 4, 3)
    rows = [
        json.loads(line)
        for line in selected_journal.events_path.read_text().splitlines()
    ]
    assert next(row for row in rows if row["kind"] == "action_proposal")["missing"] == [
        "applied_action"
    ]
    dropped = next(row for row in rows if row.get("artifact") == "wrist_world_high.npz")
    assert dropped["selected"] is False and dropped["probability"] == 0
    assert dropped["outcome_label_probability"] is None
    assert sum(
        f.stat().st_size for f in (fixed._output_dir).glob("*world_high.npz/*.npz")
    ) > sum(
        f.stat().st_size for f in (selected._output_dir).glob("*world_high.npz/*.npz")
    )


@pytest.mark.parametrize(
    "scope,arm", [("release", "D"), ("evaluation", "D"), ("development", "A")]
)
def test_evaluation_and_control_media_cannot_be_selected_away(tmp_path, scope, arm):
    state, journal = _runtime(tmp_path, feedback=True, scope=scope, arm=arm, limit=0)
    _save(state)
    assert state.load("wrist_world_high.npz").shape == (4, 4, 3)
    rows = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    assert all(
        row["selected"] for row in rows if row["kind"] == "media_retention_selection"
    )


def test_zero_budget_records_temporary_cost_and_rejects_optional_publication(tmp_path):
    state, journal = _runtime(tmp_path, limit=0)
    _save(state)
    assert not list(tmp_path.glob("*world_high.npz/*.npz"))
    assert state.load("audit.json") == {"result": "unknown"}
    rows = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    admissions = [row for row in rows if row["kind"] == "media_persistence_admission"]
    assert admissions and all(not row["admitted"] for row in admissions)
    assert admissions[-1]["cumulative_new_bytes_lower_bound"] > 0
    assert admissions[-1]["temporary_bytes"] > 0
    assert admissions[-1]["cumulative_new_bytes_lower_bound"] == sum(
        row["encoded_bytes"] for row in admissions
    )
    assert not list(tmp_path.rglob("*.tmp*"))


def test_policy_cannot_claim_raw_images_are_optional(tmp_path):
    state, journal = _runtime(tmp_path)
    journal.context["l1_media_policy"]["optional_artifacts"].append(
        "agentview_policy.png"
    )
    with pytest.raises(ValueError, match="only optional"):
        configure_media_policy(state, journal)


@pytest.mark.parametrize(
    "flags",
    [{"terminated": True}, {"truncated": True}, {"extras": {"audit_required": True}}],
)
def test_terminal_and_audit_state_remain_complete_even_at_zero_optional_budget(
    tmp_path, flags
):
    state, _ = _runtime(tmp_path, feedback=True, limit=0)
    with state.record_step(state={}, **flags):
        state.save("wrist_world_high.npz", np.zeros((4, 4, 3)))
    assert state.exists("wrist_world_high.npz")
