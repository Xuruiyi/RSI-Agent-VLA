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

"""Client-to-facade RPC mapping with CPU fakes and the real write lock."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.runtime_evidence import observed_libero_class
from robots.libero.tools import LiberoPrimitives
from rpent.session.runtime import RuntimeJournal


def _context(path, steps=20):
    return {
        "schema_version": "1.0",
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "profile.json",
        "parent_decision_id": None,
        "output_dir": str(path),
        "limits": {"environment_steps": steps},
    }


class _Vector:
    def __init__(self):
        self.actions = []
        self.fail_on = None

    def step(self, actions):
        self.actions.append(actions.copy())
        if len(self.actions) == self.fail_on:
            raise ConnectionError("worker feedback lost")
        return [{"robot0_eef_pos": np.zeros(3)}], np.zeros(1), np.array([False]), [{}]

    def close(self):
        pass


class _Native:
    def __init__(self):
        self._init_env()

    def _init_env(self):
        self.env = _Vector()

    def step(self, actions):
        _obs, reward, terminated, _info = self.env.step(actions)
        return {"states": np.zeros((1, 8))}, reward, terminated, np.array([False]), {}

    def chunk_step(self, actions):
        observations, rewards, terms, truncs = [], [], [], []
        for slot in range(actions.shape[1]):
            obs, reward, term, trunc, _info = self.step(actions[:, slot])
            observations.append(obs)
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
        return (
            observations,
            np.stack(rewards, axis=1),
            np.stack(terms, axis=1),
            np.stack(truncs, axis=1),
            {},
        )


class _Transport:
    def __init__(self, facade):
        self.facade = facade
        self.lose_response = False

    def call(self, method, *, args=(), kwargs=None, timeout_s=None):
        del timeout_s
        # This is the same registration and write-lock boundary as real RPC,
        # without a socket or simulator process.
        result = self.facade._dispatch_nosession(method, args, kwargs or {})
        if self.lose_response:
            raise ConnectionError("transport response lost")
        return result


def _chain(tmp_path, steps=20):
    context = _context(tmp_path, steps)
    server = observed_libero_class(_Native, context)()
    facade = LiberoEnvFacade(server, meta={})
    transport = _Transport(facade)
    client = LiberoEnvClient.__new__(LiberoEnvClient)
    client._client = transport
    client.return_all_frames = client.terminated = client.truncated = False
    journal = RuntimeJournal(context)
    return server, client, journal


def _events(journal):
    return [json.loads(line) for line in journal.events_path.read_text().splitlines()]


def _send(client, journal, actions, *, proposal_length=None):
    proposal = journal.emit("action_proposal", source="vla", proposal=actions)
    return journal.action(
        lambda send: client.rsi_chunk_step(
            actions,
            evidence={
                **journal.identity(),
                "client_send_sequence": send,
                "proposal_sequence": proposal,
                "proposal_indices": list(range(len(actions))),
                "proposal_length": proposal_length or len(actions),
            },
        ),
        actions,
        source="vla",
        proposal_sequence=proposal,
        proposal_length=proposal_length or len(actions),
        chunk=True,
        callback_with_send_sequence=True,
    )


def test_client_send_and_full_proposal_bind_each_native_prefix_slot(tmp_path):
    server, client, journal = _chain(tmp_path)
    _send(client, journal, np.ones((2, 7)), proposal_length=5)
    sends = [row for row in _events(journal) if row["kind"] == "action_send"]
    server_events = _events(server._rsi_journal)
    native = [row for row in server_events if row["kind"] == "action_feedback"]
    assert [row["rpc_link"]["client_send_sequence"] for row in native] == [
        sends[0]["sequence"]
    ] * 2
    assert [row["rpc_link"]["client_proposal_sequence"] for row in native] == [
        sends[0]["proposal_sequence"]
    ] * 2
    assert [row["rpc_link"]["client_proposal_index"] for row in native] == [0, 1]
    assert [row["rpc_link"]["rpc_slot_index"] for row in native] == [0, 1]
    assert server_events[-1]["execution_mask"] == [True, True, False, False, False]
    assert server_events[-1]["confirmed_proposal_indices"] == [0, 1]
    assert journal.confirmed_steps == server._rsi_journal.confirmed_steps == 2


def test_partial_chunk_lost_worker_feedback_retains_precise_confirmed_and_unknown_slots(
    tmp_path,
):
    server, client, journal = _chain(tmp_path)
    server.env._vector.fail_on = 2
    with pytest.raises(ConnectionError, match="worker feedback"):
        _send(client, journal, np.ones((3, 7)))
    end = _events(server._rsi_journal)[-1]
    assert end["kind"] == "rpc_dispatch_end"
    assert end["confirmed_proposal_indices"] == [0]
    assert end["unknown_proposal_indices"] == [1]
    assert end["not_dispatched_proposal_indices"] == [2]
    assert end["execution_mask"] is None
    assert server._rsi_journal.pending_action
    assert server._rsi_journal.reserved_steps == 2
    assert (
        server._rsi_journal.confirmed_steps == server._rsi_journal.uncertain_steps == 1
    )
    assert journal.reserved_steps == journal.uncertain_steps == 3


def test_server_budget_rejects_only_unstarted_slots_without_refund_or_unknown_application(
    tmp_path,
):
    server, client, journal = _chain(tmp_path)
    server._rsi_journal.context["limits"] = {"environment_steps": 1}
    with pytest.raises(RuntimeError, match="environment_steps"):
        _send(client, journal, np.ones((3, 7)))
    end = _events(server._rsi_journal)[-1]
    assert end["confirmed_proposal_indices"] == [0]
    assert end["unknown_proposal_indices"] == []
    assert end["not_dispatched_proposal_indices"] == [1, 2]
    assert end["execution_mask"] == [True, False, False]
    assert server._rsi_journal.reserved_steps == 1
    assert not server._rsi_journal.pending_action


def test_lost_transport_response_preserves_full_server_mapping_and_client_unknown(
    tmp_path,
):
    server, client, journal = _chain(tmp_path)
    client._client.lose_response = True
    with pytest.raises(ConnectionError, match="transport response"):
        _send(client, journal, np.ones((2, 7)))
    assert _events(server._rsi_journal)[-1]["execution_mask"] == [True, True]
    assert _events(journal)[-1]["execution_mask"] is None
    assert not server._rsi_journal.pending_action
    assert journal.reserved_steps == journal.uncertain_steps == 2


def test_mismatched_identity_is_rejected_before_native_motion(tmp_path):
    server, client, journal = _chain(tmp_path)
    evidence = {
        **journal.identity(),
        "bundle_id": "other",
        "client_send_sequence": 3,
        "proposal_sequence": 2,
        "proposal_indices": [0],
        "proposal_length": 1,
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        client.rsi_step(np.ones(7), evidence=evidence)
    assert server.env._vector.actions == []


def test_dispatch_context_is_cleared_before_legacy_rpc_after_failure(tmp_path):
    server, client, journal = _chain(tmp_path)
    server.env._vector.fail_on = 1
    with pytest.raises(ConnectionError):
        _send(client, journal, np.ones((1, 7)))
    server.env._vector.fail_on = None
    client.step(np.zeros(7))
    legacy = _events(server._rsi_journal)[-1]
    assert legacy["kind"] == "action_feedback" and "rpc_link" not in legacy


def test_concurrent_rpc_metadata_stays_local_and_motion_uses_existing_write_lock(
    tmp_path,
):
    server, client, journal = _chain(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_send, client, journal, np.full((2, 7), value))
            for value in (1, 2)
        ]
        for future in futures:
            future.result()
    native = [
        row for row in _events(server._rsi_journal) if row["kind"] == "action_send"
    ]
    ids = [row["rpc_link"]["client_send_sequence"] for row in native]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert [row["rpc_link"]["rpc_slot_index"] for row in native] == [0, 1, 0, 1]
    assert [row["sent"][0][0] for row in native] in ([1, 1, 2, 2], [2, 2, 1, 1])


def test_actual_primitive_populates_rpc_metadata_for_selected_proposal_prefix(tmp_path):
    server, client, journal = _chain(tmp_path)
    journal.chunk_steps = 2
    model = SimpleNamespace(
        predict=lambda *_args, **_kwargs: np.arange(35).reshape(5, 7)
    )
    primitives = LiberoPrimitives(
        client, model, SimpleNamespace(), lambda: None, runtime_journal=journal
    )
    primitives.set_obs({"states": np.zeros(8), "task_descriptions": "task"})
    primitives._vlm_chunk("task")
    proposal = next(row for row in _events(journal) if row["kind"] == "action_proposal")
    end = _events(server._rsi_journal)[-1]
    assert len(proposal["proposal"]) == 5
    assert end["rpc_evidence"]["proposal_sequence"] == proposal["sequence"]
    assert end["execution_mask"] == [True, True, False, False, False]
    assert len(server.env._vector.actions) == 2


def test_scripted_primitive_binds_one_send_to_one_native_slot(tmp_path):
    server, client, journal = _chain(tmp_path)
    primitives = LiberoPrimitives(
        client,
        SimpleNamespace(),
        SimpleNamespace(),
        lambda: None,
        runtime_journal=journal,
    )
    primitives._active_primitive = "release"
    primitives._step_env(np.ones(7))
    client_send = next(row for row in _events(journal) if row["kind"] == "action_send")
    native = next(
        row for row in _events(server._rsi_journal) if row["kind"] == "action_feedback"
    )
    assert client_send["source"] == "release"
    assert native["rpc_link"]["client_send_sequence"] == client_send["sequence"]
    assert (
        native["rpc_link"]["client_proposal_sequence"]
        == client_send["proposal_sequence"]
    )
    assert native["rpc_link"]["client_proposal_index"] == 0


def test_proposal_slot_count_mismatch_is_rejected_before_native_motion(tmp_path):
    server, client, journal = _chain(tmp_path)
    evidence = {
        **journal.identity(),
        "client_send_sequence": 3,
        "proposal_sequence": 2,
        "proposal_indices": [0, 1],
        "proposal_length": 2,
    }
    with pytest.raises(ValueError, match="invalid RSI RPC proposal mapping"):
        client.rsi_step(np.ones(7), evidence=evidence)
    assert server.env._vector.actions == []
