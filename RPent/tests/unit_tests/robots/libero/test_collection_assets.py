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

"""Actual writer and exporter consumers with synthetic CPU transitions."""

import json
from pathlib import Path

import numpy as np
import pytest

from robots.libero.flywheel import LIBERO_SPEC, create_episode_writer
from rpent.flywheel.episode import read_episode_arrays, validate_episode
from rpent.flywheel.export import export_lerobot
from rpent.session.runtime import RuntimeJournal
from tests.unit_tests.robots.libero.test_bounded_vla import context, driver


def writer(root):
    obs = {
        "main_images": np.zeros((256, 256, 3), np.uint8),
        "wrist_images": np.ones((256, 256, 3), np.uint8),
        "states": np.zeros(8, np.float32),
        "task_descriptions": "native task",
    }
    config = {
        "root": str(root),
        "suite": "libero_object",
        "task_id": 2,
        "seed": 1,
        "collection_version": "aligned-batches-v1",
        "flush_steps": 2,
        "information_scope": "development",
        "identity": {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
        },
    }
    return create_episode_writer(config, obs), obs


def test_batches_interrupt_tail_masks_and_readback(tmp_path):
    w, obs = writer(tmp_path)
    w.record_input(
        obs, inference_sequence=2, observation_sequence=1, segment_sequence=3
    )
    actions = np.zeros((5, 7), np.float32)
    proposal = w.add_proposal("remaining subgoal", actions)
    w.begin_dispatch(actions[:3], source="vla", proposal_id=proposal, proposal_length=5)
    for index in range(3):
        obs["states"] = np.full(8, index + 1, np.float32)
        w.add_transition(
            actions[index],
            obs,
            index,
            index == 2,
            False,
            vla_id=proposal,
            proposal_index=index,
        )
    partial = w.path.with_name(w.path.name + ".partial")
    status = json.loads((partial / "collection.json").read_text())
    assert status["persisted_steps"] == 2
    assert status["pending_dispatch"]["expected_steps"] == 3
    assert len(read_episode_arrays(partial)["actions"]) == 2
    w.end_dispatch()
    path = w.finalize()
    metadata = validate_episode(path, spec=LIBERO_SPEC)
    assert metadata["alignment_complete"] is True
    assert metadata["training_admission"] == "not_admitted"
    assert metadata["source_kind"] == "autonomous_execution"
    arrays = read_episode_arrays(path)
    assert arrays["states"][:, 0].tolist() == [0, 1, 2, 3]
    assert arrays["proposal_index"].tolist() == [0, 1, 2]
    events = [
        json.loads(line) for line in (path / "alignment.jsonl").read_text().splitlines()
    ]
    assert events[-1]["execution_mask"] == [True, True, True, False, False]
    with pytest.raises(ValueError, match="require separate training admission"):
        export_lerobot(
            [path],
            spec=LIBERO_SPEC,
            expected_metadata={"suite": "libero_object"},
            repo_id_prefix="fixture/test",
            output_root=tmp_path / "forbidden",
        )


def test_actual_vla_consumer_and_handoff_only_after_allowance(tmp_path):
    import os

    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "schema": "rsi-runtime-rules-v4",
                "chunk_steps": 5,
                "baseline_chunk_steps": 5,
                "finish_requires_native_success": False,
                "planner_finish_reserve_calls": 0,
                "planner_segment_handoff_hint": True,
            }
        )
    )
    ctx = context(tmp_path)
    ctx["runtime_rules_ref"] = str(rules)
    journal = RuntimeJournal(ctx)
    p, env, _ = driver(journal=journal)
    original = env.obs

    def observations():
        obs = original()
        obs["main_images"] = np.full((256, 256, 3), env.steps, np.uint8)
        obs["wrist_images"] = np.ones((256, 256, 3), np.uint8)
        return obs

    env.obs = observations
    w, _ = writer(tmp_path / "raw")
    p._flywheel = w
    p.set_obs(env.obs())
    out = p.vla_execute("remaining placement", max_chunks=1).data
    assert "handoff_hint" in out
    assert env.steps == 5
    env.terminal = "success"
    out = p.vla_execute("remaining placement", max_chunks=1).data
    assert "handoff_hint" not in out
    path = w.finalize()
    events = [
        json.loads(line) for line in (path / "alignment.jsonl").read_text().splitlines()
    ]
    inputs = [e for e in events if e["kind"] == "policy_input"]
    assert [e["instruction"] for e in inputs] == ["remaining placement"] * 2
    sends = [e for e in events if e["kind"] == "send"]
    assert all(type(e["client_send_sequence"]) is int for e in sends)
    with np.load(path / "input-00001.npz") as data:
        assert data["states"][0] == 5
    exported = export_lerobot(
        [path],
        spec=LIBERO_SPEC,
        expected_metadata={"suite": "libero_object", "task_id": 2},
        repo_id_prefix="fixture/autonomous",
        output_root=tmp_path / "export",
        dataset_id="diagnostic",
        purpose="diagnostic",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(exported["repo_id"], root=exported["dataset_path"])
    assert len(dataset) == 10
    assert float(dataset[5]["state"][0]) == 5
    assert exported["training_admission"] == "not_admitted"
    evidence = os.environ.get("RSI_ASSET_FIXTURE_OUTPUT")
    if evidence:
        import shutil

        target = Path(evidence)
        shutil.copytree(tmp_path, target)
        (target / "export-check.json").write_text(json.dumps(exported))


def test_supervised_configuration_and_manifest_keep_native_invalid_separate(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import rpent.session.runtime as runtime
    from robots.libero.robot_spec import flywheel_configuration

    identity = {"run_id": "r", "episode_id": "e", "bundle_id": "b", "protocol_id": "p"}
    ctx = {
        **identity,
        "output_dir": str(tmp_path),
        "information_scope": "development",
        "task": {"suite": "libero_object", "task": 2, "seed": 1},
        "collection": {
            "root": str(tmp_path / "data"),
            "collection_version": "aligned-batches-v1",
            "flush_steps": 2,
            "required_for_release": True,
        },
    }
    monkeypatch.setattr(runtime, "runtime_context", lambda: ctx)
    args = SimpleNamespace(
        flywheel_root=str(tmp_path / "data"), suite="libero_object", task=2, seed=1
    )
    config = flywheel_configuration(args)
    assert config["identity"] == identity
    args.flywheel_root = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="root"):
        flywheel_configuration(args)


def test_owner_retains_available_joint_state_without_extra_rpc(tmp_path, monkeypatch):
    from robots.libero.runtime_evidence import observed_libero_class
    from tests.unit_tests.robots.libero.test_rpc_evidence import _Native, _Vector

    def native_step(vector, actions):
        vector.actions.append(actions.copy())
        return (
            [
                {
                    "robot0_eef_pos": np.zeros(3),
                    "robot0_joint_pos": np.arange(7),
                    "robot0_joint_vel": np.ones(7),
                }
            ],
            np.zeros(1),
            np.array([False]),
            [{}],
        )

    monkeypatch.setattr(_Vector, "step", native_step)
    cls = observed_libero_class(_Native, context(tmp_path))
    env = cls()
    env.step(np.zeros((1, 7)))
    assert len(env.env._vector.actions) == 1
    events = [
        json.loads(line)
        for line in env._rsi_journal.events_path.read_text().splitlines()
    ]
    feedback = next(e for e in events if e["kind"] == "action_feedback")
    state = feedback["feedback"]["observed_state"][0]
    assert state["robot0_joint_pos"] == list(range(7))
    assert state["robot0_joint_vel"] == [1] * 7
    assert state["robot0_gripper_qvel"] is None


def test_missing_native_flag_is_not_saved_as_false(tmp_path):
    w, obs = writer(tmp_path)
    w.begin_dispatch(np.zeros(7), source="unknown")
    with pytest.raises(ValueError, match="native feedback"):
        w.add_transition(np.zeros(7), obs, 0, None, False)
    partial = w.path.with_name(w.path.name + ".partial")
    events = [
        json.loads(line)
        for line in (partial / "alignment.jsonl").read_text().splitlines()
    ]
    assert events[-1]["terminated"] is None
    assert events[-1]["missing"] == ["terminated"]
    assert w.step_count == 0
    assert (
        json.loads((partial / "collection.json").read_text())["pending_dispatch"]
        is not None
    )


def test_formal_reset_records_existing_cached_action_contract(tmp_path):
    p, env, _ = driver()
    original = env.obs

    def observation():
        obs = original()
        obs["main_images"] = np.zeros((256, 256, 3), np.uint8)
        obs["wrist_images"] = np.zeros((256, 256, 3), np.uint8)
        return obs

    env.obs = observation
    env.action_specs = {
        "default": {
            "description": "OSC_POSE base delta, normalized [-1,1], output metres/radians"
        }
    }
    p._flywheel_config = {
        "root": str(tmp_path),
        "suite": "libero_object",
        "task_id": 2,
        "seed": 1,
        "collection_version": "aligned-batches-v1",
        "identity": {"run_id": "r"},
    }
    p.reset()
    metadata = json.loads((p._flywheel._partial / "collection.json").read_text())
    assert metadata["native_action_spec"] == env.action_specs
    assert env.steps == 0


def test_raw_policy_state_precision_survives_export_view(tmp_path):
    w, obs = writer(tmp_path)
    obs["states"] = np.full(8, 1.0000000001, np.float64)
    w.record_input(obs)
    w.begin_dispatch(np.zeros((1, 7)), source="vla")
    w.add_transition(np.zeros(7), obs, 0, False, False)
    w.end_dispatch()
    path = w.finalize()
    with np.load(path / "input-00000.npz") as data:
        assert data["states"].dtype == np.float64
        np.testing.assert_array_equal(data["states"], obs["states"])
    arrays = read_episode_arrays(path)
    np.testing.assert_array_equal(arrays["raw_states"][1], obs["states"])
    assert arrays["states"].dtype == np.float32
