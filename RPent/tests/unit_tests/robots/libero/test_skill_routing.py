# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Offline production dispatch and private environment evidence for finite rhoB."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.skill_routing import load_control, record_skill_observation
from robots.libero.toolkit import LiberoToolkit
from rpent.tools import ToolResult

SIGNATURE = {"theta_ref": "theta", "processor_ref": "processor", "phi_ref": "phi", "code_ref": "code"}
CRITERION = {"schema": "rsi-skill-lift-criterion-v1", "criterion_ref": "registered-object-lift-v1",
             "object_position_key": "salad_dressing_1_pos", "min_lift_m": .03,
             "max_eef_object_distance_m": .1, "min_gripper_opening": .005, "max_gripper_opening": .06}


def control(selected="pi0_pick"):
    return {"schema": "rsi-skill-routing-v1", "rho_ref": "rho-measured", "version_signature": SIGNATURE,
            "routes": [{"route_id": "pick-salad", "state": "salad:pick", "requested_worker": "vla_execute",
                        "trigger_workers": ["pi0_pick", "vla_execute"], "selected_worker": selected,
                        "instruction": "pick up the salad dressing",
                        "allowed_workers": ["pi0_pick", "vla_execute", "handoff", "defer"], "criterion": CRITERION}]}


class Journal:
    def __init__(self, projection):
        self.context = {**SIGNATURE, "relation_control": projection}
        self.events = []

    def identity(self):
        return dict(run_id="r", episode_id="e", bundle_id="b", protocol_id="p", profile_ref="profile", parent_decision_id=None)

    def emit(self, kind, **fields):
        sequence = len(self.events)
        self.events.append({**self.identity(), "sequence": sequence, "kind": kind, **fields})
        return sequence


def toolkit(selected="pi0_pick"):
    projection = control(selected)
    private = Journal(projection)
    client = Journal(projection)
    raw = {"salad_dressing_1_pos": np.array([0., 0., .1]), "robot0_eef_pos": np.array([0., 0., .2]),
           "robot0_gripper_qpos": np.array([.04, -.04])}
    env = SimpleNamespace(terminated=False, truncated=False)
    facade = object.__new__(LiberoEnvFacade)
    facade._env = SimpleNamespace(_rsi_journal=private)
    facade.raw_obs = lambda: raw
    env.rsi_skill_observation = facade.rsi_skill_observation
    calls = []

    def action(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            raw["salad_dressing_1_pos"][2] = .2
            raw["robot0_gripper_qpos"] = np.array([.01, -.01])
            return ToolResult(data={"name": name, "task_success": False})
        return run

    primitive = SimpleNamespace(env=env, begin_primitive=lambda name: None, end_primitive=lambda: None,
                                pi0_pick=action("pi0_pick"), vla_execute=action("vla_execute"))
    result = object.__new__(LiberoToolkit)
    result._primitives = primitive
    result._policy_kind = "vla"
    result.runtime_journal = client
    result._relation_control = load_control(client, {"pi0_pick", "vla_execute"})
    return result, calls, private, raw


def test_actual_vla_trigger_dispatches_registered_pick_once_with_same_allowance():
    tk, calls, private, _ = toolkit()
    out = tk._execute_primitive("vla_execute", tk._primitives.vla_execute,
                                instruction="pick up the salad dressing", max_chunks=2)
    assert calls == [("pi0_pick", {"prompt": "pick up the salad dressing", "max_chunks": 2})]
    assert out.data["relation_route"]["trigger_worker"] == "vla_execute"
    assert out.data["relation_route"]["selected_worker"] == "pi0_pick"
    assert "measurements" not in str(out.data)
    assert [e["phase"] for e in private.events] == ["start", "handoff"]
    assert private.events[0]["measurements"]["object_pos"][2] == .1
    assert private.events[1]["measurements"]["object_pos"][2] == .2
    assert all(set(event["environment_observation"]) == {"source_sequence", "status"}
               for event in tk.runtime_journal.events if event["kind"].startswith("skill_route_"))


def test_pick_trigger_can_dispatch_vla_with_identical_goal_and_allowance():
    tk, calls, _, _ = toolkit("vla_execute")
    tk._execute_primitive("pi0_pick", tk._primitives.pi0_pick,
                           prompt="pick up the salad dressing", max_chunks=3)
    assert calls == [("vla_execute", {"instruction": "pick up the salad dressing", "max_chunks": 3})]


@pytest.mark.parametrize("selected", ["handoff", "defer"])
def test_handoff_and_defer_return_to_planner_without_dispatch(selected):
    tk, calls, private, _ = toolkit(selected)
    out = tk._execute_primitive("vla_execute", tk._primitives.vla_execute,
                                instruction="pick up the salad dressing", max_chunks=2)
    assert calls == [] and private.events == []
    assert out.data["action"] == selected and out.data["executed_steps"] == 0


def test_place_instruction_and_unregistered_contact_keep_original_handler():
    tk, calls, private, _ = toolkit()
    tk._execute_primitive("vla_execute", tk._primitives.vla_execute,
                           instruction="place in the basket", max_chunks=2)
    tk._execute_primitive("pi0_doubled", lambda **kwargs: calls.append(("contact", kwargs)))
    assert [call[0] for call in calls] == ["vla_execute", "contact"]
    assert private.events == []


def test_runtime_rejects_incompatible_versions_and_unavailable_skills():
    projection = control()
    journal = Journal(projection)
    journal.context["phi_ref"] = "changed"
    with pytest.raises(ValueError, match="version binding"):
        load_control(journal, {"pi0_pick", "vla_execute"})
    with pytest.raises(ValueError, match="compatible registered"):
        load_control(Journal(projection), {"vla_execute"})
    unsupported = deepcopy(projection)
    unsupported["routes"][0]["selected_worker"] = "new-expert"
    with pytest.raises(ValueError, match="compatible registered"):
        load_control(Journal(unsupported), {"pi0_pick", "vla_execute"})


def test_missing_environment_key_is_unknown_and_rpc_returns_no_privileged_geometry():
    projection = control()
    journal = Journal(projection)
    value = record_skill_observation(journal, {}, {**journal.identity(), "route_id": "pick-salad",
                                    "worker": "pi0_pick", "phase": "start", "client_route_sequence": 0})
    assert value == {"source_sequence": 0, "status": "unknown"}
    assert journal.events[0]["measurements"]["object_pos"] is None
    assert journal.events[0]["missing"]
    with pytest.raises(ValueError, match="identity mismatch"):
        record_skill_observation(journal, {}, {"run_id": "another"})


def test_existing_client_rpc_exposes_only_private_observation_receipt():
    calls = []
    client = object.__new__(LiberoEnvClient)
    client._client = SimpleNamespace(call=lambda *args, **kwargs: calls.append((args, kwargs)) or {"source_sequence": 2, "status": "recorded"})
    value = client.rsi_skill_observation(evidence={"route_id": "pick-salad"})
    assert value == {"source_sequence": 2, "status": "recorded"}
    assert calls[0][0] == ("env.rsi_skill_observation",)
    assert calls[0][1]["kwargs"] == {"evidence": {"route_id": "pick-salad"}}


@pytest.mark.parametrize("name", ["read_text_file", "write_text_file", "list_dir"])
def test_policy_file_tools_cannot_read_or_change_private_environment_evidence(tmp_path, name):
    tk, _, _, _ = toolkit()
    journal = tk.runtime_journal
    journal.context["output_dir"] = str(tmp_path)
    journal.admit_tool = lambda name: None
    journal.budget_snapshot = lambda: {"tool_calls": 1}
    private = tmp_path / "environment_service/action_events.jsonl"
    private.parent.mkdir()
    private.write_text("private object xyz")
    symlink = tmp_path / "alias.jsonl"
    symlink.symlink_to(private)
    for path in (private, symlink):
        value = tk.execute_tool(name, {"path": str(path), "content": "spoofed"})
        assert "private" in value.data["error"]
        assert "private object xyz" not in str(value.data)
        assert private.read_text() == "private object xyz"


def drawer_control(selected="pi0_doubled"):
    criterion = {"schema": "rsi-skill-drawer-open-criterion-v1", "criterion_ref": "registered-drawer-open-v1",
                 "region": "wooden_cabinet_1_middle_region", "fixture": "wooden_cabinet_1",
                 "joint": "wooden_cabinet_1_middle_level", "joint_type": "slide",
                 "joint_range": [-.16, .01], "open_threshold": -.14}
    return {"schema": "rsi-skill-routing-v1", "rho_ref": "rho-drawer", "version_signature": SIGNATURE,
            "routes": [{"route_id": "drawer", "task_key": "old_drawer", "state": "old_drawer",
                        "requested_worker": "vla_execute", "trigger_workers": ["vla_execute", "pi0_doubled"],
                        "selected_worker": selected, "instruction": "Open the middle layer of the drawer",
                        "allowed_workers": ["vla_execute", "pi0_doubled", "handoff", "defer"],
                        "max_chunks": 24, "criterion": criterion}]}


def drawer_toolkit(selected="pi0_doubled"):
    projection = drawer_control(selected)
    client, private = Journal(projection), Journal(projection)
    criterion = projection["routes"][0]["criterion"]
    joint = {"schema": "rsi-private-drawer-joint-observation-v1", "status": "recorded", "qpos": 0., "missing": [],
             **{key: criterion[key] for key in ("region", "fixture", "joint", "joint_type", "joint_range")}}
    calls = []
    env = SimpleNamespace(terminated=False, truncated=False,
        rsi_skill_observation=lambda *, evidence: record_skill_observation(private, {}, evidence, joint_observation=joint))
    def action(name):
        def execute(**kwargs):
            calls.append((name, kwargs))
            joint["qpos"] = -.15
            return ToolResult(data={"name": name, "success": False, "task_success": False, "chunks_used": 1})
        return execute
    tk = object.__new__(LiberoToolkit)
    tk._primitives = SimpleNamespace(env=env, begin_primitive=lambda name: None, end_primitive=lambda: None,
        vla_execute=action("vla_execute"), pi0_doubled=action("pi0_doubled"))
    tk._policy_kind = "vla"
    tk.runtime_journal = client
    tk._relation_control = load_control(client, {"vla_execute", "pi0_doubled", "pi0_pick"})
    return tk, calls, private


def test_registered_drawer_vla_trigger_dispatches_contact_once_without_private_qpos():
    tk, calls, private = drawer_toolkit()
    goal = drawer_control()["routes"][0]["instruction"]
    result = tk._execute_primitive("vla_execute", tk._primitives.vla_execute, instruction=goal, max_chunks=24)
    assert calls == [("pi0_doubled", {"prompt": goal, "max_chunks": 24})]
    assert [event["measurements"]["joint"]["qpos"] for event in private.events] == [0., -.15]
    assert "qpos" not in str(result.data) and "measurements" not in str(result.data)
    assert all(set(event["environment_observation"]) == {"source_sequence", "status"}
               for event in tk.runtime_journal.events if event["kind"].startswith("skill_route_"))


def test_registered_drawer_contact_trigger_can_dispatch_vla_and_missing_allowance_is_explicit_24():
    goal = drawer_control()["routes"][0]["instruction"]
    tk, calls, _ = drawer_toolkit("vla_execute")
    tk._execute_primitive("pi0_doubled", tk._primitives.pi0_doubled, prompt=goal, max_chunks=24)
    assert calls == [("vla_execute", {"instruction": goal, "max_chunks": 24})]
    tk, calls, _ = drawer_toolkit()
    tk._execute_primitive("pi0_doubled", tk._primitives.pi0_doubled, prompt=goal)
    assert calls == [("pi0_doubled", {"prompt": goal, "max_chunks": 24})]


def test_drawer_route_rejects_pick_group_wrong_allowance_and_wrong_versions():
    goal = drawer_control()["routes"][0]["instruction"]
    for max_chunks in (12, 23, 25):
        tk, calls, private = drawer_toolkit()
        with pytest.raises(ValueError, match="allowance|24chunk"):
            tk._execute_primitive("vla_execute", tk._primitives.vla_execute, instruction=goal, max_chunks=max_chunks)
        assert calls == [] and private.events == []
    projection = drawer_control()
    projection["routes"][0]["allowed_workers"].append("pi0_pick")
    with pytest.raises(ValueError, match="compatible registered"):
        load_control(Journal(projection), {"pi0_pick", "pi0_doubled", "vla_execute"})
    journal = Journal(drawer_control())
    journal.context["code_ref"] = "another-version"
    with pytest.raises(ValueError, match="version binding"):
        load_control(journal, {"pi0_doubled", "vla_execute"})
