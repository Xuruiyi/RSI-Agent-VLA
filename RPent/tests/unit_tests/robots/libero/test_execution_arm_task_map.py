# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""One frozen arm uses the registered program for the actual task."""

from copy import deepcopy

import pytest

from robots.libero.execution_arms import run_execution_arm, select_execution_arm
from rpent.tools import ToolResult
from tests.unit_tests.robots.libero.test_bounded_vla import make_toolkit


def programs(arm):
    result = {}
    for task_key, instruction in (
        ("base_salad", "pick salad dressing"),
        ("position_swap_salad", "pick salad dressing from its current position"),
        ("old_milk", "pick milk"),
        ("old_drawer", "open the middle drawer"),
    ):
        spec = {"schema": "libero-execution-arm-v1", "arm": arm, "max_chunks": 2}
        if arm in {"S", "R"}:
            stage = {"id": "first", "instruction": instruction, "max_chunks": 2}
            if arm == "R":
                stage.update(on_complete="stop", on_timeout="stop", on_unknown="stop")
            spec["stages"] = [stage]
        result[task_key] = spec
    return result


@pytest.mark.parametrize("arm", ["S", "R"])
@pytest.mark.parametrize(
    "task_key", ["base_salad", "position_swap_salad", "old_milk", "old_drawer"]
)
def test_task_selected_frozen_instruction_reaches_actual_vla(
    tmp_path, monkeypatch, arm, task_key
):
    tk, _env, model, _journal = make_toolkit(tmp_path, monkeypatch)
    tk.get_env_state = lambda *, result, **kwargs: ToolResult(
        data={"execution_segment": result}
    )
    mapping = programs(arm)
    profile = {"execution_arm_by_task": mapping}
    before = deepcopy(profile)
    selected = select_execution_arm(profile, task_key)
    run_execution_arm(tk, selected)
    assert [obs["task_descriptions"] for obs in model.inputs] == [
        mapping[task_key]["stages"][0]["instruction"]
    ] * 2
    assert profile == before
    selected["stages"][0]["instruction"] = "changed after selection"
    assert profile == before


def test_direct_mapping_keeps_native_instruction_and_refuses_stages(
    tmp_path, monkeypatch
):
    tk, _env, model, _journal = make_toolkit(tmp_path, monkeypatch)
    tk.get_env_state = lambda *, result, **kwargs: ToolResult(
        data={"execution_segment": result}
    )
    profile = {"execution_arm_by_task": programs("D")}
    run_execution_arm(tk, select_execution_arm(profile, "old_milk"))
    assert [obs["task_descriptions"] for obs in model.inputs] == ["native task"] * 2
    profile["execution_arm_by_task"]["old_drawer"]["stages"] = []
    with pytest.raises(ValueError, match="native instruction without stages"):
        select_execution_arm(profile, "old_milk")


@pytest.mark.parametrize("arm", ["D", "S", "R", "A"])
def test_mapping_validates_all_entries_and_never_silently_falls_back(arm):
    mapping = programs(arm)
    profile = {"execution_arm": mapping["base_salad"], "execution_arm_by_task": mapping}
    with pytest.raises(ValueError, match="no frozen execution program"):
        select_execution_arm(profile, "unregistered-task")
    profile["execution_arm_by_task"]["old_drawer"] = programs(
        "S" if arm != "S" else "D"
    )["old_drawer"]
    with pytest.raises(ValueError, match="same arm"):
        select_execution_arm(profile, "base_salad")


def test_legacy_single_program_and_unconfigured_profile_keep_original_behavior():
    assert select_execution_arm({}, "old_milk") is None
    spec = programs("S")["old_milk"]
    assert select_execution_arm({"execution_arm": spec}, "old_drawer") == spec
