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

"""Registered task hints reach Planner startup without replacing existing input."""

import json

import pytest

from robots.libero.task_input import GLOBAL_MEMORY, V2_EXCERPTS, build_task_input
from rpent.session.runtime import RuntimeJournal


def rules(hint):
    return {
        "schema": "rsi-runtime-rules-v5",
        "chunk_steps": 5,
        "baseline_chunk_steps": 5,
        "finish_requires_native_success": False,
        "planner_finish_reserve_calls": 0,
        "planner_segment_handoff_hint": False,
        "planner_task_hint": hint,
    }


def journal(tmp_path, value=None):
    context = {
        "run_id": "r",
        "episode_id": "e",
        "bundle_id": "b",
        "protocol_id": "p",
        "profile_ref": "offline",
        "output_dir": str(tmp_path / "run"),
        "limits": {"tool_calls": 60, "environment_steps": 2000},
    }
    if value is not None:
        ref = tmp_path / "rules.json"
        ref.write_text(json.dumps(value))
        context["runtime_rules_ref"] = str(ref)
    return RuntimeJournal(context)


def test_v5_records_exact_model_text_and_keeps_original_admission(tmp_path):
    hint = "  Use the current task's target label.\nInspect the latest images.  "
    runtime = journal(tmp_path, rules(hint))
    start = json.loads(runtime.events_path.read_text().splitlines()[0])
    assert runtime.planner_task_hint == hint
    assert start["effective_rules"]["planner_task_hint"] == hint
    assert runtime.chunk_steps == 5 and runtime.tool_calls == 0
    assert runtime.finish_requires_native_success is False
    assert runtime.planner_finish_reserve_calls == 0
    assert runtime.planner_segment_handoff_hint is False
    assert runtime.context["limits"] == {"tool_calls": 60, "environment_steps": 2000}


@pytest.mark.parametrize("hint", [None, True, 1, [], "", " \n\t", "a" * 1201])
def test_v5_rejects_invalid_task_hint(tmp_path, hint):
    with pytest.raises(ValueError, match="Planner task hint"):
        journal(tmp_path, rules(hint))


def test_v5_accepts_1200_characters_without_rewriting(tmp_path):
    assert journal(tmp_path, rules("任" * 1200)).planner_task_hint == "任" * 1200


@pytest.mark.parametrize("change", ["missing_hint", "missing_v4_field", "unexpected_field", "old_schema"])
def test_v5_requires_exact_schema_fields(tmp_path, change):
    value = rules("Inspect the target label.")
    if change == "missing_hint":
        value.pop("planner_task_hint")
    elif change == "missing_v4_field":
        value.pop("planner_segment_handoff_hint")
    elif change == "unexpected_field":
        value["extra_instruction"] = "not registered"
    else:
        value["schema"] = "rsi-runtime-rules-v4"
    with pytest.raises(ValueError, match="RSI runtime rule artifact"):
        journal(tmp_path, value)


@pytest.mark.parametrize("value", [None, {"schema": "rsi-runtime-rules-v1", "chunk_steps": 5, "baseline_chunk_steps": 5}])
def test_old_rules_keep_empty_task_hint(tmp_path, value):
    runtime = journal(tmp_path, value)
    start = json.loads(runtime.events_path.read_text().splitlines()[0])
    assert runtime.planner_task_hint == ""
    assert start["effective_rules"]["planner_task_hint"] == ""


@pytest.mark.parametrize("version", ["libero-startup-v1", "libero-startup-v2-bounded-vla"])
def test_task_hint_appends_only_to_planner_user_input(tmp_path, version):
    memory = tmp_path / "revision" / "version"
    memory.mkdir(parents=True)
    for relative in GLOBAL_MEMORY:
        path = memory / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(V2_EXCERPTS[relative]))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"memory": {"revision": "revision", "version": "version"}}))
    context = {
        "run_id": "r", "episode_id": "e", "bundle_id": "b", "protocol_id": "p",
        "profile_ref": str(profile),
        "task": {"suite": "libero_object_swap", "task": 2, "seed": 2},
        "task_input": {"version": version, "task_language": "Pick the salad dressing and place it in the basket"},
    }
    original = build_task_input(context, memory, context["task"])
    assert build_task_input({**context, "planner_task_hint": ""}, memory, context["task"]) == original
    hint = "Select the salad dressing by its current visible label; use the latest images."
    view = build_task_input({**context, "planner_task_hint": hint}, memory, context["task"])
    assert view == {
        **original,
        "user_message": original["user_message"] + "\n\nAdditional Planner instruction from the runtime rule:\n" + hint,
    }
    assert "Success requires native terminated=true." in view["system_prompt"]
    assert "no reset" in view["system_prompt"].lower()
