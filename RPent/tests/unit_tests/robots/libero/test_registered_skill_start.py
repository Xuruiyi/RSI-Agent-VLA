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

"""A registered controlled skill call reaches Planner without replacing its rules."""

import json

from robots.libero.task_input import GLOBAL_MEMORY, V2_EXCERPTS, build_task_input


def test_registered_pick_start_retains_task_rules_and_exact_allowance(tmp_path):
    memory = tmp_path / "revision/version"
    memory.mkdir(parents=True)
    for name in GLOBAL_MEMORY:
        path = memory / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(V2_EXCERPTS[name]))
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"memory": {"revision": "revision", "version": "version"}})
    )
    context = {
        "run_id": "run",
        "episode_id": "ep",
        "bundle_id": "bundle",
        "protocol_id": "protocol",
        "profile_ref": str(profile),
        "task": {"suite": "libero_object_swap", "task": 2, "seed": 2},
        "task_input": {
            "version": "libero-startup-v2-bounded-vla",
            "task_language": "Place salad dressing in basket",
        },
    }
    before = build_task_input(context, memory, context["task"])
    context["relation_skill_start"] = {
        "route_id": "pick",
        "tool": "vla_execute",
        "instruction": "pick up the salad dressing bottle",
        "max_chunks": 4,
    }
    after = build_task_input(context, memory, context["task"])
    assert after["system_prompt"] == before["system_prompt"]
    assert after["user_message"].startswith(before["user_message"])
    assert (
        '"instruction": "pick up the salad dressing bottle", "max_chunks": 4'
        in after["user_message"]
    )
    assert "does not establish holding or native task success" in after["user_message"]
    assert after["sources"] == before["sources"]


def test_registered_drawer_contact_start_uses_prompt_and_preserves_original_task_rules(
    tmp_path,
):
    memory = tmp_path / "revision/version"
    memory.mkdir(parents=True)
    for name in GLOBAL_MEMORY:
        path = memory / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(V2_EXCERPTS[name]))
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"memory": {"revision": "revision", "version": "version"}})
    )
    context = {
        "run_id": "run",
        "episode_id": "drawer",
        "bundle_id": "bundle",
        "protocol_id": "protocol",
        "profile_ref": str(profile),
        "task": {"suite": "libero_goal", "task": 3, "seed": 2},
        "task_input": {
            "version": "libero-startup-v2-bounded-vla",
            "task_language": "Open the middle layer of the drawer",
        },
    }
    before = build_task_input(context, memory, context["task"])
    context["relation_skill_start"] = {
        "route_id": "drawer",
        "tool": "pi0_doubled",
        "instruction": "Open the middle layer of the drawer",
        "max_chunks": 24,
    }
    after = build_task_input(context, memory, context["task"])
    assert after["system_prompt"] == before["system_prompt"]
    assert after["user_message"].startswith(before["user_message"])
    assert (
        '"prompt": "Open the middle layer of the drawer", "max_chunks": 24'
        in after["user_message"]
    )
    assert "registered subgoal" in after["user_message"]
    assert after["sources"] == before["sources"]
