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

"""Versioned startup context from explicit legal memory, without model calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GLOBAL_MEMORY = (
    "global/brand-label-over-segmentation.md",
    "global/avoid-full-task-prompt-after-miss.md",
    "global/retreat-high-before-lateral-near-lined-basket.md",
)

# Engineering-authored execution instructions; external memory remains verbatim.
OPERATIONS = """Use structured tools only. The runner owns services, saving results and cleanup.
One continuous episode: no reset, restart, installation, download or device access.
Task language from view_env_state is authoritative. Inspect step 0 before acting.
Do not read BDDL, benchmark code, object poses, evaluator/release/sealed results,
world-map files directly or historical absolute coordinates. Use images and
back_project/segment for geometry; state exposes proprioception and names only.
Agentview chooses target identity and destination by RGB/relation. Wrist refines
the same target, within 3-5 cm of the agentview anchor; basket placement uses the
open interior. Localize each relevant entity before manipulation. Select 3-8
pixels per entity; back_project_batch returns ordered per-point results/errors.
Use the same step, camera and resolution as the image; high is 1024, low is 256.
Re-observe after motion and re-localize moved objects. Do not invent coordinates.
pi0_pick is a bounded grasp tool; it can carry inadvertently. Inspect wrist/grip
before further motion. Its success flag is a heuristic, not task completion.
Use move_to for carry, release for placement; +1 CLOSE/hold and -1 OPEN. Always
pass gripper=1 to move_pose when carrying (its default opens). If move_to stalls,
inspect the error and use existing move_pose/rotation tools within the same
budget. pi0_doubled is bounded contact, not an autonomous task plan. No teleports.
On failure or uncertainty inspect available evidence, recover only within the
remaining original allowance (at most one local recovery), or stop honestly.
No automatic retries, fresh episodes or extra budgets. No hidden action batches.
Success requires native terminated=true. An active false signal is not success.
Call finish for a voluntary stop; runtime independently saves result/costs even
without finish. Budget exhaustion blocks new requests/tools, drains admitted
operations and closes automatically. Do not spend tools writing a duplicate audit.
Tool feedback shows remaining tools, actual model requests, time and environment
steps. Missing owner values are unknown; admission supervision is authoritative.
Read a guide or memory only for a specific unresolved operation, not to repeat
startup preparation. External memory can describe old resets/coordinates or
other task nouns: current task, perception and this episode's rules take priority.
"""


OPERATIONS_V2 = OPERATIONS.replace(
    "Localize each relevant entity before manipulation. Select 3-8",
    "For geometry tools, localize each relevant entity before manipulation. Select 3-8",
).replace(
    "Use move_to for carry, release for placement; +1 CLOSE/hold and -1 OPEN. Always",
    "On the fine-grained path use move_to for carry and release for placement; +1 CLOSE/hold and -1 OPEN. Always",
)
OPERATIONS_V2 += """
Execution version bounded-vla-v1: prefer vla_execute for a task or subgoal.
Pass explicit instruction and max_chunks (1-24); 24 is a provisional segment
allowance, not a task capability claim. Each prediction uses the latest VLA
observation and executes the existing five-action chunk. Instructions are local
to this call. VLA may execute pick, carry, place or another stage, without returning
to Planner after every chunk. Planner selects goals, supervises segment feedback,
continues, switches to existing tools, recovers or stops within the same budgets.
At each intervention briefly state its reason in the transcript, including why
fine-grained tools are chosen. This needs no separate audit or extra tool call.
Chunk allowance exhausted with native success=false means unfinished, not stage
failure. No holding, stage-success or stagnation detector is implied.
pi0_pick remains grasp-only; pi0_doubled remains contact-only. Fine-grained tools
remain available for specific interventions. Do not replace decisions with a
prewritten task script. Native task success automatically closes Planner admission;
no extra request or duplicate audit is needed.
Planner feedback defaults to calibrated agentview/wrist images; request
view_env_state(image_detail="full") for the additional policy view. Pixel geometry
must use image_frames camera, resolution and step, never the policy-oriented view.
External missed-pick advice applies to grasp recovery while Planner owns placement;
it does not prohibit full-task instructions on the bounded VLA path. Older hybrid
guides describe the fine-grained path; their grasp-only division is not global here.
"""

# Verbatim, condition-preserving excerpts. Full upstream files remain unchanged.
V2_EXCERPTS = {
    GLOBAL_MEMORY[0]: (
        "applies_when: Grocery or packaged-object tasks name a brand/flavor/label and similar\n  packages share shape or color.",
        "confidence: single-shot",
        "When a task names a specific grocery label, use agentview_high.png label/appearance as the identity authority and treat SAM masks as geometry candidates only after the overlay matches that identity.",
    ),
    GLOBAL_MEMORY[1]: (
        "applies_when: A Pi0 pick misses or only partially engages and the task still needs\n  the LLM to script placement.",
        "confidence: single-shot",
        "After a missed pick, escalating to the full task language can make Pi0 resume an end-to-end pick/place habit; retry with a short grasp-only prompt and a changed chunk budget instead.",
    ),
    GLOBAL_MEMORY[2]: (
        "applies_when: placing items into a woven/cloth-lined basket or any container with\n  a soft rim/liner that stands above the rigid rim; especially multi-item drop-ins\n  where you must traverse away between placements",
        "confidence: single-shot",
        "a purely vertical retreat does not (the tips exit along the same corridor they\nentered).",
    ),
}


def build_task_input(
    context: dict[str, Any], memory_root: Path, variables: dict[str, Any]
) -> dict[str, Any]:
    """Render an opt-in startup view and preserve each source or missing match."""
    options = context["task_input"]
    v2 = options["version"] == "libero-startup-v2-bounded-vla"
    if options["version"] not in {"libero-startup-v1", "libero-startup-v2-bounded-vla"}:
        raise ValueError("unsupported startup input version")
    task = context["task"]
    if any(str(variables[key]) != str(task[key]) for key in ("suite", "task", "seed")):
        raise ValueError("startup task identity mismatch")
    root = Path(memory_root).resolve()
    profile = json.loads(Path(context["profile_ref"].partition("#")[0]).read_text())
    memory = profile["memory"]
    if root.name != memory["version"] or root.parent.name != memory["revision"]:
        raise ValueError("startup memory must match frozen version/revision")
    matching = f"task-family/task-family_{task['suite']}_t{task['task']}.md"
    sources, missing = [], []
    for relative in (*GLOBAL_MEMORY, matching):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        text = path.read_text()
        excerpts = V2_EXCERPTS.get(relative, ()) if v2 else ()
        if v2 and not excerpts:
            # A newly present task recipe needs explicit relevance review before default inclusion.
            sources.append(
                {
                    "ref": str(path),
                    "text": "",
                    "selection": "on_demand_only",
                    "provenance": "external_upstream_prior",
                    "project_learning": False,
                }
            )
            continue
        if v2:
            if not all(excerpt in text for excerpt in excerpts):
                raise ValueError(f"upstream excerpt no longer matches: {relative}")
            text = "\n".join(excerpts)
        sources.append(
            {
                "ref": str(path),
                "text": text,
                "selection": "verbatim_excerpt" if v2 else "complete_original",
                "provenance": "external_upstream_prior",
                "project_learning": False,
            }
        )
    guides = [
        str(Path(__file__).parent / "guides" / name)
        for name in ("strict_hybrid_guide.md", "pro_hybrid_guide.md")
    ]
    system = (
        OPERATIONS_V2 if v2 else OPERATIONS
    ) + "\nExisting tool schemas define arguments and action limits.\n"
    user = (
        f"Task: {options['task_language']}\n"
        f"suite={task['suite']}, task={task['task']}, seed={task['seed']}\n"
        "Confirm task_language and initial images with view_env_state(step=0).\n"
        + "\n".join(f"Source: {item['ref']}\n{item['text']}" for item in sources)
        + f"\nMissing matching sources: {missing}\n"
        + "On-demand guides: "
        + ", ".join(guides)
        + f"\nOn-demand memory index: {root / 'MEMORY.md'}\n"
        + ("Skills: vla_execute (bounded task/subgoal); " if v2 else "")
        + "Skills: pi0_pick (grasp), move_to/move_pose (motion), release, set_gripper, "
        "rotate_wrist/rotate_pitch, pi0_doubled (contact); use their exposed schemas."
    )
    task_hint = context.get("planner_task_hint", "")
    if task_hint:
        user += "\n\nAdditional Planner instruction from the runtime rule:\n" + task_hint
    skill_start = context.get("relation_skill_start")
    if skill_start:
        argument = "prompt" if skill_start["tool"] in {"pi0_pick", "pi0_doubled"} else "instruction"
        user += (
            "\n\nRegistered same-goal skill comparison: after the initial observation, "
            "make one registered subgoal call using " + skill_start["tool"] + " with "
            + json.dumps({argument: skill_start["instruction"],
                          "max_chunks": skill_start["max_chunks"]})
            + ". This consumes the existing allowance. Then inspect feedback and "
            "continue the task, recover or stop under the original rules. "
            "The skill call does not establish holding or native task success."
        )
    return {
        "schema": "libero-task-input-v2" if v2 else "libero-task-input-v1",
        "version": options["version"],
        "identity": {
            key: context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        },
        "task": task,
        "system_prompt": system,
        "user_message": user,
        "sources": sources,
        "missing_sources": missing,
        "guide_refs": guides,
        "information_scope": "runtime public prior only; no evaluator/release/sealed input",
        "project_learning": False,
    }
