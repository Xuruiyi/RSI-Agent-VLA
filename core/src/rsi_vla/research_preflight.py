"""Bounded operator feasibility before any Research model admission."""

import tempfile
from pathlib import Path

from .rule_updates import write_rule_trial
from .runtime import prepare_runtime
from .storage import ResearchStore


def operator_preflight(tools, root, *, protocol_path, result_ids):
    """Materialize isolated finite phi/rho trials and pass them to the actual runtime planner."""
    tools.query_development(result_ids)
    from .task_policy import exploration_policy

    exploration = exploration_policy(tools)
    from .experience import handoff_evidence

    triggers = {
        "planner_segment_handoff_hint": handoff_evidence(tools, result_ids),
        "chunk_steps": [],
        "planner_task_hint": tools.query_development(result_ids),
        "finish_requires_native_success": tools.inspect_native_finish_opportunity(
            result_ids
        ),
        "planner_finish_reserve_calls": tools.inspect_finish_budget_opportunity(
            result_ids
        ),
    }
    proposals = {
        "planner_segment_handoff_hint": True,
        "chunk_steps": 2,
        "finish_requires_native_success": True,
        "planner_finish_reserve_calls": 1,
        "planner_task_hint": "Before acting, identify the task's named target object and verify the intended grasp from the current observation.",
    }
    development = tools.protocol["task_sets"]["development"]
    task_keys, seeds = development["task_keys"], development["seeds"]
    if not task_keys or not seeds:
        raise ValueError("current protocol has no development assignment for preflight")
    task_key = (
        "position_swap_salad" if "position_swap_salad" in task_keys else task_keys[0]
    )
    seed = seeds[0]
    if exploration:
        task_key = exploration["assignments"][0]["task_key"]
        seed = exploration["assignments"][0]["seed"]
    rows = []
    for operator, value in proposals.items():
        with tempfile.TemporaryDirectory(prefix="rsi-operator-preflight-") as tmp:
            store = ResearchStore(Path(tmp) / "research")
            base = tools.base
            for kind, ref in (
                ("rho", base.rho_ref),
                ("psi", base.psi_ref),
                ("memory", base.memory_ref),
            ):
                store.put(kind, ref, tools.store.get(kind, ref))
            store.save_bundle(base)
            row = {
                "operator": operator,
                "development_assignment": {"task_key": task_key, "seed": seed},
                "trigger_evidence": triggers[operator],
                "artifact_generated": False,
                "runtime_prepared": False,
                "required_protocol": "shared native output; matched prefix/profile and comparison conditions"
                if operator == "chunk_steps"
                else "unchanged task budgets/scoring with independent comparison",
            }
            try:
                target, diff = write_rule_trial(
                    store,
                    base,
                    bundle_id="preflight-trial",
                    artifact_id="preflight-rule",
                    **{operator: value},
                )
                row.update(
                    artifact_generated=True,
                    artifact=store.get("runtime_rules", "preflight-rule"),
                    actual_changes=diff["changes"],
                    consumer=diff["consumer"],
                )
                plan = prepare_runtime(
                    root,
                    tmp,
                    protocol_path=protocol_path,
                    run_id="preflight-only",
                    bundle_id=target.bundle_id,
                    task_key=task_key,
                    seed=seed,
                    split="development",
                    output_dir=Path(tmp) / "never-started",
                )
                row.update(runtime_prepared=True, runtime_blockers=plan["blockers"])
            except ValueError as exc:
                row["runtime_error"] = str(exc)
            # Chunk generation alone is not evidence for changing controller timing.
            approved = operator in tools.domain_policy.get("rule_operators", []) and (
                operator != "planner_segment_handoff_hint"
                or tools.domain_policy.get("handoff_scope_approved") is True
            )
            if operator == "planner_task_hint":
                from .task_policy import task_policy
                approved = approved and bool(task_policy(tools.protocol))
            row["currently_trialable"] = (
                (bool(triggers[operator]) or (exploration is not None and operator in exploration["operators"]))
                and row["runtime_prepared"]
                and approved
                and all(
                    b == "explicit single-job GPU/LLM/simulation authorization missing"
                    for b in row.get("runtime_blockers", [])
                )
            )
            row["consumer_available"] = (
                row["artifact_generated"] and row["runtime_prepared"]
            )
            row["evidence_supports_trial"] = bool(triggers[operator])
            row["research_scope_approved"] = approved
            row["exploration_scope"] = exploration if exploration and operator in exploration["operators"] else None
            row["evidence_basis"] = "observed_development" if triggers[operator] else "registered_hypothesis" if row["exploration_scope"] else "missing"
            row["execution_authorized"] = False
            rows.append(row)
    if "relation_feedback_update" in tools.domain_policy.get("rule_operators", []):
        from copy import copy
        from .relation_learning import preview_relation_feedback, write_relation_trial
        support = preview_relation_feedback(tools, result_ids)
        row = {"operator": "relation_feedback_update", "development_assignment": {"task_key": task_key, "seed": seed},
               "trigger_evidence": support, "artifact_generated": False, "runtime_prepared": False,
               "research_scope_approved": True, "evidence_supports_trial": bool(support["supported_relations"]),
               "evidence_basis": "observed_development", "execution_authorized": False}
        if support["supported_relations"]:
            with tempfile.TemporaryDirectory(prefix="rsi-relation-preflight-") as tmp:
                original = tools.store
                class Inputs(ResearchStore):
                    def get(self, kind, key):
                        return super().get(kind, key) if self.path(kind, key).exists() else original.get(kind, key)
                    def all(self, kind):
                        return original.all(kind) + super().all(kind)
                store = Inputs(Path(tmp) / "research")
                for kind, ref in (("rho", tools.base.rho_ref), ("psi", tools.base.psi_ref), ("memory", tools.base.memory_ref)):
                    store.put(kind, ref, original.get(kind, ref))
                store.save_bundle(tools.base)
                scoped = copy(tools)
                scoped.store = store
                try:
                    target, changes = write_relation_trial(scoped, experiment_id="preflight-relation",
                        bundle_id="preflight-relation", result_ids=result_ids, relations=support["supported_relations"])
                    row.update(artifact_generated=True, actual_changes=changes,
                               consumer="rhoA Research input; rhoB compatible Worker routing; rhoC research admission")
                    plan = prepare_runtime(root, tmp, protocol_path=protocol_path, run_id="preflight-only",
                        bundle_id=target.bundle_id, task_key=task_key, seed=seed,
                        split="development", output_dir=Path(tmp) / "never-started")
                    row.update(runtime_prepared=True, runtime_blockers=plan["blockers"])
                except ValueError as exc:
                    row["runtime_error"] = str(exc)
        row["consumer_available"] = row["artifact_generated"] and row["runtime_prepared"]
        row["currently_trialable"] = row["consumer_available"] and all(
            reason == "explicit single-job GPU/LLM/simulation authorization missing"
            for reason in row.get("runtime_blockers", []))
        rows.append(row)
    return rows
