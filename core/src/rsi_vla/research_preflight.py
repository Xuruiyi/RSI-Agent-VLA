"""Bounded operator feasibility before any Research model admission."""

import tempfile
from pathlib import Path
from .rule_updates import write_rule_trial
from .runtime import prepare_runtime
from .storage import ResearchStore


def operator_preflight(tools, root, *, protocol_path, result_ids):
    """Materialize isolated actual rules and pass them to the actual runtime planner."""
    tools.query_development(result_ids)
    triggers = {
        "chunk_steps": [],
        "finish_requires_native_success": tools.inspect_native_finish_opportunity(
            result_ids
        ),
        "planner_finish_reserve_calls": tools.inspect_finish_budget_opportunity(
            result_ids
        ),
    }
    proposals = {
        "chunk_steps": 2,
        "finish_requires_native_success": True,
        "planner_finish_reserve_calls": 1,
    }
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
                    task_key="position_swap_salad",
                    seed=1,
                    split="development",
                    output_dir=Path(tmp) / "never-started",
                )
                row.update(runtime_prepared=True, runtime_blockers=plan["blockers"])
            except ValueError as exc:
                row["runtime_error"] = str(exc)
            # Chunk generation alone is not evidence for changing controller timing.
            row["currently_trialable"] = (
                bool(triggers[operator])
                and row["runtime_prepared"]
                and operator in tools.domain_policy.get("rule_operators", [])
                and all(
                    b == "explicit single-job GPU/LLM/simulation authorization missing"
                    for b in row.get("runtime_blockers", [])
                )
            )
            row["execution_authorized"] = False
            rows.append(row)
    return rows
