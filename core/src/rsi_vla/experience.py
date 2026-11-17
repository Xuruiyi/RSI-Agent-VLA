"""Deterministic views over existing legal research records, without causal attribution."""

from collections import Counter
import json
from pathlib import Path


def query_experience(
    tools,
    result_ids=None,
    *,
    task_keys=None,
    bundle_ids=None,
    outcomes=None,
    opportunity_ids=None,
):
    bundles = set(
        bundle_ids
        or [
            tools.base.bundle_id,
            *tools.domain_policy.get("historical_development_bundle_ids", []),
        ]
    )
    ids = (
        result_ids
        if result_ids is not None
        else [
            r["result_id"]
            for r in tools.store.all("development_results")
            if r["bundle_id"] in bundles
        ]
    )
    results = tools.query_development(ids, bundle_ids=list(bundles))
    opportunities = [
        o
        for o in tools.store.all("opportunities")
        if o["base_bundle_id"]
        in {
            tools.base.bundle_id,
            *tools.domain_policy.get("historical_development_bundle_ids", []),
        }
        and set(o["result_ids"]) <= {value["result_id"] for value in results}
        and (not opportunity_ids or o["opportunity_id"] in opportunity_ids)
    ]
    if opportunity_ids:
        ids = {k for o in opportunities for k in o["result_ids"]}
        results = [r for r in results if r["result_id"] in ids]
    rows = []
    for result in results:
        slots = [
            a
            for a in result["assignments"]
            if (not task_keys or a["task_key"] in task_keys)
            and (not outcomes or a["outcome"] in outcomes)
        ]
        if not slots:
            continue
        view_id = result["intervention"].get("evidence_view_id")
        view = tools.store.get("evidence_views", view_id) if view_id else {"l0": []}
        segments = []
        for position, event in enumerate(view["l0"]):
            if event["kind"] in {"vla_segment_start", "vla_segment_end"}:
                raw = event["flags"].get("raw_event", {})
                segments.append(
                    {
                        "kind": event["kind"],
                        "event_ref": event["event_id"],
                        "instruction": raw.get("instruction"),
                        "max_chunks": raw.get("max_chunks"),
                        "chunks_used": raw.get("chunks_used"),
                        "stop_reason": raw.get("stop_reason"),
                        "task_success": raw.get("task_success"),
                        "executed_steps": raw.get("executed_steps"),
                        "missing": event["missing"],
                        "observation_before_ref": next(
                            (
                                e["event_id"]
                                for e in reversed(view["l0"][:position])
                                if e["kind"]
                                in {
                                    "observation",
                                    "observation_update",
                                    "action_feedback",
                                }
                            ),
                            None,
                        ),
                        "observation_after_ref": next(
                            (
                                e["event_id"]
                                for e in view["l0"][position + 1 :]
                                if e["kind"]
                                in {
                                    "observation",
                                    "observation_update",
                                    "action_feedback",
                                }
                            ),
                            None,
                        ),
                    }
                )
        independent = None
        cost_details, audit_details = None, None
        for reference in result["data_refs"]:
            path = Path(reference)
            if (
                path.name in {"cost_breakdown.json", "independent-audit.json"}
                and path.is_file()
            ):
                detail = json.loads(path.read_text())
                identity = result["intervention"].get("identity", {})
                if identity and detail.get("identity") == identity:
                    if path.name == "cost_breakdown.json":
                        cost_details = detail
                    else:
                        audit_details = detail
            if path.name == "independent_result.json" and path.is_file():
                raw = json.loads(path.read_text())
                identity = result["intervention"].get("identity", {})
                if identity and all(raw.get(k) == v for k, v in identity.items()):
                    independent = {
                        key: raw.get(key)
                        for key in (
                            "native_task_result",
                            "validity",
                            "outcome",
                            "runtime_status",
                            "runtime_stop_reason",
                            "execution_health",
                            "missing",
                        )
                    }
        bundle = tools.store.get("bundles", result["bundle_id"])
        rows.append(
            {
                "result_id": result["result_id"],
                "bundle_id": result["bundle_id"],
                "protocol_id": result["protocol_id"],
                "independent_fact": independent,
                "cost_details": cost_details,
                "audit_details": audit_details,
                "data_admission": "autonomous/development evidence; not demonstration admission; historical full capture may be absent",
                "psi_ref": bundle["psi_ref"],
                "rho_ref": bundle["rho_ref"],
                "memory_ref": bundle["memory_ref"],
                "evidence_selection": view.get("selection", {}),
                "record_layer": "cross_task_experience_derived_from_original_facts",
                "raw_events_mutated": False,
                "assignments": slots,
                "outcome": result["outcome"],
                "invalid_count": result["invalid_count"],
                "unknown_count": result["unknown_count"],
                "costs": result["costs"],
                "segments": segments,
                "execution_statistics": {
                    "event_counts": dict(Counter(e["kind"] for e in view["l0"])),
                    "model_requests": result["costs"].get("model_requests"),
                    "tool_calls": result["costs"].get("tool_calls"),
                },
                "skill_observations": [
                    {
                        "event_ref": e["event_id"],
                        "bundle_id": result["bundle_id"],
                        "command": (e.get("feedback") or {}).get("command"),
                        "result": (e.get("feedback") or {}).get("tool_result"),
                        "action_source": e["flags"].get("action_source"),
                        "generality": "this assignment only; no skill promotion",
                    }
                    for e in view["l0"]
                    if e["kind"] == "tool_transition"
                ],
                "faults": [
                    {
                        "event_ref": e["event_id"],
                        "error": e["flags"].get("error"),
                        "missing": e["missing"],
                    }
                    for e in view["l0"]
                    if e["flags"].get("error") or e["missing"]
                ],
                "evidence_refs": result["data_refs"],
                "scope": "observed assignments at this version only",
                "origin": "deterministic_fact_extraction",
                "causal_attribution": "not_established",
                "collection_manifest_refs": [
                    p
                    for p in result["data_refs"]
                    if p.endswith("dataset_manifest.json")
                ],
            }
        )
    selected = {r["result_id"] for r in rows}
    related = [o for o in opportunities if selected.intersection(o["result_ids"])]
    experiments = [
        e
        for e in tools.store.all("research_experiments")
        if e["opportunity_id"] in {o["opportunity_id"] for o in related}
    ]
    experiments_ids = {e["experiment_id"] for e in experiments}
    related_result_ids = {
        f["result_id"]
        for f in tools.store.all("research_findings")
        if f.get("experiment_id") in experiments_ids
    }
    return {
        "facts": rows,
        "opportunities": related,
        "experiments": experiments,
        "findings": [
            f
            for f in tools.store.all("research_findings")
            if f.get("experiment_id") in experiments_ids
        ],
        "decisions": [
            d
            for d in tools.store.all("research_decisions")
            if d.get("opportunity_id") in {o["opportunity_id"] for o in related}
        ],
        "candidates": [
            c
            for c in tools.store.all("candidates")
            if (selected | related_result_ids).intersection(c["development_results"])
        ],
        "interpretation_origin": "opportunities and predictions are proposals; findings do not resolve hypotheses",
        "repeat_policy": "same evidence/operator with a completed finding is not a new experiment",
    }


def handoff_evidence(tools, result_ids):
    observations = []
    for row in query_experience(tools, result_ids)["facts"]:
        segments = row["segments"]
        for index, event in enumerate(segments[:-1]):
            next_event = segments[index + 1]
            if (
                event["kind"] == "vla_segment_end"
                and event["stop_reason"] == "chunk_allowance"
                and event["task_success"] is False
                and next_event["kind"] == "vla_segment_start"
            ):
                observations.append(
                    {
                        "result_id": row["result_id"],
                        "before_ref": event["event_ref"],
                        "after_ref": next_event["event_ref"],
                        "previous_instruction": event["instruction"],
                        "next_instruction": next_event["instruction"],
                        "final_outcome": row["outcome"],
                        "cause": "unresolved; sequence does not establish handoff benefit",
                    }
                )
    return observations
