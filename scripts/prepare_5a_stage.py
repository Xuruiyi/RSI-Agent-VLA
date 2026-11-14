#!/usr/bin/env python3
"""Build an immutable review input after explicit policy/profile/protocol decisions; never execute."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    protocol = json.loads(Path(spec["protocol"]).read_text())
    profile = json.loads(Path(spec["profile"]).read_text())
    policy = spec["domain_policy"]
    blockers = []
    expected_research = {"model": "gpt-6.1-sol", "reasoning_effort": "high"}
    if (
        any(
            profile.get("research_llm", {}).get(k) != v
            for k, v in expected_research.items()
        )
        or protocol["runtime_config"].get("research_llm") != expected_research
    ):
        blockers.append(
            "new Research preparation requires a matching gpt-6.1-sol/high profile and protocol"
        )
    if protocol.get("status") != "frozen" or not protocol.get("dynamic_authorized"):
        blockers.append(
            "comparison protocol draft must be explicitly approved and frozen"
        )
    if not profile.get("dynamic_authorized"):
        blockers.append("new common runtime profile/source has not been approved")
    if policy.get("handoff_scope_approved") is not True:
        blockers.append("new handoff behavior scope has not been approved")
    if blockers:
        print(
            json.dumps(
                {"mode": "prepare", "dispatch": False, "blockers": blockers},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    from rsi_vla.storage import ResearchStore, atomic_json
    from rsi_vla.__main__ import main as initialize
    from rsi_vla.research import ResearchJob
    from rsi_vla.research_tools import ResearchTools, BoundedResearchAdapter
    from rsi_vla.research_preflight import operator_preflight

    root = Path(__file__).resolve().parents[1]
    limits = spec["job_limits"]
    caps = {
        "model_requests": 21,
        "whole_job_wall_seconds": 1200,
        "tool_calls": 60,
        "environment_steps": 2000,
        "gpu_device_seconds_envelope": 900,
        "artifact_bytes": 1084227584,
    }
    if set(limits) != set(caps) or any(
        type(v) not in (int, float) or not 0 < v <= caps[k] for k, v in limits.items()
    ):
        raise ValueError("original combined Research1/task20 proposal bounds required")
    phase = json.loads(Path(spec["phase_snapshot_ref"]).read_text())
    mapping = {
        "model_requests": "model_requests",
        "whole_job_wall_seconds": "wall_seconds",
        "tool_calls": "tool_calls",
        "environment_steps": "environment_steps",
        "gpu_device_seconds_envelope": "gpu_seconds",
        "artifact_bytes": "storage_bytes",
    }
    if (
        phase.get("unresolved_costs")
        or phase["remaining"]["simulation_episodes"] < 1
        or any(limits[k] > phase["remaining"][v] for k, v in mapping.items())
    ):
        raise ValueError("original phase budget insufficient/unresolved")
    store = ResearchStore(Path(spec["artifacts"]) / "research")
    source = ResearchStore(spec["development_source"])
    # Copy only immutable legal source records. Jobs, queues and authorizations never migrate.
    for kind in (
        "profiles",
        "bundles",
        "psi",
        "rho",
        "memory",
        "development_results",
        "evidence_views",
        "events",
        "development_imports",
    ):
        for old in source.root.joinpath(kind).glob("*.json"):
            store.put(kind, old.stem, json.loads(old.read_text()))
    legal_ids = {
        record["result_id"]
        for record in store.all("development_results")
        if record.get("role") == "development"
    }
    approved_bases = set(policy.get("historical_development_bundle_ids", []))
    related = set()
    for opportunity in source.all("opportunities"):
        if (
            opportunity["base_bundle_id"] in approved_bases
            and set(opportunity["result_ids"]) <= legal_ids
        ):
            store.put("opportunities", opportunity["opportunity_id"], opportunity)
            related.add(opportunity["opportunity_id"])
    experiment_ids = set()
    for experiment in source.all("research_experiments"):
        if experiment["opportunity_id"] in related:
            store.put("research_experiments", experiment["experiment_id"], experiment)
            experiment_ids.add(experiment["experiment_id"])
    for kind, key_name in [
        ("research_findings", "experiment_id"),
        ("research_decisions", "decision_id"),
    ]:
        for value in source.all(kind):
            if (
                value.get("experiment_id") in experiment_ids
                or value.get("opportunity_id") in related
            ):
                store.put(kind, value[key_name], value)
    for candidate in source.all("candidates"):
        if (
            candidate["base_bundle_id"] in approved_bases
            and set(candidate["development_results"]) <= legal_ids
        ):
            store.put("candidates", candidate["candidate_id"], candidate)
    initialize(
        [
            "init",
            "--artifacts",
            spec["artifacts"],
            "--profile",
            spec["profile"],
            "--protocol",
            spec["protocol"],
            "--bundle",
            spec["bundle_id"],
        ]
    )
    job = ResearchJob(store, spec["job_id"], spec["bundle_id"], limits)
    tools = ResearchTools(store, job, protocol, domain_policy=policy)
    preflight = operator_preflight(
        tools, root, protocol_path=spec["protocol"], result_ids=spec["result_ids"]
    )
    allowed = [row["operator"] for row in preflight if row["currently_trialable"]]
    if not allowed:
        print(
            json.dumps(
                {
                    "mode": "prepare",
                    "dispatch": False,
                    "blockers": ["no applicable approved operator"],
                    "preflight": preflight,
                }
            )
        )
        return
    if set(allowed) != {"planner_segment_handoff_hint"}:
        raise ValueError(
            "this staged proposal only opens the approved finite handoff hint"
        )
    tools.domain_policy = {
        **policy,
        "rule_operators": allowed,
        "operator_preflight": preflight,
    }
    adapter = BoundedResearchAdapter()
    transport = adapter.prepare_request(
        tools,
        request_id=spec["request_id"],
        result_ids=spec["result_ids"],
        max_input_tokens=64000,
        max_output_tokens=4000,
    )
    plan = {
        **spec,
        "domain_policy": tools.domain_policy,
        "status": "prepared_only",
        "authorized": False,
    }
    atomic_json(Path(spec["review_output"]) / "development-plan.json", plan)
    atomic_json(Path(spec["review_output"]) / "request.json", transport)
    atomic_json(
        Path(spec["review_output"]) / "response-schema.json",
        adapter.response_schema(spec["request_id"], tools=tools),
    )
    print(
        json.dumps(
            {
                "mode": "prepare",
                "dispatch": False,
                "request_id": spec["request_id"],
                "new_permit_required": True,
            }
        )
    )


if __name__ == "__main__":
    main()
