"""Read existing candidate ancestry and compose bounded, complete development bundles."""

from dataclasses import asdict, replace
from pathlib import Path

from .contracts import Candidate, RunBundle

COMPONENTS = (
    "theta_ref",
    "processor_ref",
    "phi_ref",
    "rho_ref",
    "psi_ref",
    "memory_ref",
    "backend_ref",
    "code_ref",
)
AFFECTED = {
    "theta_ref": ["model_load", "skill_start", "handoff", "task_result"],
    "processor_ref": ["model_load", "action_protocol", "skill_start", "handoff"],
    "phi_ref": ["execution_program", "skill_start", "handoff"],
    "rho_ref": ["media_outlet", "skill_route", "research_selection"],
    "psi_ref": ["research_program"],
    "memory_ref": ["retrieval"],
    "backend_ref": ["model_load", "action_protocol"],
    "code_ref": ["consumer_contract"],
    "action_protocol": ["action_protocol", "skill_start", "handoff"],
}


def applicability(
    dependencies, bundle, *, task_scope=None, action_protocol=None, invalidated=False
):
    """Keep invalidation, out-of-domain use and unconfirmed versions distinct."""
    if invalidated:
        return {
            "status": "invalid",
            "reasons": ["explicitly_invalidated"],
            "checks": [],
        }
    required_scope = dependencies.get("task_scope")
    if (
        required_scope is not None
        and task_scope is not None
        and task_scope not in required_scope
    ):
        return {
            "status": "inapplicable",
            "reasons": ["outside_verified_scope"],
            "checks": [],
        }
    expected = dependencies.get("versions", {})
    unknown = []
    for name, value in expected.items():
        if name not in {*COMPONENTS, "action_protocol"}:
            raise ValueError("unsupported candidate dependency: " + name)
        current = action_protocol if name == "action_protocol" else bundle[name]
        if current != value:
            unknown.append(name)
    if required_scope is not None and task_scope is None:
        unknown.append("task_scope")
    if not expected:
        unknown.append("missing_version_dependencies")
    checks = sorted(
        {check for name in unknown for check in AFFECTED.get(name, ["scope_review"])}
    )
    return {
        "status": "unconfirmed" if unknown else "applicable",
        "reasons": unknown,
        "checks": checks,
    }


def register_context(
    store,
    candidate_id,
    *,
    experiment_ids=(),
    hypothesis_refs=(),
    source_candidate_ids=(),
    selection_reason,
    dependencies=None,
    origin=None,
    contribution_ids=(),
    job_ids=(),
    invalidated=False,
):
    """Attach immutable provenance to a real Candidate, including failed branches."""
    candidate = store.get("candidates", candidate_id)
    if not selection_reason.strip():
        raise ValueError("candidate selection needs an explicit reason")
    if len(set(source_candidate_ids)) != len(source_candidate_ids):
        raise ValueError("duplicate candidate source")
    for source in source_candidate_ids:
        store.get("candidates", source)
    experiments = [store.get("research_experiments", key) for key in experiment_ids]
    for job_id in job_ids:
        job = store.get("jobs", job_id)
        if job["bundle_id"] != candidate["base_bundle_id"]:
            raise ValueError(
                "candidate cost owner belongs to a different starting bundle"
            )
    if any(row["bundle_id"] != candidate["bundle_id"] for row in experiments):
        raise ValueError("candidate provenance references a different trial bundle")
    value = {
        "candidate_id": candidate_id,
        "experiment_ids": list(experiment_ids),
        "hypothesis_refs": list(hypothesis_refs),
        "source_candidate_ids": list(source_candidate_ids),
        "selection_reason": selection_reason,
        "dependencies": dependencies or {},
        "origin": origin or {"kind": "legacy_candidate", "ref": candidate_id},
        "contribution_ids": sorted(set(contribution_ids)),
        "job_ids": sorted(set(job_ids) | {row["job_id"] for row in experiments}),
        "invalidated": invalidated,
    }
    store.put("candidate_context", candidate_id, value)
    return value


def candidate_trace(
    store,
    candidate_id,
    *,
    evaluations=None,
    release_records=None,
    target_bundle=None,
    task_scope=None,
    action_protocol=None,
):
    """Resolve facts from their owners; this view grants no evaluation qualification."""
    candidate = store.get("candidates", candidate_id)
    bundle = store.get("bundles", candidate["bundle_id"])
    base = store.get("bundles", candidate["base_bundle_id"])
    if bundle["parent_bundle_id"] != base["bundle_id"]:
        raise ValueError("candidate archive parent differs from the complete bundle")
    context_ref = store.path("candidate_context", candidate_id)
    context = (
        store.get("candidate_context", candidate_id) if context_ref.exists() else {}
    )
    experiments = [
        row
        for row in store.all("research_experiments")
        if row.get("bundle_id") == candidate["bundle_id"]
    ]
    composition = candidate.get("compatibility", {}).get("composition_ref")
    composition_jobs = []
    if composition:
        value = store.get("candidate_compositions", candidate_id)
        if (
            str(store.path("candidate_compositions", candidate_id).resolve())
            != composition
        ):
            raise ValueError("composition reference differs from its owned record")
        composition_jobs = value["job_ids"]
    jobs = sorted(
        set(context.get("job_ids", []))
        | set(composition_jobs)
        | {row["job_id"] for row in experiments}
    )
    histories = [store.get("jobs", key) for key in jobs]
    results = [
        store.get("development_results", key)
        for key in candidate["development_results"]
    ]
    findings = [
        row for row in store.all("research_findings") if row.get("job_id") in jobs
    ]
    # Per-job ledgers include failed attempts. Their complete charge vectors stay
    # separate from Candidate snapshots, which can include an enclosing phase.
    costs = {
        "candidate_snapshot": candidate["costs"],
        "job_ledgers": histories,
        "all_findings": findings,
        "unknown": [] if histories else ["job_cost_lineage_unavailable"],
    }
    if composition and candidate["compatibility"].get("costs_complete") is not True:
        costs["unknown"].append("composition_source_cost_lineage_incomplete")
    evaluated = (
        []
        if evaluations is None
        else [
            row
            for row in evaluations.all("results")
            if row.get("candidate_id") == candidate_id
        ]
    )
    decisions = (
        []
        if release_records is None
        else [
            row
            for row in release_records.all("decisions")
            if row.get("candidate_id") == candidate_id
        ]
    )
    sources = context.get("source_candidate_ids", [])
    assessment = applicability(
        context.get("dependencies", {}),
        target_bundle or bundle,
        task_scope=task_scope,
        action_protocol=action_protocol,
        invalidated=context.get("invalidated", False),
    )
    source_assessments = []
    for source_id in sources:
        source_context = (
            store.get("candidate_context", source_id)
            if store.path("candidate_context", source_id).exists()
            else {}
        )
        source_assessments.append(
            {
                "candidate_id": source_id,
                **applicability(
                    source_context.get("dependencies", {}),
                    target_bundle or bundle,
                    task_scope=task_scope,
                    action_protocol=action_protocol,
                    invalidated=source_context.get("invalidated", False),
                ),
            }
        )
    if source_assessments:
        statuses = {
            assessment["status"],
            *(row["status"] for row in source_assessments),
        }
        assessment["status"] = next(
            name
            for name in ("invalid", "inapplicable", "unconfirmed", "applicable")
            if name in statuses
        )
        assessment["checks"] = sorted(
            set(assessment["checks"])
            | {check for row in source_assessments for check in row["checks"]}
        )
        assessment["source_dependencies"] = source_assessments
    return {
        "schema": "rsi-candidate-archive-v1",
        "candidate": candidate,
        "bundle": bundle,
        "base_bundle": base,
        "context": context,
        "source_candidates": [store.get("candidates", key) for key in sources],
        "experiments": experiments,
        "development_results": results,
        "costs": costs,
        "evaluation_results": evaluated,
        "decisions": decisions,
        "evaluation_status": ("evaluated" if evaluated else "unconfirmed"),
        "applicability": assessment,
        "artifacts": [
            {
                "ref": ref,
                "available": (
                    Path(ref.split("#", 1)[0]).is_file()
                    or Path(ref.split("#", 1)[0]).is_dir()
                ),
            }
            for ref in candidate["artifact_refs"]
        ],
        "qualification_inherited": False,
    }


def query_candidates(
    store,
    *,
    protocol_id=None,
    evaluations=None,
    release_records=None,
    target_bundle=None,
    task_scope=None,
    action_protocol=None,
):
    """A bounded local scan of existing records; negative branches remain visible."""
    return [
        candidate_trace(
            store,
            row["candidate_id"],
            evaluations=evaluations,
            release_records=release_records,
            target_bundle=target_bundle,
            task_scope=task_scope,
            action_protocol=action_protocol,
        )
        for row in store.all("candidates")
        if protocol_id is None or row["protocol_id"] == protocol_id
    ]


def select_parent(tools, selection_id, candidate_id, *, reason):
    """Select a legal historical development parent without changing a running job."""
    limit = tools.domain_policy.get("max_parent_selections", 1)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("archive parent selection requires a finite registered bound")
    previous = [
        row
        for row in tools.store.all("research_parent_selections")
        if row["job_id"] == tools.job.state["job_id"]
    ]
    if (
        not tools.store.path("research_parent_selections", selection_id).exists()
        and len(previous) >= limit
    ):
        raise ValueError("archive parent selection bound exhausted")
    if tools.job.state["status"] != "running":
        raise ValueError("archive selection requires an active research job budget")
    trace = candidate_trace(tools.store, candidate_id)
    candidate = trace["candidate"]
    allowed = set(tools.domain_policy.get("historical_parent_candidate_ids", []))
    allowed.update(
        row["candidate_id"]
        for row in tools.store.all("candidates")
        if row["base_bundle_id"] == tools.base.bundle_id
    )
    if (
        candidate_id not in allowed
        or candidate["protocol_id"] != tools.protocol["protocol_id"]
        or candidate["approval_scope"] != tools.protocol["scope"]
    ):
        raise PermissionError(
            "historical candidate is outside the registered development domain"
        )
    if trace["applicability"]["status"] in {"invalid", "inapplicable"}:
        raise ValueError("historical candidate is not applicable")
    if not reason.strip() or any(not row["available"] for row in trace["artifacts"]):
        raise ValueError(
            "historical parent needs a reason and retained executable artifacts"
        )
    return tools._save(
        "research_parent_selections",
        selection_id,
        {
            "selection_id": selection_id,
            "candidate_id": candidate_id,
            "selected_bundle_id": candidate["bundle_id"],
            "reason": reason,
            "required_checks": trace["applicability"],
            "qualification_inherited": False,
        },
    )


def inherit_parent(store, selection_id, job, protocol, *, domain_policy):
    """Bind a separately admitted new Job to the selected immutable historical bundle."""
    from .research_tools import ResearchTools

    selection = store.get("research_parent_selections", selection_id)
    if (
        job.state["bundle_id"] != selection["selected_bundle_id"]
        or selection["protocol_id"] != protocol["protocol_id"]
    ):
        raise ValueError("new research job does not inherit the selected parent")
    tools = ResearchTools(store, job, protocol, domain_policy=domain_policy)
    tools._save(
        "research_parent_inheritances",
        job.state["job_id"],
        {
            "selection_id": selection_id,
            "candidate_id": selection["candidate_id"],
            "selected_bundle_id": selection["selected_bundle_id"],
        },
    )
    return tools


def compose_candidates(
    store,
    candidate_id,
    *,
    base_bundle_id,
    bundle_id,
    source_candidate_ids,
    selection_reason,
    dependencies=None,
):
    """Create a complete Bundle and fresh unevaluated Candidate, never splice eligibility."""
    if len(source_candidate_ids) < 2 or len(set(source_candidate_ids)) != len(
        source_candidate_ids
    ):
        raise ValueError("composition needs distinct candidate sources")
    sources = [candidate_trace(store, key) for key in source_candidate_ids]
    base = RunBundle(**store.get("bundles", base_bundle_id))
    first = sources[0]["candidate"]
    if any(
        trace["candidate"]["benchmark_id"] != first["benchmark_id"]
        or trace["candidate"]["approval_scope"] != first["approval_scope"]
        for trace in sources
    ):
        raise ValueError("composition source benchmark or scope differs")
    changes, contributions = {}, set()
    job_ids, snapshots = set(), []
    for trace in sources:
        source = trace["candidate"]
        if (
            source["base_bundle_id"] != base_bundle_id
            or source["protocol_id"] != base.protocol_id
        ):
            raise ValueError("composition source has a stale or different base")
        for key, value in source["changes"].items():
            if key == "diff_ref":
                continue
            if key in changes and changes[key] != value:
                raise ValueError("conflicting candidate component: " + key)
            changes[key] = value
            contributions.add((key, value))
        job_ids.update(row["job_id"] for row in trace["costs"]["job_ledgers"])
        snapshots.append(
            {"candidate_id": source["candidate_id"], "costs": source["costs"]}
        )
    if not changes:
        raise ValueError("composition has no actual component change")
    diff_ref = str(store.path("candidate_compositions", candidate_id).resolve())
    target = replace(
        base,
        bundle_id=bundle_id,
        parent_bundle_id=base_bundle_id,
        diff_ref=diff_ref,
        **changes,
    )
    diff = {
        "schema": "rsi-candidate-composition-v1",
        "base_bundle_id": base_bundle_id,
        "bundle_id": bundle_id,
        "source_candidate_ids": list(source_candidate_ids),
        "changes": changes,
        "contributions": [{"field": k, "value": v} for k, v in sorted(contributions)],
        "cost_snapshots": snapshots,
        "job_ids": sorted(job_ids),
        "qualification_inherited": False,
        "evaluation_required": True,
    }
    store.put("candidate_compositions", candidate_id, diff)
    store.save_bundle(target)
    costs = {}
    for job_id in sorted(job_ids):
        for key, value in store.get("jobs", job_id)["used"].items():
            costs[key] = costs.get(key, 0) + value
    candidate = Candidate(
        candidate_id,
        base_bundle_id,
        bundle_id,
        base.protocol_id,
        sources[0]["candidate"]["benchmark_id"],
        {**changes, "diff_ref": diff_ref},
        sorted(
            {ref for trace in sources for ref in trace["candidate"]["artifact_refs"]}
            | {diff_ref}
        ),
        [],
        costs,
        {
            "composition_ref": diff_ref,
            "evaluation_status": "unconfirmed",
            "source_cost_snapshots_retained": True,
            "costs_complete": all(
                trace["costs"]["job_ledgers"] and not trace["costs"]["unknown"]
                for trace in sources
            ),
            "cost_basis": "unique_known_job_ledgers; other source snapshots retained without assuming disjoint costs",
        },
        approval_scope=sources[0]["candidate"]["approval_scope"],
    )
    store.save_candidate(candidate)
    context = register_context(
        store,
        candidate_id,
        source_candidate_ids=source_candidate_ids,
        selection_reason=selection_reason,
        dependencies=dependencies,
        contribution_ids=[k + ":" + v for k, v in sorted(contributions)],
        origin={"kind": "composition", "ref": diff_ref},
    )
    return {
        "candidate": asdict(candidate),
        "bundle": asdict(target),
        "context": context,
        "diff": diff,
    }
