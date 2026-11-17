# Copyright 2026 RSI-Agent-VLA Authors.
"""Organize legacy Bundle references and verify frozen consumer inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .research_tools import ResearchTools

from copy import deepcopy


def bundle_references(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group existing authority references without duplicating their artifacts."""
    groups = {
        "execution": (
            "theta_ref",
            "processor_ref",
            "phi_ref",
            "rho_ref",
            "backend_ref",
            "code_ref",
        ),
        "research": ("psi_ref", "rho_ref"),
        "facts": ("memory_ref",),
        "evaluation": ("protocol_id",),
    }
    return {name: {key: bundle[key] for key in keys} for name, keys in groups.items()}


def research_binding(tools: ResearchTools) -> dict[str, Any]:
    """Freeze policy and factual starting records at the Research request boundary."""
    from dataclasses import asdict

    bundle = asdict(tools.base)
    return deepcopy(
        {
            "references": bundle_references(bundle),
            "records": {
                key: tools.store.get(kind, bundle[key])
                for key, kind in (
                    ("psi_ref", "psi"),
                    ("rho_ref", "rho"),
                    ("memory_ref", "memory"),
                )
            },
        }
    )


def validate_research_binding(tools: ResearchTools, request: dict[str, Any]) -> None:
    """Reject changed inputs before forwarding; historical requests remain readable."""
    saved = request.get("version_binding")
    if saved is not None and saved != research_binding(tools):
        raise PermissionError(
            "Research policy/factual starting records changed after preparation"
        )
    effective = request.get("effective_psi")
    if effective:
        from .effective_psi import consume_effective_psi

        if consume_effective_psi(tools, effective["snapshot_id"]) != effective:
            raise PermissionError("Research effective policy changed after preparation")


def start_research_job(store, job_id, *, bundle_id, limits, release_manager=None, scope=None):
    """Keep existing Job versions and select qualified psi only for a new Job."""
    from .research import ResearchJob
    from .storage import atomic_json
    if store.path("jobs", job_id).exists():
        return ResearchJob(store, job_id, limits=limits)
    binding = None
    if release_manager is not None:
        bundle, binding = release_manager.research_policy(scope=scope)
        bundle_id = bundle["bundle_id"]
    job = ResearchJob(store, job_id, bundle_id, limits)
    if binding is not None:
        job.state["research_policy_binding"] = binding
        atomic_json(job.path, job.state)
    return job
