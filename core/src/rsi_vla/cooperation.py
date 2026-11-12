"""Executable fixed offline baselines for the initial rhoB/rhoC snapshots."""


def fixed_route(rho_snapshot, requested_worker, allowed_workers):
    """Retain the existing requested Worker only when explicitly authorized."""
    policy = rho_snapshot["rhoB"]
    if policy.get("type") != "fixed" or policy.get("route") != "existing_authorized_worker":
        raise ValueError("unsupported rhoB policy")
    return requested_worker if requested_worker in allowed_workers else "defer"


def fixed_research_step(rho_snapshot, job, *, evidence_available, legal_repairs=(),
                        inspected=False, experiment_permitted=False):
    """Fixed inspect -> bounded experiment -> defer, under the original budget."""
    policy = rho_snapshot["rhoC"]
    if policy.get("type") != "fixed" or policy.get("order") != ["inspect", "bounded_experiment", "defer"]:
        raise ValueError("unsupported rhoC policy")
    if job.state["status"] != "running" or not evidence_available:
        return "defer"
    if not inspected:
        return "inspect"
    return "bounded_experiment" if experiment_permitted and legal_repairs else "defer"
