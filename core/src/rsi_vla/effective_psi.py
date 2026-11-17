"""Immutable effective research policies, with factual memory recorded separately."""

from math import isfinite
from pathlib import Path

from .task_policy import task_policy


def freeze_effective_psi(tools, snapshot_id, *, program_refs, prompt_rules, retrieval_rules,
                         dependencies, development_costs, development_families, reserved_families,
                         development_mechanisms=(), reserved_mechanisms=(), result_ids=(),
                         origin_kind="human", origin_ref, parent_snapshot_id=None):
    """Save actual policy sources and held-out problem partitions; never activate a version."""
    if not task_policy(tools.protocol):
        raise ValueError("effective psi snapshots require a registered new task")
    if (origin_kind not in {"human", "program", "model"} or not origin_ref
            or not program_refs or not prompt_rules or not retrieval_rules or not dependencies):
        raise ValueError("effective psi needs program/prompt/retrieval/dependencies and change provenance")
    if (set(development_families) & set(reserved_families)
            or set(development_mechanisms) & set(reserved_mechanisms)
            or not reserved_families or not reserved_mechanisms):
        raise ValueError("reserve unexposed task families and mechanisms separately from development")
    if any(v is not None and (type(v) not in (int, float) or not isfinite(v) or v < 0)
           for v in development_costs.values()):
        raise ValueError("development costs are nonnegative observations or explicit unknown")
    programs = [{"ref": str(Path(ref).resolve()), "source": Path(ref).read_text()}
                for ref in program_refs]
    rho = tools.store.get("rho", tools.base.rho_ref)
    components = {"programs": programs, "declared_psi": tools.store.get("psi", tools.base.psi_ref),
                  "prompt_rules": prompt_rules, "retrieval_rules": retrieval_rules,
                  "rhoC": {"rho_ref": tools.base.rho_ref, "policy": rho["rhoC"]},
                  "dependencies": dependencies}
    if parent_snapshot_id is not None:
        parent = tools.store.get("effective_psi", parent_snapshot_id)
        if parent["components"] == components:
            raise ValueError("no policy change; factual additions do not create a later psi")
    facts = {"schema": "rsi-fact-memory-snapshot-v1", "snapshot_id": snapshot_id,
             "memory_ref": tools.base.memory_ref, "memory": tools.store.get("memory", tools.base.memory_ref),
             "development_facts": tools.query_development(list(result_ids)),
             "legal_scope": "development_only", "role": "facts_not_policy"}
    tools.store.put("fact_memory_snapshots", snapshot_id, facts)
    snapshot = {"schema": "rsi-effective-psi-v1", "snapshot_id": snapshot_id,
                "bundle_id": tools.base.bundle_id, "psi_ref": tools.base.psi_ref,
                "protocol_id": tools.protocol["protocol_id"], "components": components,
                "fact_memory_ref": str(tools.store.path("fact_memory_snapshots", snapshot_id).resolve()),
                "development_costs": development_costs,
                "problem_partition": {"development_families": list(development_families),
                    "reserved_families": list(reserved_families),
                    "development_mechanisms": list(development_mechanisms),
                    "reserved_mechanisms": list(reserved_mechanisms)},
                "origin": {"kind": origin_kind, "source_ref": origin_ref},
                "parent_snapshot_id": parent_snapshot_id, "policy_improvement": None,
                "execution_authorized": False}
    tools.store.put("effective_psi", snapshot_id, snapshot)
    return snapshot


def consume_effective_psi(tools, snapshot_id, *, facts_ref=None):
    """Bind actual Research input to its frozen effective policy and separate legal facts."""
    snapshot = tools.store.get("effective_psi", snapshot_id)
    if ((facts_ref is None and snapshot["bundle_id"] != tools.base.bundle_id) or snapshot["psi_ref"] != tools.base.psi_ref
            or snapshot["protocol_id"] != tools.protocol["protocol_id"]
            or snapshot["components"]["rhoC"]["rho_ref"] != tools.base.rho_ref
            or snapshot["components"]["rhoC"]["policy"] != tools.store.get("rho", tools.base.rho_ref)["rhoC"]):
        raise ValueError("effective psi differs from the bound research policy")
    if snapshot["components"]["declared_psi"] != tools.store.get("psi", tools.base.psi_ref):
        raise ValueError("effective psi declared policy changed")
    if facts_ref is not None:
        import json
        facts = json.loads(Path(facts_ref).read_text())
        if (facts.get("legal_scope") != "development_only" or facts.get("role") != "facts_not_policy"
                or facts.get("memory_ref") != tools.base.memory_ref
                or facts.get("memory") != tools.store.get("memory", tools.base.memory_ref)):
            raise ValueError("effective psi factual rebinding is not a matched legal start")
    for program in snapshot["components"]["programs"]:
        if Path(program["ref"]).read_text() != program["source"]:
            raise ValueError("effective psi program changed; freeze a new policy snapshot")
    return {"snapshot_id": snapshot_id,
            "snapshot_ref": str(tools.store.path("effective_psi", snapshot_id).resolve()),
            "psi_ref": snapshot["psi_ref"], "components": snapshot["components"],
            "fact_memory_ref": str(Path(facts_ref).resolve()) if facts_ref else snapshot["fact_memory_ref"], "problem_partition": snapshot["problem_partition"],
            "origin": snapshot["origin"], "policy_improvement": None}


def validate_policy_snapshot(store, snapshot):
    """Check the frozen executable policy sources without consuming exam scores."""
    bundle = store.get("bundles", snapshot["bundle_id"])
    if (bundle["psi_ref"] != snapshot["psi_ref"]
            or store.get("psi", snapshot["psi_ref"]) != snapshot["components"]["declared_psi"]
            or store.get("rho", snapshot["components"]["rhoC"]["rho_ref"])["rhoC"] != snapshot["components"]["rhoC"]["policy"]):
        raise ValueError("frozen research policy records changed")
    for program in snapshot["components"]["programs"]:
        if Path(program["ref"]).read_text() != program["source"]:
            raise ValueError("frozen research policy program changed")
