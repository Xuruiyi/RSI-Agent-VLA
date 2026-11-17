"""Independent offline tasks for traced theta artifact publication interfaces."""

import json
import time
from pathlib import Path

from .contracts import Candidate, ExperimentResult, RunBundle
from .runtime import resolve_reference
from .theta_b import training_theta_version


def close_theta_candidate_costs(candidate: Candidate, *, scope_id: str) -> dict:
    """Close serialization costs from original settled Host receipts, preserving history."""
    from .theta_b import _training_candidate_costs

    source_ref = Path(candidate.compatibility["source_receipt_ref"]).resolve(
        strict=True
    )
    source = json.loads(source_ref.read_text())
    if (
        source.get("schema") != "rsi-training-candidate-sources-v1"
        or source.get("candidate_id") != candidate.candidate_id
    ):
        raise PermissionError(
            "theta artifact cost provenance belongs to another candidate"
        )
    references = [row["receipt_ref"] for row in source["host_cost_receipts"]]
    historical, _ = _training_candidate_costs(
        references, scope_id=scope_id, required_operations=set()
    )
    if historical != candidate.costs:
        raise PermissionError("theta artifact serialized cost history changed")
    pending = source.get("pending_cost_operations", [])
    if pending != candidate.compatibility.get(
        "pending_serialization_cost_operations", []
    ):
        raise PermissionError("theta artifact pending serialization identity mismatch")
    accounting_ref = source.get("accounting_ref")
    if pending and not accounting_ref:
        raise PermissionError("theta artifact serialization cost still unresolved")
    if pending:
        accounting = json.loads(Path(accounting_ref).read_text())
        if accounting.get("scope_id") != scope_id:
            raise PermissionError("theta artifact accounting belongs to another scope")
        for operation in pending:
            reservation = accounting["reservations"].get(operation, {})
            if (
                reservation.get("status") != "settled"
                or reservation.get("cleanup_confirmed") is not True
                or reservation.get("unknown_dimensions")
            ):
                raise PermissionError(
                    "theta artifact serialization cost or cleanup still unresolved"
                )
            receipt_ref = reservation["receipt_ref"]
            receipt = json.loads(Path(receipt_ref).read_text())
            if receipt.get("operation_id") != operation or receipt.get(
                "actual"
            ) != reservation.get("actual"):
                raise PermissionError(
                    "theta artifact serialization receipt differs from settled accounting"
                )
            references.append(receipt_ref)
    complete, receipts = _training_candidate_costs(
        references, scope_id=scope_id, required_operations=set(pending)
    )
    return {
        "schema": "rsi-theta-candidate-cost-closure-v1",
        "source_candidate_id": candidate.candidate_id,
        "source_ref": str(source_ref),
        "candidate_costs_as_serialized": dict(candidate.costs),
        "closed_serialization_operations": list(pending),
        "pending_cost_operations": [],
        "all_attempt_costs": complete,
        "host_cost_receipts": receipts,
        "accounting_ref": accounting_ref,
    }


def evaluate_theta_artifact_matrix(
    protocol: dict,
    bundle: dict,
    *,
    result_id: str,
    candidate_id: str | None,
    problem_ref: str | Path,
) -> ExperimentResult:
    """Check saved-artifact handoff tasks without executing model weights.

    The scope establishes publication and binding of an independently traced
    artifact. It establishes neither real model compatibility, learning gain,
    physical safety nor robot-task competence.
    """
    if (
        protocol.get("scope") != "offline"
        or protocol.get("synthetic_engineering") is not True
        or protocol.get("theta_artifact_engineering")
        != "rsi-theta-artifact-engineering-v1"
        or set(protocol["hard_constraints"])
        != {"no_device_safety", "artifact_interface"}
    ):
        raise PermissionError(
            "theta artifact evidence requires its explicit synthetic offline protocol"
        )
    started = time.monotonic()
    problem_path = Path(problem_ref).resolve(strict=True)
    problem = json.loads(problem_path.read_text())
    if (
        problem.get("schema") != "rsi-theta-artifact-problem-v1"
        or problem.get("information_scope") != "release"
    ):
        raise ValueError("unsupported theta artifact engineering problem")
    expected = {
        (key, seed)
        for key in protocol["task_sets"]["release"]["task_keys"]
        for seed in protocol["task_sets"]["release"]["seeds"]
    }
    tasks = problem["tasks"]
    if (
        len(tasks) != len(expected)
        or {(row["task_key"], row["seed"]) for row in tasks} != expected
    ):
        raise ValueError(
            "theta artifact assignments differ from the frozen denominator"
        )
    version = training_theta_version(
        training_result_ref=problem["training_result_ref"],
        load_result_ref=problem["load_result_ref"],
        scope_id=problem["training_scope_id"],
    )
    source_path = Path(problem["source_candidate_ref"]).resolve(strict=True)
    source = Candidate(**json.loads(source_path.read_text()))
    if (
        source.changes.get("theta_ref") != version["theta_ref"]
        or source.changes.get("processor_ref") != version["processor_ref"]
    ):
        raise ValueError(
            "registered theta source candidate differs from its trained/loaded artifact"
        )
    closure = close_theta_candidate_costs(source, scope_id=problem["training_scope_id"])
    complete = RunBundle(**bundle)
    if complete.protocol_id != protocol["protocol_id"]:
        raise ValueError("theta artifact bundle belongs to another evaluation protocol")
    # Parsing the frozen adapter configuration is an interface check only. The
    # saved synthetic byte fixture is deliberately never loaded as model weights.
    descriptor = resolve_reference(complete.theta_ref)
    profile = resolve_reference(complete.backend_ref)
    program = resolve_reference(complete.phi_ref)
    interface = (
        isinstance(descriptor, dict)
        and isinstance(profile, dict)
        and isinstance(program, dict)
    )
    if not interface:
        raise ValueError("theta artifact bundle has no parsed execution interface")
    handoff = (
        complete.theta_ref == version["theta_ref"]
        and complete.processor_ref == version["processor_ref"]
    )
    assignments = []
    for task in tasks:
        criterion = task["criterion"]
        if criterion not in {"parsed_interface", "traced_training_handoff"}:
            raise ValueError("unsupported theta artifact task criterion")
        successful = interface if criterion == "parsed_interface" else handoff
        assignments.append(
            {
                "task_key": task["task_key"],
                "seed": task["seed"],
                "outcome": "success" if successful else "failure",
                "validity": "valid",
                "evidence_ref": str(problem_path),
                "outcome_source": "synthetic_task",
            }
        )
    groups = {task["task_key"]: task["group"] for task in protocol["tasks"]}
    metrics = {}
    for group in {groups[key] for key, _ in expected}:
        rows = [row for row in assignments if groups[row["task_key"]] == group]
        metrics["success_" + group] = sum(
            row["outcome"] == "success" for row in rows
        ) / len(rows)
    elapsed = time.monotonic() - started
    costs = dict(closure["all_attempt_costs"]) if handoff else {}
    costs["whole_job_wall_seconds"] = costs.get("whole_job_wall_seconds", 0) + elapsed
    costs["evaluation_wall_seconds"] = elapsed
    costs.setdefault("model_requests", 0)
    costs.setdefault("gpu_device_seconds_envelope", 0)
    return ExperimentResult(
        result_id,
        complete.bundle_id,
        protocol["protocol_id"],
        protocol["benchmark_id"],
        "release",
        "offline",
        {
            "information_scope": "release",
            "synthetic_engineering": True,
            "evaluation_kind": "synthetic_theta_artifact_engineering",
            "theta_ref": complete.theta_ref,
            "processor_ref": complete.processor_ref,
            "artifact_handoff_complete": handoff,
            "candidate_cost_closure": closure,
            "candidate_acquisition_costs_included": handoff,
            "real_model_compatibility": "unmeasured",
            "physical_safety": "unmeasured",
            "learning_gain": "unmeasured",
            "robot_task_gain": "unmeasured",
        },
        [
            str(problem_path),
            str(source_path),
            version["training_result_ref"],
            version["load_receipt_ref"],
            complete.theta_ref,
            complete.processor_ref,
        ],
        protocol["protocol_id"],
        metrics,
        {"no_device_safety": True, "artifact_interface": interface},
        costs,
        candidate_id=candidate_id,
        assigned_count=len(expected),
        assignments=assignments,
        outcome="success"
        if all(row["outcome"] == "success" for row in assignments)
        else "failure",
    )
