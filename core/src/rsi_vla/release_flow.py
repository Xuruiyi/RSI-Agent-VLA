# ruff: noqa: UP017, FURB162 - Python 3.10 support
"""Fixed Host consumers for qualification, independent safety and publication."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .qualification_import import _readonly, require_role
from .storage import atomic_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manager(config):
    from .evaluation import EvaluationStore
    from .release import ReleaseManager
    from .storage import ResearchStore

    return ReleaseManager(
        config["release_root"],
        ResearchStore(config["research_store_ref"]),
        EvaluationStore(config["evaluation_store_ref"]),
    )


def _observe(config, task, registration, manager):
    """Produce independent engineering observations for a no-device offline domain.

    Real device observations require their supported runtime/operator producers;
    this consumer never promotes a synthetic observation to physical evidence.
    """
    role = "operator" if task["action"] == "release_observe" else "runtime"
    require_role(registration, role)
    protocol = _read(config["protocol_ref"])
    if (
        protocol["scope"] != "offline"
        or registration.get("service_mode") != "no_service"
        or config.get("offline_fixture") is not True
    ):
        raise PermissionError(
            "automatic no-device observation is restricted to synthetic offline domains"
        )
    # Observe the completed, owned Host operations rather than trusting booleans
    # supplied in a release request. Original unresolved receipts remain blockers.
    references = task.get("completed_operation_refs", [])
    for reference in references:
        value = _read(reference)
        if (
            value.get("status") != "completed"
            or value.get("cleanup_complete") is not True
            or value.get("cost_closed") is not True
        ):
            raise ValueError("publication predecessor cost/cleanup is unresolved")
    phase = _read(config["phase_ref"])
    if phase.get("unresolved_costs"):
        raise ValueError("publication phase still has unresolved costs")
    active = manager.active()
    now = datetime.now(timezone.utc)
    common = {
        "scope": registration["scope"],
        "active_snapshot": active,
        "observed_at": now.isoformat(),
        "expires_at": min(
            now + timedelta(minutes=10),
            datetime.fromisoformat(registration["expires_at"].replace("Z", "+00:00")),
        ).isoformat(),
        "producer_role": role,
        "evidence_level": "synthetic_offline",
        "evidence_refs": [config["phase_ref"], *references],
        "device_access": "none",
    }
    if role == "operator":
        value = {
            **common,
            "schema": "rsi-release-safety-observation-v1",
            "operations": ["initialize", "activate", "inherit", "rollback"],
            "safe_point": "task_boundary",
            "queues_empty": True,
            "pending_actions": [],
            "unresolved_costs": [],
            "cleanup_complete": True,
            "physical_state": "known_safe",
        }
        destination = registration["safety_ref"]
    else:
        value = {
            **common,
            "schema": "rsi-service-loading-observation-v1",
            "service_stopped": True,
            "loaded_binding": None,
        }
        destination = registration["loaded_ref"]
    atomic_json(destination, value)
    return value


def role_action(config, task):
    """Use the same ReleaseManager and actual Runtime selector from a Host graph."""
    manager = _manager(config)
    action = task["action"]
    registration_ref = task["registration_ref"]
    registration = _read(registration_ref)
    role = {
        "release_observe": "operator",
        "release_loaded": "runtime",
        "release_inherit": "runtime",
        "release_matrix_evaluation": "evaluation",
    }.get(action, "release")
    require_role(registration, role)
    if not _readonly(registration_ref):
        raise PermissionError("publication registration must be operator protected")
    if action in {"release_observe", "release_loaded"}:
        return _observe(config, task, registration, manager)
    if action == "release_matrix_evaluation":
        from dataclasses import asdict

        from .synthetic_sequence import evaluate_sequence_matrix

        slot = next(
            row
            for row in registration["matrices"]
            if row["result_id"] == task["result_id"]
        )
        if any(slot.get(key) != task.get(key) for key in ("bundle_id", "candidate_id")):
            raise ValueError("publication matrix identity differs from registered task")
        problem_ref = Path(slot["problem_ref"]).resolve(strict=True)
        if not problem_ref.is_relative_to(
            manager.evaluations.root.resolve()
        ) or not _readonly(problem_ref):
            raise PermissionError(
                "publication task requires protected independent problem input"
            )
        if slot.get("evaluation_kind") == "theta_artifact_engineering":
            from .theta_artifact_evaluation import evaluate_theta_artifact_matrix

            evaluator = evaluate_theta_artifact_matrix
        else:
            evaluator = evaluate_sequence_matrix
        value = evaluator(
            _read(config["protocol_ref"]),
            manager.research.get("bundles", task["bundle_id"]),
            result_id=task["result_id"],
            candidate_id=task.get("candidate_id"),
            problem_ref=problem_ref,
        )
        manager.evaluations.save_result(value)
        return asdict(value)
    if action == "release_initialize":
        manager.register_safety(registration_ref)
        qualification = manager.qualify_initial(
            task["qualification_id"],
            result_id=task["result_id"],
            evidence_import_id=task.get("evidence_import_id"),
        )
        return {
            "qualification": qualification,
            "active": manager.establish_initial(
                task["qualification_id"], safe_point="task_boundary", queues_empty=True
            ),
        }
    if action == "release_evaluate":
        from dataclasses import asdict

        return asdict(
            manager.evaluate(
                task["decision_id"],
                task["candidate_id"],
                candidate_result=task.get("candidate_result"),
                baseline_result=task.get("baseline_result"),
                anchor_result=task.get("anchor_result"),
            )
        )
    if action == "release_activate":
        decision = manager.records.get("decisions", task["decision_id"])
        if decision["verdict"] != "accept":
            return {
                "status": "kept_current",
                "active": manager.active(),
                "decision": decision,
            }
        return manager.activate(
            task["decision_id"],
            safe_point="task_boundary",
            queues_empty=True,
            scope=task["scope"],
        )
    if action == "release_rollback":
        return manager.rollback(
            task["target_bundle_id"],
            safe_point="task_boundary",
            queues_empty=True,
            reason=task["reason"],
        )
    if action == "release_inherit":
        import contextlib
        import io

        from .runtime import main

        argv = task["runtime_argv"]
        if (
            "--bundle" in argv
            or "--execute" in argv
            or "--deployment-release" not in argv
        ):
            raise ValueError(
                "offline publication inheritance must use the actual effective Runtime selector"
            )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = main(argv)
        plan = json.loads(stream.getvalue())
        return {
            "runtime_exit_code": result,
            "plan": plan,
            "active": plan["context"]["release_binding"],
            "selection": plan["selection"],
        }
    raise ValueError("unsupported fixed release flow action")


def protected_paths(config, task):
    """Keep the operator registration immutable even in its producing role."""
    operation = task.get("operation", task)
    payload = operation.get("payload", operation)
    reference = payload.get("registration_ref")
    if not reference or not operation.get("kind", payload.get("action", "")).startswith(
        "release_"
    ):
        return ()
    registration = _read(reference)
    paths = [Path(reference).resolve(strict=True)]
    if operation.get("kind") == "release_matrix_evaluation":
        paths.extend(
            Path(row["problem_ref"]).resolve(strict=True)
            for row in registration["matrices"]
        )
    if operation.get("kind") == "release_inherit":
        paths.append(Path(registration["loaded_ref"]).resolve(strict=True))
    return tuple(paths)
