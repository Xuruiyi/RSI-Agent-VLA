# Copyright 2026 RSI-Agent-VLA contributors.

"""Direct Q16 qualification and task-boundary publication in a fresh task domain.

Baseline qualification creates the first effective version from new evidence.
Candidate acceptance requires both useful comparison and initial capabilities.
The CLI executes only fixed role operations through the existing mount boundary.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .contracts import ExperimentResult, GateDecision
from .evaluation import EvaluationStore, assignment_metrics, load_protocol
from .evaluator import evaluate_matrix
from .qualification_import import _readonly, require_role
from .release import (
    ReleaseManager,
    _validate_initial_capability,
    inspect_initial_result,
)
from .runtime import (
    dynamic_job_lock,
    execution_binding,
    prepare_runtime,
    resolve_reference,
)
from .storage import ResearchStore, identifier

ROLES = {
    "evaluate-baseline": "evaluation",
    "evaluate-candidate": "evaluation",
    "initialize": "release",
    "decide": "release",
    "activate": "release",
    "inherit-plan": "runtime",
    "inherit-run": "runtime",
}
TASKS = {"base_salad", "position_swap_salad", "old_milk", "old_drawer"}


def _read(path):
    return json.loads(Path(path).read_text())


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deployment registration requires UTC timestamps")
    return parsed


def _configuration(reference, role=None):
    reference = Path(reference).resolve(strict=True)
    value = _read(reference)
    scope = _read(value["scope_authorization_ref"])
    root = Path(value["task_root"]).resolve(strict=True)
    domains = {
        name: str((root / relative).resolve())
        for name, relative in {
            "operator": "operator",
            "research": "state/research",
            "evaluation": "state/evaluation",
            "release": "release",
            "runtime": "runtime",
        }.items()
    }
    if (
        value.get("version") != "rsi-midterm-direct-deployment-v1"
        or value.get("authorized") is not True
        or scope.get("schema") != "rsi-midterm-scope-v1"
        or scope.get("authorized") is not True
        or scope.get("dynamic_authorized") is not True
        or value.get("scope") != scope.get("scope_id")
        or scope.get("status") != "active"
        or scope.get("owner_uid") != os.getuid()
        or not root.is_relative_to(Path(scope["write_root"]).resolve())
        or not _time(scope["accepted_at"])
        <= _time(value["registered_at"])
        < _time(value["expires_at"])
        <= _time(scope["expires_at"])
        or datetime.now(timezone.utc) >= _time(value["expires_at"])
        or value.get("role_domains") != domains
        or Path(value["artifacts_root"]).resolve() != root / "state"
        or not reference.is_relative_to(Path(domains["operator"]))
        or value.get("registration_ref") != str(reference)
        or not all(
            isinstance(value.get("controller_selection_rules", {}).get(side), str)
            and value["controller_selection_rules"][side].strip()
            for side in ("baseline", "candidate")
        )
    ):
        raise PermissionError(
            "fresh task deployment registration, authority or domains mismatch"
        )
    if role is not None:
        require_role(value, role)
        if not _readonly(reference) or not _readonly(
            Path(value["scope_authorization_ref"])
        ):
            raise PermissionError(
                "deployment registration and scope must be protected inputs"
            )
    protocol = load_protocol(value["protocol_ref"])
    requirements = {
        "metric_minima": {"success_base": 1, "success_old": 1},
        "required_old_tasks": ["old_milk", "old_drawer"],
    }
    if (
        protocol != value["protocol_snapshot"]
        or protocol.get("comparison_only") is not False
        or protocol.get("scope_id") != value["scope"]
        or protocol.get("scope") != "simulation"
        or protocol.get("dynamic_authorized") is not True
        or protocol.get("deployment_derivation")
        or protocol.get("initial_requirements", {}).get("metric_minima")
        != requirements["metric_minima"]
        or set(protocol.get("initial_requirements", {}).get("required_old_tasks", []))
        != set(requirements["required_old_tasks"])
        or protocol.get("min_gain") != 0.02
        or protocol.get("anchor_max_drop") != 0.02
        or protocol["budget"]["per_attempt_proposal"]["max_episode_steps"] != 400
        or "initial_capabilities" not in protocol["hard_constraints"]
        or set(protocol["task_sets"]["release"]["task_keys"]) != TASKS
        or protocol["task_sets"]["release"]["seeds"] != [16, 17]
    ):
        raise ValueError(
            "Q16 protocol must preserve its fixed denominator and qualification thresholds"
        )
    manager = ReleaseManager(
        domains["release"],
        ResearchStore(domains["research"]),
        EvaluationStore(domains["evaluation"]),
    )
    return value, protocol, manager


def _candidate_selection(registration, *, require_accept=True):
    selection_path = Path(registration["candidate"]["selection_ref"]).resolve(
        strict=True
    )
    if not selection_path.is_relative_to(
        Path(registration["role_domains"]["operator"])
    ) or not _readonly(selection_path):
        raise PermissionError("candidate selection must be protected operator evidence")
    selected = _read(selection_path)
    if (
        selected.get("registration_ref") != registration["registration_ref"]
        or selected.get("selection_rule")
        != registration["controller_selection_rules"]["candidate"]
        or not selected.get("source_result_refs")
    ):
        raise ValueError(
            "candidate differs from the predeclared controller selection rule"
        )
    for ref in selected["source_result_refs"]:
        path = Path(ref).resolve(strict=True)
        if not path.is_relative_to(
            Path(registration["role_domains"]["evaluation"])
        ) or not _readonly(path):
            raise PermissionError(
                "candidate selection needs independent legal P4 results"
            )
    gate_path = Path(selected["theta_effect_admission_ref"]).resolve(strict=True)
    if not gate_path.is_relative_to(
        Path(registration["role_domains"]["evaluation"])
    ) or not _readonly(gate_path):
        raise PermissionError(
            "theta-effect Gate must be independent protected evidence"
        )
    from .midterm_science import verify_scientific_admission

    prospective = selected["execution_binding"]["bundle"]
    original = registration["baseline"]["execution_binding"]["bundle"]
    gate = verify_scientific_admission(
        gate_path,
        expected_scope=registration["scope"],
        expected_theta_ref=prospective["theta_ref"],
        expected_processor_ref=prospective["processor_ref"],
        expected_baseline_theta_ref=original["theta_ref"],
        expected_baseline_processor_ref=original["processor_ref"],
        expected_scope_authorization_ref=registration["scope_authorization_ref"],
        evaluation_root=registration["role_domains"]["evaluation"],
        require_accept=require_accept,
    )
    if (
        require_accept
        and gate.get("verdict") != "accept"
        or gate.get("protocol_id") != registration["p4_protocol_id"]
        or gate.get("release_eligible") is not False
    ):
        raise ValueError("candidate needs independent P4 theta-effect acceptance")
    return selected, gate


def _side(registration, protocol, manager, side):
    slot = registration[side]
    if side == "baseline":
        binding = slot["execution_binding"]
        bundle_id, candidate_id = binding["bundle"]["bundle_id"], None
        cutoff = max(
            _time(registration["registered_at"]),
            datetime.fromtimestamp(
                Path(registration["registration_ref"]).stat().st_mtime, timezone.utc
            ),
        )
    else:
        selected, gate = _candidate_selection(registration)
        binding = selected["execution_binding"]
        bundle_id, candidate_id = (
            binding["bundle"]["bundle_id"],
            selected["candidate_id"],
        )
        candidate = manager.research.get("candidates", candidate_id)
        baseline = registration["baseline"]["execution_binding"]["bundle"]
        if (
            candidate["bundle_id"] != bundle_id
            or candidate["base_bundle_id"] != baseline["bundle_id"]
            or candidate["protocol_id"] != protocol["protocol_id"]
            or binding["bundle"]["theta_ref"] == baseline["theta_ref"]
        ):
            raise ValueError(
                "candidate must bind a distinct admitted theta1 and the real baseline"
            )
        cutoff = max(
            _time(selected["selected_at"]),
            datetime.fromtimestamp(
                Path(slot["selection_ref"]).stat().st_mtime, timezone.utc
            ),
        )
        if cutoff < _time(registration["registered_at"]):
            raise ValueError("candidate selection predates its frozen rule")
    bundle = manager.research.get("bundles", bundle_id)
    if (
        bundle["protocol_id"] != protocol["protocol_id"]
        or not binding["profile"].get("common_execution")
        or execution_binding(
            manager.research, bundle, resolve_reference(bundle["backend_ref"])
        )
        != binding
    ):
        raise ValueError("qualification execution inputs changed after freezing")
    return slot, binding, candidate_id, cutoff


def evaluate_qualification(registration_ref, side):
    """Evaluate actual Q16 outputs and independently derive initial capabilities."""
    from robots.libero.execution_arms import select_execution_arm

    registration, protocol, manager = _configuration(registration_ref, "evaluation")
    slot, binding, candidate_id, cutoff = _side(registration, protocol, manager, side)
    expected = {(task, seed) for task in TASKS for seed in (16, 17)}
    rows = slot["manifest"]
    if len(rows) != 8 or {(row["task_key"], row["seed"]) for row in rows} != expected:
        raise ValueError(
            "qualification requires all eight registered task/initial-state slots"
        )
    confirmed, unavailable = [], []
    for row in rows:
        output = Path(row["result_path"]).parent.resolve()
        if not output.is_relative_to(Path(registration["role_domains"]["runtime"])):
            raise PermissionError(
                "qualification output is outside the isolated runtime domain"
            )
        try:
            context = _read(output / "runtime_context.json")
            first = json.loads(
                (output / "action_events.jsonl").read_text().splitlines()[0]
            )
        except (OSError, ValueError, IndexError) as exc:
            unavailable.append(
                {
                    "task_key": row["task_key"],
                    "seed": row["seed"],
                    "run_id": row["run_id"],
                    "reason": str(exc),
                }
            )
            continue
        if (
            context.get("execution_binding") != binding
            or context.get("run_id") != row["run_id"]
            or context.get("execution_arm")
            != select_execution_arm(binding["profile"], row["task_key"])
            or context.get("common_execution")
            != binding["profile"].get("common_execution")
            or context.get("limits", {}).get("environment_steps")
            != protocol["budget"]["per_attempt_proposal"]["max_episode_steps"]
            or datetime.fromtimestamp(first["utc_ns"] / 1e9, timezone.utc) <= cutoff
        ):
            raise ValueError(
                "qualification inputs or outcome precede the frozen controller"
            )
        confirmed.append(row)
    result = evaluate_matrix(
        protocol,
        binding["profile"],
        confirmed,
        result_id=slot["result_id"],
        bundle_id=binding["bundle"]["bundle_id"],
        candidate_id=candidate_id,
        role="release",
        audit_output_root=manager.evaluations.root
        / "result_audits"
        / identifier(slot["result_id"]),
    )
    capability = _capability(protocol, result)
    result = replace(
        result,
        hard_constraints={
            **result.hard_constraints,
            "initial_capabilities": capability,
        },
        intervention={
            **result.intervention,
            "direct_deployment_registration": registration,
            "qualified_side": side,
            "unavailable_input_bindings": unavailable,
        },
    )
    manager.evaluations.put("protocols", protocol["protocol_id"], protocol)
    manager.evaluations.save_result(result)
    return asdict(result)


def _capability(protocol, result):
    required = set(protocol["initial_requirements"]["required_old_tasks"])
    required.update(
        task["task_key"] for task in protocol["tasks"] if task["group"] == "base"
    )
    if any(
        row["task_key"] in required
        and row["validity"] == "valid"
        and row["outcome"] == "failure"
        for row in result.assignments
    ):
        return False
    if result.unknown_count or result.invalid_count:
        return None
    try:
        _validate_initial_capability(
            protocol["initial_requirements"],
            assignment_metrics(protocol, result),
            result,
        )
        return True
    except ValueError:
        return False


def _result(registration, protocol, manager, side):
    slot, binding, candidate_id, _ = _side(registration, protocol, manager, side)
    result = ExperimentResult(**manager.evaluations.get("results", slot["result_id"]))
    if (
        result.bundle_id != binding["bundle"]["bundle_id"]
        or result.candidate_id != candidate_id
        or result.intervention.get("direct_deployment_registration") != registration
        or result.intervention.get("qualified_side") != side
        or result.hard_constraints.get("initial_capabilities")
        is not _capability(protocol, result)
        or manager.evaluations.get("protocols", protocol["protocol_id"]) != protocol
    ):
        raise ValueError(
            "qualified result does not match its actual registered consumer"
        )
    assignment_metrics(protocol, result)
    return result


def initialize(registration_ref, *, queues_empty):
    """Qualify new evidence before creating a bootstrap staging record."""
    registration, protocol, manager = _configuration(registration_ref, "release")
    if queues_empty is not True:
        raise ValueError(
            "initial activation requires a task boundary and empty motion queues"
        )
    result = _result(registration, protocol, manager, "baseline")
    check = inspect_initial_result(
        protocol, manager.research.get("bundles", result.bundle_id), result
    )
    if check["passed"] is not True:
        if manager.active_path.exists():
            raise ValueError(
                "initial rejection cannot relabel an existing active history"
            )
        value = {
            "status": "no_qualified_initial",
            "verdict": "reject"
            if any(v is False for v in result.hard_constraints.values())
            else "inconclusive",
            "check": check,
            "active_created": False,
            "active": None,
        }
        manager.records.put(
            "midterm_initializations", registration["initial_qualification_id"], value
        )
        return value
    key = registration["initial_qualification_id"]
    receipt = manager.records.path("midterm_initializations", key)
    if receipt.exists():
        return _read(receipt)
    qualification = manager.qualify_initial(key, result_id=result.result_id)
    if manager.active_path.exists():
        active = manager.active()
        if (
            active.get("activation") == "qualified_initial"
            and active.get("qualification_ref") == key
            and active["bundle_id"] == result.bundle_id
        ):
            value = {
                "qualification": qualification,
                "active": active,
                "bootstrap_claims_activation": False,
            }
            manager.records.put("midterm_initializations", key, value)
            return value
        if (
            active.get("activation") != "bootstrap"
            or active.get("bundle_id") != result.bundle_id
        ):
            raise ValueError(
                "fresh deployment domain already has another effective version"
            )
    manager.bootstrap(result.bundle_id, scope=protocol["scope"])
    active = manager.establish_initial(
        key, safe_point="task_boundary", queues_empty=queues_empty
    )
    value = {
        "qualification": qualification,
        "active": active,
        "bootstrap_claims_activation": False,
    }
    manager.records.put("midterm_initializations", key, value)
    return value


def decide(registration_ref):
    """Use actual current/anchor results; rejection keeps the effective version."""
    registration, protocol, manager = _configuration(registration_ref, "release")
    terminal = _no_initial(registration, manager)
    if terminal:
        return terminal
    selected, science = _candidate_selection(registration, require_accept=False)
    if science["verdict"] != "accept":
        baseline = _result(registration, protocol, manager, "baseline")
        with dynamic_job_lock():
            before = manager.active()
            if (
                before.get("activation") == "bootstrap"
                or before["bundle_id"] != baseline.bundle_id
            ):
                raise ValueError(
                    "scientific refusal requires the real qualified baseline"
                )
            manager.qualification(baseline.bundle_id, scope=protocol["scope"])
            decision = GateDecision(
                registration["decision_id"],
                selected["candidate_id"],
                baseline.bundle_id,
                selected["execution_binding"]["bundle"]["bundle_id"],
                protocol["protocol_id"],
                protocol["benchmark_id"],
                science["verdict"],
                [
                    "independent scientific theta effect was not accepted; candidate Q/publication is blocked"
                ],
                False,
                protocol["scope"],
                {},
                [selected["theta_effect_admission_ref"]],
            )
            value = {
                "decision": asdict(decision),
                "active_before": before,
                "active_after": manager.active(),
                "scientific_admission_ref": selected["theta_effect_admission_ref"],
                "scientific_admission_blocked": True,
                "candidate_runtime_started": any(
                    (Path(row["result_path"]).parent / "runtime_context.json").exists()
                    for row in registration["candidate"]["manifest"]
                ),
            }
            manager.records.put("midterm_decisions", registration["decision_id"], value)
            return value
    candidate = _result(registration, protocol, manager, "candidate")
    baseline = _result(registration, protocol, manager, "baseline")
    key = registration["decision_id"]
    with dynamic_job_lock():
        existing = manager.records.path("midterm_decisions", key)
        if existing.exists():
            return _read(existing)
        before = manager.active()
        if (
            before.get("activation") == "bootstrap"
            or before["bundle_id"] != baseline.bundle_id
        ):
            raise ValueError(
                "candidate decision requires the truly activated qualified baseline"
            )
        manager.qualification(baseline.bundle_id, scope=protocol["scope"])
        result = manager._evaluate(
            key,
            candidate.candidate_id,
            candidate_result=candidate.result_id,
            baseline_result=baseline.result_id,
            anchor_result=baseline.result_id,
        )
        value = {
            "decision": asdict(result),
            "active_before": before,
            "active_after": manager.active(),
        }
        manager.records.put("midterm_decisions", key, value)
        return value


def activate(registration_ref, *, queues_empty):
    registration, protocol, manager = _configuration(registration_ref, "release")
    terminal = _no_initial(registration, manager)
    if terminal:
        return terminal
    saved = manager.records.get("midterm_decisions", registration["decision_id"])
    if saved.get("scientific_admission_blocked"):
        selected, science = _candidate_selection(registration, require_accept=False)
        if (
            science["verdict"] == "accept"
            or science["verdict"] != saved["decision"]["verdict"]
            or selected["theta_effect_admission_ref"]
            != saved["scientific_admission_ref"]
        ):
            raise ValueError("scientific refusal is stale against its current source")
    else:
        _result(registration, protocol, manager, "candidate")
    if saved["decision"]["verdict"] != "accept":
        if manager.active() != saved["active_before"]:
            raise ValueError(
                "rejected/inconclusive candidate changed the effective version"
            )
        return {
            "status": "kept_current",
            "decision": saved["decision"],
            "active": manager.active(),
        }
    active = manager.active()
    if active.get("activation") == registration["decision_id"]:
        return {"status": "active", "active": active}
    return {
        "status": "active",
        "active": manager.activate(
            registration["decision_id"],
            safe_point="task_boundary",
            queues_empty=queues_empty,
            scope=protocol["scope"],
        ),
    }


def inheritance_plan(registration_ref):
    """Select through ReleaseManager, then retain live selection for Runtime."""
    registration, protocol, manager = _configuration(registration_ref, "runtime")
    terminal = _no_initial(registration, manager)
    if terminal:
        return terminal
    bundle, active = manager.deployment_bundle(
        scope=protocol["scope"], safe_point="task_boundary"
    )
    side = (
        "baseline"
        if bundle["bundle_id"]
        == registration["baseline"]["execution_binding"]["bundle"]["bundle_id"]
        else "candidate"
    )
    _, binding, _, _ = _side(registration, protocol, manager, side)
    if bundle != binding["bundle"]:
        raise ValueError(
            "effective version is outside this task's frozen deployment bindings"
        )
    task = registration["inheritance"]
    authorization = task.get("authorization_ref")
    permit = None
    if authorization and Path(authorization).exists():
        authorization = Path(authorization).resolve(strict=True)
        permit = _read(authorization)
        if (
            not authorization.is_relative_to(
                Path(registration["role_domains"]["operator"])
            )
            or not _readonly(authorization)
            or permit.get("scope") != registration["scope"]
            or Path(permit.get("approval_reference", "")).resolve()
            != Path(registration["scope_authorization_ref"]).resolve()
            or _time(permit["expires_at"]) <= datetime.now(timezone.utc)
            or len(permit.get("gpu_devices", [])) != 1
        ):
            raise PermissionError(
                "inheritance requires its current host-issued task permit"
            )
    cuda_device = permit["gpu_devices"][0] if permit else task.get("cuda_device", 0)
    output = Path(task["output_dir"]).resolve()
    if not output.is_relative_to(Path(registration["role_domains"]["runtime"])):
        raise PermissionError("inheritance output escapes the isolated task domain")
    plan = prepare_runtime(
        registration["repository_root"],
        registration["artifacts_root"],
        protocol_path=registration["protocol_ref"],
        run_id=task["run_id"],
        bundle_id=bundle["bundle_id"],
        task_key=task["task_key"],
        seed=task["seed"],
        split="deployment",
        output_dir=output,
        authorization_path=authorization if permit else None,
        cuda_device=cuda_device,
        deployment_binding=active,
        storage_write_roots=task.get("storage_write_roots"),
    )
    plan["context"].update(
        deployment_release_root=str(manager.records.root.resolve()),
        deployment_artifacts_root=str(Path(registration["artifacts_root"]).resolve()),
        deployment_authorization_ref=registration["registration_ref"],
    )
    plan["selection"] = {
        "mode": "deployment",
        "selected_bundle_id": bundle["bundle_id"],
        "active": active,
        "registration_ref": registration["registration_ref"],
    }
    return plan


def _no_initial(registration, manager):
    path = manager.records.path(
        "midterm_initializations", registration["initial_qualification_id"]
    )
    if path.exists() and _read(path).get("status") == "no_qualified_initial":
        if manager.active_path.exists():
            raise ValueError(
                "no-initial receipt conflicts with an actual active version"
            )
        return {
            "status": "blocked_no_qualified_initial",
            "initial_receipt": _read(path),
            "active": None,
            "runtime_started": False,
        }
    return None


def role_operation(registration_ref, operation, *, queues_empty=False):
    if operation not in ROLES:
        raise ValueError("unsupported fixed deployment role operation")
    if operation.startswith("evaluate-"):
        return evaluate_qualification(registration_ref, operation.split("-", 1)[1])
    if operation == "initialize":
        return initialize(registration_ref, queues_empty=queues_empty)
    if operation == "decide":
        return decide(registration_ref)
    if operation == "activate":
        return activate(registration_ref, queues_empty=queues_empty)
    plan = inheritance_plan(registration_ref)
    if plan.get("status") == "blocked_no_qualified_initial":
        return plan
    if operation == "inherit-run":
        from .runtime import execute_runtime

        return execute_runtime(plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--operation", required=True, choices=tuple(ROLES))
    parser.add_argument("--queues-empty", action="store_true")
    parser.add_argument("--launch-role", action="store_true")
    args = parser.parse_args(argv)
    if args.launch_role:
        registration, _, _ = _configuration(args.registration)
        runner_path = (
            Path(registration["repository_root"]) / "scripts/run_isolated_role.py"
        )
        spec = importlib.util.spec_from_file_location("_midterm_isolation", runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        child = [
            registration["python_executable"],
            "-m",
            "rsi_vla.midterm_deployment",
            "--registration",
            args.registration,
            "--operation",
            args.operation,
        ] + (["--queues-empty"] if args.queues_empty else [])
        return subprocess.call(
            runner.command(
                registration["task_root"],
                ROLES[args.operation],
                child,
                gpu=args.operation == "inherit-run",
                protected_paths=registration["protected_paths"],
            )
        )
    print(
        json.dumps(
            role_operation(
                args.registration, args.operation, queues_empty=args.queues_empty
            ),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
