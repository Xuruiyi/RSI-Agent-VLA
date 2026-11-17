# ruff: noqa: UP017, FURB162 - Python 3.10 support
"""Frozen policy examinations over the existing Host and isolated Research jobs.

Each registered slot runs the inherited policy through the trusted rule broker,
submits its chosen development candidate, then uses its registered independent
robot evaluation operations. Only the evaluation role seals and scores results.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .qualification_import import _readonly, require_role
from .storage import identifier


def _read(ref):
    return json.loads(Path(ref).read_text(encoding="utf-8"))


def _manager(config):
    from .evaluation import EvaluationStore
    from .release import ReleaseManager
    from .storage import ResearchStore

    return ReleaseManager(
        config["release_root"],
        ResearchStore(config["research_store_ref"]),
        EvaluationStore(config["evaluation_store_ref"]),
    )


def _registration(config, reference, role):
    path = Path(reference).resolve(strict=True)
    value = _read(path)
    if (
        value.get("schema") != "rsi-meta-exam-v1"
        or not path.is_relative_to(Path(value["role_domains"]["operator"]).resolve())
        or not _readonly(path)
    ):
        raise PermissionError(
            "meta examination requires a protected operator registration"
        )
    require_role(value, role)
    manager = _manager(config)
    domains = value["role_domains"]
    if any(
        Path(domains[domain]).resolve() != actual.resolve()
        for domain, actual in (
            ("research", manager.research.root),
            ("evaluation", manager.evaluations.root),
            ("release", manager.records.root),
        )
    ):
        raise PermissionError(
            "meta examination registered stores differ from consumers"
        )
    if datetime.fromisoformat(
        value["expires_at"].replace("Z", "+00:00")
    ) <= datetime.now(timezone.utc):
        raise PermissionError("meta examination registration expired")
    return value, manager


def build_operations(registration_ref):
    """Build a fixed registered Host graph; this function does not execute work.

    A slot supplies its existing ``evaluation_operations`` and legal development
    ``action_candidates``. The Host retains its own permits, budgets, failure
    handling and native receipts for every operation in the returned graph.
    """
    registration = _read(registration_ref)
    operations, requests = [], []
    points = registration["budget_points"]
    slots = registration["slots"]
    keys = [(s["problem_id"], s["repeat"], s["arm"], s["budget_point"]) for s in slots]
    expected = {
        (p["problem_id"], repeat, arm, point)
        for p in registration["problems"]
        for repeat in registration["repetitions"]
        for arm in registration["arms"]
        for point in points
    }
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(
            "meta graph must preserve every registered problem/process/arm/budget slot"
        )
    for slot in slots:
        identity = identifier(slot["slot_id"])
        request = deepcopy(slot["research_request"])
        requests.append(request)
        operations.append(
            {
                "operation_id": identity + "-research",
                "kind": "meta_exam_research",
                "stage": "meta_exam",
                "dynamic": False,
                "payload": {
                    "registration_ref": str(Path(registration_ref).resolve()),
                    "slot_id": identity,
                    "request_id": request["request_id"],
                },
            }
        )
        operations.extend(deepcopy(slot["evaluation_operations"]))
    for kind in ("meta_exam_score", "meta_policy_qualify", "meta_policy_activate"):
        operations.append(
            {
                "operation_id": identifier(registration["exam_id"] + "-" + kind),
                "kind": kind,
                "stage": "meta_exam",
                "dynamic": False,
                "payload": {
                    "registration_ref": str(Path(registration_ref).resolve()),
                    "exam_id": registration["exam_id"],
                    "qualification_id": registration["exam_id"] + "-policy",
                },
            }
        )
    identities = [op["operation_id"] for op in operations]
    if len(identities) != len(set(identities)):
        raise ValueError("meta graph operation identities must be unique")
    nodes = {
        identity: {"operation_id": identity, "transitions": [{"next": successor}]}
        for identity, successor in zip(identities, identities[1:] + ["meta-done"])
    }
    nodes["meta-done"] = {"terminal": "meta_exam_complete"}
    return {
        "operations": operations,
        "research_requests": requests,
        "continuation": {
            "schema": "rsi-task-continuation-v1",
            "start": identities[0],
            "nodes": nodes,
        },
    }


def _slot(registration, slot_id):
    matches = [row for row in registration["slots"] if row["slot_id"] == slot_id]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate meta examination slot")
    return matches[0]


def research(config, task):
    """Run the actual inherited program without exposing independent exam scores."""
    registration, manager = _registration(config, task["registration_ref"], "research")
    slot = _slot(registration, task["slot_id"])
    from .effective_psi import consume_effective_psi
    from .task_completion import TaskCompletionBackend

    backend = TaskCompletionBackend(config)
    tools, request = backend.tools(task["request_id"])
    arm = registration["arms"][slot["arm"]]
    if (
        request != slot["research_request"]
        or task["request_id"] != request["request_id"]
    ):
        raise ValueError("meta Research request differs from the registered slot")
    policy = consume_effective_psi(
        tools, arm["policy_snapshot_id"], facts_ref=arm["facts_ref"]
    )
    frozen = _read(arm["policy_snapshot_ref"])
    facts = _read(arm["facts_ref"])
    if (
        frozen != manager.research.get("effective_psi", arm["policy_snapshot_id"])
        or facts.get("legal_scope") != "development_only"
        or facts.get("role") != "facts_not_policy"
        or tools.base.theta_ref != registration["common"]["theta_ref"]
        or tools.base.phi_ref != registration["common"]["phi_ref"]
        or tools.base.memory_ref != facts["memory_ref"]
        or tools.store.get("memory", tools.base.memory_ref) != facts["memory"]
    ):
        raise ValueError(
            "meta policy, common execution or legal factual starting point differs"
        )
    if (
        slot["arm"] in {"psi0", "psi1"}
        and arm["facts_ref"] != registration["shared_facts_ref"]
    ):
        raise ValueError("core policies need the same legal factual start")
    non_test = manager.research.get(
        "rho", registration["common"]["non_test_policy_ref"]
    )
    actual_rho = manager.research.get("rho", tools.base.rho_ref)
    if any(actual_rho[key] != non_test[key] for key in ("rhoA", "rhoB")):
        raise ValueError("non-tested research policy components differ")
    process_limits = {
        json.dumps(row["research_request"]["job_limits"], sort_keys=True)
        for row in registration["slots"]
    }
    if len(process_limits) != 1:
        raise ValueError(
            "compared research policies must share the same hard Job resource limits"
        )
    tools.query_development(request["result_ids"])
    feedback = tools.run_inherited_research_rule(
        request["request_id"] + "-psi-selection",
        inputs={
            "evidence_available": bool(request["result_ids"]),
            "budget_available": True,
        },
        config=config,
        timeout_seconds=min(3, slot.get("timeout_seconds", 3)),
    )
    action = (
        feedback["selection"]["action"] if feedback.get("technical_valid") else None
    )
    candidate_ref = slot.get("action_candidates", {}).get(action)
    candidate = None
    if action in slot.get("action_experiments", {}):
        from .contracts import ExperimentResult
        from .rule_updates import load_runtime_rules

        experiment = deepcopy(slot["action_experiments"][action])
        opportunity_id = request["request_id"] + "-opportunity"
        tools.discover_opportunity(
            opportunity_id,
            source="normal",
            result_ids=request["result_ids"],
            external_value=slot["development_value"],
            competing_explanations=slot["hypotheses"],
            expected_cost=experiment["cost_limit"],
        )
        tools.plan_rule_experiment(opportunity_id=opportunity_id, **experiment)
        trial = tools.prepare_rule_trial(experiment["experiment_id"])
        development = _read(slot["development_problem_ref"])
        if (
            development.get("schema") != "rsi-synthetic-sequence-problem-v1"
            or development.get("information_scope") != "development"
        ):
            raise PermissionError(
                "meta development consumer requires a public development task"
            )

        def evaluate(plan, actual_trial):
            rules = load_runtime_rules(actual_trial["runtime_rules_ref"])
            rows = _sequence_assignments(
                development["tasks"], rules, slot["development_problem_ref"]
            )
            return ExperimentResult(
                request["request_id"] + "-development",
                trial["bundle_id"],
                tools.protocol["protocol_id"],
                tools.protocol["benchmark_id"],
                "development",
                "offline",
                {"information_scope": "development", **plan["intervention"]},
                [slot["development_problem_ref"], actual_trial["runtime_rules_ref"]],
                plan["criterion_ref"],
                {},
                {},
                {"offline_calls": 1},
                assigned_count=len(rows),
                assignments=rows,
                outcome="success"
                if all(row["outcome"] == "success" for row in rows)
                else "failure",
            )

        tools.run_offline_experiment(
            experiment["experiment_id"], evaluate, cost={"offline_calls": 1}
        )
        candidate = tools.submit_candidate(
            slot["candidate_id"], experiment_id=experiment["experiment_id"]
        )
        candidate_ref = str(
            tools.store.path("candidates", slot["candidate_id"]).resolve()
        )
    if candidate_ref:
        path = Path(candidate_ref).resolve(strict=True)
        if not path.is_relative_to(manager.research.root.resolve()):
            raise PermissionError(
                "meta selection requires a legal development candidate"
            )
        candidate = _read(path)
        if manager.research.get("candidates", candidate["candidate_id"]) != candidate:
            raise ValueError(
                "meta selected candidate differs from registered research artifact"
            )
        if not candidate.get("development_results"):
            raise ValueError("meta submission requires an actually developed candidate")
        tools.query_development(
            candidate["development_results"], bundle_ids=[candidate["bundle_id"]]
        )
    value = {
        "schema": "rsi-meta-research-selection-v1",
        "exam_id": registration["exam_id"],
        "slot_id": slot["slot_id"],
        "problem_id": slot["problem_id"],
        "repeat": slot["repeat"],
        "arm": slot["arm"],
        "budget_point": slot["budget_point"],
        "action": action,
        "candidate_ref": candidate_ref,
        "candidate_snapshot": candidate,
        "policy_binding": policy,
        "facts_ref": arm["facts_ref"],
        "common": registration["common"],
        "feedback": feedback,
        "job_cost_snapshot": dict(tools.job.state["used"]),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitted"
        if candidate
        else "technical_failure"
        if action is None
        else "missing",
    }
    manager.research.put("meta_exam_selections", slot["slot_id"], value)
    return value


def _sequence_assignments(tasks, rules, reference):
    rows = []
    for task in tasks:
        if len(task["sequence"]) < rules["chunk_steps"]:
            raise ValueError("synthetic sequence cannot execute the candidate chunk")
        observed = task["sequence"][: rules["chunk_steps"]][-1]
        rows.append(
            {
                "task_key": task["task_key"],
                "seed": task["seed"],
                "outcome": "success"
                if observed == task["expected_observation"]
                else "failure",
                "validity": "valid",
                "evidence_ref": str(reference),
                "outcome_source": "synthetic_task",
            }
        )
    return rows


def _evaluate_synthetic_problem(registration, manager, slot, selection):
    """Execute a registered offline sequence task using the submitted real rules.

    Each task observes the sequence element at the candidate's chunk boundary;
    the independent expected observation lives only in the evaluation domain.
    This is an engineering task consumer, not robot or model evidence.
    """
    from dataclasses import asdict, replace

    from .synthetic_sequence import evaluate_sequence_matrix

    problem = next(
        p for p in registration["problems"] if p["problem_id"] == slot["problem_id"]
    )
    if problem.get("problem_kind") != "synthetic_rule_choice":
        return
    if not selection["candidate_ref"]:
        return
    ref = Path(problem["problem_ref"]).resolve(strict=True)
    if not ref.is_relative_to(manager.evaluations.root.resolve()) or not _readonly(ref):
        raise PermissionError(
            "synthetic questions require protected independent evaluation ownership"
        )
    candidate = _read(selection["candidate_ref"])
    bundle = manager.research.get("bundles", candidate["bundle_id"])
    value = evaluate_sequence_matrix(
        _read(problem["protocol_ref"]),
        bundle,
        result_id=slot["result_id"],
        candidate_id=candidate["candidate_id"],
        problem_ref=ref,
    )
    value = replace(
        value,
        intervention={
            **value.intervention,
            "submitted_selection_ref": str(
                manager.research.path("meta_exam_selections", slot["slot_id"]).resolve()
            ),
        },
    )
    path = manager.evaluations.path("results", value.result_id)
    if path.exists():
        existing, proposed = (
            manager.evaluations.get("results", value.result_id),
            asdict(value),
        )
        if any(existing[key] != proposed[key] for key in proposed if key != "costs"):
            raise ValueError("independent synthetic evaluation inputs changed")
    else:
        manager.evaluations.save_result(value)


def _host_attempt(config, identity):
    root = Path(
        config.get(
            "host_root", Path(config["isolation_task_root"]) / "operator/midterm-host"
        )
    )
    completion_ref = root / "jobs" / identifier(identity) / "completion.json"
    completion = _read(completion_ref)
    supervisor = _read(completion["supervisor_ref"])
    if (
        completion.get("operation_id") != identity
        or completion.get("scope_id") != _read(config["protocol_ref"])["scope_id"]
        or completion.get("cleanup_confirmed") is not True
        or supervisor.get("cleanup_confirmed") is not True
        or completion.get("unknown_dimensions")
        or not {"wall_seconds", "storage_bytes"} <= completion.get("actual", {}).keys()
    ):
        raise ValueError(
            "original meta Host attempt has unresolved identity, costs or cleanup"
        )
    return {
        key: value
        for key, value in completion["actual"].items()
        if key not in {"current_disk_bytes", "concurrent_gpu_devices"}
    }


def _seal_submissions(registration, manager, config):
    """Evaluation normalizes actual rule and original Host all-attempt receipts."""
    submissions = []
    for slot in registration["slots"]:
        selection = manager.research.get("meta_exam_selections", slot["slot_id"])
        arm = registration["arms"][slot["arm"]]
        if (
            selection["exam_id"] != registration["exam_id"]
            or selection["policy_binding"]["snapshot_id"] != arm["policy_snapshot_id"]
            or selection["facts_ref"] != arm["facts_ref"]
            or selection["common"] != registration["common"]
        ):
            raise ValueError("meta selection policy or facts differ from registration")
        if (
            selection["candidate_ref"]
            and _read(selection["candidate_ref"]) != selection["candidate_snapshot"]
        ):
            raise ValueError("meta candidate changed after its development submission")
        _evaluate_synthetic_problem(registration, manager, slot, selection)
        identity = {
            key: selection[key]
            for key in ("exam_id", "problem_id", "repeat", "arm", "budget_point")
        }
        attempts = [_host_attempt(config, slot["slot_id"] + "-research")]
        for op in slot["evaluation_operations"]:
            attempts.append(_host_attempt(config, op["operation_id"]))
        if (
            selection["candidate_ref"]
            and manager.evaluations.path("results", slot["result_id"]).exists()
        ):
            measured = manager.evaluations.get("results", slot["result_id"])
            if measured["intervention"].get("synthetic_engineering"):
                attempts.append(measured["costs"])
        # CPU work units are independent from Host wall/storage observations.
        # Per-point snapshots preserve increments instead of charging cumulative
        # Job costs repeatedly at later registered milestones.
        previous = [
            row
            for row in manager.research.all("meta_exam_selections")
            if row["exam_id"] == selection["exam_id"]
            and row["arm"] == selection["arm"]
            and row["repeat"] == selection["repeat"]
            and row["problem_id"] == selection["problem_id"]
            and row["budget_point"] < selection["budget_point"]
        ]
        prior = (
            max(previous, key=lambda row: row["budget_point"])["job_cost_snapshot"]
            if previous
            else {}
        )
        work = {
            key: value - prior.get(key, 0)
            for key, value in selection["job_cost_snapshot"].items()
            if key not in {"whole_job_wall_seconds", "artifact_bytes"}
        }
        if work:
            attempts.append(work)
        for ref in slot.get("attempt_receipt_refs", []):
            receipt = _read(ref)
            if receipt.get("cleanup_confirmed") is not True or receipt.get(
                "unknown_dimensions"
            ):
                raise ValueError(
                    "meta original attempt costs or cleanup remain unresolved"
                )
            attempts.append(receipt["actual"])
        row = {
            key: selection[key]
            for key in (
                "problem_id",
                "repeat",
                "arm",
                "budget_point",
                "candidate_ref",
                "status",
            )
        }
        row.update(
            result_id=slot["result_id"] if selection["candidate_ref"] else None,
            attempt_costs=attempts,
            submission_receipt_ref=str(
                manager.evaluations.path("meta_submissions", slot["slot_id"]).resolve()
            ),
            cost_receipt_ref=str(
                manager.evaluations.path(
                    "meta_attempt_costs", slot["slot_id"]
                ).resolve()
            ),
        )
        manager.evaluations.put(
            "meta_submissions",
            slot["slot_id"],
            {
                **identity,
                "schema": "rsi-meta-submission-v1",
                "candidate_ref": row["candidate_ref"],
                "result_id": row["result_id"],
                "status": row["status"],
                "common": registration["common"],
                "policy_snapshot_id": arm["policy_snapshot_id"],
                "facts_ref": arm["facts_ref"],
                "submitted_at": selection["submitted_at"],
            },
        )
        manager.evaluations.put(
            "meta_attempt_costs",
            slot["slot_id"],
            {
                **identity,
                "schema": "rsi-meta-attempt-cost-v1",
                "attempt_costs": attempts,
            },
        )
        submissions.append(row)
    return submissions


def role_action(config, task):
    """Fixed TaskCompletion consumers; role and graph are selected by the Host."""
    action = task["action"]
    if action == "meta_exam_research":
        return research(config, task)
    role = "evaluation" if action == "meta_exam_score" else "release"
    registration, manager = _registration(config, task["registration_ref"], role)
    if action == "meta_exam_score":
        from .meta_scoring import score_exam

        value = score_exam(
            registration,
            _seal_submissions(registration, manager, config),
            manager.evaluations,
        )
        manager.evaluations.put("meta_exam_scores", registration["exam_id"], value)
        return value
    if action == "meta_policy_qualify":
        return manager.qualify_research_policy(
            task["qualification_id"], exam_id=task["exam_id"]
        )
    if action == "meta_policy_activate":
        qualification = manager.records.get(
            "research_policy_qualifications", task["qualification_id"]
        )
        if qualification["verdict"] != "accept":
            return {"status": "kept_current", "qualification": qualification}
        return manager.activate_research_policy(
            task["qualification_id"], scope=registration["scope"]
        )
    raise ValueError("unsupported fixed meta examination action")


def protected_paths(config, task):
    """Pin frozen policy, facts, protocols and independent question inputs."""
    operation = task.get("operation", task)
    payload = operation.get("payload", operation)
    reference = payload.get("registration_ref")
    if not reference or not operation.get("kind", payload.get("action", "")).startswith(
        "meta_"
    ):
        return ()
    registration = _read(reference)
    paths = {Path(reference).resolve(strict=True)}
    for arm in registration["arms"].values():
        paths.update(
            Path(arm[key]).resolve(strict=True)
            for key in ("policy_snapshot_ref", "facts_ref")
        )
    for problem in registration["problems"]:
        paths.add(Path(problem["protocol_ref"]).resolve(strict=True))
        if problem.get("problem_ref") and operation.get("kind") != "meta_exam_research":
            paths.add(Path(problem["problem_ref"]).resolve(strict=True))
    return tuple(sorted(paths))
