# Copyright 2026 RSI-Agent-VLA Authors.
"""Fixed third-batch consumers dispatched by the existing isolated Host roles."""

from pathlib import Path

from .storage import atomic_json

ROLES = {
    "p1_acceptance": "evaluation",
    "p1_boundary_check": "operator",
    "data_review": "evaluation",
    "training_prepare": "runtime",
    "evaluation_feedback": "evaluation",
    "release_initialize": "release",
    "release_evaluate": "release",
    "release_matrix_evaluation": "evaluation",
    "release_activate": "release",
    "release_inherit": "runtime",
    "release_rollback": "release",
    "release_observe": "operator",
    "release_loaded": "runtime",
    "training_output_parse": "runtime",
    "training_bundle_prepare": "research",
    "goal_materialization": "evaluation",
    "materialized_task_prepare": "runtime",
    "meta_exam_research": "research",
    "meta_exam_score": "evaluation",
    "meta_policy_qualify": "release",
    "meta_policy_activate": "release",
}


def execute(backend, kind, payload, *, operation_id=None):
    """Consume registered inputs within their role-owned output domain."""
    import os
    import sys

    from .task_completion import read

    role = ROLES[kind]
    if os.environ.get("RSI_ISOLATED_ROLE") != role:
        raise PermissionError("framework operation requires its independent OS role")
    scope, _ = backend.authority()
    root = Path(backend.config["isolation_task_root"]).resolve()
    domains = {
        "evaluation": root / "state/evaluation",
        "runtime": root / "runtime",
        "research": root / "state/research",
        "release": Path(backend.config.get("release_root", root / "release")),
    }

    def output(value):
        path = Path(value).resolve()
        if not path.is_relative_to(domains[role]):
            raise PermissionError("framework output escapes its role domain")
        return path

    if kind == "p1_acceptance":
        from .p1_acceptance import evaluate_p1

        output(payload["output_ref"])
        return evaluate_p1(payload["configuration_ref"], payload["output_ref"])
    if kind.startswith("release_"):
        from .release_flow import role_action

        return role_action(backend.config, {"action": kind, **payload})
    if kind == "evaluation_feedback":
        return backend.evaluations.public_feedback(
            payload["result_id"],
            protocol=backend.protocol,
            information_scope=payload.get("information_scope"),
        )
    if kind.startswith("meta_"):
        from .meta_exam import role_action

        return role_action(backend.config, {"action": kind, **payload})
    if kind == "data_review":
        from .collection import review_training_dataset

        args = dict(payload)
        output(args["output_root"])
        for field in ("quality_review_ref", "evaluation_isolation_review_ref"):
            if args.get(field):
                review = Path(args[field]).resolve(strict=True)
                if not any(
                    review.is_relative_to(domain)
                    for domain in (root / "operator", root / "state/evaluation")
                ):
                    raise PermissionError(
                        "data admission requires independently owned review files"
                    )
        if args.pop("scope_id", scope["scope_id"]) != scope["scope_id"]:
            raise PermissionError("data review scope mismatch")
        if (
            args.pop("protocol_id", backend.protocol["protocol_id"])
            != backend.protocol["protocol_id"]
        ):
            raise PermissionError("data review protocol mismatch")
        return review_training_dataset(
            scope_id=scope["scope_id"],
            protocol_id=backend.protocol["protocol_id"],
            **args,
        )
    if kind == "training_prepare":
        from .adapters.pi05_training import prepare_admitted_training
        from .training import TrainingBudget

        args = dict(payload)
        output(args["output_root"])
        admission = Path(args["admission_ref"]).resolve(strict=True)
        if not any(
            admission.is_relative_to(domain)
            for domain in (root / "operator", root / "state/evaluation")
        ):
            raise PermissionError(
                "training preparation requires independently owned admission"
            )
        if args.pop("scope_id", scope["scope_id"]) != scope["scope_id"]:
            raise PermissionError("training preparation scope mismatch")
        if (
            args.pop("protocol_id", backend.protocol["protocol_id"])
            != backend.protocol["protocol_id"]
        ):
            raise PermissionError("training preparation protocol mismatch")
        args["budget"] = TrainingBudget(**args["budget"])
        data_protocol_id = args.pop("data_protocol_id", backend.protocol["protocol_id"])
        return prepare_admitted_training(
            scope_id=scope["scope_id"],
            protocol_id=data_protocol_id,
            **args,
        )
    if kind == "training_output_parse":
        from .training import finalize_training_job

        args = dict(payload)
        output(args["artifact_root"])
        permit_path = Path(args.pop("permit_ref")).resolve(strict=True)
        if not permit_path.is_relative_to(root / "operator"):
            raise PermissionError(
                "training parser requires the registered operator permit"
            )
        output(args["prepared_root"])
        permit = read(permit_path)
        supervision_ref = Path(args.pop("supervision_ref")).resolve(strict=True)
        if permit.get("scope_id") != scope["scope_id"]:
            raise PermissionError("training output belongs to another scope")
        supervision = read(supervision_ref)
        if (
            supervision_ref
            != Path(args["artifact_root"]).resolve() / "supervision.json"
        ):
            raise PermissionError(
                "training output must consume its original supervision file"
            )
        return finalize_training_job(
            permit=permit,
            supervision=supervision,
            python_executable=sys.executable,
            **args,
        )
    if kind == "training_bundle_prepare":
        from .theta_b import register_training_bundle

        args = dict(payload)
        artifacts = Path(args.pop("artifacts")).resolve()
        if artifacts / "research" != backend.store.root.resolve():
            raise PermissionError(
                "training candidate must use the registered ResearchStore"
            )
        if args.pop("scope_id", scope["scope_id"]) != scope["scope_id"]:
            raise PermissionError("training candidate scope mismatch")
        args.setdefault(
            "accounting_ref",
            backend.config.get("host_accounting_ref", str(root / "accounting.json")),
        )
        args.setdefault("approval_scope", backend.protocol["scope"])
        args.setdefault("current_operation_id", operation_id)
        return register_training_bundle(artifacts, scope_id=scope["scope_id"], **args)
    if kind == "goal_materialization":
        from .trusted_goal import materialize_goal

        args = dict(payload)
        policy_ref = Path(args.pop("policy_ref")).resolve(strict=True)
        if not policy_ref.is_relative_to(root / "operator"):
            raise PermissionError("goal policy requires operator ownership")
        policy = read(policy_ref)
        if Path(policy["policy_ref"]).resolve() != policy_ref:
            raise ValueError("goal policy identity mismatch")
        opportunity = backend.store.get("opportunities", args.pop("opportunity_id"))
        output(args["output_root"])
        result = materialize_goal(backend.protocol, opportunity, policy=policy, **args)
        return {
            **result,
            "materialization_ref": result["task"]["materialization_ref"],
            "artifact_ref": result["task"]["materialization_ref"],
        }
    if kind == "materialized_task_prepare":
        from robots.libero.materialized_task import runtime_binding

        args = dict(payload)
        directory = output(args.pop("output_root"))
        receipt_ref = Path(args.pop("materialization_ref")).resolve(strict=True)
        if not receipt_ref.is_relative_to(root / "state/evaluation"):
            raise PermissionError("new task requires independently materialized inputs")
        receipt = read(receipt_ref)
        result = runtime_binding(receipt["task_ref"], **args)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "task-binding.json"
        if destination.exists() and read(destination) != result:
            raise ValueError("conflicting materialized task binding")
        atomic_json(destination, result)
        return {**result, "artifact_ref": str(destination)}
    raise ValueError("unsupported framework operation")
