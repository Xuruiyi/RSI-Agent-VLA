"""Single prepared trial through the existing Runtime and independent development importer."""

import json
from dataclasses import replace
from pathlib import Path

from .phase_receipts import read_phase, write_phase
from .runtime import execute_runtime, prepare_runtime
from .storage import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def prepare_trial(
    tools,
    experiment_id,
    *,
    root,
    artifacts,
    protocol_path,
    output,
    run_id,
    permit_path=None,
    task_key="position_swap_salad",
    seed=1,
    cuda_device=0,
    storage_write_roots=None,
):
    experiment = tools._owned("research_experiments", experiment_id)
    trial = tools._owned("rule_trials", experiment_id)
    exploration = experiment.get("exploration")
    if exploration and {"task_key": task_key, "seed": seed} not in exploration["assignments"]:
        raise ValueError("trial assignment outside registered exploration slots")
    if tools.store.path("research_findings", experiment_id).exists():
        return {
            "mode": "already_recorded",
            "dispatch": False,
            "experiment_id": experiment_id,
        }
    plan = prepare_runtime(
        root,
        artifacts,
        protocol_path=protocol_path,
        run_id=run_id,
        bundle_id=trial["bundle_id"],
        task_key=task_key,
        seed=seed,
        split="development",
        output_dir=output,
        authorization_path=permit_path,
        cuda_device=cuda_device,
        storage_write_roots=storage_write_roots,
    )
    for runtime_key, job_key in [
        ("wall_seconds", "whole_job_wall_seconds"),
        ("model_requests", "model_requests"),
        ("tool_calls", "tool_calls"),
        ("environment_steps", "environment_steps"),
        ("gpu_seconds", "gpu_device_seconds_envelope"),
        ("storage_bytes", "artifact_bytes"),
    ]:
        if plan["context"]["limits"][runtime_key] > min(
            experiment["cost_limit"].get(job_key, 0),
            tools.job.remaining().get(job_key, 0),
        ):
            raise ValueError(
                "trial exceeds experiment/job remaining budget: " + job_key
            )
    plan["experiment_id"] = experiment_id
    return plan


def _collect(
    tools,
    experiment_id,
    output,
    *,
    revision_id=None,
    derived_root=None,
    read_source_ref=None,
):
    from .adapters.runtime_acceptance import inspect_attempt
    from .critical_progress import inspect_progress
    from .development import import_development

    output = Path(output)
    audit = inspect_attempt(output, require_vla=True)
    audit_path = (
        Path(derived_root) / "independent-audit.json"
        if derived_root
        else output / "independent-audit.json"
    )
    atomic_json(audit_path, audit)
    result_prefix = experiment_id + ("-" + revision_id if revision_id else "")
    try:
        imported = import_development(
            tools.store,
            tools.protocol,
            output,
            result_id=result_prefix + "-result",
            revision_id=revision_id,
            audit_path=audit_path,
        )
    except (OSError, ValueError, KeyError) as exc:
        # Interruptions may lack states/transcript. Preserve unknown and measured costs, not a fabricated failure.
        from .contracts import ExperimentResult

        context = read(output / "runtime_context.json")
        raw = read(output / "independent_result.json")
        costs = dict(audit["costs"])
        detail = read(output / "cost_breakdown.json")
        launcher = read(output / "launcher_costs.json")
        for key, value in {
            "whole_job_wall_seconds": launcher.get("whole_job_wall_seconds"),
            "gpu_device_seconds_envelope": launcher.get("gpu_device_seconds_envelope"),
            "model_requests": detail.get("actual_model_requests"),
            "tool_calls": detail.get("tool_calls"),
            "environment_steps": detail.get("environment_steps"),
            "artifact_bytes": sum(
                p.stat().st_size
                for p in output.rglob("*")
                if p.is_file() and not p.is_symlink()
            ),
        }.items():
            if type(value) in (int, float):
                costs[key] = value
        task = context["task"]
        task_keys = [
            entry["task_key"]
            for entry in tools.protocol["tasks"]
            if (entry["suite"], entry["task_id"]) == (task["suite"], task["task"])
        ]
        split = tools.protocol["task_sets"]["development"]
        if (
            len(task_keys) != 1
            or task_keys[0] not in split["task_keys"]
            or task["seed"] not in split["seeds"]
        ):
            raise ValueError(
                "interrupted trial is outside its registered development split"
            )
        imported = ExperimentResult(
            experiment_id + "-interrupted",
            context["bundle_id"],
            tools.protocol["protocol_id"],
            tools.protocol["task_sets"]["development"].get(
                "benchmark_id", tools.protocol["benchmark_id"]
            ),
            "development",
            tools.protocol["scope"],
            {
                "information_scope": "development",
                "identity": {
                    k: context[k]
                    for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
                },
                "native_task_fact": raw.get("native_task_result"),
            },
            [
                str(output / "independent_result.json"),
                str(output / "cost_breakdown.json"),
            ],
            tools.protocol["protocol_id"],
            {},
            {},
            costs,
            assigned_count=1,
            unknown_count=1,
            invalid_count=1,
            outcome="unknown",
            assignments=[
                {
                    "task_key": task_keys[0],
                    "seed": task["seed"],
                    "outcome": "unknown",
                    "validity": "invalid",
                    "evidence_ref": str(output / "independent_result.json"),
                    "outcome_source": None,
                }
            ],
            notes=["Incomplete development evidence: " + str(exc)],
        )
    experiment = tools._owned("research_experiments", experiment_id)
    trial = tools._owned("rule_trials", experiment_id)
    events = (
        [
            json.loads(line)
            for line in (output / "action_events.jsonl").read_text().splitlines()
            if line
        ]
        if (output / "action_events.jsonl").exists()
        else []
    )
    effective = next(
        (e.get("effective_rules", {}) for e in events if e["kind"] == "run_start"), {}
    )
    loaded = all(effective.get(k) == v for k, v in experiment["intervention"].items())
    relation_consumption = None
    selected_relations = experiment["intervention"].get("relation_feedback_update")
    if selected_relations is not None:
        consumption_path = Path(derived_root) / "relation-consumption.json" if derived_root else None
        loaded = None
        if consumption_path is not None and consumption_path.is_file():
            relation_consumption = read(consumption_path)
            identity = imported.intervention["identity"]
            matches = (relation_consumption.get("schema") == "rsi-relation-consumption-v1"
                       and relation_consumption.get("identity") == identity
                       and relation_consumption.get("rho_ref") == trial["rho_ref"])
            entries = [relation_consumption.get("relations", {}).get(name, {}) for name in selected_relations]
            loaded = matches and all(entry.get("value") is True and entry.get("source_refs") for entry in entries)
    try:
        progress = inspect_progress(
            output, imported.intervention["identity"], read_source_ref=read_source_ref
        )
        if derived_root:
            atomic_json(Path(derived_root) / "critical-progress.json", progress)
    except (OSError, ValueError, KeyError):
        progress = {"value": None}
    collection = (
        read(output / "dataset_manifest.json")
        if (output / "dataset_manifest.json").exists()
        else {}
    )
    checks = {
        "native_success": imported.outcome == "success",
        "technical_valid": not imported.invalid_count and audit["passed"],
        "runtime_intervention_loaded": loaded,
        "critical_progress": progress["value"],
        "collection_complete": collection.get("alignment_complete"),
    }
    enriched = replace(
        imported,
        result_id=result_prefix + "-bound-result",
        intervention={
            **imported.intervention,
            **experiment["intervention"],
            "runtime_loaded": loaded,
            **({"relation_consumption": relation_consumption,
                "relation_consumption_ref": str(consumption_path) if consumption_path else None}
               if selected_relations is not None else {}),
            "screening_checks": checks,
            "screening_criterion": "development-native-valid-loaded-progress-data-v1",
            **(
                {"audit_revision_id": revision_id, "audit_ref": str(audit_path)}
                if revision_id
                else {}
            ),
        },
        criteria_ref=experiment["criterion_ref"],
        costs=imported.costs,
    )
    return enriched, checks


def _phase_charge(
    permit, run_id, costs, output, *, charge_kind="development_trial_charges"
):
    path = Path(permit["phase_snapshot_ref"])
    phase = read_phase(path)
    mapped = {
        "whole_job_wall_seconds": "wall_seconds",
        "gpu_device_seconds_envelope": "gpu_seconds",
        "model_requests": "model_requests",
        "tool_calls": "tool_calls",
        "environment_steps": "environment_steps",
        "artifact_bytes": "storage_bytes",
    }
    charge = {v: costs[k] for k, v in mapped.items() if k in costs}
    charge["simulation_episodes"] = 1
    entry = {
        "run_id": run_id,
        "costs": charge,
        "permit_ref": permit["permit_ref"],
        "output_ref": str(output),
    }
    entries = phase.setdefault(charge_kind, [])
    existing = [e for e in entries if e["run_id"] == run_id]
    if existing:
        if existing == [entry]:
            return
        if (
            len(existing) != 1
            or existing[0]["permit_ref"] != entry["permit_ref"]
            or existing[0]["output_ref"] != entry["output_ref"]
            or any(
                key not in charge or charge[key] != value
                for key, value in existing[0]["costs"].items()
            )
        ):
            raise ValueError("conflicting phase trial charge")
        # Reconcile only previously unknown dimensions from the same receipt.
        # Retain the original observation and append the newly measured delta.
        from math import isfinite

        added = {
            key: value
            for key, value in charge.items()
            if key not in existing[0]["costs"]
        }
        if any(
            type(value) not in (int, float) or not isfinite(value) or value < 0
            for value in added.values()
        ):
            raise ValueError("invalid reconciled trial cost")
        phase.setdefault("trial_cost_reconciliations", []).append(
            {
                "run_id": run_id,
                "previous_costs": dict(existing[0]["costs"]),
                "added_costs": added,
                "permit_ref": entry["permit_ref"],
                "output_ref": str(output),
            }
        )
        existing[0]["costs"].update(added)
        for key, value in added.items():
            phase["used"][key] += value
            phase["remaining"][key] = max(0, phase["limits"][key] - phase["used"][key])
        missing = sorted(
            key
            for key, dimension in mapped.items()
            if dimension not in existing[0]["costs"]
        )
        unresolved = phase.get("unresolved_costs", [])
        for item in unresolved:
            if item.get("run_id") == run_id and item.get("output_ref") == str(output):
                item["missing"] = missing
        phase["unresolved_costs"] = [
            item for item in unresolved if item.get("missing") != []
        ]
        write_phase(path, phase)
        return
    entries.append(entry)
    missing = sorted(set(mapped) - costs.keys())
    if missing:
        phase.setdefault("unresolved_costs", []).append(
            {"run_id": run_id, "missing": missing, "output_ref": str(output)}
        )
    for key, value in charge.items():
        phase["used"][key] += value
        phase["remaining"][key] = max(0, phase["limits"][key] - phase["used"][key])
    write_phase(path, phase)


def _settle_trial_output(
    tools, experiment_id, output, permit_path, context, *, cost_observer=None
):
    """Book a returned trial before importing or screening its development result.

    The immutable measurement survives a collector exception. Missing dimensions
    remain unresolved in the phase ledger; subsequent parsing cannot invent them.
    """
    from math import isfinite

    from .adapters.runtime_acceptance import inspect_attempt
    from .process_accounting import accounted_storage_bytes

    permit = read(permit_path)
    identity = {
        key: context[key]
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    receipt_path = tools.store.path("trial_cost_receipts", experiment_id)
    had_receipt = receipt_path.exists()
    if had_receipt:
        receipt = tools._owned("trial_cost_receipts", experiment_id)
        if (
            receipt["identity"] != identity
            or receipt["output_ref"] != str(Path(output).resolve())
            or receipt["permit_ref"] != str(Path(permit_path).resolve())
        ):
            raise ValueError("trial cost receipt identity mismatch")
    else:
        try:
            audit = (cost_observer or inspect_attempt)(output, require_vla=False)
            if audit.get("identity") != identity:
                raise ValueError("trial cost observation identity mismatch")
            costs = {
                key: value
                for key, value in audit.get("costs", {}).items()
                if type(value) in (int, float) and isfinite(value) and value >= 0
            }
        except (OSError, KeyError):
            costs = {}
        costs["artifact_bytes"] = max(
            costs.get("artifact_bytes", 0), accounted_storage_bytes(output)
        )
        receipt = tools._save(
            "trial_cost_receipts",
            experiment_id,
            {
                "identity": identity,
                "output_ref": str(Path(output).resolve()),
                "permit_ref": str(Path(permit_path).resolve()),
                "costs": costs,
            },
        )
    costs = dict(receipt["costs"])
    for observation in tools.store.all("trial_cost_additions"):
        if observation["experiment_id"] == experiment_id:
            if observation["identity"] != identity:
                raise ValueError("trial cost supplement identity mismatch")
            costs.update(observation["costs"])
    required = {
        "whole_job_wall_seconds",
        "gpu_device_seconds_envelope",
        "model_requests",
        "tool_calls",
        "environment_steps",
        "artifact_bytes",
    }
    missing = required - costs.keys()
    if had_receipt and missing:
        audit = (cost_observer or inspect_attempt)(output, require_vla=False)
        if audit.get("identity") != identity:
            raise ValueError("trial cost observation identity mismatch")
        added = {
            key: value
            for key, value in audit.get("costs", {}).items()
            if key in missing
            and type(value) in (int, float)
            and isfinite(value)
            and value >= 0
        }
        if added:
            tools._save(
                "trial_cost_additions",
                experiment_id + "-" + "-".join(sorted(added)),
                {
                    "experiment_id": experiment_id,
                    "identity": identity,
                    "costs": added,
                    "output_ref": str(Path(output).resolve()),
                    "permit_ref": str(Path(permit_path).resolve()),
                },
            )
            costs.update(added)
    _phase_charge(
        {**permit, "permit_ref": str(Path(permit_path).resolve())},
        context["run_id"],
        costs,
        output,
    )
    return costs


def _reconcile_trial(
    tools,
    experiment_id,
    *,
    output,
    permit_path,
    collector=None,
    auto_submit=True,
    settle_before_collect=False,
    cost_observer=None,
):
    """Reparse immutable output; never invoke Runtime or a model again."""
    key = experiment_id + "-cycle"
    if tools.store.path("research_cycles", key).exists():
        return tools._owned("research_cycles", key)
    experiment = tools._owned("research_experiments", experiment_id)
    permit = read(permit_path)
    if (
        permit.get("experiment_id") != experiment_id
        or permit.get("job_id") != tools.job.state["job_id"]
    ):
        raise ValueError("trial reconciliation belongs to another job")
    dispatch = tools._owned("trial_dispatches", experiment_id + "-dispatch")
    context = read(Path(output) / "runtime_context.json")
    if (
        Path(dispatch["output"]).resolve() != Path(output).resolve()
        or context.get("bundle_id") != experiment["bundle_id"]
        or context.get("protocol_id") != tools.protocol["protocol_id"]
        or permit.get("run_id") != context.get("run_id")
        or Path(permit["output_dir"]).resolve() != Path(output).resolve()
    ):
        raise ValueError("result output/identity differs from actual trial dispatch")
    settled_costs = (
        _settle_trial_output(
            tools,
            experiment_id,
            output,
            permit_path,
            context,
            cost_observer=cost_observer,
        )
        if settle_before_collect
        else None
    )
    if (
        settled_costs is not None
        and not {
            "whole_job_wall_seconds",
            "gpu_device_seconds_envelope",
            "model_requests",
            "tool_calls",
            "environment_steps",
            "artifact_bytes",
        }
        <= settled_costs.keys()
    ):
        raise RuntimeError(
            "trial actual costs remain unknown; reconcile original output before evaluation"
        )
    receipt_path = tools.store.path("trial_collections", experiment_id)
    if receipt_path.exists():
        from .contracts import ExperimentResult

        receipt = tools._owned("trial_collections", experiment_id)
        result, checks = ExperimentResult(**receipt["result"]), receipt["checks"]
    else:
        from dataclasses import asdict

        result, checks = (collector or _collect)(tools, experiment_id, output)
        if settled_costs is not None:
            hard_fields = {
                "whole_job_wall_seconds",
                "gpu_device_seconds_envelope",
                "model_requests",
                "tool_calls",
                "environment_steps",
                "artifact_bytes",
            }
            result = replace(
                result,
                costs={
                    **{
                        key: value
                        for key, value in result.costs.items()
                        if key not in hard_fields
                    },
                    **settled_costs,
                },
            )
        tools._save(
            "trial_collections",
            experiment_id,
            {"result": asdict(result), "checks": checks},
        )
    if result.bundle_id != experiment["bundle_id"]:
        raise ValueError("trial result belongs to another artifact")
    # File parsing does not infer natural-language predictions or improvement.
    try:
        finding = tools.record_experiment_result(experiment_id, result)
        checks["resource_accounting_complete"] = True
    except ValueError as exc:
        # Actual failed/over-budget attempts are still charged and retained; never mint a candidate.
        if checks.get("technical_valid") is True and result.outcome != "unknown":
            raise
        known = {
            k: v for k, v in result.costs.items() if k in tools.job.state["limits"]
        }
        reason = "incomplete runtime experiment " + experiment_id
        previous = [c for c in tools.job.state["charges"] if c["reason"] == reason]
        if previous and previous != [{"reason": reason, "costs": known}]:
            raise ValueError("conflicting incomplete attempt charge")
        if not previous:
            tools.job.consume(known, reason)
        tools.store.save_development_result(result)
        finding = tools._save(
            "research_findings",
            experiment_id,
            {
                "experiment_id": experiment_id,
                "result_id": result.result_id,
                "outcome": result.outcome,
                "costs": result.costs,
                "hard_costs": known,
                "resource_accounting_complete": False,
                "missing": sorted(set(experiment["cost_limit"]) - result.costs.keys()),
                "reason": str(exc),
                "hypotheses_resolved": False,
            },
        )
        checks["resource_accounting_complete"] = False
    _phase_charge(
        {**permit, "permit_ref": str(Path(permit_path).resolve())},
        read(Path(output) / "runtime_context.json")["run_id"],
        result.costs,
        output,
    )
    if all(value is True for value in checks.values()) and auto_submit:
        candidate = tools.submit_candidate(
            experiment_id + "-candidate", experiment_id=experiment_id
        )
        action = "candidate_for_independent_evaluation"
    elif all(value is True for value in checks.values()):
        candidate = None
        action = "development_complete_for_registered_candidate"
    else:
        candidate = None
        action = "stop"
        tools.defer(
            experiment_id + "-stop",
            opportunity_id=experiment["opportunity_id"],
            reason="registered deterministic development screening incomplete or unsatisfied",
            missing_evidence=[k for k, v in checks.items() if v is not True],
        )
    return tools._save(
        "research_cycles",
        key,
        {
            "experiment_id": experiment_id,
            "action": action,
            "result_id": result.result_id,
            "finding_ref": finding["experiment_id"],
            "candidate": candidate,
            "screening_checks": checks,
            "model_reanalysis": False,
            "hypotheses_resolved": False,
            "criterion": "development-native-valid-loaded-progress-data-v1",
            "evidence_level": result.evidence_level,
            "costs": result.costs,
        },
    )


def reconcile_trial(
    tools,
    experiment_id,
    *,
    output,
    permit_path,
    collector=None,
    auto_submit=True,
    settle_before_collect=False,
    cost_observer=None,
):
    from .artifact_accounting import begin, finish

    key = experiment_id + "-derived"
    begin(tools, key)
    permit = read(permit_path)
    from .process_accounting import snapshot

    billing_path = tools.store.path("trial_raw_output_billing", experiment_id)
    if not billing_path.exists():
        tools._save(
            "trial_raw_output_billing",
            experiment_id,
            {
                "output_ref": str(Path(output).resolve()),
                "raw_output_bytes_at_measurement": sum(snapshot([output]).values()),
            },
        )
    raw_billing = tools._owned("trial_raw_output_billing", experiment_id)
    if raw_billing["output_ref"] != str(Path(output).resolve()):
        raise ValueError("trial raw billing belongs to another output")
    try:
        return _reconcile_trial(
            tools,
            experiment_id,
            output=output,
            permit_path=permit_path,
            collector=collector,
            auto_submit=auto_submit,
            settle_before_collect=settle_before_collect,
            cost_observer=cost_observer,
        )
    finally:
        receipt = (
            tools.store.get("trial_collections", experiment_id)
            if tools.store.path("trial_collections", experiment_id).exists()
            else {}
        )
        settled = (
            tools.store.get("trial_cost_receipts", experiment_id)
            if tools.store.path("trial_cost_receipts", experiment_id).exists()
            else {}
        )
        billed = (
            raw_billing["raw_output_bytes_at_measurement"]
            if receipt.get("result", {}).get("costs", {}).get("artifact_bytes")
            is not None
            or settled.get("costs", {}).get("artifact_bytes") is not None
            else None
        )
        finish(
            tools,
            key,
            phase_path=permit["phase_snapshot_ref"],
            output=output,
            billed_output_bytes=billed,
        )


def execute_trial(
    tools,
    plan,
    *,
    permit_path,
    executor=None,
    collector=None,
    auto_submit=True,
    settle_before_collect=False,
    cost_observer=None,
):
    """Explicit single dispatch; production uses execute_runtime, tests use in-process substitutes."""
    experiment_id = plan["experiment_id"]
    trial = tools._owned("rule_trials", experiment_id)
    permit = read(permit_path)
    context = plan["context"]
    output = Path(context["output_dir"])
    expected = {
        "schema": "rsi-development-trial-permit-v1",
        "authorized": True,
        "experiment_id": experiment_id,
        "job_id": tools.job.state["job_id"],
        "bundle_id": trial["bundle_id"],
        "protocol_id": tools.protocol["protocol_id"],
        "intervention": tools._owned("research_experiments", experiment_id)[
            "intervention"
        ],
        "runtime_rules": read(trial["runtime_rules_ref"]),
        "output_dir": str(output.resolve()),
        "run_id": context["run_id"],
        "limits": context["limits"],
        "collection": context.get("collection"),
        "code_ref": context["code_ref"],
    }
    if any(permit.get(k) != v for k, v in expected.items()) or plan["blockers"]:
        raise PermissionError(
            "trial requires matching artifact/runtime/collection/code/budget permit"
        )
    if tools.store.path("research_cycles", experiment_id + "-cycle").exists():
        return reconcile_trial(
            tools,
            experiment_id,
            output=output,
            permit_path=permit_path,
            collector=collector,
            auto_submit=auto_submit,
            settle_before_collect=settle_before_collect,
            cost_observer=cost_observer,
        )
    from .artifact_accounting import begin

    begin(tools, experiment_id + "-derived")
    phase = read_phase(permit["phase_snapshot_ref"])
    mapping = {
        "wall_seconds": "wall_seconds",
        "gpu_seconds": "gpu_seconds",
        "model_requests": "model_requests",
        "tool_calls": "tool_calls",
        "environment_steps": "environment_steps",
        "storage_bytes": "storage_bytes",
    }
    if phase.get("unresolved_costs"):
        raise PermissionError(
            "phase has unresolved actual costs; reconcile before new dispatch"
        )
    if phase["remaining"]["simulation_episodes"] < 1 or any(
        context["limits"][k] > phase["remaining"][v] for k, v in mapping.items()
    ):
        raise PermissionError("original phase remaining budget insufficient")
    marker = experiment_id + "-dispatch"
    if not output.exists():
        if tools.store.path("trial_dispatches", marker).exists():
            raise RuntimeError(
                "trial was already dispatched without output; stop, never retry"
            )
        tools._save(
            "trial_dispatches",
            marker,
            {
                "experiment_id": experiment_id,
                "output": str(output),
                "status": "dispatched",
            },
        )
        try:
            (executor or execute_runtime)(plan)
        except BaseException:
            # Retain any independently closable output; never retry execution.
            if not (output / "independent_result.json").exists():
                raise
    elif not tools.store.path("trial_dispatches", marker).exists():
        raise FileExistsError("output belongs to another dispatch")
    return reconcile_trial(
        tools,
        experiment_id,
        output=output,
        permit_path=permit_path,
        collector=collector,
        auto_submit=auto_submit,
        settle_before_collect=settle_before_collect,
        cost_observer=cost_observer,
    )
