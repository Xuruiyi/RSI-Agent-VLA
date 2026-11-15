# Copyright 2026 RSI-Agent-VLA contributors.

"""Pre-registered finite-matrix scientific admission, separate from point Gate.

The exact test concerns paired exchangeability under no theta effect. Admission
also requires the observed finite-matrix engineering gain; it is not a lower
confidence bound that a population gain exceeds two percentage points.
"""

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .contracts import Candidate, ExperimentResult
from .evaluation import (
    EvaluationStore,
    apply_comparison_attempt_evidence,
    assignment_metrics,
    load_protocol,
)
from .midterm_evaluation import (
    evaluate_midterm_gate,
    inspect_midterm_bindings,
)
from .qualification_import import _readonly, require_role
from .runtime import resolve_reference
from .storage import Records, identifier

WEIGHTS = {"base_salad": 16, "position_swap_salad": 14, "old_milk": 5, "old_drawer": 5}
CELLS = ("theta0-D", "theta1-D", "theta1-S", "theta1-R", "theta1-A")
DESIGN = {
    "method": "paired_sign_flip_exact_v1",
    "alternative": "greater",
    "alpha": [1, 20],
    "useful_observed_gain": [1, 50],
    "anchor_max_drop": [1, 50],
    "denominator": 240,
    "integer_weights": WEIGHTS,
    "multiplicity": "fixed_sequence_theta_then_e1",
    "null": "within each task/init pair the two outcomes are exchangeable under no effect",
    "claim": "registered finite-matrix useful point gain with one-sided evidence of a positive effect; no population minimum-gain confidence claim",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("scientific registration needs timezone-aware timestamps")
    return result


def exact_weighted_paired(differences: list[int]) -> dict:
    """Inclusive exact upper tail by integer DP, with all zero pairs retained."""
    if any(type(value) is not int for value in differences):
        raise ValueError("exact paired statistic needs integer weighted differences")
    distribution = Counter({0: 1})
    for weight in (abs(value) for value in differences if value):
        updated = Counter()
        for total, count in distribution.items():
            updated[total - weight] += count
            updated[total + weight] += count
        distribution = updated
    observed = sum(differences)
    total = sum(distribution.values())
    tail = sum(count for value, count in distribution.items() if value >= observed)
    return {
        "statistic_numerator": observed,
        "p_numerator": tail,
        "p_denominator": total,
        "p_value": tail / total,
        "assigned_pairs": len(differences),
        "discordant_pairs": sum(value != 0 for value in differences),
        "distribution": [
            [value, count] for value, count in sorted(distribution.items())
        ],
    }


def make_block_schedule(
    cells: dict, task_keys: list[str], seeds: list[int], random_seed: int
) -> list[dict]:
    """Shuffle independent five-cell blocks before any formal outcome exists."""
    if (
        set(cells) != set(CELLS)
        or set(task_keys) != set(WEIGHTS)
        or len(seeds) != 6
        or len(set(seeds)) != 6
    ):
        raise ValueError(
            "formal schedule requires five cells and four tasks by six initial states"
        )
    rng = random.Random(random_seed)
    blocks = [(task, seed) for task in task_keys for seed in seeds]
    rng.shuffle(blocks)
    schedule = []
    for task, seed in blocks:
        order = list(CELLS)
        rng.shuffle(order)
        for cell in order:
            rows = [
                row
                for row in cells[cell]["manifest"]
                if row["task_key"] == task and row["seed"] == seed
            ]
            if len(rows) != 1:
                raise ValueError(
                    "registered cell manifest lacks a unique task/init slot"
                )
            schedule.append(
                {
                    "cell": cell,
                    "task_key": task,
                    "seed": seed,
                    "run_id": rows[0]["run_id"],
                }
            )
    if len({row["run_id"] for row in schedule}) != len(schedule):
        raise ValueError("formal run identities must be unique across all cells")
    return schedule


def select_theta0_controllers(development: list[dict]) -> dict:
    """Use frozen P2 rules; invalid/unknown rows remain recorded, never cheap wins.

    Each row is an independently closed development outcome with an arm,
    task_key, program_id, seed and complete all-attempt cost vector.
    """
    allowed = set(WEIGHTS)
    chosen, scores = {}, {}
    for arm in ("S", "R"):
        chosen[arm] = {}
        for task in WEIGHTS:
            rows = [r for r in development if r["arm"] == arm and r["task_key"] == task]
            expected = (
                {"v1", "v2"}
                if task in {"position_swap_salad", "old_drawer"}
                else {"v1"}
            )
            if (
                {r["program_id"] for r in rows} != expected
                or len(rows) != len(expected)
                or any(r["seed"] != 1 for r in rows)
            ):
                raise ValueError(
                    "P2 task/program development denominator differs from registration"
                )
            valid = [
                r
                for r in rows
                if r["validity"] == "valid" and r["outcome"] in {"success", "failure"}
            ]
            if not valid:
                return {
                    "verdict": "inconclusive",
                    "reason": "no valid frozen program for task",
                    "arm": arm,
                    "task_key": task,
                }
            chosen[arm][task] = min(
                valid,
                key=lambda r: (
                    r["outcome"] != "success",
                    r["costs"]["environment_steps"],
                    r["costs"]["model_requests"],
                    r["costs"]["whole_job_wall_seconds"],
                    r["program_id"],
                ),
            )
    for arm in ("D", "S", "R"):
        rows = (
            [r for r in development if r["arm"] == "D"]
            if arm == "D"
            else list(chosen[arm].values())
        )
        if {r["task_key"] for r in rows} != allowed:
            raise ValueError("simple selection must cover all four tasks")
        if arm == "D" and (
            {(r["task_key"], r["seed"]) for r in rows}
            != {(task, seed) for task in WEIGHTS for seed in (1, 2)}
            or len(rows) != 8
        ):
            raise ValueError(
                "D development selection retains its two-seed task denominator"
            )
        if any(
            r["validity"] != "valid" or r["outcome"] not in {"success", "failure"}
            for r in rows
        ):
            return {
                "verdict": "inconclusive",
                "reason": "simple arm development incomplete",
                "arm": arm,
            }
        for row in rows:
            if any(
                type(row["costs"].get(k)) not in (int, float)
                or not math.isfinite(row["costs"][k])
                or row["costs"][k] < 0
                for k in (
                    "environment_steps",
                    "model_requests",
                    "whole_job_wall_seconds",
                )
            ):
                raise ValueError(
                    "simple selection requires actual complete cost vectors"
                )
        macro = (
            sum(
                Fraction(
                    sum(
                        r["outcome"] == "success" for r in rows if r["task_key"] == task
                    ),
                    sum(r["task_key"] == task for r in rows),
                )
                for task in WEIGHTS
            )
            / 4
        )
        scores[arm] = {
            "macro_success_numerator": macro.numerator,
            "macro_success_denominator": macro.denominator,
            "model_requests": sum(r["costs"]["model_requests"] for r in rows),
            "mean_wall_seconds": sum(r["costs"]["whole_job_wall_seconds"] for r in rows)
            / len(rows),
        }
    best = min(
        scores,
        key=lambda arm: (
            -Fraction(
                scores[arm]["macro_success_numerator"],
                scores[arm]["macro_success_denominator"],
            ),
            scores[arm]["model_requests"],
            scores[arm]["mean_wall_seconds"],
            ("D", "S", "R").index(arm),
        ),
    )
    return {
        "verdict": "selected",
        "simple_arm": best,
        "selected_programs": {
            a: {t: r["program_id"] for t, r in tasks.items()}
            for a, tasks in chosen.items()
        },
        "selected_program_specs": {
            a: {t: r["program_spec"] for t, r in tasks.items()}
            for a, tasks in chosen.items()
        },
        "scores": scores,
        "all_development_rows": development,
    }


def development_rows(reference: str | Path) -> list[dict]:
    """Read native development outcomes and audit costs, never submitted labels."""
    rows = []
    used = set()
    for item in _read(reference)["runs"]:
        native, audit = _read(item["result_ref"]), _read(item["audit_ref"])
        context = _read(Path(item["run_root"]) / "runtime_context.json")
        protocol = load_protocol(item["protocol_ref"])
        identity = {
            key: context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        if (
            context.get("information_scope") != "development"
            or context["protocol_id"] != protocol["protocol_id"]
            or context.get("execution_arm", {}).get("arm") != item["arm"]
            or any(
                native.get(k) != v or audit.get("identity", {}).get(k) != v
                for k, v in identity.items()
            )
            or native["task"]["seed"] != item["seed"]
            or item["task_key"] not in protocol["task_sets"]["development"]["task_keys"]
            or item["seed"] not in protocol["task_sets"]["development"]["seeds"]
        ):
            raise ValueError(
                "simple selection source is not exact legal development evidence"
            )
        if context["run_id"] in used:
            raise ValueError(
                "a development run cannot be counted as two program trials"
            )
        used.add(context["run_id"])
        audits = [audit] + [
            _read(ref) for ref in item.get("related_attempt_audit_refs", [])
        ]
        costs = {
            key: sum(doc["costs"][key] for doc in audits)
            for key in ("environment_steps", "model_requests", "whole_job_wall_seconds")
        }
        valid = (
            audit.get("passed") is True
            and native.get("validity") == "valid"
            and native.get("attempt_closed") is True
            and type(native.get("benchmark_success")) is bool
        )
        rows.append(
            {
                "arm": item["arm"],
                "program_id": item["program_id"],
                "task_key": item["task_key"],
                "seed": item["seed"],
                "outcome": "success"
                if valid and native["benchmark_success"]
                else "failure"
                if valid
                else "unknown",
                "validity": "valid" if valid else "invalid",
                "costs": costs,
                "program_spec": context["execution_arm"],
                "source_refs": {
                    key: item[key]
                    for key in ("result_ref", "audit_ref", "run_root", "protocol_ref")
                },
            }
        )
    return rows


def _registration(reference):
    path = Path(reference).resolve(strict=True)
    registration = _read(path)
    protocol = load_protocol(registration["protocol_ref"])
    split = protocol["task_sets"]["release"]
    if (
        registration.get("schema") != "rsi-midterm-science-registration-v1"
        or registration.get("design") != DESIGN
        or registration.get("protocol_snapshot") != protocol
        or protocol.get("scope_id") != registration["scope"]
        or protocol["scope"] != "simulation"
        or protocol.get("min_gain") != 0.02
        or protocol.get("anchor_max_drop") != 0.02
        or protocol.get("cost_weights", {})
        or set(split["task_keys"]) != set(WEIGHTS)
        or len(split["seeds"]) != 6
        or len(set(split["seeds"])) != 6
        or set(registration["cells"]) != set(CELLS)
        or not path.is_relative_to(
            Path(registration["role_domains"]["operator"]).resolve()
        )
        or registration["schedule"]
        != make_block_schedule(
            registration["cells"],
            split["task_keys"],
            split["seeds"],
            registration["schedule_seed"],
        )
    ):
        raise ValueError("scientific registration or fixed paired design mismatch")
    if {
        key: Fraction(str(value["weight"]))
        for key, value in protocol["metrics"].items()
    } != {
        "success_base": Fraction(2, 5),
        "success_controlled": Fraction(7, 20),
        "success_old": Fraction(1, 4),
    }:
        raise ValueError(
            "exact integer utility weights differ from the frozen J criterion"
        )
    for cell in registration["cells"].values():
        for original in cell["manifest"]:
            for row in (original, original.get("technical_recovery")):
                if row is not None and any(
                    not row.get(key)
                    for key in (
                        "run_id",
                        "result_path",
                        "audit_ref",
                        "host_completion_ref",
                        "attempt_receipt_ref",
                    )
                ):
                    raise ValueError(
                        "formal rows require frozen Host/native/independent attempt references"
                    )
    selection = select_theta0_controllers(
        development_rows(registration["development_selection_ref"])
    )
    if (
        selection["verdict"] != "selected"
        or selection != registration["simple_selection_snapshot"]
    ):
        raise ValueError(
            "E1 simple baseline is not the registered theta0 development selection"
        )
    return path, registration, protocol, selection["simple_arm"]


def training_sources(
    registration: dict, theta_ref: str, processor_ref: str, baseline_theta_ref: str
) -> dict:
    """Consume the actual native CLI's start/update/save/load and host receipts."""
    source = registration["training"]
    recipe, manifest = _read(source["recipe_ref"]), _read(source["manifest_ref"])
    descriptor = resolve_reference(theta_ref)
    initial_descriptor = resolve_reference(baseline_theta_ref)
    initial_path = Path(
        initial_descriptor.get("model_path", initial_descriptor.get("checkpoint", ""))
    )
    if not initial_path.is_absolute():
        initial_path = Path(registration["repository_root"]) / initial_path
    if Path(recipe["actor"]["model"]["model_path"]).resolve() != initial_path.resolve():
        raise ValueError(
            "native training did not start from the registered theta0 asset"
        )
    scope = _read(registration["scope_authorization_ref"])
    records = {}
    for kind, key in (
        ("pi05-training-learning", "training"),
        ("pi05-load-check", "load"),
    ):
        job = _read(source[key + "_job_ref"])
        permit = _read(source[key + "_permit_ref"])
        supervision, start = (
            _read(job["supervision_ref"]),
            _read(job["native_start_ref"]),
        )
        reservation = _read(permit["accounting_ref"])["reservations"][
            permit["operation_id"]
        ]
        if (
            job.get("schema") != "rsi-native-training-job-result-v1"
            or job.get("status") != "completed"
            or job.get("kind") != kind
            or job.get("scope_id") != registration["scope"]
            or permit.get("authorized") is not True
            or permit.get("kind") != kind
            or Path(permit["scope_ref"]).resolve()
            != Path(registration["scope_authorization_ref"]).resolve()
            or start.get("schema") != "rsi-native-training-start-v1"
            or any(
                job.get(k) != permit.get(k) or start.get(k) != permit.get(k)
                for k in ("scope_id", "operation_id", "run_id")
            )
            or supervision.get("returncode") != 0
            or supervision.get("stop_reason") is not None
            or supervision.get("cleanup_confirmed") is not True
            or supervision.get("resource_limits_satisfied") is not True
            or reservation.get("status") != "settled"
            or reservation.get("cleanup_confirmed") is not True
            or reservation.get("missing")
            or not reservation.get("submitted_at")
            or reservation.get("scope_id") != registration["scope"]
            or reservation.get("operation_id") != permit["operation_id"]
            or reservation.get("actual", {}).get(
                "training_starts" if key == "training" else "load_checks"
            )
            != 1
            or reservation.get("actual", {}).get("gpu_seconds", 0) <= 0
            or _time(start["started_at"]) < _time(scope["accepted_at"])
            or _time(reservation["submitted_at"]) > _time(start["started_at"])
            or _time(start["started_at"])
            >= min(_time(permit["expires_at"]), _time(scope["expires_at"]))
        ):
            raise ValueError(
                "native training/loading or live host settlement evidence is incomplete"
            )
        records[key] = {
            "job": job,
            "permit": permit,
            "supervision": supervision,
            "native_start": start,
            "reservation": reservation,
        }
    job = records["training"]["job"]
    root = Path(source["training_job_ref"]).parent
    update, census, optimizer = (
        _read(root / name)
        for name in (
            "weight-update-check.json",
            "parameter-census.json",
            "optimizer-role.json",
        )
    )
    load = _read(records["load"]["job"]["handoff_ref"])
    check = _read(load["model_descriptor"]["actual_load_evidence_ref"])
    steps = recipe["runner"]["max_steps"]
    expected_checkpoint = root / "checkpoints" / f"global_step_{steps}"
    trained = _read(job["handoff_ref"])["model_descriptor"]
    if (
        records["load"]["job"]["run_id"] != job["run_id"]
        or Path(descriptor["model_path"]).resolve() != expected_checkpoint.resolve()
        or Path(load["selected_weights"]).resolve()
        != (expected_checkpoint / "actor/model_state_dict/full_weights.pt").resolve()
        or descriptor.get("training_recipe_ref") != source["recipe_ref"]
        or descriptor.get("weight_update_ref") != str(root / "weight-update-check.json")
        or any(
            trained.get(key) != descriptor.get(key)
            for key in (
                "model_path",
                "norm_stats_path",
                "training_recipe_ref",
                "weight_update_ref",
            )
        )
    ):
        raise ValueError(
            "saved trained checkpoint and actual native load do not share the same producer lineage"
        )
    if (
        job["run_id"] != recipe["runner"]["logger"]["experiment_name"]
        or recipe["rsi_training"].get("mode") != "learning"
        or records["training"]["permit"]["budget"]["max_steps"] != steps
        or update.get("step") != steps
        or steps < 1
        or update.get("scope") != "action_expert_only"
        or update.get("nonzero_sampled_weight_update") is not True
        or not update.get("probes")
        or not any(
            p.get("changed_elements", 0) > 0 and p.get("max_abs_difference", 0) > 0
            for p in update["probes"].values()
        )
        or not update.get("sampled_frozen_parameters_equal")
        or not all(update["sampled_frozen_parameters_equal"].values())
        or not math.isfinite(update["metrics"]["loss"])
        or not math.isfinite(update["metrics"]["grad_norm"])
        or update["metrics"]["grad_norm"] < 0
        or census.get("scope") != "action_expert_only"
        or census.get("trainable_elements", 0) <= 0
        or census.get("frozen_elements", 0) <= 0
        or census["trainable_elements"] != update["trainable_elements"]
        or optimizer.get("class") != "FP32MasterAdamW"
        or recipe["data"]["train_data_paths"] != manifest["dataset_root"]
        or load.get("actual_model_load") != "passed"
        or load["model_descriptor"] != descriptor
        or check.get("prediction_shape") != [1, 5, 7]
        or check.get("prediction_finite") is not True
        or not check.get("saved_probe_equal")
        or not all(check["saved_probe_equal"].values())
        or Path(check["loader_config"]["model_path"]).resolve()
        != Path(descriptor["model_path"]).resolve()
    ):
        raise ValueError(
            "P3 actual update/freeze/optimizer/dataset/native-loader bindings do not establish theta1"
        )
    from .adapters.pi05_training import frozen_vlm_key, require_training_data_admission

    data_admission = require_training_data_admission(
        manifest,
        admission_ref=records["training"]["permit"]["data_admission_ref"],
        scope_id=registration["scope"],
        protocol_id=records["training"]["permit"]["protocol_id"],
        authority=records["training"]["permit"]["data_admission_authority"],
    )
    if (
        any(
            row.get("trainable") is not False
            for key, row in census.get("parameters", {}).items()
            if frozen_vlm_key(key)
        )
        or not census.get("parameters")
        or recipe["rsi_training"]["learning_plan"]["data_admission_ref"]
        != records["training"]["permit"]["data_admission_ref"]
        or not Path(load["selected_weights"]).is_file()
    ):
        raise ValueError(
            "P3 frozen census/admitted data/final checkpoint evidence is incomplete"
        )
    resources = [
        json.loads(line)
        for line in (root / "resources-rank0.jsonl").read_text().splitlines()
    ]
    if sum(row.get("phase") == "native_optimizer_step" for row in resources) != steps:
        raise ValueError(
            "native optimizer resource trace does not cover every actual update step"
        )
    normalization = (
        resolve_reference(processor_ref) if "#" in processor_ref else processor_ref
    )
    normalization = Path(normalization)
    if not normalization.is_absolute():
        normalization = Path(descriptor["model_path"]) / normalization
    if (
        Path(check["loader_config"]["openpi_data"]["norm_stats_path"]).resolve()
        != normalization.resolve()
    ):
        raise ValueError("loaded processor differs from formal theta1 processor")
    prediction = check.get("prediction")
    if (
        check.get("prediction_count") != 1
        or not isinstance(prediction, list)
        or len(prediction) != 1
        or len(prediction[0]) != 5
        or any(
            len(row) != 7
            or not all(type(v) in (int, float) and math.isfinite(v) for v in row)
            for row in prediction[0]
        )
    ):
        raise ValueError(
            "actual decoded prediction does not match the claimed native load probe"
        )
    return {
        "records": records,
        "update": update,
        "census": census,
        "optimizer": optimizer,
        "load": load,
        "load_check": check,
        "recipe": recipe,
        "manifest": manifest,
        "resources": resources,
        "data_admission": data_admission,
    }


def _registered_attempt(registration, cell_name, run_id):
    cell = registration["cells"][cell_name]
    for original in cell["manifest"]:
        for row in (original, original.get("technical_recovery")):
            if row and row["run_id"] == run_id:
                return {**original, **row}, original
    raise ValueError("attempt is outside the frozen formal cell registration")


def inspect_scientific_attempt(
    registration_ref: str | Path, cell_name: str, run_id: str
) -> dict:
    """Classify one actual closed attempt from Host, native audit and progression.

    This CPU consumer reads saved evidence only. It does not dispatch Runtime or
    authorize recovery. Unknown native outcomes caused by hard budget stops are
    ineligible for technical recovery.
    """
    _, registration, _, _ = _registration(registration_ref)
    row, _ = _registered_attempt(registration, cell_name, run_id)
    return _inspect_attempt(registration, row)


def _inspect_attempt(registration, row):
    from .critical_progress import inspect_progress

    root = Path(row["result_path"]).parent.resolve(strict=True)
    context = _read(root / "runtime_context.json")
    identity = {
        key: context[key]
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    audit = _read(row["audit_ref"])
    permit = _read(context["authorization_ref"])
    completion_path = Path(row["host_completion_ref"]).resolve(strict=True)
    completion = _read(completion_path)
    submission = _read(completion_path.parent / "submission.json")
    operation = submission["operation"]
    operation_id = operation["operation_id"]
    ended = completion.get("schema") == "rsi-host-ended-operation-receipt-v1"
    closure = completion["value"] if ended else completion
    supervisor = _read(closure["supervisor_ref"])
    scope = _read(registration["scope_authorization_ref"])
    ledger_path = Path(permit["accounting_ref"]).resolve(strict=True)
    if not ledger_path.is_relative_to(Path(scope["write_root"]).resolve()):
        raise ValueError("attempt ledger is outside its actual task authority")
    reservation = _read(ledger_path)["reservations"][operation_id]
    actual = reservation.get("actual", {})
    if (
        identity["run_id"] != row["run_id"]
        or audit.get("identity") != identity
        or operation.get("kind") != "runtime"
        or operation.get("payload", {}).get("run_id") != row["run_id"]
        or Path(operation["payload"]["output_ref"]).resolve() != root
        or permit.get("scope") != registration["scope"]
        or Path(permit["approval_reference"]).resolve()
        != Path(registration["scope_authorization_ref"]).resolve()
        or permit.get("run_id") != row["run_id"]
        or permit.get("operation_id") != operation_id
        or permit.get("authorized") is not True
        or completion.get("scope_id") != registration["scope"]
        or (not ended and completion.get("operation_id") != operation_id)
        or (ended and completion.get("operation") != operation)
        or supervisor.get("identity", {}).get("run_id") != operation_id
        or reservation.get("scope_id") != registration["scope"]
        or reservation.get("operation_id") != operation_id
        or Path(reservation["receipt_ref"]).resolve()
        != (completion_path.parent / "completion.json" if ended else completion_path)
    ):
        raise ValueError("attempt Host/context/permit/ledger identity mismatch")
    completed_normally = (
        supervisor.get("status") == "completed"
        and supervisor.get("exit_code") == 0
        and supervisor.get("cleanup_confirmed") is True
    )
    if (
        not ended
        and completion.get("status")
        != ("completed" if completed_normally else "unknown")
        or ended
        and (
            completion.get("status") != "completed"
            or supervisor.get("status") != "interrupted"
            or closure.get("execution_ended") is not True
        )
    ):
        raise ValueError("attempt Host completion contradicts its actual supervisor")
    closed = (
        reservation.get("status") == "settled"
        and not reservation.get("missing")
        and reservation.get("cleanup_confirmed") is True
        and closure.get("cleanup_confirmed", supervisor.get("cleanup_confirmed"))
        is True
        and supervisor.get("cleanup_confirmed") is True
        and supervisor.get("remaining_process_ids") == []
        and reservation.get("submitted_at") is not None
        and closure.get("actual") == actual
        and actual.get("runtime_starts") == 1
        and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in actual.values()
        )
    )
    if not closed:
        raise ValueError(
            "attempt costs or owner cleanup are not closed by the actual Host"
        )
    progress = inspect_progress(root, identity)
    if progress["value"] is None and audit.get("passed") is True:
        raise ValueError(
            "passed Runtime audit still lacks independent critical progression"
        )
    valid = (
        audit.get("passed") is True
        and progress["value"] is True
        and audit.get("outcome") in {"success", "failure"}
    )
    budget_stop = (
        supervisor.get("stop_reason")
        in {
            "storage_bytes",
            "wall_seconds",
            "gpu_seconds_envelope",
            "model_requests",
            "llm_input_tokens",
            "llm_output_tokens",
        }
        or ended
        and closure.get("termination_class") == "budget_stop"
    )
    valid = valid and not budget_stop
    failure_class = (
        None
        if valid
        else "governance"
        if (
            budget_stop
            or progress["value"] is False
            or audit.get("confirmed_violations")
        )
        else "technical"
    )
    return {
        "schema": "rsi-midterm-attempt-evaluation-v1",
        "scope_id": registration["scope"],
        "run_id": row["run_id"],
        "identity": identity,
        "validity": "valid" if valid else "invalid",
        "outcome": audit.get("outcome", "unknown"),
        "failure_class": failure_class,
        "critical_progress": progress["value"],
        "cost_closed": True,
        "cleanup_complete": True,
        "technical_recovery_authorized": False,
        "sources": {
            "runtime_context": context,
            "audit": audit,
            "progress": progress,
            "permit": permit,
            "completion": completion,
            "submission": submission,
            "supervisor": supervisor,
            "reservation": reservation,
        },
        "costs": {"native": audit.get("costs", {}), "whole_command": actual},
        "source_refs": {
            "run_root": str(root),
            "audit_ref": row["audit_ref"],
            "host_completion_ref": str(completion_path),
            "accounting_ref": str(ledger_path),
        },
    }


def write_scientific_attempt(
    registration_ref: str | Path, cell_name: str, run_id: str
) -> tuple[str, dict]:
    """Write one independently evaluated attempt in the current task namespace."""
    _, registration, _, _ = _registration(registration_ref)
    require_role(registration, "evaluation")
    row, _ = _registered_attempt(registration, cell_name, run_id)
    store = Records(registration["evaluation_store_ref"])
    target = store.path("attempts", identifier(run_id)).resolve()
    if Path(row["attempt_receipt_ref"]).resolve() != target:
        raise ValueError(
            "attempt receipt differs from the frozen independent output path"
        )
    receipt = _inspect_attempt(registration, row)
    store.put("attempts", run_id, receipt)
    return str(target), receipt


def _current_attempt(registration, row):
    path = Path(row["attempt_receipt_ref"]).resolve(strict=True)
    expected = (
        Records(registration["evaluation_store_ref"])
        .path("attempts", row["run_id"])
        .resolve()
    )
    if path != expected:
        raise ValueError("formal attempt must use its independent evaluation receipt")
    receipt = _read(path)
    if receipt != _inspect_attempt(registration, row):
        raise ValueError(
            "formal attempt receipt is stale against actual current evidence"
        )
    return receipt


def _selected_formal_manifest(registration, cell):
    """Select the first valid attempt; retain all costs and every fixed slot."""
    selected, histories = [], []
    for original in cell["manifest"]:
        row = original
        attempts = []
        initial = (
            _current_attempt(registration, original)
            if Path(original["attempt_receipt_ref"]).exists()
            else None
        )
        if initial:
            attempts.append(initial)
        recovery = original.get("technical_recovery")
        if (
            recovery
            and (Path(recovery["result_path"]).parent / "runtime_context.json").exists()
        ):
            if (
                not initial
                or initial["validity"] == "valid"
                or initial["failure_class"] != "technical"
                or initial["cost_closed"] is not True
                or initial["cleanup_complete"] is not True
            ):
                raise ValueError(
                    "recovery cannot replace a valid outcome or a nontechnical/unclosed fault"
                )
            recovery = {**original, **recovery}
            basis = _read(recovery["recovery_basis_ref"])
            cause_ref = Path(basis["cause_resolution_ref"]).resolve(strict=True)
            cause = _read(cause_ref)
            operator = Path(registration["role_domains"]["operator"]).resolve()
            if (
                not Path(recovery["recovery_basis_ref"])
                .resolve()
                .is_relative_to(operator)
                or not cause_ref.is_relative_to(operator)
                or any(
                    basis.get(key) != initial.get(key)
                    for key in (
                        "run_id",
                        "validity",
                        "failure_class",
                        "cost_closed",
                        "cleanup_complete",
                    )
                )
                or Path(basis["attempt_receipt_ref"]).resolve()
                != Path(original["attempt_receipt_ref"]).resolve()
                or basis.get("recovery_cause_resolved") is not True
                or cause.get("scope_id") != registration["scope"]
                or cause.get("run_id") != original["run_id"]
                or cause.get("resolved") is not True
                or not cause.get("repair_ref")
                or not Path(cause["repair_ref"])
                .resolve(strict=True)
                .is_relative_to(operator)
            ):
                raise ValueError(
                    "recovery needs the actual independent technical fault and protected concrete cause repair"
                )
            recovered = _current_attempt(registration, recovery)
            attempts.append(recovered)
            row = recovery
        selected.append(row)
        histories.append(
            {
                "slot_run_id": original["run_id"],
                "selected_run_id": row["run_id"],
                "attempts": attempts,
            }
        )
    return selected, histories


def _cell(registration, protocol, name, cutoff):
    cell = registration["cells"][name]
    evaluation = EvaluationStore(registration["evaluation_store_ref"])
    result_ref = evaluation.path("results", cell["result_id"])
    result = ExperimentResult(**evaluation.get("results", cell["result_id"]))
    assignment_metrics(protocol, result)
    if (
        result.bundle_id != cell["bundle_id"]
        or result.protocol_id != protocol["protocol_id"]
        or result.benchmark_id != protocol["benchmark_id"]
        or result.role != "release"
        or result.evidence_level != "simulation"
        or resolve_reference(cell["profile_ref"]) != cell["profile_snapshot"]
    ):
        raise ValueError(
            "scientific result/current profile identity differs from registration"
        )
    expected = {
        (task, seed)
        for task in protocol["task_sets"]["release"]["task_keys"]
        for seed in protocol["task_sets"]["release"]["seeds"]
    }
    if (
        len(cell["manifest"]) != 24
        or {(r["task_key"], r["seed"]) for r in cell["manifest"]} != expected
    ):
        raise ValueError(
            "scientific cell does not retain its complete 24-slot denominator"
        )
    selected, attempt_histories = _selected_formal_manifest(registration, cell)
    loaded = inspect_midterm_bindings(cell["profile_snapshot"], selected)
    plan = {"slots": [], "final_attempt_violations": []}
    for row, history in zip(cell["manifest"], attempt_histories):
        attempts = []
        for attempt in history["attempts"]:
            costs = dict(attempt["costs"]["native"])
            for outer, native in (
                ("gpu_seconds", "gpu_device_seconds_envelope"),
                ("storage_bytes", "artifact_bytes"),
                ("wall_seconds", "whole_job_wall_seconds"),
            ):
                if outer in attempt["costs"]["whole_command"]:
                    costs[native] = max(
                        costs.get(native, 0), attempt["costs"]["whole_command"][outer]
                    )
            attempts.append({"costs": costs})
            violations = list(
                attempt["sources"]["audit"].get("confirmed_violations", [])
            )
            if attempt["critical_progress"] is False:
                violations.append("critical_progress")
            plan["final_attempt_violations"].extend(
                {"side": name, "constraint": key} for key in violations
            )
        plan["slots"].append(
            {"side": name, "task_key": row["task_key"], "attempts": attempts}
        )
    apply_comparison_attempt_evidence(plan, name, result)
    theta_id, arm = name.split("-")
    if loaded["cell"]["theta_id"] != theta_id or loaded["cell"]["arm"] != arm:
        raise ValueError(
            "scientific theta/arm registration differs from actual model binding"
        )
    snapshots, starts = [], []
    by_pair = {(r["task_key"], r["seed"]): r for r in result.assignments}
    for row in selected:
        pair = by_pair[(row["task_key"], row["seed"])]
        root = Path(row["result_path"]).parent
        try:
            context = _read(root / "runtime_context.json")
            events = (root / "action_events.jsonl").read_text().splitlines()
            first = json.loads(events[0])
        except (OSError, ValueError, IndexError) as exc:
            snapshots.append(
                {
                    "pair": [row["task_key"], row["seed"]],
                    "status": "unavailable",
                    "reason": str(exc),
                }
            )
            continue
        identity = {
            k: context[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        if (
            any(first.get(k) != v for k, v in identity.items())
            or first.get("kind") != "run_start"
            or context["run_id"] != row["run_id"]
            or context["bundle_id"] != cell["bundle_id"]
            or datetime.fromtimestamp(first["utc_ns"] / 1e9, timezone.utc) <= cutoff
        ):
            raise ValueError(
                "scientific evidence predates registration or has a mismatched run identity"
            )
        budget = protocol["budget"]["per_attempt_proposal"]
        if any(
            context.get("limits", {}).get(key) != budget[field]
            for key, field in (
                ("environment_steps", "max_episode_steps"),
                ("wall_seconds", "full_job_wall_seconds"),
                ("tool_calls", "tool_calls"),
            )
        ):
            raise ValueError(
                "scientific Runtime consumed a different physical/time/tool bound"
            )
        starts.append({"run_id": row["run_id"], "utc_ns": first["utc_ns"]})
        audit_path = Path(row["audit_ref"]).resolve(strict=True)
        if (
            not audit_path.is_relative_to(
                Path(registration["evaluation_store_ref"]).resolve()
            )
            or str(audit_path) not in result.data_refs
        ):
            raise ValueError(
                "scientific native audit is not an independently referenced current receipt"
            )
        audit = _read(audit_path)
        loader = _read(root / "model_load.json")
        descriptor = resolve_reference(loaded["cell"]["theta_ref"])
        model_path = Path(
            descriptor.get("model_path", descriptor.get("checkpoint", ""))
        )
        if not model_path.is_absolute():
            model_path = Path(registration["repository_root"]) / model_path
        processor = loaded["cell"]["processor_ref"]
        norm_path = Path(
            resolve_reference(processor) if "#" in processor else processor
        )
        if not norm_path.is_absolute():
            norm_path = model_path / norm_path
        if (
            Path(loader["model_path"]).resolve() != model_path.resolve()
            or Path(loader["norm_stats_path"]).resolve() != norm_path.resolve()
        ):
            raise ValueError(
                "actual runtime loaded a different theta or processor path"
            )
        native = _read(pair["evidence_ref"]) if pair.get("evidence_ref") else None
        known = (
            pair["outcome"] in {"success", "failure"} and pair["validity"] == "valid"
        )
        if known and (
            not native
            or any(native.get(k) != v for k, v in identity.items())
            or type(native.get("benchmark_success")) is not bool
            or native["benchmark_success"] != (pair["outcome"] == "success")
            or Path(pair["evidence_ref"]).resolve()
            != Path(row["result_path"]).resolve()
            or audit.get("outcome") != pair["outcome"]
            or native.get("attempt_closed") is not True
            or native.get("validity") != "valid"
            or any(audit.get("identity", {}).get(k) != v for k, v in identity.items())
        ):
            raise ValueError(
                "scientific assignment conflicts with actual native closure/audit evidence"
            )
        snapshots.append(
            {
                "pair": [row["task_key"], row["seed"]],
                "native_ref": pair.get("evidence_ref"),
                "native": native,
                "audit_ref": str(audit_path),
                "audit": audit,
                "model_load": loader,
            }
        )
    return result, {
        "result_ref": str(result_ref.resolve()),
        "binding": loaded,
        "snapshots": snapshots,
        "starts": starts,
        "attempt_histories": attempt_histories,
        "all_attempt_costs": dict(result.costs),
        "attempt_evaluations_complete": all(
            history["attempts"] and history["attempts"][-1]["validity"] == "valid"
            for history in attempt_histories
        ),
    }


def _registered_starts(registration, cutoff):
    """Keep original randomized order and insert at most one immediate recovery."""
    observed, recoveries = [], []
    ranks = {row["run_id"]: index for index, row in enumerate(registration["schedule"])}
    all_ids = set(ranks)
    for cell in registration["cells"].values():
        for original in cell["manifest"]:
            recovery = original.get("technical_recovery")
            if recovery:
                if recovery["run_id"] in all_ids:
                    raise ValueError("technical recovery identity is not unique")
                all_ids.add(recovery["run_id"])
            for number, row in enumerate((original, recovery)):
                if not row:
                    continue
                root = Path(row["result_path"]).parent
                if not (root / "runtime_context.json").exists():
                    continue
                context = _read(root / "runtime_context.json")
                try:
                    first = json.loads(
                        (root / "action_events.jsonl").read_text().splitlines()[0]
                    )
                except (OSError, ValueError, IndexError):
                    continue
                if (
                    context["run_id"] != row["run_id"]
                    or first.get("kind") != "run_start"
                    or any(
                        first.get(key) != context[key]
                        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                    )
                    or datetime.fromtimestamp(first["utc_ns"] / 1e9, timezone.utc)
                    <= cutoff
                ):
                    raise ValueError(
                        "registered run_start has another identity or predates formal registration"
                    )
                item = {
                    "run_id": row["run_id"],
                    "slot_run_id": original["run_id"],
                    "utc_ns": first["utc_ns"],
                    "recovery": bool(number),
                    "order": [ranks[original["run_id"]], number],
                }
                observed.append(item)
                if number:
                    recoveries.append(item)
    limit = _read(registration["scope_authorization_ref"]).get(
        "runtime_technical_recovery_max", 0
    )
    if len(recoveries) > min(8, limit):
        raise ValueError(
            "formal technical recoveries exceed the actual scope/global cap"
        )
    ordered = sorted(observed, key=lambda row: row["utc_ns"])
    if len({row["utc_ns"] for row in ordered}) != len(ordered) or [
        row["order"] for row in ordered
    ] != sorted(row["order"] for row in ordered):
        raise ValueError(
            "actual run_start order did not consume the pre-registered randomized schedule with immediate slot recovery"
        )
    # Validate all started recoveries, including cells outside the current contrast.
    for cell in registration["cells"].values():
        if any(
            row.get("technical_recovery")
            and (
                Path(row["technical_recovery"]["result_path"]).parent
                / "runtime_context.json"
            ).exists()
            for row in cell["manifest"]
        ):
            _selected_formal_manifest(registration, cell)
    return {
        "all_original_starts": [row for row in ordered if not row["recovery"]],
        "technical_recovery_starts": recoveries,
        "ordered_all_starts": ordered,
        "technical_recoveries_used": len(recoveries),
        "global_cap": min(8, limit),
    }


def compute_admission(registration_ref: str | Path, contrast: str = "theta") -> dict:
    path, registration, protocol, simple = _registration(registration_ref)
    if contrast not in {"theta", "e1"}:
        raise ValueError("unsupported scientific contrast")
    cutoff = max(
        _time(registration["registered_at"]),
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
        datetime.fromtimestamp(
            Path(registration["development_selection_ref"]).stat().st_mtime,
            timezone.utc,
        ),
    )
    names = (
        ("theta1-D", "theta0-D")
        if contrast == "theta"
        else ("theta1-A", "theta1-" + simple)
    )
    if contrast == "e1" and simple in {"S", "R"}:
        from robots.libero.execution_arms import select_execution_arm

        selected = registration["simple_selection_snapshot"]["selected_program_specs"][
            simple
        ]
        profile = registration["cells"]["theta1-" + simple]["profile_snapshot"]
        if any(
            select_execution_arm(profile, task) != selected[task] for task in WEIGHTS
        ):
            raise ValueError(
                "E1 simple profile did not inherit the actual frozen theta0 winning programs"
            )
    results, sources = {}, {}
    for name in set(names) | {"theta0-D"}:
        results[name], sources[name] = _cell(registration, protocol, name, cutoff)
    initial, changed = results[names[1]], results[names[0]]
    bindings = [sources[name]["binding"]["cell"] for name in names]
    if bindings[0]["b0_ref"] != bindings[1]["b0_ref"]:
        raise ValueError(
            "scientific contrast changed B0 outside theta/arm registration"
        )
    if contrast == "theta" and bindings[0]["theta_ref"] == bindings[1]["theta_ref"]:
        raise ValueError("theta0 cannot substitute for a real theta1")
    if contrast == "theta" and all(
        sources[name]["binding"]["value"] is True for name in names
    ):
        norms = []
        for name in names:
            reference = sources[name]["snapshots"][0]["model_load"]["norm_stats_path"]
            path_norm = Path(reference)
            norms.append(
                _read(
                    path_norm / "norm_stats.json" if path_norm.is_dir() else path_norm
                )
            )
        if norms[0] != norms[1]:
            raise ValueError(
                "processor normalization changed; this registration cannot attribute the contrast to theta alone"
            )
    p3 = training_sources(
        registration,
        bindings[0]["theta_ref"],
        bindings[0]["processor_ref"],
        sources["theta0-D"]["binding"]["cell"]["theta_ref"],
    )
    start_order = _registered_starts(registration, cutoff)
    candidate = (
        Candidate(**_read(registration["theta_candidate_ref"]))
        if contrast == "theta"
        else Candidate(**_read(registration["e1_candidate_ref"]))
    )
    engineering = evaluate_midterm_gate(
        protocol,
        candidate,
        candidate_result=changed,
        baseline_result=initial,
        anchor_result=results["theta0-D"],
        current_bundle_id=initial.bundle_id,
        decision_id=registration["experiment_id"] + "-" + contrast + "-point",
    )
    maps = [
        {(r["task_key"], r["seed"]): r for r in result.assignments}
        for result in (changed, initial)
    ]
    complete = all(
        not result.unknown_count and not result.invalid_count
        for result in (changed, initial)
    )
    bound = all(
        source["binding"]["value"] is True
        and source["attempt_evaluations_complete"]
        and all(
            snap.get("audit", {}).get("passed") is True for snap in source["snapshots"]
        )
        for source in sources.values()
    )
    test = None
    if complete:
        differences = [
            WEIGHTS[key]
            * (
                int(maps[0][(key, seed)]["outcome"] == "success")
                - int(maps[1][(key, seed)]["outcome"] == "success")
            )
            for key in protocol["task_sets"]["release"]["task_keys"]
            for seed in protocol["task_sets"]["release"]["seeds"]
        ]
        test = exact_weighted_paired(differences)
    verdict = (
        "reject" if engineering["decision"]["verdict"] == "reject" else "inconclusive"
    )
    reason = (
        "engineering rejection"
        if verdict == "reject"
        else "incomplete/uncertain evidence or insufficient exact-test precision"
    )
    if (
        engineering["decision"]["verdict"] == "accept"
        and complete
        and bound
        and test
        and Fraction(test["statistic_numerator"], 240) >= Fraction(1, 50)
        and Fraction(test["p_numerator"], test["p_denominator"]) <= Fraction(1, 20)
    ):
        verdict, reason = (
            "accept",
            "finite-matrix engineering gain and pre-registered one-sided exact test passed",
        )
    confirmatory = True
    if contrast == "e1":
        theta = compute_admission(registration_ref, "theta")
        confirmatory = theta["verdict"] == "accept"
        if not confirmatory and verdict != "reject":
            verdict, reason = (
                "inconclusive",
                "E1 is descriptive because the first theta hypothesis did not pass",
            )
    return {
        "schema": "rsi-midterm-scientific-admission-v1",
        "registration_ref": str(path),
        "registration_snapshot": registration,
        "protocol_id": protocol["protocol_id"],
        "scope_id": registration["scope"],
        "contrast": contrast,
        "bundle_id": changed.bundle_id,
        "theta_ref": bindings[0]["theta_ref"],
        "processor_ref": bindings[0]["processor_ref"],
        "theta0_ref": sources["theta0-D"]["binding"]["cell"]["theta_ref"],
        "theta0_processor_ref": sources["theta0-D"]["binding"]["cell"]["processor_ref"],
        "verdict": verdict,
        "reason": reason,
        "release_eligible": False,
        "sources": sources,
        "formal_run_order": start_order,
        "P3": p3,
        "engineering": engineering,
        "exact_test": test,
        "design": DESIGN,
        "population_minimum_gain_proven": False,
        "confirmatory_test_allowed": confirmatory,
        "descriptive_other_E1_contrasts": True,
        "simple_arm": simple,
    }


def write_admission(
    registration_ref: str | Path, receipt_id: str, contrast: str = "theta"
) -> tuple[str, dict]:
    _, registration, _, _ = _registration(registration_ref)
    require_role(registration, "evaluation")
    receipt = compute_admission(registration_ref, contrast)
    store = Records(registration["evaluation_store_ref"])
    store.put("scientific_admissions", identifier(receipt_id), receipt)
    return str(store.path("scientific_admissions", receipt_id)), receipt


def verify_scientific_admission(
    reference: str | Path,
    *,
    expected_scope: str,
    expected_theta_ref: str,
    expected_processor_ref: str,
    evaluation_root: str | Path,
    expected_baseline_theta_ref: str | None = None,
    expected_baseline_processor_ref: str | None = None,
    expected_scope_authorization_ref: str | Path | None = None,
    require_accept: bool = True,
) -> dict:
    """Recompute a protected scientific receipt before admission or real refusal."""
    path = Path(reference).resolve(strict=True)
    if not path.is_relative_to(Path(evaluation_root).resolve()) or not _readonly(path):
        raise PermissionError(
            "scientific admission must be protected independent evidence"
        )
    receipt = _read(path)
    if (
        receipt.get("schema") != "rsi-midterm-scientific-admission-v1"
        or receipt.get("contrast") != "theta"
    ):
        raise PermissionError("point Gate is not scientific theta-effect admission")
    try:
        current = compute_admission(receipt["registration_ref"], "theta")
    except (OSError, ValueError, KeyError) as exc:
        raise PermissionError(
            "scientific admission is stale or its current source evidence is incomplete"
        ) from exc
    if (
        expected_scope_authorization_ref is not None
        and Path(current["registration_snapshot"]["scope_authorization_ref"]).resolve()
        != Path(expected_scope_authorization_ref).resolve()
    ):
        raise PermissionError("scientific admission belongs to another task authority")
    if (
        receipt != current
        or require_accept
        and current["verdict"] != "accept"
        or current["scope_id"] != expected_scope
        or current["theta_ref"] != expected_theta_ref
        or current["processor_ref"] != expected_processor_ref
    ):
        raise PermissionError(
            "scientific theta-effect admission is unaccepted, stale or bound to another model"
        )
    if (
        expected_baseline_theta_ref is not None
        and current["theta0_ref"] != expected_baseline_theta_ref
        or expected_baseline_processor_ref is not None
        and current["theta0_processor_ref"] != expected_baseline_processor_ref
    ):
        raise PermissionError(
            "scientific theta0 differs from the real initial deployment model"
        )
    return current


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--receipt-id")
    parser.add_argument("--contrast", choices=("theta", "e1"), default="theta")
    parser.add_argument("--attempt-cell", choices=CELLS)
    parser.add_argument("--attempt-run-id")
    args = parser.parse_args(argv)
    if args.attempt_cell or args.attempt_run_id:
        if not args.attempt_cell or not args.attempt_run_id or args.receipt_id:
            parser.error(
                "attempt operation needs both attempt fields and no admission receipt-id"
            )
        path, receipt = write_scientific_attempt(
            args.registration, args.attempt_cell, args.attempt_run_id
        )
        print(
            json.dumps(
                {
                    "attempt_receipt_ref": path,
                    "validity": receipt["validity"],
                    "failure_class": receipt["failure_class"],
                }
            )
        )
    else:
        if not args.receipt_id:
            parser.error("admission operation requires --receipt-id")
        path, receipt = write_admission(
            args.registration, args.receipt_id, args.contrast
        )
        print(
            json.dumps(
                {
                    "receipt_ref": path,
                    "verdict": receipt["verdict"],
                    "exact_test": receipt["exact_test"],
                }
            )
        )


if __name__ == "__main__":
    main()
