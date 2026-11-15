# Copyright 2026 RSI-Agent-VLA contributors.

"""Exact finite tests and actual file-bound scientific admission on CPU facts."""

import itertools
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rsi_vla.adapters.pi05_training import ACTION_CONTRACT, dataset_identity
from rsi_vla.contracts import Candidate, ExperimentResult
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.midterm_evaluation import freeze_midterm_b0
from rsi_vla.midterm_science import (
    CELLS,
    DESIGN,
    WEIGHTS,
    compute_admission,
    development_rows,
    exact_weighted_paired,
    make_block_schedule,
    select_theta0_controllers,
    verify_scientific_admission,
    write_admission,
)
from rsi_vla.storage import atomic_json

from robots.libero import execution_arms

ROOT = Path(execution_arms.__file__).resolve().parents[3]


class ScienceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=getattr(self, "fixture_parent", None)
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.operator = self.root / "operator"
        self.evaluation = EvaluationStore(self.root / "state/evaluation")
        self.now = datetime.now(timezone.utc)
        self.progress_patch = patch(
            "rsi_vla.critical_progress.inspect_progress",
            return_value={"value": True, "evidence_refs": [], "offline_fixture": True},
        )
        self.progress_mock = self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)
        self.scope_ref = self.operator / "scope.json"
        atomic_json(
            self.scope_ref,
            {
                "scope_id": "science-task",
                "write_root": str(self.root),
                "runtime_technical_recovery_max": 8,
                "accepted_at": (self.now - timedelta(hours=1)).isoformat(),
                "expires_at": (self.now + timedelta(hours=2)).isoformat(),
            },
        )
        self.protocol = json.loads((ROOT / "config/pilot_v2.json").read_text())
        self.protocol.update(
            protocol_id="science-P4",
            scope_id="science-task",
            dynamic_authorized=True,
            comparison_only=True,
            anchor_bundle_id="theta0-D",
        )
        self.protocol["task_sets"]["release"]["seeds"] = list(range(10, 16))
        self.protocol["budget"]["per_attempt_proposal"].update(
            max_episode_steps=400, tool_calls=512, full_job_wall_seconds=1200
        )
        self.protocol["midterm_policy"] = {
            "schema": "rsi-midterm-evaluation-v1",
            "critical_old_min_fraction": 0.5,
        }
        self.protocol_ref = self.operator / "formal.json"
        atomic_json(self.protocol_ref, self.protocol)
        dev = deepcopy(self.protocol)
        dev["protocol_id"] = "science-dev"
        self.dev_ref = self.operator / "dev.json"
        atomic_json(self.dev_ref, dev)
        runs = []
        for arm in ("D", "S", "R"):
            for task in WEIGHTS:
                variants = (
                    ["v1"]
                    if arm == "D" or task not in ("position_swap_salad", "old_drawer")
                    else ["v1", "v2"]
                )
                for program in variants:
                    for seed in (1, 2) if arm == "D" else (1,):
                        folder = (
                            self.root / "development" / f"{arm}-{task}-{program}-{seed}"
                        )
                        identity = {
                            "run_id": folder.name,
                            "episode_id": task + str(seed),
                            "bundle_id": arm,
                            "protocol_id": "science-dev",
                        }
                        success = not (
                            arm == "D"
                            and task in ("position_swap_salad", "old_drawer")
                            and seed == 1
                        )
                        program_spec = {
                            "schema": "libero-execution-arm-v1",
                            "arm": arm,
                            "max_chunks": 80,
                        }
                        if arm in ("S", "R"):
                            program_spec["stages"] = [
                                {
                                    "id": "native",
                                    "instruction": "registered goal",
                                    "max_chunks": 80,
                                }
                            ]
                            if arm == "R":
                                program_spec["stages"][0].update(
                                    on_complete="stop",
                                    on_unknown="stop",
                                    on_timeout="stop",
                                )
                        atomic_json(
                            folder / "runtime_context.json",
                            {
                                **identity,
                                "information_scope": "development",
                                "execution_arm": program_spec,
                            },
                        )
                        atomic_json(
                            folder / "native.json",
                            {
                                **identity,
                                "task": {"seed": seed},
                                "validity": "valid",
                                "attempt_closed": True,
                                "benchmark_success": success,
                            },
                        )
                        audit = (
                            self.evaluation.root
                            / "dev-audits"
                            / folder.name
                            / "audit.json"
                        )
                        atomic_json(
                            audit,
                            {
                                "identity": identity,
                                "passed": True,
                                "costs": {
                                    "environment_steps": 80 if program == "v2" else 100,
                                    "model_requests": 0,
                                    "whole_job_wall_seconds": (10 if arm == "S" else 14)
                                    + 2 * (program == "v2"),
                                },
                            },
                        )
                        runs.append(
                            {
                                "arm": arm,
                                "program_id": program,
                                "task_key": task,
                                "seed": seed,
                                "run_root": str(folder),
                                "result_ref": str(folder / "native.json"),
                                "audit_ref": str(audit),
                                "protocol_ref": str(self.dev_ref),
                            }
                        )
        self.dev_selection_ref = self.operator / "development-selection.json"
        atomic_json(self.dev_selection_ref, {"runs": runs})
        selection = select_theta0_controllers(development_rows(self.dev_selection_ref))
        self.assertEqual(selection["simple_arm"], "S")
        self.training = self.make_training()
        model0 = self.operator / "theta0.json"
        norm0 = self.root / "theta0-norm"
        norm0.mkdir()
        atomic_json(norm0 / "norm_stats.json", {"same_normalization": [0, 1]})
        checkpoint0 = self.root / "theta0"
        checkpoint0.mkdir()
        atomic_json(
            model0, {"checkpoint": str(checkpoint0), "normalization": str(norm0)}
        )
        self.theta0 = str(model0)
        self.theta1 = self.training["verified_handoff"] + "#/model_descriptor"
        self.processor0 = str(norm0)
        self.processor1 = self.training["normalization"]
        b0_path = self.operator / "B0.json"
        cells = {}
        b0_bindings = {}
        for name in CELLS:
            theta, arm = name.split("-")
            changed = theta == "theta1"
            theta_ref = self.theta1 if changed else self.theta0
            processor = self.processor1 if changed else self.processor0
            spec = {"schema": "libero-execution-arm-v1", "arm": arm, "max_chunks": 80}
            if arm in ("S", "R"):
                spec["stages"] = [
                    {"id": "native", "instruction": "registered goal", "max_chunks": 80}
                ]
                if arm == "R":
                    spec["stages"][0].update(
                        on_complete="stop", on_unknown="stop", on_timeout="stop"
                    )
            profile = {
                "execution_arm": spec,
                "common_execution": {
                    "schema": "libero-common-execution-v1",
                    "observation_cadence": "prediction",
                    "allowed_motion_tools": ["vla_execute"],
                    "action_prefix_steps": 5,
                },
                "midterm_cell": {
                    "cell_id": name,
                    "theta_id": theta,
                    "theta_ref": theta_ref,
                    "processor_ref": processor,
                    "b0_ref": str(b0_path),
                    "arm": arm,
                },
            }
            ref = self.operator / (name + "-profile.json")
            atomic_json(ref, profile)
            binding = {
                "bundle": {
                    "bundle_id": name,
                    "theta_ref": theta_ref,
                    "processor_ref": processor,
                    "phi_ref": "phi",
                    "rho_ref": "rho",
                    "psi_ref": "psi",
                    "memory_ref": "memory",
                },
                "profile": profile,
                "references": {
                    "phi_ref": {"chunk": 5},
                    "theta_ref": json.loads(
                        Path(self.training["verified_handoff"]).read_text()
                    )["model_descriptor"]
                    if changed
                    else json.loads(model0.read_text()),
                },
                "records": {"rho_ref": {}, "psi_ref": {}, "memory_ref": {}},
            }
            if changed:
                b0_bindings[arm] = binding
            manifest = []
            for task in WEIGHTS:
                for seed in range(10, 16):
                    folder = self.root / "runtime" / f"{name}-{task}-{seed}"
                    run_id = folder.name
                    audit = self.evaluation.root / "audits" / run_id / "audit.json"
                    manifest.append(
                        {
                            "task_key": task,
                            "seed": seed,
                            "run_id": run_id,
                            "result_path": str(folder / "independent_result.json"),
                            "audit_ref": str(audit),
                            "attempt_receipt_ref": str(
                                self.evaluation.path("attempts", run_id)
                            ),
                            "host_completion_ref": str(
                                self.root
                                / "host/jobs"
                                / (run_id + "-runtime")
                                / "completion.json"
                            ),
                        }
                    )
            cells[name] = {
                "result_id": name + "-result",
                "bundle_id": name,
                "profile_ref": str(ref),
                "profile_snapshot": profile,
                "manifest": manifest,
                "binding": binding,
            }
        atomic_json(b0_path, freeze_midterm_b0(b0_bindings))
        for key, baseline in (("theta", "theta0-D"), ("e1", "theta1-S")):
            candidate = Candidate(
                key + "-candidate",
                baseline,
                "theta1-D" if key == "theta" else "theta1-A",
                self.protocol["protocol_id"],
                self.protocol["benchmark_id"],
                {},
                [],
                [],
                {},
                {},
                "simulation",
            )
            atomic_json(self.operator / (key + "-candidate.json"), asdict(candidate))
        self.registration = {
            "schema": "rsi-midterm-science-registration-v1",
            "experiment_id": "science-example",
            "scope": "science-task",
            "scope_authorization_ref": str(self.scope_ref),
            "registered_at": self.now.isoformat(),
            "protocol_ref": str(self.protocol_ref),
            "protocol_snapshot": self.protocol,
            "evaluation_store_ref": str(self.evaluation.root),
            "role_domains": {
                "operator": str(self.operator),
                "evaluation": str(self.evaluation.root),
            },
            "repository_root": str(ROOT),
            "design": DESIGN,
            "cells": {
                k: {a: b for a, b in v.items() if a != "binding"}
                for k, v in cells.items()
            },
            "development_selection_ref": str(self.dev_selection_ref),
            "simple_selection_snapshot": selection,
            "schedule_seed": 12419,
            "theta_candidate_ref": str(self.operator / "theta-candidate.json"),
            "e1_candidate_ref": str(self.operator / "e1-candidate.json"),
            "training": {
                k: v
                for k, v in self.training.items()
                if k not in ("verified_handoff", "normalization")
            },
        }
        self.registration["schedule"] = make_block_schedule(
            self.registration["cells"], list(WEIGHTS), list(range(10, 16)), 12419
        )
        self.reg_ref = self.operator / "registration.json"
        atomic_json(self.reg_ref, self.registration)
        order = {
            row["run_id"]: i for i, row in enumerate(self.registration["schedule"])
        }
        for name, cell in cells.items():
            assignments = []
            theta, arm = name.split("-")
            for row in cell["manifest"]:
                folder = Path(row["result_path"]).parent
                identity = {
                    "run_id": row["run_id"],
                    "episode_id": row["task_key"] + str(row["seed"]),
                    "bundle_id": name,
                    "protocol_id": self.protocol["protocol_id"],
                }
                success = not (
                    name == "theta0-D" and row["task_key"] == "position_swap_salad"
                )
                native = {
                    **identity,
                    "benchmark_success": success,
                    "attempt_closed": True,
                    "validity": "valid",
                }
                atomic_json(row["result_path"], native)
                atomic_json(
                    folder / "runtime_context.json",
                    {
                        **identity,
                        "execution_binding": cell["binding"],
                        "theta_ref": cell["binding"]["bundle"]["theta_ref"],
                        "processor_ref": cell["binding"]["bundle"]["processor_ref"],
                        "execution_arm": cell["profile_snapshot"]["execution_arm"],
                        "common_execution": cell["profile_snapshot"][
                            "common_execution"
                        ],
                        "limits": {
                            "environment_steps": 400,
                            "wall_seconds": 1200,
                            "tool_calls": 512,
                        },
                    },
                )
                timestamp = (
                    int((self.now + timedelta(minutes=1)).timestamp() * 1e9)
                    + order[row["run_id"]] * 1000000
                )
                (folder / "action_events.jsonl").write_text(
                    json.dumps({**identity, "kind": "run_start", "utc_ns": timestamp})
                    + "\n"
                )
                desc = cell["binding"]["references"]["theta_ref"]
                atomic_json(
                    folder / "model_load.json",
                    {
                        "identity": {
                            **identity,
                            "theta_ref": cell["binding"]["bundle"]["theta_ref"],
                            "processor_ref": cell["binding"]["bundle"]["processor_ref"],
                        },
                        "loaded": True,
                        "prediction_count": 1,
                        "last_actions_finite": True,
                        "action_chunk": 5,
                        "model_path": desc.get("model_path", desc.get("checkpoint")),
                        "norm_stats_path": cell["binding"]["bundle"]["processor_ref"],
                    },
                )
                atomic_json(
                    row["audit_ref"],
                    {
                        "identity": identity,
                        "passed": True,
                        "outcome": "success" if success else "failure",
                        "checks": {
                            "identity": True,
                            "complete_usage": True,
                            "model_reference": True,
                        },
                        "costs": {
                            "whole_job_wall_seconds": 1,
                            "gpu_device_seconds_envelope": 1,
                            "model_requests": 0,
                            "tool_calls": 2,
                            "environment_steps": 5,
                            "artifact_bytes": 1000,
                        },
                    },
                )
                assignments.append(
                    {
                        "task_key": row["task_key"],
                        "seed": row["seed"],
                        "outcome": "success" if success else "failure",
                        "validity": "valid",
                        "evidence_ref": row["result_path"],
                        "outcome_source": "native_benchmark",
                    }
                )
            metrics = {
                "success_base": 1,
                "success_controlled": 0 if name == "theta0-D" else 1,
                "success_old": 1,
            }
            result = ExperimentResult(
                cell["result_id"],
                name,
                self.protocol["protocol_id"],
                self.protocol["benchmark_id"],
                "release",
                "simulation",
                {},
                [row["audit_ref"] for row in cell["manifest"]],
                self.protocol["protocol_id"],
                metrics,
                dict.fromkeys(self.protocol["hard_constraints"], True),
                {},
                candidate_id="theta-candidate"
                if name == "theta1-D"
                else "e1-candidate"
                if name == "theta1-A"
                else None,
                assigned_count=24,
                assignments=assignments,
            )
            self.evaluation.save_result(result)
        self.make_host_attempts()

    def make_host_attempts(self):
        accounting = self.root / "runtime-accounting.json"
        reservations = {}
        for cell in self.registration["cells"].values():
            for row in cell["manifest"]:
                root = Path(row["result_path"]).parent
                context = json.loads((root / "runtime_context.json").read_text())
                operation_id = row["run_id"] + "-runtime"
                job = Path(row["host_completion_ref"]).parent
                actual = {
                    "runtime_starts": 1,
                    "gpu_seconds": 2,
                    "storage_bytes": 1000,
                    "model_requests": 0,
                }
                reservation = {
                    "scope_id": "science-task",
                    "operation_id": operation_id,
                    "status": "settled",
                    "cleanup_confirmed": True,
                    "missing": [],
                    "submitted_at": self.now.isoformat(),
                    "actual": actual,
                    "receipt_ref": row["host_completion_ref"],
                }
                reservations[operation_id] = reservation
                operation = {
                    "operation_id": operation_id,
                    "kind": "runtime",
                    "payload": {"run_id": row["run_id"], "output_ref": str(root)},
                }
                atomic_json(job / "submission.json", {"operation": operation})
                atomic_json(
                    job / "supervisor.json",
                    {
                        "identity": {"run_id": operation_id},
                        "status": "completed",
                        "exit_code": 0,
                        "cleanup_confirmed": True,
                        "remaining_process_ids": [],
                        "stop_reason": None,
                    },
                )
                atomic_json(
                    row["host_completion_ref"],
                    {
                        "scope_id": "science-task",
                        "operation_id": operation_id,
                        "status": "completed",
                        "supervisor_ref": str(job / "supervisor.json"),
                        "actual": actual,
                        "cleanup_confirmed": True,
                    },
                )
                permit_ref = self.operator / "permits" / (row["run_id"] + ".json")
                atomic_json(
                    permit_ref,
                    {
                        "scope": "science-task",
                        "approval_reference": str(self.scope_ref),
                        "run_id": row["run_id"],
                        "operation_id": operation_id,
                        "authorized": True,
                        "accounting_ref": str(accounting),
                    },
                )
                context["authorization_ref"] = str(permit_ref)
                atomic_json(root / "runtime_context.json", context)
        atomic_json(accounting, {"reservations": reservations})
        self.refresh_attempts()

    def refresh_attempts(self, name=None):
        from rsi_vla.midterm_science import inspect_scientific_attempt

        for cell_name, cell in self.registration["cells"].items():
            if name is None or name == cell_name:
                for row in cell["manifest"]:
                    atomic_json(
                        row["attempt_receipt_ref"],
                        inspect_scientific_attempt(
                            self.reg_ref, cell_name, row["run_id"]
                        ),
                    )

    def make_training(self):
        folder = self.root / "training"
        data = folder / "dataset"
        (data / "meta").mkdir(parents=True)
        (data / "data").mkdir()
        atomic_json(
            data / "meta/info.json",
            {
                "features": {
                    "state": {"shape": [8]},
                    "actions": {"shape": [7]},
                    "image": {"dtype": "image"},
                    "wrist_image": {"dtype": "image"},
                },
                "fps": 20,
                "total_episodes": 1,
            },
        )
        (data / "meta/episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": ["drawer"], "length": 1}) + "\n"
        )
        (data / "meta/tasks.jsonl").write_text("{}\n")
        (data / "data/example.parquet").write_bytes(b"offline metadata fixture")
        manifest = {
            "source_kind": "public_demonstration",
            "provenance_ref": "public-fixture",
            "revision": "fixed",
            "action_contract": ACTION_CONTRACT,
            "dataset_root": str(data),
            "episodes": [
                {"episode_index": 0, "task": "drawer", "source_ref": "public-episode"}
            ],
        }
        manifest_ref = folder / "dataset-manifest.json"
        atomic_json(manifest_ref, manifest)
        identity = dataset_identity(manifest)
        audit_ref = folder / "payload-audit.json"
        atomic_json(
            audit_ref,
            {
                "schema": "rsi-demonstration-payload-audit-v1",
                "dataset": identity,
                "checks_passed": True,
                "frames_checked": 1,
                "episodes": [
                    {
                        "episode_index": 0,
                        "checks": {"offline_fixture": True},
                        "frames_checked": 1,
                    }
                ],
            },
        )
        admission_ref = folder / "data-admission.json"
        reviews = {}
        for key, schema, checks in [
            (
                "quality_review",
                "rsi-demonstration-quality-review-v1",
                [
                    "image_orientation_verified",
                    "state_action_alignment_verified",
                    "task_relevance_verified",
                    "license_verified",
                ],
            ),
            (
                "evaluation_isolation_review",
                "rsi-training-evaluation-isolation-v1",
                [
                    "source_episode_mapping_checked",
                    "development_formal_split_checked",
                    "sealed_labels_not_consumed",
                ],
            ),
        ]:
            ref = folder / (key + ".json")
            atomic_json(
                ref,
                dict(
                    schema=schema,
                    scope_id="science-task",
                    protocol_id="training-protocol",
                    dataset=identity,
                    reviewer_id="independent-fixture",
                    decision="accepted",
                    evidence_refs=[str(manifest_ref)],
                    **dict.fromkeys(checks, True),
                ),
            )
            reviews[key + "_ref"] = str(ref)
        atomic_json(
            admission_ref,
            dict(
                schema="rsi-training-data-admission-v1",
                decision="admitted",
                scope_id="science-task",
                protocol_id="training-protocol",
                reviewer_id="independent-fixture",
                dataset=identity,
                original_manifest_ref=str(manifest_ref),
                payload_audit_ref=str(audit_ref),
                approved_training_tasks=["drawer"],
                **reviews,
            ),
        )
        norm = folder / "normalization"
        norm.mkdir()
        atomic_json(norm / "norm_stats.json", {"same_normalization": [0, 1]})
        checkpoint = folder / "train/checkpoints/global_step_2"
        checkpoint.mkdir(parents=True)
        weights = checkpoint / "actor/model_state_dict/full_weights.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"offline weights path only")
        descriptor = {
            "kind": "pi05_libero_native_full_weights",
            "model_path": str(checkpoint),
            "norm_stats_path": str(norm),
            "actual_load_verified": True,
            "config_name": "pi05_libero",
            "action_contract": ACTION_CONTRACT,
            "training_recipe_ref": str(folder / "recipe.json"),
            "weight_update_ref": str(folder / "train/weight-update-check.json"),
        }
        load_root = folder / "load"
        load_root.mkdir()
        load_check = load_root / "load-check.json"
        descriptor["actual_load_evidence_ref"] = str(load_check)
        atomic_json(
            load_check,
            {
                "prediction_shape": [1, 5, 7],
                "prediction_finite": True,
                "prediction_count": 1,
                "prediction": [[[0.0] * 7 for _ in range(5)]],
                "saved_probe_equal": {"actual_probe": True},
                "loader_config": {
                    "model_path": str(checkpoint),
                    "openpi_data": {"norm_stats_path": str(norm)},
                },
            },
        )
        handoff = load_root / "deployment-handoff-verified.json"
        atomic_json(
            handoff,
            {
                "actual_model_load": "passed",
                "model_descriptor": descriptor,
                "selected_weights": str(weights),
            },
        )
        recipe = folder / "recipe.json"
        atomic_json(
            recipe,
            {
                "runner": {"max_steps": 2, "logger": {"experiment_name": "train-run"}},
                "rsi_training": {
                    "mode": "learning",
                    "learning_plan": {"data_admission_ref": str(admission_ref)},
                },
                "data": {"train_data_paths": str(data)},
                "actor": {"model": {"model_path": str(self.root / "theta0")}},
            },
        )
        train_root = folder / "train"
        train_root.mkdir(exist_ok=True)
        atomic_json(
            train_root / "weight-update-check.json",
            {
                "scope": "action_expert_only",
                "step": 2,
                "trainable_elements": 100,
                "metrics": {"loss": 0.2, "grad_norm": 0.1},
                "probes": {"p": {"changed_elements": 5, "max_abs_difference": 0.01}},
                "sampled_frozen_parameters_equal": {"f": True},
                "nonzero_sampled_weight_update": True,
            },
        )
        atomic_json(
            train_root / "parameter-census.json",
            {
                "scope": "action_expert_only",
                "trainable_elements": 100,
                "frozen_elements": 1000,
                "parameters": {
                    "img.stem.bias": {"trainable": False},
                    "action_out_proj.weight": {"trainable": True},
                },
            },
        )
        atomic_json(train_root / "optimizer-role.json", {"class": "FP32MasterAdamW"})
        (train_root / "resources-rank0.jsonl").write_text(
            "\n".join(
                json.dumps({"phase": "native_optimizer_step", "step": i})
                for i in range(2)
            )
            + "\n"
        )
        result = {
            "recipe_ref": str(recipe),
            "manifest_ref": str(manifest_ref),
            "verified_handoff": str(handoff),
            "normalization": str(norm),
        }
        reservations = {}
        accounting = folder / "accounting.json"
        for key, kind, where in [
            ("training", "pi05-training-learning", train_root),
            ("load", "pi05-load-check", load_root),
        ]:
            operation = key + "-operation"
            permit = folder / (key + "-permit.json")
            job = where / "job-result.json"
            supervision = where / "supervision.json"
            native = where / "native-start.json"
            atomic_json(
                permit,
                {
                    "scope_id": "science-task",
                    "operation_id": operation,
                    "run_id": "train-run",
                    "accounting_ref": str(accounting),
                    "expires_at": (self.now + timedelta(hours=1)).isoformat(),
                    "budget": {"max_steps": 2 if key == "training" else 1},
                    "data_admission_ref": str(admission_ref),
                    "protocol_id": "training-protocol",
                    "data_admission_authority": "independent-fixture",
                    "scope_ref": str(self.scope_ref),
                    "kind": kind,
                    "authorized": True,
                },
            )
            atomic_json(
                native,
                {
                    "schema": "rsi-native-training-start-v1",
                    "scope_id": "science-task",
                    "operation_id": operation,
                    "run_id": "train-run",
                    "started_at": self.now.isoformat(),
                },
            )
            atomic_json(
                supervision,
                {
                    "returncode": 0,
                    "stop_reason": None,
                    "cleanup_confirmed": True,
                    "resource_limits_satisfied": True,
                },
            )
            atomic_json(
                job,
                {
                    "schema": "rsi-native-training-job-result-v1",
                    "scope_id": "science-task",
                    "operation_id": operation,
                    "run_id": "train-run",
                    "kind": kind,
                    "status": "completed",
                    "supervision_ref": str(supervision),
                    "native_start_ref": str(native),
                    "handoff_ref": str(handoff),
                },
            )
            reservations[operation] = {
                "status": "settled",
                "cleanup_confirmed": True,
                "missing": [],
                "submitted_at": self.now.isoformat(),
                "scope_id": "science-task",
                "operation_id": operation,
                "actual": {
                    "training_starts" if key == "training" else "load_checks": 1,
                    "gpu_seconds": 10,
                },
            }
            result[key + "_job_ref"] = str(job)
            result[key + "_permit_ref"] = str(permit)
        atomic_json(accounting, {"reservations": reservations})
        return result


class ScientificAttemptTests(ScienceFixture):
    def first(self):
        return self.registration["cells"]["theta1-D"]["manifest"][0]

    def test_attempt_is_an_evaluation_role_cpu_operation_with_canonical_output(self):
        from rsi_vla.midterm_science import write_scientific_attempt

        row = self.first()
        with self.assertRaisesRegex(ValueError, "OS role"):
            write_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])
        with patch("rsi_vla.midterm_science.require_role") as role:
            ref, receipt = write_scientific_attempt(
                self.reg_ref, "theta1-D", row["run_id"]
            )
        role.assert_called_once_with(self.registration, "evaluation")
        self.assertEqual(ref, row["attempt_receipt_ref"])
        self.assertEqual(receipt["validity"], "valid")
        self.assertFalse(receipt["technical_recovery_authorized"])

    def test_actual_supervisor_ledger_and_progress_determine_classification(self):
        from rsi_vla.midterm_science import inspect_scientific_attempt

        row = self.first()
        root = Path(row["result_path"]).parent
        self.progress_mock.return_value = {"value": None, "evidence_refs": []}
        with self.assertRaisesRegex(ValueError, "critical progression"):
            inspect_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])
        audit = json.loads(Path(row["audit_ref"]).read_text())
        audit.update(passed=False, outcome="unknown")
        atomic_json(row["audit_ref"], audit)
        technical = inspect_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])
        self.assertEqual(technical["failure_class"], "technical")
        job = Path(row["host_completion_ref"]).parent
        supervisor = json.loads((job / "supervisor.json").read_text())
        supervisor.update(
            status="interrupted", exit_code=-15, stop_reason="storage_bytes"
        )
        atomic_json(job / "supervisor.json", supervisor)
        completion = json.loads(Path(row["host_completion_ref"]).read_text())
        completion["status"] = "unknown"
        atomic_json(row["host_completion_ref"], completion)
        budget = inspect_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])
        self.assertEqual(budget["failure_class"], "governance")
        self.progress_mock.assert_called_with(root, technical["identity"])
        context = json.loads((root / "runtime_context.json").read_text())
        permit = json.loads(Path(context["authorization_ref"]).read_text())
        ledger = json.loads(Path(permit["accounting_ref"]).read_text())
        ledger["reservations"][permit["operation_id"]]["missing"] = ["gpu_seconds"]
        atomic_json(permit["accounting_ref"], ledger)
        with self.assertRaisesRegex(ValueError, "costs or owner cleanup"):
            inspect_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])


class ScientificAdmissionTests(ScienceFixture):
    def set_controlled(self, name, count):
        path = self.evaluation.path("results", name + "-result")
        value = json.loads(path.read_text())
        for row in value["assignments"]:
            if row["task_key"] == "position_swap_salad":
                success = row["seed"] < 10 + count
                row["outcome"] = "success" if success else "failure"
                native = json.loads(Path(row["evidence_ref"]).read_text())
                native["benchmark_success"] = success
                atomic_json(row["evidence_ref"], native)
                audit_ref = next(
                    item["audit_ref"]
                    for item in self.registration["cells"][name]["manifest"]
                    if item["task_key"] == row["task_key"]
                    and item["seed"] == row["seed"]
                )
                audit = json.loads(Path(audit_ref).read_text())
                audit["outcome"] = row["outcome"]
                atomic_json(audit_ref, audit)
        value["metrics"]["success_controlled"] = count / 6
        atomic_json(path, value)
        self.refresh_attempts(name)

    def add_recovery(self, *, technical=True):
        import shutil

        from rsi_vla.midterm_science import inspect_scientific_attempt

        name = "theta1-D"
        row = self.registration["cells"][name]["manifest"][0]
        original_root = Path(row["result_path"]).parent
        recovery_id = row["run_id"] + "-recovery"
        recovery_root = original_root.with_name(recovery_id)
        shutil.copytree(original_root, recovery_root)
        recovered = {
            "run_id": recovery_id,
            "result_path": str(recovery_root / "independent_result.json"),
            "audit_ref": str(
                self.evaluation.root / "audits" / recovery_id / "audit.json"
            ),
            "attempt_receipt_ref": str(self.evaluation.path("attempts", recovery_id)),
            "host_completion_ref": str(
                self.root / "host/jobs" / (recovery_id + "-runtime") / "completion.json"
            ),
            "recovery_basis_ref": str(self.operator / "recovery-basis.json"),
        }
        row["technical_recovery"] = recovered
        original_context = json.loads(
            (original_root / "runtime_context.json").read_text()
        )
        recovery_context = json.loads(
            (recovery_root / "runtime_context.json").read_text()
        )
        recovery_context["run_id"] = recovery_id
        permit_ref = self.operator / "permits" / (recovery_id + ".json")
        recovery_context["authorization_ref"] = str(permit_ref)
        atomic_json(recovery_root / "runtime_context.json", recovery_context)
        native = json.loads((recovery_root / "independent_result.json").read_text())
        native["run_id"] = recovery_id
        atomic_json(recovered["result_path"], native)
        loader = json.loads((recovery_root / "model_load.json").read_text())
        loader["identity"]["run_id"] = recovery_id
        atomic_json(recovery_root / "model_load.json", loader)
        first = json.loads((recovery_root / "action_events.jsonl").read_text())
        first["run_id"] = recovery_id
        first["utc_ns"] += 1
        (recovery_root / "action_events.jsonl").write_text(json.dumps(first) + "\n")
        audit = json.loads(Path(row["audit_ref"]).read_text())
        audit["identity"]["run_id"] = recovery_id
        atomic_json(recovered["audit_ref"], audit)
        original_permit = json.loads(
            Path(original_context["authorization_ref"]).read_text()
        )
        operation_id = recovery_id + "-runtime"
        permit = {
            **original_permit,
            "run_id": recovery_id,
            "operation_id": operation_id,
        }
        atomic_json(permit_ref, permit)
        job = Path(recovered["host_completion_ref"]).parent
        atomic_json(
            job / "submission.json",
            {
                "operation": {
                    "operation_id": operation_id,
                    "kind": "runtime",
                    "payload": {
                        "run_id": recovery_id,
                        "output_ref": str(recovery_root),
                    },
                }
            },
        )
        actual = {
            "runtime_starts": 1,
            "gpu_seconds": 2,
            "storage_bytes": 1000,
            "model_requests": 0,
        }
        atomic_json(
            job / "supervisor.json",
            {
                "identity": {"run_id": operation_id},
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
                "stop_reason": None,
            },
        )
        atomic_json(
            recovered["host_completion_ref"],
            {
                "scope_id": "science-task",
                "operation_id": operation_id,
                "status": "completed",
                "supervisor_ref": str(job / "supervisor.json"),
                "actual": actual,
                "cleanup_confirmed": True,
            },
        )
        ledger = json.loads(Path(permit["accounting_ref"]).read_text())
        ledger["reservations"][operation_id] = {
            "scope_id": "science-task",
            "operation_id": operation_id,
            "status": "settled",
            "cleanup_confirmed": True,
            "missing": [],
            "submitted_at": self.now.isoformat(),
            "actual": actual,
            "receipt_ref": recovered["host_completion_ref"],
        }
        atomic_json(permit["accounting_ref"], ledger)
        if technical:
            original_audit = json.loads(Path(row["audit_ref"]).read_text())
            original_audit.update(passed=False, outcome="unknown")
            atomic_json(row["audit_ref"], original_audit)
        atomic_json(self.reg_ref, self.registration)
        original_attempt = inspect_scientific_attempt(self.reg_ref, name, row["run_id"])
        atomic_json(row["attempt_receipt_ref"], original_attempt)
        atomic_json(
            recovered["attempt_receipt_ref"],
            inspect_scientific_attempt(self.reg_ref, name, recovery_id),
        )
        repair_ref = self.operator / "repair.txt"
        repair_ref.write_text(
            "CPU fixture: repair the independently classified transient transport fault"
        )
        cause_ref = self.operator / "cause-resolution.json"
        atomic_json(
            cause_ref,
            {
                "scope_id": "science-task",
                "run_id": row["run_id"],
                "resolved": True,
                "repair_ref": str(repair_ref),
            },
        )
        basis = {
            key: original_attempt[key]
            for key in (
                "run_id",
                "validity",
                "failure_class",
                "cost_closed",
                "cleanup_complete",
            )
        }
        atomic_json(
            recovered["recovery_basis_ref"],
            dict(
                **basis,
                attempt_receipt_ref=row["attempt_receipt_ref"],
                recovery_cause_resolved=True,
                cause_resolution_ref=str(cause_ref),
            ),
        )
        result_path = self.evaluation.path("results", name + "-result")
        result = json.loads(result_path.read_text())
        pair = next(
            pair
            for pair in result["assignments"]
            if pair["task_key"] == row["task_key"] and pair["seed"] == row["seed"]
        )
        pair["evidence_ref"] = recovered["result_path"]
        result["data_refs"].append(recovered["audit_ref"])
        atomic_json(result_path, result)
        return row, recovered

    def test_single_registered_technical_recovery_keeps_original_clock_cost_and_denominator(
        self,
    ):
        row, recovered = self.add_recovery()
        receipt = compute_admission(self.reg_ref)
        self.assertEqual(receipt["verdict"], "accept")
        self.assertEqual(receipt["exact_test"]["assigned_pairs"], 24)
        self.assertEqual(receipt["formal_run_order"]["technical_recoveries_used"], 1)
        history = receipt["sources"]["theta1-D"]["attempt_histories"][0]
        self.assertEqual(history["selected_run_id"], recovered["run_id"])
        self.assertEqual(len(history["attempts"]), 2)
        self.assertEqual(
            receipt["sources"]["theta1-D"]["all_attempt_costs"][
                "gpu_device_seconds_envelope"
            ],
            50,
        )
        cause_ref = json.loads(Path(recovered["recovery_basis_ref"]).read_text())[
            "cause_resolution_ref"
        ]
        cause = json.loads(Path(cause_ref).read_text())
        cause["resolved"] = False
        atomic_json(cause_ref, cause)
        with self.assertRaisesRegex(ValueError, "concrete cause repair"):
            compute_admission(self.reg_ref)

    def test_budget_unknown_cannot_acquire_technical_recovery_even_after_registration(
        self,
    ):
        from rsi_vla.midterm_science import inspect_scientific_attempt

        row, _ = self.add_recovery()
        job = Path(row["host_completion_ref"]).parent
        supervisor = json.loads((job / "supervisor.json").read_text())
        supervisor.update(
            status="interrupted", exit_code=-15, stop_reason="storage_bytes"
        )
        atomic_json(job / "supervisor.json", supervisor)
        completion = json.loads(Path(row["host_completion_ref"]).read_text())
        completion["status"] = "unknown"
        atomic_json(row["host_completion_ref"], completion)
        attempt = inspect_scientific_attempt(self.reg_ref, "theta1-D", row["run_id"])
        atomic_json(row["attempt_receipt_ref"], attempt)
        self.assertEqual(attempt["failure_class"], "governance")
        with self.assertRaisesRegex(ValueError, "nontechnical"):
            compute_admission(self.reg_ref)

    def test_valid_outcome_cannot_be_replaced_by_a_registered_recovery(self):
        self.add_recovery(technical=False)
        with self.assertRaisesRegex(ValueError, "valid outcome"):
            compute_admission(self.reg_ref)

    def test_exact_upper_tail_matches_all_sign_assignments_and_ties(self):
        values = [16, 14, -5, 0]
        result = exact_weighted_paired(values)
        samples = [
            sum(sign * abs(value) for sign, value in zip(signs, [16, 14, 5]))
            for signs in itertools.product((-1, 1), repeat=3)
        ]
        self.assertEqual(result["p_value"], sum(v >= sum(values) for v in samples) / 8)
        self.assertEqual(exact_weighted_paired([0] * 24)["p_value"], 1)

    def test_registered_large_positive_effect_accepts_and_receipt_revalidates(self):
        result = compute_admission(self.reg_ref)
        self.assertEqual(result["verdict"], "accept")
        self.assertEqual(result["exact_test"]["p_value"], 1 / 64)
        self.assertEqual(result["exact_test"]["statistic_numerator"], 84)
        self.assertFalse(result["population_minimum_gain_proven"])
        with (
            patch("rsi_vla.midterm_science.require_role"),
            patch("rsi_vla.midterm_science._readonly", return_value=True),
        ):
            ref, receipt = write_admission(self.reg_ref, "scientific-theta")
            self.assertEqual(
                verify_scientific_admission(
                    ref,
                    expected_scope="science-task",
                    expected_theta_ref=self.theta1,
                    expected_processor_ref=self.processor1,
                    evaluation_root=self.evaluation.root,
                ),
                receipt,
            )

    def test_point_gain_without_precision_is_inconclusive_not_useful_learning(self):
        self.set_controlled("theta1-D", 1)
        result = compute_admission(self.reg_ref)
        self.assertEqual(result["engineering"]["decision"]["verdict"], "accept")
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["exact_test"]["p_value"], 0.5)

    def test_native_loader_and_training_failure_or_changed_order_block_admission(self):
        update = (
            Path(self.training["training_job_ref"]).parent / "weight-update-check.json"
        )
        value = json.loads(update.read_text())
        value["nonzero_sampled_weight_update"] = False
        atomic_json(update, value)
        with self.assertRaisesRegex(ValueError, "P3 actual"):
            compute_admission(self.reg_ref)

    def test_e1_significance_is_only_descriptive_when_theta_did_not_pass(self):
        self.set_controlled("theta1-D", 1)
        self.set_controlled("theta1-S", 0)
        result = compute_admission(self.reg_ref, "e1")
        self.assertEqual(result["exact_test"]["p_value"], 1 / 64)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertFalse(result["confirmatory_test_allowed"])

    def test_incomplete_pair_is_not_dropped_and_source_receipt_change_is_stale(self):
        with (
            patch("rsi_vla.midterm_science.require_role"),
            patch("rsi_vla.midterm_science._readonly", return_value=True),
        ):
            ref, _ = write_admission(self.reg_ref, "before-change")
            audit = Path(
                self.registration["cells"]["theta1-D"]["manifest"][0]["audit_ref"]
            )
            value = json.loads(audit.read_text())
            value["new_evaluation_revision"] = "v2"
            atomic_json(audit, value)
            with self.assertRaisesRegex(PermissionError, "stale"):
                verify_scientific_admission(
                    ref,
                    expected_scope="science-task",
                    expected_theta_ref=self.theta1,
                    expected_processor_ref=self.processor1,
                    evaluation_root=self.evaluation.root,
                )
        self.refresh_attempts("theta1-D")
        path = self.evaluation.path("results", "theta1-D-result")
        value = json.loads(path.read_text())
        value["assignments"][0].update(
            outcome="unknown", evidence_ref=None, outcome_source=None
        )
        value["unknown_count"] = 1
        value["metrics"]["success_base"] = 5 / 6
        atomic_json(path, value)
        result = compute_admission(self.reg_ref)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIsNone(result["exact_test"])
        self.assertEqual(result["engineering"]["comparison"]["assigned_pairs"], 24)

    def test_same_identity_copy_or_conflicting_audit_cannot_substitute_for_native_evidence(
        self,
    ):
        row = self.registration["cells"]["theta1-D"]["manifest"][0]
        audit = json.loads(Path(row["audit_ref"]).read_text())
        changed = {**audit, "outcome": "failure"}
        atomic_json(row["audit_ref"], changed)
        with self.assertRaisesRegex(ValueError, "stale|native closure/audit"):
            compute_admission(self.reg_ref)
        atomic_json(row["audit_ref"], audit)
        result_path = self.evaluation.path("results", "theta1-D-result")
        result = json.loads(result_path.read_text())
        assignment = next(
            item
            for item in result["assignments"]
            if item["task_key"] == row["task_key"] and item["seed"] == row["seed"]
        )
        copy_path = Path(row["result_path"]).parent / "same-identity-copy.json"
        copy_path.write_bytes(Path(row["result_path"]).read_bytes())
        assignment["evidence_ref"] = str(copy_path)
        atomic_json(result_path, result)
        with self.assertRaisesRegex(ValueError, "stale|native closure/audit"):
            compute_admission(self.reg_ref)

    def test_actual_schedule_and_actual_loader_paths_are_consumed(self):
        cell = self.registration["cells"]["theta1-D"]
        first, second = cell["manifest"][:2]
        p1 = Path(first["result_path"]).parent / "action_events.jsonl"
        p2 = Path(second["result_path"]).parent / "action_events.jsonl"
        e1 = json.loads(p1.read_text())
        e2 = json.loads(p2.read_text())
        e1["utc_ns"], e2["utc_ns"] = e2["utc_ns"], e1["utc_ns"]
        p1.write_text(json.dumps(e1) + "\n")
        p2.write_text(json.dumps(e2) + "\n")
        with self.assertRaisesRegex(ValueError, "run_start order"):
            compute_admission(self.reg_ref)
        e1["utc_ns"], e2["utc_ns"] = e2["utc_ns"], e1["utc_ns"]
        p1.write_text(json.dumps(e1) + "\n")
        p2.write_text(json.dumps(e2) + "\n")
        loader = Path(first["result_path"]).parent / "model_load.json"
        value = json.loads(loader.read_text())
        value["model_path"] = "/different-model"
        atomic_json(loader, value)
        with self.assertRaisesRegex(ValueError, "different theta"):
            compute_admission(self.reg_ref)


if __name__ == "__main__":
    unittest.main()
