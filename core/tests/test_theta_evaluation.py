"""Consume the Host's saved/loaded theta in the full frozen evaluation matrix."""

import contextlib
import io
import json
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from rsi_vla.__main__ import main as core_main
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import EvaluationStore, Gate
from rsi_vla.evaluator import evaluate_matrix
from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.theta_b import (
    evaluate_theta_b_cell,
    freeze_theta_b,
    prepare_theta_b_matrix,
    training_theta_version,
)

ROOT = Path(__file__).resolve().parents[2]


def consume_training_matrix(
    *, training_result_ref, load_result_ref, scope_id, candidate_ref=None
):
    """Evaluate the real saved-model binding with a synthetic Env result boundary.

    The caller has already run the controlled Host native trainer/loader graph.
    This callback runs no trainer, model request, Env, GPU or simulator. Native
    outcome files below are isolated synthetic facts; unmeasured safety and
    compatibility keep the actual Gate inconclusive.
    """
    version = training_theta_version(
        training_result_ref=training_result_ref,
        load_result_ref=load_result_ref,
        scope_id=scope_id,
    )
    with tempfile.TemporaryDirectory(prefix="theta-matrix-consumer-") as directory:
        artifacts = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(
                [
                    "init",
                    "--artifacts",
                    str(artifacts),
                    "--profile",
                    str(ROOT / "config/run_profile_v2.json"),
                    "--protocol",
                    str(ROOT / "config/pilot_v2.json"),
                ]
            )
        store = ResearchStore(artifacts / "research")
        original = RunBundle(**store.get("bundles", "initial-medium"))
        protocol = json.loads((ROOT / "config/pilot_v2.json").read_text())
        protocol.update(
            protocol_id="theta-matrix-offline",
            benchmark_id="theta-matrix-release",
            scope="offline",
            scope_id="theta-matrix-offline",
            dynamic_authorized=False,
            anchor_bundle_id="matrix-base",
            task_policy={"schema": "rsi-task-policy-v1"},
        )
        protocol["task_sets"]["release"]["benchmark_id"] = protocol["benchmark_id"]
        protocol["objective_comparison"] = {
            "schema": "rsi-objective-comparison-v1",
            "objective": "success_gain",
            "task_keys": protocol["task_sets"]["release"]["task_keys"],
            "minimum_gain": 0.1,
            "anchor_max_drop": 0,
        }
        protocol_ref = artifacts / "protocol.json"
        atomic_json(protocol_ref, protocol)
        base = replace(
            original, bundle_id="matrix-base", protocol_id=protocol["protocol_id"]
        )
        store.save_bundle(base)
        memory = deepcopy(store.get("memory", base.memory_ref))
        memory["retrieval_prompt"] = "frozen B1 retrieval rule"
        store.put("memory", "matrix-B1-memory", memory)
        store.save_bundle(
            replace(base, bundle_id="matrix-B1", memory_ref="matrix-B1-memory")
        )
        experiment = freeze_theta_b(
            artifacts,
            experiment_id="loaded-theta-matrix",
            base_bundle_id=base.bundle_id,
            b_bundle_ids={"B0": base.bundle_id, "B1": "matrix-B1"},
            theta_versions={
                "theta0": {
                    "theta_ref": base.theta_ref,
                    "processor_ref": base.processor_ref,
                },
                "theta1": version,
            },
        )
        split = protocol["task_sets"]["release"]
        cells = [
            {
                "theta_id": theta,
                "b_id": b,
                "bundle_id": base.bundle_id
                if (theta, b) == ("theta0", "B0")
                else theta + "-" + b,
                "protocol_path": protocol_ref,
                "run_id": theta + "-" + b,
                "task_key": split["task_keys"][0],
                "seed": split["seeds"][0],
                "split": "release",
                "output_dir": artifacts / "planned" / theta / b,
            }
            for theta in ["theta0", "theta1"]
            for b in ["B0", "B1"]
        ]
        # The actual base uses its identity as the frozen anchor/current cell.
        # prepare_theta_b_cell creates a derived bundle; avoid rewriting base.
        cells[0]["bundle_id"] = "theta0-B0"
        prepared = prepare_theta_b_matrix(ROOT, artifacts, experiment, cells=cells)
        assert len(prepared["cells"]) == 4
        assert all(row["plan"] is not None for row in prepared["cells"])
        unavailable = deepcopy(experiment)
        unavailable["theta"]["theta1"].update(
            theta_ref=None, processor_ref=None, snapshot=None, processor_snapshot=None
        )
        unavailable["theta"]["theta1"].pop("load_receipt_ref", None)
        missing = prepare_theta_b_matrix(
            ROOT,
            artifacts,
            unavailable,
            cells=[
                {**cell, "bundle_id": "missing-" + cell["bundle_id"]} for cell in cells
            ],
        )
        assert [
            row["status"]
            for row in missing["cells"]
            if row["binding"]["theta_id"] == "theta1"
        ] == ["blocked_theta_unavailable"] * 2
        profile = json.loads((ROOT / "config/run_profile_v2.json").read_text())
        evaluations = EvaluationStore(artifacts / "evaluation")
        results = {}
        for cell in prepared["cells"]:
            metadata = cell["binding"]
            label = metadata["theta_id"] + "-" + metadata["effective_b"]
            bundle_id = cell["plan"]["context"]["bundle_id"]
            manifest = []
            assert (
                metadata["requested_b"] == metadata["effective_b"]
                and metadata["fallback"] is False
            )
            if metadata["theta_id"] == "theta1":
                assert metadata["theta_ref"] == version["theta_ref"]
                assert metadata["load_receipt_ref"] == version["load_receipt_ref"]
            assert metadata["applicability"] == "unknown"
            for task in protocol["tasks"]:
                if task["task_key"] not in split["task_keys"]:
                    continue
                for seed in split["seeds"]:
                    run_id = label + "-" + task["task_key"] + "-" + str(seed)
                    run = artifacts / "synthetic-native" / run_id
                    run.mkdir(parents=True)
                    native = {
                        "run_id": run_id,
                        "episode_id": run_id,
                        "bundle_id": bundle_id,
                        "protocol_id": protocol["protocol_id"],
                        "task": {
                            "suite": task["suite"],
                            "task": task["task_id"],
                            "seed": seed,
                        },
                        "planner": profile["planner"],
                        "benchmark_success": True,
                        "terminated": True,
                        "truncated": False,
                        "native_result_source": {
                            "terminated": "isolated synthetic Env boundary; not simulator evidence"
                        },
                    }
                    path = run / "result.json"
                    atomic_json(path, native)
                    manifest.append(
                        {
                            "task_key": task["task_key"],
                            "seed": seed,
                            "run_id": run_id,
                            "episode_id": run_id,
                            "result_path": str(path),
                        }
                    )
            independent = evaluate_matrix(
                protocol,
                profile,
                manifest,
                result_id=label,
                bundle_id=bundle_id,
                candidate_id="theta-matrix-candidate" if label == "theta1-B1" else None,
            )
            independent.intervention["theta_b_experiment"] = deepcopy(metadata)
            evaluations.save_result(independent)
            results[label] = independent
            identity = {
                key: cell["plan"]["context"][key]
                for key in ["run_id", "episode_id", "bundle_id", "protocol_id"]
            }
            linked = evaluate_theta_b_cell(cell, {**identity, "outcome": "unknown"})
            assert linked["intervention"]["theta_ref"] == metadata["theta_ref"]
            assert independent.unknown_count == 0 and independent.assigned_count == len(
                split["task_keys"]
            ) * len(split["seeds"])
            assert independent.hard_constraints["safety"] is None
        baseline = results["theta0-B0"]
        changed = results["theta1-B1"]
        # The paired registration uses the prepared current/anchor identity.
        protocol["anchor_bundle_id"] = baseline.bundle_id
        source_candidate = (
            Candidate(**json.loads(Path(candidate_ref).read_text()))
            if candidate_ref
            else None
        )
        if source_candidate:
            assert source_candidate.changes["theta_ref"] == version["theta_ref"]
            assert source_candidate.changes["processor_ref"] == version["processor_ref"]
        candidate = Candidate(
            "theta-matrix-candidate",
            baseline.bundle_id,
            changed.bundle_id,
            protocol["protocol_id"],
            protocol["benchmark_id"],
            {
                "theta_ref": version["theta_ref"],
                "processor_ref": version["processor_ref"],
                "memory_ref": "matrix-B1-memory",
            },
            list(
                dict.fromkeys(
                    [str(training_result_ref), str(load_result_ref)]
                    + (
                        source_candidate.artifact_refs + [str(candidate_ref)]
                        if source_candidate
                        else []
                    )
                )
            ),
            [],
            dict(source_candidate.costs) if source_candidate else {},
            {},
            "offline",
        )
        store.put("candidates", candidate.candidate_id, candidate)
        decision = Gate(protocol).evaluate(
            "theta-matrix-gate",
            candidate,
            current_bundle_id=baseline.bundle_id,
            candidate_result=changed,
            baseline_result=baseline,
            anchor_result=baseline,
        )
        evaluations.put("decisions", decision.decision_id, decision)
        assert decision.verdict == "inconclusive" and decision.release_eligible is False
        assert len(decision.result_refs) == 3
        publication = None
        if candidate_ref:
            from theta_publication import consume_training_publication

            publication = consume_training_publication(
                training_result_ref=training_result_ref,
                load_result_ref=load_result_ref,
                scope_id=scope_id,
                candidate_ref=candidate_ref,
            )
        return {
            "publication": publication,
            "schema": "rsi-theta-matrix-offline-integration-v1",
            "prepared_cells": 4,
            "independent_matrices": 4,
            "assigned_per_matrix": changed.assigned_count,
            "theta1_artifact_source": "actual_host_train_and_load_receipts",
            "source_candidate_ref": str(candidate_ref) if candidate_ref else None,
            "candidate_costs": dict(candidate.costs),
            "missing_theta1": "blocked_theta_unavailable",
            "gate_verdict": decision.verdict,
            "model_requests": 0,
            "gpu_seconds": 0,
            "environment_execution": "synthetic_boundary_only",
        }
