"""Real local audit/receipt consumers, with only native heavy boundaries replaced."""

import io
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from rsi_vla.adapters.pi05_training import (
    ACTION_CONTRACT,
    dataset_identity,
    native_train,
    native_verify_load,
    prepare_admitted_training,
    require_training_data_admission,
)
from rsi_vla.collection import review_training_dataset
from rsi_vla.theta_b import training_theta_version
from rsi_vla.training import TrainingBudget, finalize_training_job, training_handoff


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


@pytest.fixture
def local(tmp_path):
    """Two licensed synthetic frames, explicitly unrelated to project data."""
    data = tmp_path / "data"
    source = tmp_path / "source"
    info = {
        "fps": 20,
        "total_episodes": 1,
        "total_frames": 2,
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "features": {
            "state": {"shape": [8]},
            "actions": {"shape": [7]},
            "image": {"dtype": "image", "shape": [4, 4, 3]},
            "wrist_image": {"dtype": "image", "shape": [4, 4, 3]},
        },
    }
    image = io.BytesIO()
    Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3)).save(
        image, format="PNG"
    )
    rows = [
        {
            "state": [0.0] * 8,
            "actions": [0.0] * 6 + [-1.0],
            "frame_index": i,
            "episode_index": 0,
            "timestamp": i / 20,
            "image": {"bytes": image.getvalue()},
            "wrist_image": {"bytes": image.getvalue()},
        }
        for i in range(2)
    ]
    for root in (data, source):
        write(root / "meta/info.json", info)
        write(
            root / "meta/episodes.jsonl",
            {"episode_index": 0, "tasks": ["synthetic task"], "length": 2},
        )
        write(root / "meta/tasks.jsonl", {"task_index": 0, "task": "synthetic task"})
        (root / "data").mkdir()
        pq.write_table(pa.Table.from_pylist(rows), root / "data/episode_000000.parquet")
    provenance = write(source / "source.json", {"revision": "synthetic-v1"})
    manifest = {
        "source_kind": "public_demonstration",
        "dataset_root": str(data),
        "provenance_ref": provenance,
        "revision": "synthetic-v1",
        "action_contract": ACTION_CONTRACT,
        "information_scope": "training",
        "episodes": [
            {
                "episode_index": 0,
                "source_episode_index": 0,
                "task": "synthetic task",
                "source_ref": provenance,
            }
        ],
    }
    manifest_ref = write(tmp_path / "manifest.json", manifest)
    evidence = write(
        tmp_path / "synthetic-license-and-review.json", {"fixture_only": True}
    )
    common = {
        "scope_id": "fixture-scope",
        "protocol_id": "fixture-protocol",
        "reviewer_id": "fixture-independent",
        "dataset": dataset_identity(manifest),
        "decision": "accepted",
        "evidence_refs": [evidence],
    }
    quality = write(
        tmp_path / "quality.json",
        {
            **common,
            "schema": "rsi-demonstration-quality-review-v1",
            "image_orientation_verified": True,
            "state_action_alignment_verified": True,
            "task_relevance_verified": True,
            "license_verified": True,
        },
    )
    isolation = write(
        tmp_path / "isolation.json",
        {
            **common,
            "schema": "rsi-training-evaluation-isolation-v1",
            "source_episode_mapping_checked": True,
            "development_formal_split_checked": True,
            "sealed_labels_not_consumed": True,
        },
    )
    norm = write(
        tmp_path / "norm.json",
        {
            "norm_stats": {
                key: {field: [0.0] * 32 for field in ("mean", "std", "q01", "q99")}
                for key in ("state", "actions")
            }
        },
    )
    weights = tmp_path / "base/model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"synthetic-layout-only-not-model-weights")
    return types.SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        manifest_ref=manifest_ref,
        data=data,
        source=source,
        quality=quality,
        isolation=isolation,
        norm=norm,
        rows=rows,
    )


def review(local, **kwargs):
    args = dict(
        manifest_ref=local.manifest_ref,
        output_root=local.root / "review",
        scope_id="fixture-scope",
        protocol_id="fixture-protocol",
        reviewer_id="fixture-independent",
        quality_review_ref=local.quality,
        evaluation_isolation_review_ref=local.isolation,
    )
    args.update(kwargs)
    return review_training_dataset(**args)


def prepare(local):
    admitted = review(local)
    assert admitted["decision"] == "admitted"
    with (
        patch(
            "rsi_vla.adapters.pi05_training.verify_installed_sources", return_value={}
        ),
        patch(
            "rsi_vla.adapters.pi05_training.inspect_native_config",
            return_value={"native": "fixture"},
        ),
        patch(
            "rsi_vla.adapters.pi05_training.read_native_batch",
            return_value={"external_loader_fixture": True},
        ) as loader,
    ):
        result = prepare_admitted_training(
            manifest_ref=local.manifest_ref,
            admission_ref=admitted["admission_ref"],
            scope_id="fixture-scope",
            protocol_id="fixture-protocol",
            reviewer_id="fixture-independent",
            model_path=local.root / "base",
            norm_stats_path=local.norm,
            output_root=local.root / "prepared",
            run_id="train-fixture",
            budget=TrainingBudget(3, 3, 1, 1000000, 2),
        )
        loader.assert_called_once()
    assert result["training_steps"] == 0
    return result


def test_formal_audit_then_native_preparation_and_changed_payload_rejected(local):
    prepared = prepare(local)
    assert Path(prepared["recipe_ref"]).is_file()
    assert prepared["dataset_root"] == str(local.data)
    path = local.data / "data/episode_000000.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(PermissionError, match="payload changed"):
        require_training_data_admission(
            local.manifest,
            admission_ref=local.root / "review/data-admission.json",
            scope_id="fixture-scope",
            protocol_id="fixture-protocol",
            authority="fixture-independent",
        )


@pytest.mark.parametrize(
    "defect", ["clock", "missing_frame", "units", "evaluation", "provenance", "license"]
)
def test_actual_audit_rejects_misalignment_scope_and_unreviewed_sources(local, defect):
    if defect in {"clock", "missing_frame", "units"}:
        rows = local.rows.copy()
        if defect == "clock":
            rows[1] = {**rows[1], "timestamp": 0.5}
        if defect == "missing_frame":
            rows = rows[:1]
        if defect == "units":
            rows[1] = {**rows[1], "actions": [2.0] * 7}
        pq.write_table(
            pa.Table.from_pylist(rows), local.data / "data/episode_000000.parquet"
        )
    else:
        manifest = dict(local.manifest)
        if defect == "evaluation":
            manifest["information_scope"] = "sealed"
        if defect == "provenance":
            manifest["provenance_ref"] = ""
        if defect == "license":
            quality = json.loads(Path(local.quality).read_text())
            quality["license_verified"] = False
            write(Path(local.quality), quality)
        write(Path(local.manifest_ref), manifest)
    result = review(local)
    assert result["decision"] == "rejected"
    assert result["model_capability"] == "unknown"


def test_missing_review_is_unknown_and_repeat_dispatch_requires_receipt(local):
    result = review(local, quality_review_ref=None)
    assert result["decision"] == "unknown"
    with pytest.raises(FileExistsError):
        review(local)
    receipt = json.loads(Path(result["admission_ref"]).read_text())
    assert receipt["allowed_uses"] == ["preservation"]


def native_output(local):
    prepared = prepare(local)
    root = Path(prepared["recipe_ref"]).parent
    artifact = root / "train-fixture"
    weight = (
        artifact / "checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
    )
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"boundary-fixture-not-trained-weights")
    write(
        artifact / "weight-update-check.json",
        {
            "step": 2,
            "nonzero_sampled_weight_update": True,
            "sampled_frozen_parameters_equal": {"fixture": True},
        },
    )
    permit = {
        "scope_id": "fixture-scope",
        "operation_id": "train",
        "run_id": "train-fixture",
        "kind": "pi05-training-learning",
        "budget": {"max_steps": 2},
    }
    report = {
        "returncode": 0,
        "stop_reason": None,
        "cleanup_confirmed": True,
        "resource_limits_satisfied": True,
    }
    write(artifact / "supervision.json", report)
    write(
        artifact / "native-start.json",
        {
            "schema": "rsi-native-training-start-v1",
            **{
                key: permit[key]
                for key in ("scope_id", "operation_id", "run_id", "kind")
            },
            "entrypoint": "native_train",
            "fixture_only": True,
        },
    )
    result = finalize_training_job(
        prepared_root=root,
        artifact_root=artifact,
        permit=permit,
        supervision=report,
        python_executable=sys.executable,
    )
    return root, artifact, permit, report, result


def test_save_parse_load_bind_theta_and_processor_drift(local):
    root, artifact, permit, report, result = native_output(local)
    assert result["status"] == "completed"
    handoff = json.loads(Path(result["handoff_ref"]).read_text())
    load_root = root / "train-fixture-load-check"
    load_root.mkdir()
    load = {
        "actual_model_load": "passed",
        "prediction_shape": [1, 5, 7],
        "prediction_finite": True,
        "decoded_action_contract": ACTION_CONTRACT,
        "saved_probe_equal": {"external_model_fixture": True},
        "model_path": result["model_path"],
        "norm_stats_path": result["norm_stats_path"],
        "processor_snapshot": handoff["processor_snapshot"],
        "weights_identity": handoff["weights_identity"],
    }
    write(load_root / "load-check.json", load)
    write(load_root / "supervision.json", report)
    write(
        load_root / "native-start.json",
        {
            "schema": "rsi-native-training-start-v1",
            "scope_id": "fixture-scope",
            "operation_id": "load",
            "run_id": "train-fixture",
            "kind": "pi05-load-check",
            "entrypoint": "native_verify_load",
            "fixture_only": True,
        },
    )
    trained_ref = str(artifact / "job-result.json")
    loaded = finalize_training_job(
        prepared_root=root,
        artifact_root=load_root,
        permit={**permit, "kind": "pi05-load-check", "operation_id": "load"},
        supervision=report,
        python_executable=sys.executable,
        training_result_ref=trained_ref,
    )
    assert loaded["status"] == "completed"
    version = training_theta_version(
        training_result_ref=trained_ref,
        load_result_ref=load_root / "job-result.json",
        scope_id="fixture-scope",
    )
    assert version["theta_ref"] == loaded["theta_ref"]
    assert version["capability_gain"] == "unknown"
    write(Path(loaded["norm_stats_path"]), {"norm_stats": {}})
    with pytest.raises((ValueError, PermissionError, KeyError)):
        training_handoff(load_root / "job-result.json", require_load=True)


@pytest.mark.parametrize("defect", ["over_budget", "half_save", "path_drift"])
def test_failure_or_partial_model_never_exports_theta(local, defect):
    root, artifact, permit, report, _ = native_output(local)
    if defect == "over_budget":
        report["resource_limits_satisfied"] = False
    if defect == "half_save":
        (
            artifact
            / "checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
        ).unlink()
    if defect == "path_drift":
        (artifact / "checkpoints/global_step_2").rename(
            artifact / "checkpoints/global_step_1"
        )
    result = finalize_training_job(
        prepared_root=root,
        artifact_root=artifact,
        permit=permit,
        supervision=report,
        python_executable=sys.executable,
    )
    assert result["status"] == "failed"
    assert result["handoff_ref"] is None
    assert "theta_ref" not in result
    assert result["errors"]


def test_native_trainer_calls_original_runner_and_cleans_ray(local, monkeypatch):
    calls = []

    class Worker:
        @classmethod
        def create_group(cls, cfg):
            calls.append("native_worker")
            return types.SimpleNamespace(launch=lambda *a, **k: "actors")

    class Runner:
        def __init__(self, **kwargs):
            calls.append(("SFTRunner", kwargs["actor"]))

        def init_workers(self):
            calls.append("init_workers")

        def run(self):
            calls.append("run")

    class Placement:
        def __init__(self, *args):
            pass

        def get_strategy(self, name):
            return name

    modules = {
        "torch": types.SimpleNamespace(
            multiprocessing=types.SimpleNamespace(set_start_method=lambda *a, **k: None)
        ),
        "torch.multiprocessing": types.SimpleNamespace(
            set_start_method=lambda *a, **k: None
        ),
        "omegaconf": types.SimpleNamespace(
            OmegaConf=types.SimpleNamespace(
                load=lambda p: types.SimpleNamespace(
                    cluster={}, actor=types.SimpleNamespace(group_name="actor")
                )
            )
        ),
        "rlinf.config": types.SimpleNamespace(validate_cfg=lambda x: x),
        "rlinf.runners.sft_runner": types.SimpleNamespace(SFTRunner=Runner),
        "rlinf.scheduler": types.SimpleNamespace(Cluster=lambda **k: "cluster"),
        "rlinf.utils.placement": types.SimpleNamespace(
            HybridComponentPlacement=Placement
        ),
        "rlinf.workers.sft.fsdp_vla_sft_worker": types.SimpleNamespace(
            FSDPVlaSftWorker=Worker
        ),
        "ray": types.SimpleNamespace(shutdown=lambda: calls.append("ray_shutdown")),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    recipe = write(
        local.root / "recipe.json",
        {"runner": {"logger": {"experiment_name": "fixture"}}},
    )
    env = {
        "RSI_TRAINING_JOB_ROOT": str(local.root),
        "RSI_TRAINING_SCOPE_ID": "fixture-scope",
        "RSI_TRAINING_OPERATION_ID": "train",
        "RSI_TRAINING_KIND": "pi05-training-learning",
    }
    with patch.dict(os.environ, env):
        native_train(recipe)
    assert calls == [
        "native_worker",
        ("SFTRunner", "actors"),
        "init_workers",
        "run",
        "ray_shutdown",
    ]
    assert (
        json.loads((local.root / "native-start.json").read_text())[
            "weight_update_established"
        ]
        is False
    )


def test_load_check_uses_actual_rpent_facade_with_external_model_boundary(
    local, monkeypatch
):
    import torch

    from rpent.robots.components import pi05_vla_server

    root, artifact, _permit, _report, trained = native_output(local)
    calls = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def named_parameters(self):
            return [("probe", torch.tensor([3.0]))]

        def predict_action_batch(self, wire, mode):
            calls.append(("prediction", wire["states"].shape, mode))
            return torch.zeros((1, 5, 7)), None

    def loader(config, torch_dtype):
        calls.append(("loader", config.model_path, config.openpi_data.norm_stats_path))
        return Model()

    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi",
        types.SimpleNamespace(get_model=loader),
    )
    sample = {
        "state": np.zeros(8),
        "image": np.zeros((3, 4, 4)),
        "wrist_image": np.zeros((3, 4, 4)),
        "task": "synthetic task",
    }
    monkeypatch.setitem(
        sys.modules,
        "lerobot.datasets.lerobot_dataset",
        types.SimpleNamespace(LeRobotDataset=lambda *a, **k: [sample]),
    )
    probe = artifact / "probes-after.pt"
    torch.save({"probe": torch.tensor([3.0])}, probe)
    report_ref = local.root / "native-load-check.json"
    env = {
        "RSI_TRAINING_JOB_ROOT": str(local.root),
        "RSI_TRAINING_SCOPE_ID": "fixture-scope",
        "RSI_TRAINING_OPERATION_ID": "load",
        "RSI_TRAINING_KIND": "pi05-load-check",
    }
    # Keep Pi05VLAFacade.__init__/predict and native adapter unchanged.
    with patch.dict(os.environ, env):
        report = native_verify_load(
            root / "recipe.json", trained["model_path"], probe, report_ref
        )
    assert report["actual_model_load"] == "passed"
    assert report["service_class"].endswith(".Pi05VLAFacade")
    assert report["prediction_shape"] == [1, 5, 7]
    assert calls == [
        ("loader", trained["model_path"], trained["norm_stats_path"]),
        ("prediction", (1, 8), "eval"),
    ]
    assert (
        pi05_vla_server.Pi05VLAFacade.__module__
        == "rpent.robots.components.pi05_vla_server"
    )


def test_complete_theta_matrix_blocks_missing_models_and_labels_fallbacks():
    from rsi_vla.theta_b import prepare_theta_b_matrix, record_theta_b_applicability
    from test_theta_b import ThetaBTests

    fixture = ThetaBTests()
    fixture.setUp()
    try:
        cells = [
            dict(
                theta_id=theta,
                b_id=b,
                bundle_id=theta + "-" + b,
                protocol_path=Path(__file__).resolve().parents[2]
                / "config/pilot_v2.json",
                run_id=theta + "-" + b,
                task_key="position_swap_salad",
                seed=0,
                split="smoke",
                output_dir=fixture.artifacts / theta / b,
            )
            for theta in ("theta0", "theta1")
            for b in ("B0", "B1")
        ]
        cells[1]["fallback_b_id"] = "B0"
        matrix = prepare_theta_b_matrix(
            Path(__file__).resolve().parents[2],
            fixture.artifacts,
            fixture.experiment,
            cells=cells,
        )
        assert len(matrix["cells"]) == 4
        assert matrix["cells"][1]["binding"]["effective_b"] == "B0"
        assert matrix["cells"][1]["binding"]["fallback"] is True
        assert matrix["missing_effective_cells"] == [
            ["theta0", "B1"],
            ["theta1", "B0"],
            ["theta1", "B1"],
        ]
        assert [row["status"] for row in matrix["cells"][2:]] == [
            "blocked_theta_unavailable"
        ] * 2
        with pytest.raises(ValueError, match="all unique"):
            prepare_theta_b_matrix(
                Path(__file__).resolve().parents[2],
                fixture.artifacts,
                fixture.experiment,
                cells=cells[:-1] + [cells[0]],
            )
        theta = fixture.experiment["theta"]["theta0"]
        evidence = {
            "schema": "rsi-theta-b-applicability-v1",
            "reviewer_role": "evaluation",
            "experiment_id": fixture.experiment["experiment_id"],
            "theta_ref": theta["theta_ref"],
            "processor_ref": theta["processor_ref"],
            "b_id": "B1",
            "decision": "inapplicable",
        }
        reference = write(fixture.artifacts / "applicability.json", evidence)
        reviewed = record_theta_b_applicability(
            fixture.experiment,
            theta_id="theta0",
            b_id="B1",
            decision="inapplicable",
            evidence_ref=reference,
            fallback_b_id="B0",
        )
        blocked = prepare_theta_b_matrix(
            Path(__file__).resolve().parents[2],
            fixture.artifacts,
            reviewed,
            cells=[
                {**row, "bundle_id": "reviewed-" + row["bundle_id"]} for row in cells
            ],
        )
        assert (
            blocked["cells"][1]["binding"]["fallback_reason"]
            == "explicit_inapplicable_applicability"
        )
        write(Path(reference), {**evidence, "decision": "applicable"})
        with pytest.raises(ValueError, match="applicability evidence changed"):
            fixture.experiment = reviewed
            fixture.prepare()
    finally:
        fixture.doCleanups()


def test_native_dataloader_rejects_recipe_pointing_at_another_dataset(local):
    from rsi_vla.adapters.pi05_training import build_recipe, read_native_batch

    recipe = build_recipe(
        dataset_root=local.root / "unreviewed",
        model_path=local.root / "base",
        norm_stats_path=local.norm,
        output_root=local.root,
        run_id="mismatch",
        budget=TrainingBudget(3, 3, 1, 1000000, 2),
    )
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}):
        with pytest.raises(ValueError, match="reviewed dataset manifest"):
            read_native_batch(recipe, local.manifest)
