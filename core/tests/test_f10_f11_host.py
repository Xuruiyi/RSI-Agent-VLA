# Copyright 2026 RSI-Agent-VLA Authors.
"""Real CPU Host graph; native model/trainer/data factories are synthetic boundaries."""

import json
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rsi_vla.contracts import RunBundle
from rsi_vla.midterm_host import MidtermHost, build_configuration, read
from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.training import TrainingBudget, supervise_process

ROOT = Path(__file__).resolve().parents[2]


def native_boundaries(config):
    """Replace RLinf/OpenPI heavy components; keep all owning adapters unchanged."""
    from rsi_vla.adapters.pi05_training import ACTION_CONTRACT

    os.environ.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    task_root = Path(config["isolation_task_root"])
    os.environ["OPENPI_DATA_HOME"] = str(task_root / "operator/tokenizer")
    transforms = types.SimpleNamespace(inputs=[])
    data = types.SimpleNamespace(
        norm_stats={},
        use_quantile_norm=True,
        repack_transforms=transforms,
        data_transforms=transforms,
        model_transforms=transforms,
    )
    native = types.SimpleNamespace(
        model=types.SimpleNamespace(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            max_token_len=200,
        ),
        assets_dirs=[],
        data=types.SimpleNamespace(extra_delta_transform=False, create=lambda *a: data),
    )

    def batches(*args):
        obs = types.SimpleNamespace(
            state=np.zeros((1, 32)),
            images={
                key: np.zeros((1, 3, 224, 224))
                for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
        )
        return [(obs, np.zeros((1, 10, 32)))], None

    sys.modules["rlinf.models.embodiment.openpi.dataconfig"] = types.SimpleNamespace(
        get_openpi_config=lambda *a, **k: native
    )
    sys.modules["rlinf.data.datasets.openpi"] = types.SimpleNamespace(
        build_openpi_sft_dataloader=batches
    )

    # The boundary worker emits explicit synthetic fixture save/probe artifacts;
    # these establish framework wiring only, never gradient-learning evidence.
    class Worker:
        @classmethod
        def create_group(cls, cfg):
            return types.SimpleNamespace(
                launch=lambda *a, **k: "external-fixture-actor"
            )

    class Runner:
        def __init__(self, cfg, actor):
            self.cfg = cfg

        def init_workers(self):
            pass

        def run(self):
            import torch

            artifact = Path(os.environ["RSI_TRAINING_JOB_ROOT"])
            step = self.cfg.runner.max_steps
            weights = (
                artifact
                / f"checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"
            )
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"SYNTHETIC_EXTERNAL_BOUNDARY_NOT_MODEL_WEIGHTS")
            torch.save({"probe": torch.tensor([1.0])}, artifact / "probes-after.pt")
            atomic_json(
                artifact / "weight-update-check.json",
                {
                    "step": step,
                    "nonzero_sampled_weight_update": True,
                    "sampled_frozen_parameters_equal": {"external_fixture": True},
                    "fixture_only": True,
                    "real_gradient_steps": 0,
                },
            )

    class Placement:
        def __init__(self, *a):
            pass

        def get_strategy(self, key):
            return key

    sys.modules["rlinf.config"] = types.SimpleNamespace(validate_cfg=lambda cfg: cfg)
    sys.modules["rlinf.runners.sft_runner"] = types.SimpleNamespace(SFTRunner=Runner)
    sys.modules["rlinf.scheduler"] = types.SimpleNamespace(Cluster=lambda **k: None)
    sys.modules["rlinf.utils.placement"] = types.SimpleNamespace(
        HybridComponentPlacement=Placement
    )
    sys.modules["rlinf.workers.sft.fsdp_vla_sft_worker"] = types.SimpleNamespace(
        FSDPVlaSftWorker=Worker
    )
    sys.modules["ray"] = types.SimpleNamespace(shutdown=lambda: None)

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def named_parameters(self):
            import torch

            return [("probe", torch.tensor([1.0]))]

        def predict_action_batch(self, obs, mode):
            import torch

            return torch.zeros((1, ACTION_CONTRACT["execution_chunk"], 7)), None

    sys.modules["rlinf.models.embodiment.openpi"] = types.SimpleNamespace(
        get_model=lambda *a, **k: Model()
    )
    sample = {
        "state": np.zeros(8),
        "image": np.zeros((3, 4, 4)),
        "wrist_image": np.zeros((3, 4, 4)),
        "task": "synthetic task",
    }
    sys.modules["lerobot.datasets.lerobot_dataset"] = types.SimpleNamespace(
        LeRobotDataset=lambda *a, **k: [sample]
    )
    return {}


def run_native_boundary(config_ref, mode):
    """Host command subprocess invokes the real native adapter under supervision."""
    config = read(config_ref)
    native_boundaries(config)
    root = Path(config["isolation_task_root"])
    inputs = read(os.environ["RSI_OPERATION_INPUT_REF"])
    permit = read(os.environ["RSI_CPU_PERMIT_REF"])
    prepared = root / "runtime/prepared"
    artifact = prepared / (
        "train-fixture" if mode == "train" else "train-fixture-load-check"
    )
    identity = "train-boundary" if mode == "train" else "load-boundary"
    kind = "pi05-training-learning" if mode == "train" else "pi05-load-check"
    env = dict(
        os.environ,
        RSI_TRAINING_SCOPE_ID="fixture-scope",
        RSI_TRAINING_OPERATION_ID=identity,
        RSI_TRAINING_KIND=kind,
        HF_HUB_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    code = (
        "from test_f10_f11_host import native_boundaries; from rsi_vla.storage import atomic_json; "
        "import json,sys; from pathlib import Path; "
        "native_boundaries(json.loads(Path(sys.argv[1]).read_text())); "
    )
    if mode == "train":
        code += "from rsi_vla.adapters.pi05_training import native_train; native_train(sys.argv[2])"
        command = [
            sys.executable,
            "-c",
            code,
            config_ref,
            str(prepared / "recipe.json"),
        ]
    else:
        from rsi_vla.training import training_handoff

        trained, _handoff = training_handoff(
            inputs["references"]["training_result_ref"]
        )
        assert trained["status"] == "completed"
        code += "from rsi_vla.adapters.pi05_training import native_verify_load; native_verify_load(*sys.argv[2:])"
        command = [
            sys.executable,
            "-c",
            code,
            config_ref,
            str(prepared / "recipe.json"),
            trained["model_path"],
            str(prepared / "train-fixture/probes-after.pt"),
            str(artifact / "load-check.json"),
        ]
    budget = TrainingBudget(15, 15, 1, 3 * 1024**2, 2)
    result = supervise_process(command, budget=budget, artifact_root=artifact, env=env)
    assert result["returncode"] == 0, (artifact / "trainer.log").read_text()
    assert result["cleanup_confirmed"] and result["resource_limits_satisfied"]
    atomic_json(
        permit["result_ref"],
        {
            "schema": "rsi-cpu-operation-result-v1",
            "scope_id": permit["scope_id"],
            "operation_id": permit["operation_id"],
            "version": permit["version"],
            "status": "completed",
            "artifact_ref": str(artifact / "supervision.json"),
            "fixture_only": True,
            "real_training_steps": 0,
            "gpu_devices": [],
        },
    )


def run_host_graph(*, after_bundle=None):
    from test_f10_f11_pipeline import local as local_fixture
    from test_host_receipt_inputs import HostReceiptInputsTests

    fixture = HostReceiptInputsTests()
    fixture.setUp()
    try:
        root = fixture.root
        local = local_fixture.__wrapped__(fixture.operator / "synthetic")
        prepared = root / "runtime/prepared"
        protocol = {
            "protocol_id": "fixture-protocol",
            "scope_id": "fixture-scope",
            "scope": "offline",
            "status": "frozen",
            "benchmark_id": "synthetic",
            "anchor_bundle_id": "base",
            "metrics": {
                "success": {"direction": "max", "min": 0, "max": 1, "weight": 1}
            },
            "hard_constraints": ["resource_limits"],
            "min_gain": 0,
            "anchor_max_drop": 0,
            "task_policy": {"schema": "rsi-task-policy-v1"},
        }
        atomic_json(fixture.operator / "protocol.json", protocol)
        artifacts = root / "state"
        store = ResearchStore(artifacts / "research")
        for kind in ("psi", "rho", "memory"):
            store.put(kind, kind, {})
        store.save_bundle(
            RunBundle(
                "base",
                "theta0",
                "processor0",
                "phi",
                "rho",
                "psi",
                "memory",
                "fixture-protocol",
                "backend",
                "code",
            )
        )
        budget = TrainingBudget(15, 15, 1, 3 * 1024**2, 2)
        plan = {
            "schema": "rsi-pi05-learning-plan-v1",
            "candidate_id": "trained",
            "protocol_id": "fixture-protocol",
            "data_admission_ref": str(
                root / "state/evaluation/review/data-admission.json"
            ),
            "design_source": "offline_fixture",
            "stop_rule": "fixed_optimizer_steps",
            "checkpoint_selection": "final_step",
            "max_steps": 2,
            "global_batch_size": 1,
            "micro_batch_size": 1,
            "warmup_steps": 0,
            "seed": 0,
            "learning_rate": 1e-5,
            "lr_scheduler": "constant",
        }
        train_permit = {
            "scope_id": "fixture-scope",
            "operation_id": "train-boundary",
            "run_id": "train-fixture",
            "kind": "pi05-training-learning",
            "budget": budget.__dict__,
        }
        load_permit = {
            **train_permit,
            "operation_id": "load-boundary",
            "kind": "pi05-load-check",
        }
        atomic_json(fixture.operator / "train-permit.json", train_permit)
        atomic_json(fixture.operator / "load-permit.json", load_permit)
        tokenizer = fixture.operator / "tokenizer/big_vision/paligemma_tokenizer.model"
        tokenizer.parent.mkdir(parents=True)
        tokenizer.write_text("SYNTHETIC_TOKENIZER_FACTORY_BOUNDARY")

        def op(name, kind, payload):
            return {
                "operation_id": name,
                "stage": "engineering",
                "kind": kind,
                "dynamic": False,
                "payload": payload,
                "version": "base",
            }

        def export(field):
            return {"field": field, "version": "base", "type": "path"}

        def dependency(operation, name, target=None):
            value = {"operation_id": operation, "export": name, "version": "base"}
            if target:
                value["target"] = ["payload", target]
            return value

        operations = [
            op(
                "review",
                "data_review",
                {
                    "manifest_ref": local.manifest_ref,
                    "output_root": str(root / "state/evaluation/review"),
                    "reviewer_id": "fixture-independent",
                    "quality_review_ref": local.quality,
                    "evaluation_isolation_review_ref": local.isolation,
                },
            ),
            op(
                "prepare",
                "training_prepare",
                {
                    "manifest_ref": local.manifest_ref,
                    "output_root": str(prepared),
                    "reviewer_id": "fixture-independent",
                    "model_path": str(local.root / "base"),
                    "norm_stats_path": local.norm,
                    "run_id": "train-fixture",
                    "budget": budget.__dict__,
                    "learning_plan": plan,
                },
            ),
            op("train-boundary", "host_command", {}),
            op(
                "parse-train",
                "training_output_parse",
                {
                    "prepared_root": str(prepared),
                    "artifact_root": str(prepared / "train-fixture"),
                    "permit_ref": str(fixture.operator / "train-permit.json"),
                    "supervision_ref": str(prepared / "train-fixture/supervision.json"),
                },
            ),
            op("load-boundary", "host_command", {}),
            op(
                "parse-load",
                "training_output_parse",
                {
                    "prepared_root": str(prepared),
                    "artifact_root": str(prepared / "train-fixture-load-check"),
                    "permit_ref": str(fixture.operator / "load-permit.json"),
                    "supervision_ref": str(
                        prepared / "train-fixture-load-check/supervision.json"
                    ),
                },
            ),
            op(
                "bundle",
                "training_bundle_prepare",
                {
                    "artifacts": str(artifacts),
                    "base_bundle_id": "base",
                    "bundle_id": "trained",
                    "candidate_id": "trained-candidate",
                    "benchmark_id": "synthetic",
                    "cost_receipt_refs": [
                        str(
                            root
                            / "operator/midterm-host/jobs"
                            / name
                            / "completion.json"
                        )
                        for name in (
                            "review",
                            "prepare",
                            "train-boundary",
                            "parse-train",
                            "load-boundary",
                            "parse-load",
                        )
                    ],
                },
            ),
        ]
        operations[0]["exports"] = {"admission": export("admission_ref")}
        operations[1]["late_refs"] = {
            "admission_ref": dependency("review", "admission", "admission_ref")
        }
        operations[1]["exports"] = {"recipe": export("recipe_ref")}
        operations[2].update(cpu_purpose="analysis")
        operations[2]["late_refs"] = {"recipe_ref": dependency("prepare", "recipe")}
        operations[3]["exports"] = {"result": export("result_ref")}
        operations[4].update(
            cpu_purpose="analysis",
            late_refs={"training_result_ref": dependency("parse-train", "result")},
        )
        operations[5]["late_refs"] = {
            "training_result_ref": dependency(
                "parse-train", "result", "training_result_ref"
            )
        }
        operations[5]["exports"] = {"result": export("result_ref")}
        operations[6]["late_refs"] = {
            "training_result_ref": dependency(
                "parse-train", "result", "training_result_ref"
            ),
            "load_result_ref": dependency("parse-load", "result", "load_result_ref"),
        }
        for item in operations:
            role = {"review": "state/evaluation", "bundle": "state/research"}.get(
                item["operation_id"], "runtime"
            )
            item["storage_write_roots"] = [str(root / role)]
        fixture.scope.update(
            scope_id="fixture-scope",
            cpu_operation_ids=[o["operation_id"] for o in operations],
        )
        fixture.scope["outer_limits"].update(
            wall_seconds=180,
            storage_bytes=80 * 1024**2,
            current_disk_bytes=80 * 1024**2,
        )
        atomic_json(fixture.scope_ref, fixture.scope)
        limits = {
            k: v
            for k, v in fixture.scope["outer_limits"].items()
            if k != "concurrent_gpu_devices"
        }
        atomic_json(
            fixture.phase,
            {
                "limits": limits,
                "used": dict.fromkeys(limits, 0),
                "remaining": limits,
                "unresolved_costs": [],
            },
        )
        admin = fixture.operator / "admin.json"
        atomic_json(
            admin,
            {
                "authorized": True,
                "scope": "fixture-scope",
                "protocol_id": "fixture-protocol",
                "expires_at": fixture.scope["expires_at"],
                "phase_snapshot_ref": str(fixture.phase),
                "approval_reference": str(fixture.scope_ref),
            },
        )
        names = [o["operation_id"] for o in operations]
        fixture.core.update(
            operations=operations,
            base_bundle_id="base",
            repository_root=str(ROOT),
            offline_fixture=True,
            offline_boundary_factory="test_f10_f11_host:native_boundaries",
            administrative_authorization_ref=str(admin),
            frozen_protocol_ref=str(fixture.operator / "protocol.json"),
            research_store_ref=str(store.root),
            evaluation_store_ref=str(root / "state/evaluation"),
            continuation={
                "schema": "rsi-task-continuation-v1",
                "start": names[0],
                "nodes": {
                    **{
                        name: {
                            "operation_id": name,
                            "transitions": [
                                {"next": names[i + 1] if i + 1 < len(names) else "done"}
                            ],
                        }
                        for i, name in enumerate(names)
                    },
                    "done": {"terminal": "offline_complete"},
                },
            },
        )
        atomic_json(fixture.core_ref, fixture.core)
        configuration = build_configuration(
            fixture.core_ref, fixture.host_ref, fixture.operator / "hub.json"
        )
        for name, template in configuration["operations"].items():
            template["budget"].update(
                wall_seconds=25,
                storage_bytes=10 * 1024**2,
                current_disk_bytes=10 * 1024**2,
            )
            template["supervisor_limits"].update(
                wall_seconds=25, storage_bytes=10 * 1024**2
            )
        for name, mode in [("train-boundary", "train"), ("load-boundary", "load")]:
            template = configuration["operations"][name]
            template.update(
                cpu_role="runtime",
                storage_write_roots=[str(root / "runtime")],
                result_ref=str(root / "runtime/cpu_operations" / name / "result.json"),
            )
            template["command"] = [
                sys.executable,
                "-c",
                "from test_f10_f11_host import run_native_boundary; import sys; run_native_boundary(*sys.argv[1:])",
                str(fixture.core_ref),
                mode,
            ]
        atomic_json(configuration["inputs_ref"], {})
        atomic_json(fixture.host_ref, configuration)
        env = {
            "PYTHONPATH": str(ROOT / "core/src") + ":" + str(ROOT / "core/tests"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
        }
        with patch.dict(os.environ, env):
            deadline = time.monotonic() + 150
            host = MidtermHost(fixture.host_ref)
            while time.monotonic() < deadline:
                state = host.advance()
                if state["status"] in {"completed", "unknown", "failed"}:
                    break
                time.sleep(0.05)
            if state["status"] != "completed":
                raise AssertionError(
                    json.dumps(
                        {
                            "state": state,
                            "pending_receipt": host.coordinator().status()[
                                "pending_receipt"
                            ],
                            "logs": {
                                str(p.relative_to(root)): p.read_text()
                                for p in root.rglob("*.log")
                            },
                            "completions": {
                                str(p.relative_to(root)): read(p)
                                for p in root.rglob("completion.json")
                            },
                        },
                        indent=2,
                    )
                )
            used = host.ledger.read()["used"]
            assert used["model_requests"] == 0 and used["gpu_seconds"] == 0
            assert all(
                r["status"] == "settled" and r["cleanup_confirmed"]
                for r in host.ledger.read()["reservations"].values()
            )
            assert used["wall_seconds"] > 0 and used["storage_bytes"] > 0
            assert host.advance()["status"] == "completed"
        candidate = store.get("bundles", "trained")
        assert candidate["theta_ref"].endswith(
            "deployment-handoff-verified.json#/model_descriptor"
        )
        assert candidate["processor_ref"].startswith(str(prepared))
        preparation = read(prepared / "preparation.json")
        assert preparation["batch"]["dataset"] == preparation["dataset"]
        assert preparation["batch"]["episodes"] == 1
        assert preparation["batch"]["source_frames"] == 2
        submitted = store.get("candidates", "trained-candidate")
        assert submitted["changes"]["theta_ref"] == candidate["theta_ref"]
        assert submitted["compatibility"]["task_gain"] == "unknown"
        assert submitted["costs"]["whole_job_wall_seconds"] > 0
        assert submitted["costs"]["gpu_device_seconds_envelope"] == 0
        assert submitted["compatibility"]["pending_serialization_cost_operations"] == [
            "bundle"
        ]
        import pytest
        from rsi_vla.theta_b import _training_candidate_costs

        history = [
            root / "operator/midterm-host/jobs" / name / "completion.json"
            for name in ("train-boundary", "load-boundary")
        ]
        with pytest.raises(PermissionError, match="omit"):
            _training_candidate_costs(
                history[:1],
                scope_id="fixture-scope",
                required_operations={"train-boundary", "load-boundary"},
            )
        with pytest.raises(PermissionError, match="duplicate"):
            _training_candidate_costs(
                history + history[:1],
                scope_id="fixture-scope",
                required_operations={"train-boundary", "load-boundary"},
            )
        result = read(prepared / "train-fixture-load-check/job-result.json")
        assert result["capability_evaluation"] == "not_run"
        assert (
            read(prepared / "train-fixture/weight-update-check.json")[
                "real_gradient_steps"
            ]
            == 0
        )
        if after_bundle is not None:
            after_bundle(
                training_result_ref=prepared / "train-fixture/job-result.json",
                load_result_ref=prepared / "train-fixture-load-check/job-result.json",
                scope_id="fixture-scope",
                candidate_ref=store.path("candidates", "trained-candidate"),
            )
    finally:
        fixture.doCleanups()


def test_real_host_data_training_load_bundle_graph():
    from test_theta_evaluation import consume_training_matrix

    run_host_graph(after_bundle=consume_training_matrix)


@pytest.mark.parametrize(
    "audit_case",
    [
        None,
        "forged",
        "stale",
        "wrong_identity",
        "missing_coverage",
        "valid",
        "research_admission",
    ],
)
def test_real_host_independent_review_and_payload_cache(audit_case):
    """Evaluation owns admission and checks cached claims against the actual payload."""
    from test_f10_f11_pipeline import local as local_fixture
    from test_host_receipt_inputs import HostReceiptInputsTests

    fixture = HostReceiptInputsTests()
    fixture.setUp()
    try:
        root = fixture.root
        scope_id = "f10-independent-review-rejection"
        protocol_id = "f10-review-boundary-offline"
        local = local_fixture.__wrapped__(fixture.operator / "synthetic")
        research_review = root / "state/research/forged-quality-review.json"
        quality = read(local.quality)
        quality.update(scope_id=scope_id, protocol_id=protocol_id)
        atomic_json(research_review, quality)
        isolation = read(local.isolation)
        isolation.update(scope_id=scope_id, protocol_id=protocol_id)
        atomic_json(local.isolation, isolation)
        quality_ref = str(research_review)
        cache_ref = None
        if audit_case is not None:
            from rsi_vla.adapters.pi05_training import audit_demonstration_payload
            from rsi_vla.collection import dataset_file_snapshot

            atomic_json(local.quality, quality)
            quality_ref = local.quality
            audit = audit_demonstration_payload(local.manifest)
            assert audit["checks_passed"] is True
            cache_ref = root / "state/research/payload-audit.json"
            if audit_case == "forged":
                import pyarrow as pa
                import pyarrow.parquet as pq

                rows = [dict(row, actions=[99.0] * 7) for row in local.rows]
                for data_root in (local.data, local.source):
                    pq.write_table(
                        pa.Table.from_pylist(rows),
                        data_root / "data/episode_000000.parquet",
                    )
                # Preserve the old successful claims while updating only the writable snapshot.
                audit["payload_snapshot"] = dataset_file_snapshot(local.manifest)
            elif audit_case == "stale":
                audit["payload_snapshot"] = {}
            elif audit_case == "wrong_identity":
                audit["dataset"] = {**audit["dataset"], "revision": "other"}
            elif audit_case == "missing_coverage":
                audit["episodes"] = []
            atomic_json(cache_ref, audit)
        protocol_ref = fixture.operator / "protocol.json"
        atomic_json(
            protocol_ref,
            {
                "protocol_id": protocol_id,
                "scope_id": scope_id,
                "scope": "offline",
                "status": "frozen",
                "benchmark_id": "synthetic",
                "anchor_bundle_id": "base",
                "metrics": {
                    "success": {"direction": "max", "min": 0, "max": 1, "weight": 1}
                },
                "hard_constraints": ["resource_limits"],
                "min_gain": 0,
                "anchor_max_drop": 0,
                "task_policy": {"schema": "rsi-task-policy-v1"},
            },
        )
        store = ResearchStore(root / "state/research")
        for kind in ("psi", "rho", "memory"):
            store.put(kind, kind, {})
        store.save_bundle(
            RunBundle(
                "base",
                "theta0",
                "processor0",
                "phi",
                "rho",
                "psi",
                "memory",
                protocol_id,
                "backend",
                "code",
            )
        )
        output = root / "state/evaluation/rejected-review"
        operation = {
            "operation_id": "review",
            "stage": "engineering",
            "kind": "data_review",
            "dynamic": False,
            "version": "base",
            "storage_write_roots": [str(root / "state/evaluation")],
            "payload": {
                "manifest_ref": local.manifest_ref,
                "output_root": str(output),
                "reviewer_id": "fixture-independent",
                "quality_review_ref": quality_ref,
                "evaluation_isolation_review_ref": local.isolation,
            },
        }
        if cache_ref is not None:
            operation["payload"]["payload_audit_ref"] = str(cache_ref)
        if audit_case == "research_admission":
            from rsi_vla.collection import review_training_dataset

            # Even a complete receipt cannot grant trust to a Research-writable path.
            untrusted = review_training_dataset(
                manifest_ref=local.manifest_ref,
                output_root=root / "state/research/claimed-admission",
                scope_id=scope_id,
                protocol_id=protocol_id,
                reviewer_id="fixture-independent",
                quality_review_ref=local.quality,
                evaluation_isolation_review_ref=local.isolation,
            )
            assert untrusted["decision"] == "admitted"
            output = root / "runtime/rejected-preparation"
            operation.update(
                kind="training_prepare", storage_write_roots=[str(root / "runtime")]
            )
            operation["payload"] = {
                "manifest_ref": local.manifest_ref,
                "admission_ref": untrusted["admission_ref"],
                "output_root": str(output),
                "reviewer_id": "fixture-independent",
                "model_path": str(local.root / "base"),
                "norm_stats_path": str(local.norm),
                "run_id": "untrusted-admission",
                "budget": TrainingBudget(3, 3, 1, 1000000, 2).__dict__,
                "read_batch": False,
            }
        fixture.scope.update(scope_id=scope_id, cpu_operation_ids=["review"])
        atomic_json(fixture.scope_ref, fixture.scope)
        admin_ref = fixture.operator / "admin.json"
        atomic_json(
            admin_ref,
            {
                "authorized": True,
                "scope": scope_id,
                "protocol_id": protocol_id,
                "expires_at": fixture.scope["expires_at"],
                "phase_snapshot_ref": str(fixture.phase),
                "approval_reference": str(fixture.scope_ref),
            },
        )
        fixture.core.update(
            operations=[operation],
            base_bundle_id="base",
            repository_root=str(ROOT),
            administrative_authorization_ref=str(admin_ref),
            frozen_protocol_ref=str(protocol_ref),
            research_store_ref=str(store.root),
            evaluation_store_ref=str(root / "state/evaluation"),
            continuation={
                "schema": "rsi-task-continuation-v1",
                "start": "review",
                "nodes": {
                    "review": {
                        "operation_id": "review",
                        "transitions": [{"next": "done"}],
                    },
                    "done": {"terminal": "offline_complete"},
                },
            },
        )
        atomic_json(fixture.core_ref, fixture.core)
        configuration = build_configuration(
            fixture.core_ref, fixture.host_ref, fixture.operator / "hub.json"
        )
        configuration["operations"]["review"]["budget"].update(
            wall_seconds=8, storage_bytes=2 * 1024**2, current_disk_bytes=2 * 1024**2
        )
        configuration["operations"]["review"]["supervisor_limits"].update(
            wall_seconds=8, storage_bytes=2 * 1024**2
        )
        atomic_json(configuration["inputs_ref"], {})
        atomic_json(fixture.host_ref, configuration)
        env = {
            "PYTHONPATH": str(ROOT / "core/src") + ":" + str(ROOT / "core/tests"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
        }
        with patch.dict(os.environ, env):
            host = MidtermHost(fixture.host_ref)
            assert host.advance()["status"] == "running"
            host, completion_ref = fixture.end_job(
                "review",
                expected="unknown"
                if audit_case in (None, "research_admission")
                else "completed",
            )
            if audit_case not in (None, "research_admission"):
                admission = read(output / "data-admission.json")
                expected_decision = "admitted" if audit_case == "valid" else "rejected"
                assert admission["decision"] == expected_decision, admission
                if expected_decision == "rejected":
                    from rsi_vla.adapters.pi05_training import prepare_admitted_training

                    with pytest.raises(
                        PermissionError, match="identity/version/authority"
                    ):
                        prepare_admitted_training(
                            manifest_ref=local.manifest_ref,
                            admission_ref=output / "data-admission.json",
                            scope_id=scope_id,
                            protocol_id=protocol_id,
                            reviewer_id="fixture-independent",
                            model_path=local.root / "base",
                            norm_stats_path=local.norm,
                            output_root=root / "runtime/forbidden-preparation",
                            run_id="rejected-training",
                            budget=TrainingBudget(3, 3, 1, 1000000, 2),
                            read_batch=False,
                        )
                    assert not (root / "runtime/forbidden-preparation").exists()
                completion = read(completion_ref)
                reservation = host.ledger.read()["reservations"]["review"]
                assert completion["cleanup_confirmed"] is True
                assert completion["unknown_dimensions"] == []
                assert reservation["status"] == "settled"
                assert reservation["cleanup_confirmed"] is True
                assert reservation["actual"] == completion["actual"]
                assert completion["actual"]["model_requests"] == 0
                assert completion["actual"]["gpu_seconds"] == 0
                assert completion["actual"]["wall_seconds"] > 0
                assert completion["actual"]["storage_bytes"] > 0
                assert host.advance()["status"] == "completed"
                return
            state = host.coordinator().status()
            expected = (
                "training preparation requires independently owned admission"
                if audit_case == "research_admission"
                else "data admission requires independently owned review files"
            )
            expected_role = (
                "runtime" if audit_case == "research_admission" else "evaluation"
            )
            assert expected in state["pending_receipt"]["error"]
            receipt_domain = root / (
                "runtime" if expected_role == "runtime" else "state/evaluation"
            )
            receipts = list(
                (receipt_domain / "role_receipts").glob(
                    "review-execute-revision-*.json"
                )
            )
            assert len(receipts) == 1
            role_return = read(receipts[0])
            assert role_return["role"] == expected_role
            assert "PermissionError: " + expected in role_return["error"]
            assert not output.exists()
            assert not list((root / "state/evaluation").rglob("data-admission.json"))
            completion = read(completion_ref)
            assert completion["cleanup_confirmed"] is True
            assert completion["unknown_dimensions"] == []
            accounting = host.ledger.read()
            reservation = accounting["reservations"]["review"]
            assert reservation["status"] == "settled"
            assert reservation["cleanup_confirmed"] is True
            assert reservation["actual"] == completion["actual"]
            assert accounting["used"]["model_requests"] == 0
            assert accounting["used"]["gpu_seconds"] == 0
            assert accounting["used"]["wall_seconds"] > 0
            assert accounting["used"]["storage_bytes"] > 0
            assert host.advance()["status"] == "unknown"
    finally:
        fixture.doCleanups()
