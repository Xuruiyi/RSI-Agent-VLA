# Copyright 2026 RSI-Agent-VLA Authors.
"""Real CLI/supervision receipts with explicitly synthetic native model factories."""

import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from rsi_vla.adapters.pi05_training import prepare_admitted_training
from rsi_vla.fixed_command import fixed_measurements
from rsi_vla.midterm_host import MidtermLedger
from rsi_vla.storage import atomic_json
from rsi_vla.training import TrainingBudget, finalize_native_model_meter
from test_f10_f11_pipeline import local as data_fixture
from test_f10_f11_pipeline import review

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native_cli(tmp_path):
    pytest.importorskip("torch")
    data = data_fixture.__wrapped__(tmp_path / "synthetic")
    admission = review(data)
    assert admission["decision"] == "admitted"
    budget = TrainingBudget(20, 20, 1, 16 * 1024**2, 3)
    prepared = tmp_path / "runtime/prepared"
    with (
        patch(
            "rsi_vla.adapters.pi05_training.verify_installed_sources", return_value={}
        ),
        patch("rsi_vla.adapters.pi05_training.inspect_native_config", return_value={}),
    ):
        prepare_admitted_training(
            manifest_ref=data.manifest_ref,
            admission_ref=admission["admission_ref"],
            scope_id="fixture-scope",
            protocol_id="fixture-protocol",
            reviewer_id="fixture-independent",
            model_path=data.root / "base",
            norm_stats_path=data.norm,
            output_root=prepared,
            run_id="native-fixture",
            budget=budget,
            read_batch=False,
        )
    config = tmp_path / "boundary.json"
    atomic_json(config, {"isolation_task_root": str(tmp_path)})
    tokenizer = tmp_path / "operator/tokenizer/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("SYNTHETIC_TOKENIZER_FACTORY_BOUNDARY")
    site = tmp_path / "external-boundaries"
    site.mkdir()
    # Only the external trainer/model/data factories are synthetic. Preserve the
    # actual native entries, CLI, process supervisor, marker, parser and meter.
    (site / "sitecustomize.py").write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "from test_f10_f11_host import native_boundaries\n"
        "native_boundaries(json.loads(Path(os.environ['NATIVE_METER_BOUNDARY']).read_text()))\n"
        "import torch\n"
        "torch.cuda.max_memory_allocated=lambda:0\n"
        "from rsi_vla.adapters import pi05_training\n"
        "pi05_training.verify_installed_sources=lambda:{}\n"
        "from rsi_vla import training\n"
        "actual_supervision=training.supervise_process\n"
        "def supervise(*args,**kwargs):\n"
        "    env=dict(kwargs['env'])\n"
        "    env['PYTHONPATH']=os.environ['NATIVE_METER_TEST_PATH']\n"
        "    kwargs['env']=env\n"
        "    return actual_supervision(*args,**kwargs)\n"
        "training.supervise_process=supervise\n"
        "if os.environ.get('NATIVE_METER_FAIL')=='1':\n"
        "    def fail(self):raise RuntimeError('synthetic native factory failure')\n"
        "    sys.modules['rlinf.runners.sft_runner'].SFTRunner.init_workers=fail\n"
    )
    now = datetime.now(timezone.utc)
    limits = {
        "wall_seconds": 120,
        "storage_bytes": 64 * 1024**2,
        "current_disk_bytes": 64 * 1024**2,
        "gpu_seconds": 60,
        "training_starts": 1,
        "training_pilots": 1,
        "load_checks": 1,
        "concurrent_gpu_devices": 1,
        "model_requests": 0,
    }
    p1 = tmp_path / "p1.json"
    atomic_json(
        p1,
        {
            "schema": "rsi-midterm-p1-acceptance-v1",
            "scope_id": "fixture-scope",
            "status": "passed",
        },
    )
    scope_ref = tmp_path / "scope.json"
    scope = {
        "schema": "rsi-midterm-scope-v1",
        "scope_id": "fixture-scope",
        "authorized": True,
        "dynamic_authorized": True,
        "status": "active",
        "owner_uid": os.getuid(),
        "write_root": str(tmp_path),
        "storage_write_roots": [str(tmp_path)],
        "accepted_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "outer_limits": limits,
        "p1_acceptance_ref": str(p1),
    }
    atomic_json(scope_ref, scope)
    phase = tmp_path / "phase.json"
    atomic_json(
        phase,
        {
            "limits": limits,
            "remaining": limits,
            "used": dict.fromkeys(limits, 0),
            "unresolved_costs": [],
        },
    )
    ledger = MidtermLedger(scope_ref)
    path = os.pathsep.join(
        map(str, [site, ROOT / "core/src", ROOT / "core/tests", ROOT / "RPent"])
    )
    env = dict(
        os.environ,
        PYTHONPATH=path,
        NATIVE_METER_TEST_PATH=path,
        NATIVE_METER_BOUNDARY=str(config),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return types.SimpleNamespace(
        root=tmp_path,
        data=data,
        prepared=prepared,
        budget=budget,
        scope_ref=scope_ref,
        scope=scope,
        phase=phase,
        ledger=ledger,
        env=env,
        admission=admission,
    )


def run_cli(fixture, *, load=False, fail=False):
    kind = "pi05-load-check" if load else "pi05-training-pilot"
    operation = "load-fixture" if load else "train-fixture"
    budget = TrainingBudget(20, 20, 1, 16 * 1024**2, 1) if load else fixture.budget
    artifact = fixture.prepared / (
        "native-fixture-load-check" if load else "native-fixture"
    )
    model = (
        fixture.prepared / "native-fixture/checkpoints/global_step_3"
        if load
        else fixture.data.root / "base"
    )
    outer = {
        "gpu_seconds": budget.gpu_device_seconds,
        "storage_bytes": budget.artifact_bytes,
        "current_disk_bytes": budget.artifact_bytes,
    }
    outer.update(
        {"load_checks": 1} if load else {"training_starts": 1, "training_pilots": 1}
    )
    fixture.ledger.reserve(
        operation,
        budget=outer,
        phase_ref=fixture.phase,
        gpu_devices=[0],
        receipt_ref=artifact / "job-result.json",
    )
    fixture.ledger.mark_submitted(operation)
    permit = {
        "authorized": True,
        "kind": kind,
        "scope_id": "fixture-scope",
        "scope_ref": str(fixture.scope_ref),
        "write_root": str(fixture.root),
        "accounting_ref": str(fixture.ledger.path),
        "phase_ref": str(fixture.phase),
        "owner_uid": os.getuid(),
        "expires_at": fixture.scope["expires_at"],
        "operation_id": operation,
        "run_id": "native-fixture",
        "cuda_device": 0,
        "budget": budget.__dict__,
        "dataset_root": str(fixture.data.data),
        "model_path": str(model),
        "protocol_id": "fixture-protocol",
        "data_admission_ref": fixture.admission["admission_ref"],
        "data_admission_authority": "fixture-independent",
        "pilot_disposition": "resource_only_no_capability_selection",
    }
    if load:
        permit["training_result_ref"] = str(
            fixture.prepared / "native-fixture/job-result.json"
        )
    permit_ref = fixture.root / (operation + "-permit.json")
    atomic_json(permit_ref, permit)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_pi05_training.py"),
        "--prepared",
        str(fixture.prepared),
        "--permit",
        str(permit_ref),
        "--cuda-device",
        "0",
    ]
    if load:
        command.append("--verify-load")
    result = subprocess.run(
        command,
        env=dict(fixture.env, NATIVE_METER_FAIL="1" if fail else "0"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    meter = json.loads((artifact / "model-meter.json").read_text())
    host = types.SimpleNamespace(
        config={"scope_id": "fixture-scope"}, ledger=fixture.ledger
    )
    template = {
        "permit_ref": str(permit_ref),
        "result_ref": str(artifact / "job-result.json"),
        "model_meter_ref": str(artifact / "model-meter.json"),
        "budget": {"model_requests": 0},
    }
    measured = fixed_measurements(
        host, operation, template, {"cleanup_confirmed": True}
    )
    # This fixture records no physical GPU use; release its logical test lease.
    fixture.ledger.settle(
        operation, actual={key: 0 for key in outer}, cleanup_confirmed=True
    )
    return result, meter, measured, artifact, permit


def test_real_cli_native_train_and_load_close_zero_request_meter(native_cli):
    for load, component in [
        (False, "native_sft"),
        (True, "native_loader_one_prediction"),
    ]:
        result, meter, measured, artifact, _permit = run_cli(native_cli, load=load)
        assert result.returncode == 0, (
            result.stdout + result.stderr + (artifact / "trainer.log").read_text()
        )
        assert meter["status"] == "closed"
        assert meter["components"] == [component]
        assert measured["actual"]["model_requests"] == 0
        assert measured["unknown_dimensions"] == []


def test_real_cli_failed_native_entry_cannot_close_zero_request_meter(native_cli):
    result, meter, measured, artifact, _permit = run_cli(native_cli, fail=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads((artifact / "job-result.json").read_text())["status"] == "failed"
    assert meter["status"] == "unknown"
    assert meter["project_model_requests"] is None
    assert measured["unknown_dimensions"] == ["model_requests"]


def test_incomplete_cleanup_revokes_native_zero_request_closure(native_cli):
    result, _meter, _measured, artifact, permit = run_cli(native_cli)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((artifact / "supervision.json").read_text())
    report["cleanup_confirmed"] = False
    atomic_json(artifact / "supervision.json", report)
    meter = finalize_native_model_meter(
        artifact_root=artifact,
        permit=permit,
        supervision=report,
        result=json.loads((artifact / "job-result.json").read_text()),
    )
    assert meter["status"] == "unknown"
    assert meter["project_model_requests"] is None
