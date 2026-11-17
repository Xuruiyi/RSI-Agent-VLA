# Copyright 2026 RSI-Agent-VLA Authors.
"""A pilot consumes independently admitted data without rewriting its protocol."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_f10_f11_pipeline as pipeline
from rsi_vla.framework_operations import execute
from rsi_vla.training import TrainingBudget

local = pipeline.local


@pytest.mark.parametrize('case', ['explicit', 'missing', 'wrong_data', 'wrong_scope'])
def test_registered_training_keeps_data_and_runtime_protocols_separate(local, monkeypatch, case):
    admitted = pipeline.review(local, output_root=local.root / 'state/evaluation/review')
    assert admitted['decision'] == 'admitted'
    backend = SimpleNamespace(
        config={'isolation_task_root': str(local.root)},
        protocol={'protocol_id': 'runtime-pilot-protocol'},
        authority=lambda: ({'scope_id': 'fixture-scope'}, {}),
    )
    payload = {
        'manifest_ref': local.manifest_ref,
        'admission_ref': admitted['admission_ref'],
        'protocol_id': 'runtime-pilot-protocol',
        'data_protocol_id': 'fixture-protocol',
        'reviewer_id': 'fixture-independent',
        'model_path': str(local.root / 'base'),
        'norm_stats_path': str(local.norm),
        'output_root': str(local.root / 'runtime/prepared'),
        'run_id': 'resource-only-pilot',
        'budget': TrainingBudget(3, 3, 1, 1000000, 2).__dict__,
        'read_batch': False,
    }
    if case == 'missing':
        payload.pop('data_protocol_id')
    elif case == 'wrong_data':
        payload['data_protocol_id'] = 'unreviewed-data'
    elif case == 'wrong_scope':
        payload['scope_id'] = 'foreign-scope'
    monkeypatch.setenv('RSI_ISOLATED_ROLE', 'runtime')
    with (
        patch('rsi_vla.adapters.pi05_training.verify_installed_sources', return_value={}),
        patch('rsi_vla.adapters.pi05_training.inspect_native_config', return_value={'external_factory': 'CPU fixture'}),
    ):
        if case == 'explicit':
            result = execute(backend, 'training_prepare', payload)
            assert result['training_steps'] == 0
            assert result['protocol_id'] == 'fixture-protocol'
            assert result['dataset_root'] == str(local.data)
        else:
            with pytest.raises(PermissionError):
                execute(backend, 'training_prepare', payload)
            assert not (local.root / 'runtime/prepared').exists()
