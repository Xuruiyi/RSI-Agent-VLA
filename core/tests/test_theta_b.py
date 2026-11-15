"""Offline theta x B binding through the actual Runtime consumer."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace

from rsi_vla.__main__ import main as core_main
from rsi_vla.contracts import RunBundle
from rsi_vla.storage import ResearchStore
from rsi_vla.theta_b import freeze_theta_b, prepare_theta_b_cell, evaluate_theta_b_cell

ROOT = Path(__file__).resolve().parents[2]


class ThetaBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.artifacts = Path(self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(['init', '--artifacts', str(self.artifacts), '--profile', str(ROOT / 'config/run_profile_v2.json'),
                       '--protocol', str(ROOT / 'config/pilot_v2.json')])
        self.store = ResearchStore(self.artifacts / 'research')
        self.base = RunBundle(**self.store.get('bundles', 'initial-medium'))
        memory = deepcopy(self.store.get('memory', self.base.memory_ref))
        memory['retrieval_prompt'] = 'frozen B1 retrieval rule'
        self.store.put('memory', 'B1-memory', memory)
        self.store.save_bundle(replace(self.base, bundle_id='B1-source', memory_ref='B1-memory'))
        self.experiment = freeze_theta_b(self.artifacts, experiment_id='theta-b-test',
            base_bundle_id=self.base.bundle_id,
            b_bundle_ids={'B0': self.base.bundle_id, 'B1': 'B1-source'},
            theta_versions={'theta0': {'theta_ref': self.base.theta_ref, 'processor_ref': self.base.processor_ref},
                            'theta1': {'theta_ref': None, 'processor_ref': None}})

    def prepare(self, **kwargs):
        options = dict(theta_id='theta0', b_id='B1', bundle_id='theta-b-cell',
            protocol_path=ROOT / 'config/pilot_v2.json', run_id='theta-b-run',
            task_key='position_swap_salad', seed=0, split='smoke', output_dir=self.artifacts / 'run')
        options.update(kwargs)
        return prepare_theta_b_cell(ROOT, self.artifacts, self.experiment, **options)

    def test_real_runtime_consumes_frozen_B_and_unknown_theta_stays_blocked(self):
        blocked = self.prepare(theta_id='theta1')
        self.assertEqual(blocked['status'], 'blocked_theta_unavailable')
        self.assertIsNone(blocked['plan'])
        prepared = self.prepare()
        binding = prepared['plan']['context']['execution_binding']
        self.assertEqual(binding['bundle']['memory_ref'], 'B1-memory')
        self.assertEqual(binding['records']['memory_ref']['retrieval_prompt'], 'frozen B1 retrieval rule')
        self.assertEqual(prepared['binding']['theta_b_applicability'], 'unknown')
        self.assertFalse(prepared['plan']['context']['dynamic_authorized'])
        self.assertFalse((self.artifacts / 'run').exists())

    def test_fallback_is_an_explicit_effective_cell(self):
        prepared = self.prepare(fallback_b_id='B0')
        self.assertTrue(prepared['binding']['fallback'])
        self.assertEqual(prepared['binding']['requested_b'], 'B1')
        self.assertEqual(prepared['binding']['effective_b'], 'B0')
        self.assertEqual(prepared['plan']['context']['execution_binding']['bundle']['memory_ref'], self.base.memory_ref)

    def test_changed_B_prompt_refused(self):
        path = self.store.path('memory', 'B1-memory')
        changed = json.loads(path.read_text())
        changed['retrieval_prompt'] = 'late change'
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'frozen B content changed'):
            self.prepare()

    def test_independent_evaluation_binding_does_not_grant_release(self):
        prepared = self.prepare()
        result = {key: prepared['plan']['context'][key]
                  for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
        result.update(outcome='unknown', validity='unknown')
        linked = evaluate_theta_b_cell(prepared, result)
        self.assertFalse(linked['production_eligible'])
        self.assertEqual(linked['capability_gain'], 'unknown')
        for field in ('run_id', 'episode_id', 'bundle_id', 'protocol_id'):
            with self.subTest(field=field):
                wrong = {**result, field: 'other'}
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    evaluate_theta_b_cell(prepared, wrong)

    def test_interface_batch_does_not_admit_training_or_autonomous_data(self):
        from rsi_vla.adapters.pi05_training import require_training_data_admission
        for source in ("public_demonstration", "autonomous_execution"):
            with self.assertRaisesRegex(PermissionError, "explicit demonstration-data admission"):
                require_training_data_admission({"source_kind": source,
                    "training_admission": "not_admitted_interface_batch_only"})

    def test_future_trainer_descriptor_reaches_real_loader_configuration(self):
        from rsi_vla.adapters.pi05_training import ACTION_CONTRACT
        descriptor = self.artifacts / "trained-model.json"
        norm = self.artifacts / "norm_stats.json"
        norm.write_text('{"norm_stats": {}}')
        descriptor.write_text(json.dumps({"kind": "pi05_libero_native_full_weights",
            "model_path": str(self.artifacts / "new-checkpoint"), "norm_stats_path": str(norm),
            "config_name": "pi05_libero", "action_contract": ACTION_CONTRACT,
            "actual_load_verified": False}))
        self.experiment = freeze_theta_b(self.artifacts, experiment_id='future-theta',
            base_bundle_id=self.base.bundle_id,
            b_bundle_ids={'B0': self.base.bundle_id, 'B1': 'B1-source'},
            theta_versions={'trained': {'theta_ref': str(descriptor), 'processor_ref': str(norm)}})
        prepared = self.prepare(theta_id='trained')
        self.assertEqual(prepared['plan']['environment']['PI05_CHECKPOINT_PATH'],
                         str(self.artifacts / 'new-checkpoint'))
        self.assertEqual(prepared['plan']['environment']['PI05_NORM_STATS_PATH'], str(norm))
        self.assertFalse(prepared['plan']['context']['dynamic_authorized'])
        norm.write_text('{"norm_stats": {"changed": true}}')
        with self.assertRaisesRegex(ValueError, 'processor content changed'):
            self.prepare(theta_id='trained', bundle_id='changed-processor')
