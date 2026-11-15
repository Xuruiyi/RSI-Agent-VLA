"""Offline finite-learning, independent-data admission and live-ledger contracts."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest

from rsi_vla.adapters.pi05_training import (ACTION_CONTRACT, build_recipe,
    build_learning_recipe, dataset_identity, require_training_data_admission,
    audit_demonstration_payload, _record_native_start)
from rsi_vla.training import TrainingBudget, require_scoped_training_permit
from rsi_vla.midterm_host import MidtermLedger


class MidtermTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.budget = TrainingBudget(60, 60, 1, 100000, 20)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    def learning_plan(self):
        return {"schema": "rsi-pi05-learning-plan-v1", "candidate_id": "candidate-1",
                "protocol_id": "offline-protocol", "data_admission_ref": "receipt.json",
                "design_source": "engineering", "stop_rule": "fixed_optimizer_steps",
                "checkpoint_selection": "final_step", "max_steps": 20,
                "micro_batch_size": 1, "global_batch_size": 4,
                "seed": 0, "learning_rate": 2.5e-5, "lr_scheduler": "constant", "warmup_steps": 2}

    def recipe(self, plan=None, budget=None):
        return build_learning_recipe(dataset_root=self.root / "data", model_path=self.root / "theta0",
            norm_stats_path=self.root / "processor", output_root=self.root,
            run_id="learning", budget=budget or self.budget,
            learning_plan=plan or self.learning_plan())

    def test_finite_learning_has_native_accumulation_final_save_and_frozen_vlm(self):
        cfg = self.recipe()
        self.assertEqual(cfg['runner']['max_steps'], 20)
        self.assertEqual(cfg['runner']['save_interval'], 20)
        self.assertEqual(cfg['actor']['global_batch_size'], 4)
        self.assertEqual(cfg['actor']['micro_batch_size'], 1)
        self.assertEqual(cfg['rsi_training']['scope'], 'action_expert_only')
        self.assertTrue(cfg['actor']['fsdp_config']['gradient_checkpointing'])
        self.assertTrue(cfg['actor']['optim']['use_fp32_master_params'])
        self.assertEqual(cfg['rsi_training']['learning_plan'], self.learning_plan())

    def test_pilot_cannot_be_relabelled_as_finite_learning(self):
        with self.assertRaisesRegex(ValueError, 'three steps'):
            build_recipe(dataset_root='d', model_path='m', norm_stats_path='n',
                         output_root=self.root, run_id='pilot', budget=self.budget)
        plan = self.learning_plan()
        plan['checkpoint_selection'] = 'best_formal_success'
        with self.assertRaisesRegex(ValueError, 'final-checkpoint'):
            self.recipe(plan)
        with self.assertRaisesRegex(ValueError, '10000'):
            self.recipe(budget=TrainingBudget(60, 60, 1, 100000, 10001))

    def fixture_admission(self):
        data = self.root / 'dataset'
        self.write('dataset/meta/info.json', {'fps': 10, 'total_episodes': 1, 'features': {
            'state': {'shape': [8]}, 'actions': {'shape': [7]},
            'image': {'dtype': 'image'}, 'wrist_image': {'dtype': 'image'}}})
        (data / 'meta/episodes.jsonl').write_text('{"episode_index":0,"tasks":["fixture"],"length":2}\n')
        (data / 'meta/tasks.jsonl').write_text('{}\n')
        (data / 'data').mkdir()
        (data / 'data/episode.parquet').write_bytes(b'not-consumed-by-native-loader')
        manifest = {'dataset_root': str(data), 'source_kind': 'public_demonstration',
                    'revision': 'fixed-source', 'provenance_ref': 'source-record',
                    'action_contract': ACTION_CONTRACT, 'training_admission': 'not_admitted_interface_batch_only',
                    'episodes': [{'episode_index': 0, 'source_episode_index': 8,
                                  'task': 'fixture', 'source_ref': 'fixed-source:8'}]}
        identity = dataset_identity(manifest)
        original = self.write('original.json', manifest)
        audit = self.write('audit.json', {'schema': 'rsi-demonstration-payload-audit-v1',
            'dataset': identity, 'checks_passed': True, 'frames_checked': 2,
            'episodes': [{'episode_index': 0, 'frames_checked': 2, 'checks': {'source_pixels_equal': True}}]})
        common = {'scope_id': 'scope', 'protocol_id': 'protocol', 'dataset': identity,
                  'reviewer_id': 'quality-reviewer', 'decision': 'accepted', 'evidence_refs': [original]}
        quality = self.write('quality.json', {**common, 'schema': 'rsi-demonstration-quality-review-v1',
            'image_orientation_verified': True, 'state_action_alignment_verified': True,
            'task_relevance_verified': True, 'license_verified': True})
        isolation = self.write('isolation.json', {**common, 'schema': 'rsi-training-evaluation-isolation-v1',
            'source_episode_mapping_checked': True, 'development_formal_split_checked': True,
            'sealed_labels_not_consumed': True})
        receipt = {'schema': 'rsi-training-data-admission-v1', 'scope_id': 'scope',
            'protocol_id': 'protocol', 'reviewer_id': 'quality-reviewer', 'decision': 'admitted',
            'dataset': identity, 'original_manifest_ref': original, 'payload_audit_ref': audit,
            'approved_training_tasks': ['fixture'], 'quality_review_ref': quality,
            'evaluation_isolation_review_ref': isolation}
        ref = self.write('admission.json', receipt)
        kwargs = {'admission_ref': ref, 'scope_id': 'scope', 'protocol_id': 'protocol',
                  'authority': 'quality-reviewer'}
        return manifest, receipt, kwargs

    def test_new_receipt_preserves_old_not_admitted_and_binds_episode_coverage(self):
        manifest, receipt, kwargs = self.fixture_admission()
        require_training_data_admission(manifest, **kwargs)
        self.assertEqual(json.loads(Path(receipt['original_manifest_ref']).read_text())['training_admission'],
                         'not_admitted_interface_batch_only')
        manifest['episodes'][0]['source_episode_index'] = 99
        with self.assertRaisesRegex(PermissionError, 'all consumed'):
            require_training_data_admission(manifest, **kwargs)

    def test_admitted_string_and_files_alone_do_not_admit(self):
        manifest, receipt, kwargs = self.fixture_admission()
        manifest['training_admission'] = 'admitted'
        with self.assertRaises(PermissionError):
            require_training_data_admission(manifest)
        quality = json.loads(Path(receipt['quality_review_ref']).read_text())
        quality['image_orientation_verified'] = False
        Path(receipt['quality_review_ref']).write_text(json.dumps(quality))
        with self.assertRaisesRegex(PermissionError, 'incomplete'):
            require_training_data_admission(manifest, **kwargs)

    def test_payload_audit_detects_a_corrupted_middle_frame(self):
        import io
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        manifest, _, _ = self.fixture_admission()
        root = Path(manifest['dataset_root'])
        source = self.root / 'source'
        (source / 'data').mkdir(parents=True)
        info = json.loads((root / 'meta/info.json').read_text())
        info.update(chunks_size=1000, data_path='data/episode_{episode_index:06d}.parquet')
        for camera in ('image', 'wrist_image'):
            info['features'][camera]['shape'] = [4, 4, 3]
        self.write('dataset/meta/info.json', info)
        self.write('source/meta/info.json', info)
        self.write('source/source.json', {'revision': manifest['revision']})
        manifest['provenance_ref'] = str(source / 'source.json')
        buf = io.BytesIO()
        Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3)).save(buf, format='PNG')
        rows = [{'image': {'bytes': buf.getvalue(), 'path': None},
                 'wrist_image': {'bytes': buf.getvalue(), 'path': None},
                 'state': [0.] * 8, 'actions': [0.] * 6 + [-1.],
                 'timestamp': frame / 10, 'episode_index': 0, 'frame_index': frame}
                for frame in range(3)]
        pq.write_table(pa.Table.from_pylist([{**row, 'episode_index': 8} for row in rows]),
                       source / 'data/episode_000008.parquet')
        (root / 'data/episode.parquet').unlink()
        rows[1]['actions'][0] = 0.5
        pq.write_table(pa.Table.from_pylist(rows), root / 'data/episode_000000.parquet')
        report = audit_demonstration_payload(manifest)
        self.assertEqual(report['frames_checked'], 3)
        self.assertFalse(report['checks_passed'])
        self.assertIn('episode 0: source_states_actions_equal', report['errors'])
        self.assertEqual(report['training_admission'], 'pending_independent_quality_and_isolation_review')

    def fixture_scope(self):
        now = datetime.now(timezone.utc)
        limits = {'training_starts': 6, 'training_pilots': 2, 'load_checks': 6,
                  'gpu_seconds': 10000, 'storage_bytes': 10000000,
                  'current_disk_bytes': 10000000, 'concurrent_gpu_devices': 2}
        scope = {'schema': 'rsi-midterm-scope-v1', 'scope_id': 'scope', 'authorized': True,
                 'dynamic_authorized': True, 'status': 'active', 'owner_uid': os.getuid(),
                 'write_root': str(self.root), 'storage_write_roots': [str(self.root)],
                 'accepted_at': (now - timedelta(minutes=1)).isoformat(),
                 'expires_at': (now + timedelta(hours=1)).isoformat(), 'outer_limits': limits,
                 'p1_acceptance_ref': self.write('p1.json', {'schema': 'rsi-midterm-p1-acceptance-v1', 'scope_id': 'scope', 'status': 'passed'})}
        scope_ref = self.write('scope.json', scope)
        phase = self.write('phase.json', {'limits': limits, 'remaining': limits, 'unresolved_costs': []})
        ledger = MidtermLedger(scope_ref)
        ledger.reserve('training-job', budget={'training_starts': 1, 'gpu_seconds': 60,
            'storage_bytes': 100000, 'current_disk_bytes': 100000}, phase_ref=phase,
            gpu_devices=[0], receipt_ref=self.root / 'receipt.json')
        permit = {'kind': 'pi05-training-learning', 'authorized': True, 'run_id': 'learning',
                  'dataset_root': str(self.root / 'dataset'), 'model_path': str(self.root / 'model'),
                  'budget': self.budget.__dict__, 'cuda_device': 0, 'scope_ref': scope_ref,
                  'scope_id': 'scope', 'owner_uid': os.getuid(), 'write_root': str(self.root),
                  'expires_at': scope['expires_at'], 'accounting_ref': str(ledger.path),
                  'operation_id': 'training-job', 'phase_ref': phase}
        kwargs = {'run_id': 'learning', 'budget': self.budget,
                  'dataset_root': permit['dataset_root'], 'model_path': permit['model_path'],
                  'artifact_root': self.root / 'prepared/learning',
                  'kind': permit['kind'], 'cuda_device': 0}
        return ledger, permit, kwargs

    def test_actual_launch_requires_live_reserved_and_submitted_host_operation(self):
        ledger, permit, kwargs = self.fixture_scope()
        with self.assertRaisesRegex(PermissionError, 'persist the submitted'):
            require_scoped_training_permit(permit, **kwargs, require_submission=True)
        ledger.mark_submitted('training-job')
        require_scoped_training_permit(permit, **kwargs, require_submission=True)
        phase = json.loads(Path(permit['phase_ref']).read_text())
        phase['remaining']['gpu_seconds'] = 1
        Path(permit['phase_ref']).write_text(json.dumps(phase))
        with self.assertRaisesRegex(PermissionError, 'phase'):
            require_scoped_training_permit(permit, **kwargs)

    def test_native_process_marker_keeps_identity_and_refuses_duplicate_start(self):
        from unittest.mock import patch
        recipe = self.write('recipe.json', {'runner': {'logger': {'experiment_name': 'cpu-fixture'}}})
        with patch.dict(os.environ, {'RSI_TRAINING_JOB_ROOT': str(self.root),
            'RSI_TRAINING_SCOPE_ID': 'offline-fixture', 'RSI_TRAINING_OPERATION_ID': 'fixture-op',
            'RSI_TRAINING_KIND': 'pi05-training-learning'}):
            _record_native_start('cpu_fixture_only', recipe)
            marker = json.loads((self.root / 'native-start.json').read_text())
            self.assertEqual(marker['operation_id'], 'fixture-op')
            self.assertEqual(marker['pid'], os.getpid())
            self.assertFalse(marker['weight_update_established'])
            with self.assertRaises(FileExistsError):
                _record_native_start('cpu_fixture_only', recipe)

    def test_scope_expiry_p1_write_domain_and_owner_are_consumed(self):
        _, permit, kwargs = self.fixture_scope()
        scope = json.loads(Path(permit['scope_ref']).read_text())
        for field, value in [('owner_uid', -1), ('expires_at', '2020-01-01T00:00:00Z'), ('status', 'paused')]:
            changed = {**scope, field: value}
            Path(permit['scope_ref']).write_text(json.dumps(changed))
            with self.assertRaises(PermissionError):
                require_scoped_training_permit(permit, **kwargs)
        Path(permit['scope_ref']).write_text(json.dumps(scope))
        self.write('p1.json', {'scope_id': 'scope', 'status': 'pending'})
        with self.assertRaisesRegex(PermissionError, 'P1'):
            require_scoped_training_permit(permit, **kwargs)
        self.write('p1.json', {'schema': 'rsi-midterm-p1-acceptance-v1', 'scope_id': 'scope', 'status': 'passed'})
        with self.assertRaisesRegex(PermissionError, 'write domain'):
            require_scoped_training_permit(permit, **{**kwargs, 'artifact_root': self.root.parent / 'elsewhere'})


if __name__ == '__main__':
    unittest.main()
