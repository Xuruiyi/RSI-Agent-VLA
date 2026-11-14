"""Fresh deployment authority uses immutable qualification and effective state."""

import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_initial_deployment as fixture
from test_initial_deployment import ROOT
from rsi_vla.runtime import main as runtime_main, prepare_runtime
from rsi_vla.storage import atomic_json


class RenewalTests(unittest.TestCase):
    def setUp(self):
        fixture.InitialDeploymentTests.setUp(self)
        self.qualification, self.active = fixture.InitialDeploymentTests.publish(self)
        self.ref = self.operator / 'renewal.json'
        task_ref = self.operator / 'new-task-authority.json'
        atomic_json(task_ref, dict(scope='new-task-scope', reuse_effective_version_authorized=True))
        self.renewal = dict(version='effective-version-renewal-v1', authorized=True,
            publication_authorized=False, scope='new-task-scope',
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            role_domains=self.domains, protected_paths=[str(self.operator)],
            isolation_acceptance_ref='isolated-fixture', task_authorization_ref=str(task_ref),
            source_release_root=str(self.manager.records.root.resolve()),
            source_research_root=str(self.research.root.resolve()),
            source_evaluation_root=str(self.evaluation.root.resolve()),
            active_snapshot=self.active, qualification_snapshot=self.qualification,
            tasks=[dict(task_key='position_swap_salad', seed=1)])
        atomic_json(self.ref, self.renewal)

    def select(self):
        return self.manager.deployment_bundle(scope='new-task-scope', safe_point='task_boundary',
                                              authorization_ref=self.ref)

    def test_expired_original_denied_but_fresh_renewal_consumed_by_real_cli(self):
        # Freeze time after the original expiry, preserving its immutable file/snapshot.
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        self.renewal['expires_at'] = (future + timedelta(hours=1)).isoformat()
        atomic_json(self.ref, self.renewal)
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return future
        with patch('rsi_vla.qualification_import.datetime', Clock):
            with self.assertRaisesRegex(ValueError, 'expired'):
                self.manager.deployment_bundle(scope='simulation', safe_point='task_boundary')
            selected, active = self.select()
            self.assertEqual(active, self.active)
            self.assertEqual(selected['bundle_id'], self.active['bundle_id'])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                runtime_main(['--root', str(ROOT), '--artifacts', str(self.artifacts),
                    '--protocol', str(self.protocol_ref), '--deployment-release', self.domains['release'],
                    '--deployment-authorization', str(self.ref), '--run', 'new-task',
                    '--seed', '1', '--output', str(Path(self.domains['runtime']) / 'new-task'),
                    '--storage-write-root', str(self.root / 'scratch')])
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['context']['deployment_renewal_scope'], 'new-task-scope')
            self.assertEqual(plan['context']['release_binding'], self.active)
            self.assertEqual(plan['context']['deployment_authorization_ref'], str(self.ref))
            self.assertEqual(self.manager.active(), self.active)
            self.assertEqual(self.manager.records.all('qualifications'), [self.qualification])

    def test_fresh_permission_requires_applicable_task_authority(self):
        task_ref = Path(self.renewal['task_authorization_ref'])
        atomic_json(task_ref, dict(scope='other', reuse_effective_version_authorized=True))
        with self.assertRaisesRegex(ValueError, 'task authorization'):
            self.select()

    def test_new_scope_permit_binds_scratch_and_explicit_renewal(self):
        stream = io.StringIO()
        scratch = self.root / 'scratch'
        output = Path(self.domains['runtime']) / 'new-run'
        with contextlib.redirect_stdout(stream):
            runtime_main(['--root', str(ROOT), '--artifacts', str(self.artifacts),
                '--protocol', str(self.protocol_ref), '--deployment-release', self.domains['release'],
                '--deployment-authorization', str(self.ref), '--run', 'new-run', '--seed', '1',
                '--output', str(output), '--storage-write-root', str(scratch)])
        context = json.loads(stream.getvalue())['context']
        permit = {key: context[key] for key in ('run_id', 'bundle_id', 'protocol_id', 'gpu_devices')}
        permit.update(authorized=True, scope='new-task-scope', task_key='position_swap_salad', seed=1,
                      simulation_episodes=1, limits=context['limits'],
                      storage_write_roots=context['storage_write_roots'],
                      deployment_authorization_ref=str(self.ref), release_binding=self.active)
        permit_ref = self.operator / 'new-permit.json'
        atomic_json(permit_ref, permit)
        args = dict(protocol_path=self.protocol_ref, run_id='new-run', bundle_id=self.active['bundle_id'],
            task_key='position_swap_salad', seed=1, split='deployment', output_dir=output,
            deployment_binding=self.active, deployment_release_root=self.domains['release'],
            deployment_authorization_ref=self.ref, storage_write_roots=[scratch], authorization_path=permit_ref)
        self.assertEqual(prepare_runtime(ROOT, self.artifacts, **args)['blockers'], [])
        permit['storage_write_roots'] = [str(output)]
        atomic_json(permit_ref, permit)
        with self.assertRaisesRegex(ValueError, 'storage write domains'):
            prepare_runtime(ROOT, self.artifacts, **args)
        self.assertFalse(output.exists())

    def test_expired_renewal_and_mutated_original_or_active_are_rejected(self):
        self.renewal['expires_at'] = '2000-01-01T00:00:00+00:00'
        atomic_json(self.ref, self.renewal)
        with self.assertRaisesRegex(ValueError, 'expired'):
            self.select()
        self.renewal['expires_at'] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        atomic_json(self.ref, self.renewal)
        self.auth['expires_at'] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        atomic_json(self.auth_ref, self.auth)
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.select()

    def test_renewal_task_boundary_and_source_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'safe point'):
            self.manager.deployment_bundle(scope='new-task-scope', safe_point='inside-action', authorization_ref=self.ref)
        self.renewal['active_snapshot'] = {**self.active, 'revision': 2}
        atomic_json(self.ref, self.renewal)
        with self.assertRaisesRegex(ValueError, 'effective source mismatch'):
            self.select()


if __name__ == '__main__':
    unittest.main()
