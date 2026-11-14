"""Offline imported-initial consumers; OS probe fixtures do not claim isolation."""

import contextlib
import copy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.__main__ import main as core_main
from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.qualification_import import (BEHAVIOR_CHECKS, deployment_authorization,
                                          import_initial_evidence, require_role)
from rsi_vla.release import ReleaseManager
from rsi_vla.release_preparation import derive_initial_deployment
from rsi_vla.runtime import main as runtime_main, prepare_runtime
from rsi_vla.storage import ResearchStore, atomic_json

ROOT = Path(__file__).resolve().parents[2]


class InitialDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'state'
        self.domains = {name: str(self.artifacts / name) for name in ('research', 'evaluation', 'release')}
        self.domains.update(operator=str(self.root / 'operator'), runtime=str(self.root / 'runtime'))
        for path in self.domains.values():
            Path(path).mkdir(parents=True)
        self.operator = Path(self.domains['operator'])
        self.source_protocol = json.loads((ROOT / 'config/pilot_5a_release08.json').read_text())
        self.source_profile = json.loads((ROOT / 'config/run_profile_5a_release08.json').read_text())
        # This synthetic consumer fixture needs no collection/assets or services.
        self.source_profile.pop('collection', None)
        self.source_profile.pop('auxiliary_models', None)
        source_profile_ref = self.operator / 'source-profile.json'
        atomic_json(source_profile_ref, self.source_profile)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(['init', '--artifacts', str(self.artifacts), '--profile', str(source_profile_ref),
                       '--protocol', str(ROOT / 'config/pilot_5a_release08.json')])
        self.research = ResearchStore(self.domains['research'])
        self.evaluation = EvaluationStore(self.domains['evaluation'])
        self.source_bundle = self.research.get('bundles', self.source_protocol['anchor_bundle_id'])
        rows = [dict(task_key=key, seed=seed, outcome='success', validity='valid',
                     evidence_ref='synthetic-native-result', outcome_source='native_benchmark')
                for key in self.source_protocol['task_sets']['release']['task_keys']
                for seed in self.source_protocol['task_sets']['release']['seeds']]
        self.source_result = asdict(ExperimentResult('synthetic-initial-matrix',
            self.source_bundle['bundle_id'], self.source_protocol['protocol_id'],
            self.source_protocol['benchmark_id'], 'release', 'simulation', {}, ['synthetic'],
            'synthetic-criteria', dict(success_base=1., success_controlled=1., success_old=1.),
            dict.fromkeys(self.source_protocol['hard_constraints'], True), {},
            assigned_count=len(rows), assignments=rows))
        self.auth_ref = self.operator / 'deployment-authorization.json'
        self.compat_ref = self.operator / 'compatibility.json'
        self.approval_ref = self.operator / 'evidence-approval.json'
        self.profile_ref = self.operator / 'deployment-profile.json'
        self.protocol_ref = self.operator / 'deployment-protocol.json'
        self.target = derive_initial_deployment(self.source_bundle, self.source_protocol,
            self.source_profile, bundle_id='deployment-initial', protocol_id='deployment-protocol',
            profile_id='deployment-profile', backend_ref=self.profile_ref, code_ref='control-v2',
            authorization_ref=self.auth_ref, compatibility_ref=self.compat_ref,
            evidence_import_id='approved-source', deployment_scope='synthetic-publication')
        atomic_json(self.profile_ref, self.target['profile'])
        atomic_json(self.protocol_ref, self.target['protocol'])
        self.research.save_bundle(RunBundle(**self.target['bundle']))
        self.evaluation.put('protocols', 'deployment-protocol', self.target['protocol'])
        self.auth = dict(version='initial-deployment-v1', scope='synthetic-publication', authorized=True,
            publication_authorized=True, inheritance_authorized=True, bundle_id='deployment-initial',
            protocol_id='deployment-protocol', source_bundle_id=self.source_bundle['bundle_id'],
            source_protocol_id=self.source_protocol['protocol_id'], source_result_id=self.source_result['result_id'],
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            role_domains=self.domains, protected_paths=[str(self.operator)],
            isolation_acceptance_ref=str(self.operator / 'isolation.json'),
            inheritance_task=dict(task_key='position_swap_salad', seed=1))
        atomic_json(self.auth_ref, self.auth)
        checks = dict.fromkeys(BEHAVIOR_CHECKS, True)
        self.approval = dict(scope=self.auth['scope'], approved=True, source_bundle=self.source_bundle,
            source_protocol=self.source_protocol, source_profile=self.source_profile,
            source_result=self.source_result, checks=checks, evidence_refs=['synthetic-evidence'],
            historical_os_isolation_claimed=False)
        atomic_json(self.approval_ref, self.approval)
        self.compat = dict(version='initial-deployment-compatibility-v1', checks=checks,
            source_bundle=self.source_bundle, source_protocol=self.source_protocol,
            source_profile=self.source_profile, bundle=self.target['bundle'],
            protocol=self.target['protocol'], profile=self.target['profile'],
            source_manifest_ref='synthetic-source-manifest', target_manifest_ref='synthetic-target-manifest',
            historical_os_isolation_claimed=False)
        atomic_json(self.compat_ref, self.compat)
        self.manager = ReleaseManager(self.domains['release'], self.research, self.evaluation)
        self.role = patch('rsi_vla.qualification_import.require_role')
        self.readonly = patch('rsi_vla.qualification_import._readonly', return_value=True)
        self.role_mock = self.role.start()
        self.readonly.start()
        self.addCleanup(self.role.stop)
        self.addCleanup(self.readonly.stop)

    def publish(self):
        imported = import_initial_evidence(self.evaluation, import_id='approved-source',
            approval_ref=self.approval_ref, authorization_ref=self.auth_ref)
        self.assertEqual(imported, import_initial_evidence(self.evaluation, import_id='approved-source',
            approval_ref=self.approval_ref, authorization_ref=self.auth_ref))
        qualification = self.manager.qualify_initial('initial-qualified',
            result_id=self.source_result['result_id'], evidence_import_id='approved-source')
        self.manager.bootstrap('deployment-initial', scope='simulation')
        active = self.manager.establish_initial('initial-qualified', safe_point='task_boundary', queues_empty=True)
        self.assertEqual(active, self.manager.establish_initial('initial-qualified',
            safe_point='task_boundary', queues_empty=True))
        return qualification, active

    def test_import_publish_and_real_cli_deployment_selection_preserve_source_identity(self):
        before = copy.deepcopy((self.source_protocol, self.source_bundle, self.source_result))
        qualification, active = self.publish()
        self.assertFalse(qualification['relative_gain_claimed'])
        self.assertEqual(qualification['result_refs'], [self.source_result['result_id']])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            runtime_main(['--root', str(ROOT), '--artifacts', str(self.artifacts),
                '--protocol', str(self.protocol_ref), '--deployment-release', self.domains['release'],
                '--run', 'inherited', '--task-key', 'position_swap_salad', '--seed', '1',
                '--output', str(Path(self.domains['runtime']) / 'attempt')])
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['selection']['mode'], 'deployment')
        self.assertEqual(plan['context']['release_binding'], active)
        self.assertEqual(plan['context']['bundle_id'], 'deployment-initial')
        self.assertFalse(plan['counts_as_release'])
        permit = {key: plan['context'][key] for key in ('run_id', 'bundle_id', 'protocol_id', 'gpu_devices')}
        permit.update(authorized=True, task_key='position_swap_salad', seed=1, simulation_episodes=1,
                      limits=plan['context']['limits'])
        permit_ref = self.operator / 'job-permit.json'
        atomic_json(permit_ref, permit)
        args = dict(protocol_path=self.protocol_ref, run_id='inherited', bundle_id='deployment-initial',
            task_key='position_swap_salad', seed=1, split='deployment',
            output_dir=Path(self.domains['runtime']) / 'attempt', deployment_binding=active,
            deployment_release_root=self.domains['release'], authorization_path=permit_ref)
        with self.assertRaisesRegex(ValueError, 'job permit lacks matching'):
            prepare_runtime(ROOT, self.artifacts, **args)
        permit.update(deployment_authorization_ref=str(self.auth_ref), release_binding=active)
        atomic_json(permit_ref, permit)
        self.assertEqual(prepare_runtime(ROOT, self.artifacts, **args)['blockers'], [])
        self.assertEqual(before, (self.source_protocol, self.source_bundle, self.source_result))
        self.assertIn(unittest.mock.call(self.auth, 'evaluation'), self.role_mock.call_args_list)
        self.assertIn(unittest.mock.call(self.auth, 'release'), self.role_mock.call_args_list)
        self.assertIn(unittest.mock.call(self.auth, 'runtime'), self.role_mock.call_args_list)

    def test_invalid_source_or_authority_does_not_create_qualification(self):
        for field, replacement in (('candidate_id', 'rejected-candidate'), ('bundle_id', 'rejected-trial'),
                                   ('unknown_count', 1)):
            approval = copy.deepcopy(self.approval)
            approval['source_result'][field] = replacement
            atomic_json(self.approval_ref, approval)
            with self.subTest(field=field), self.assertRaises(ValueError):
                import_initial_evidence(self.evaluation, import_id='bad', approval_ref=self.approval_ref,
                                        authorization_ref=self.auth_ref)
        self.assertEqual(self.manager.records.all('qualifications'), [])
        atomic_json(self.approval_ref, self.approval)
        for field, value in (('authorized', False), ('bundle_id', 'other'),
                             ('expires_at', '2000-01-01T00:00:00+00:00')):
            auth = {**self.auth, field: value}
            atomic_json(self.auth_ref, auth)
            with self.subTest(authority=field), self.assertRaises(ValueError):
                deployment_authorization(self.auth_ref, bundle_id='deployment-initial', role='evaluation')

    def test_target_drift_and_forged_explicit_binding_are_rejected(self):
        _, active = self.publish()
        kwargs = dict(protocol_path=self.protocol_ref, run_id='inherited', bundle_id='deployment-initial',
            task_key='position_swap_salad', seed=1, split='deployment',
            output_dir=Path(self.domains['runtime']) / 'attempt', deployment_binding=active)
        with self.assertRaisesRegex(ValueError, 'trusted deployment release'):
            prepare_runtime(ROOT, self.artifacts, **kwargs)
        kwargs['deployment_release_root'] = self.domains['release']
        kwargs['seed'] = 3
        with self.assertRaisesRegex(ValueError, 'task scope'):
            prepare_runtime(ROOT, self.artifacts, **kwargs)
        profile = copy.deepcopy(self.target['profile'])
        profile['control']['action_chunk'] = 2
        atomic_json(self.profile_ref, profile)
        with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
            self.manager.deployment_bundle(scope='simulation', safe_point='task_boundary')

    def test_os_boundary_checks_mount_state_and_protected_parent(self):
        self.role.stop()
        reports = {}
        for role in self.domains:
            ref = self.operator / (role + '-probe.json')
            atomic_json(ref, dict(role=role, scope=self.auth['scope'], allowed_write=True, denied_writes=True))
            reports[role] = dict(exit_code=0, report_ref=str(ref))
        atomic_json(self.auth['isolation_acceptance_ref'], dict(scope=self.auth['scope'],
            role_domains=self.domains, roles=reports))
        writable = Path(self.domains['evaluation'])
        with patch('rsi_vla.qualification_import._readonly', side_effect=lambda p: Path(p) != writable):
            require_role(self.auth, 'evaluation')
            with self.assertRaisesRegex(ValueError, 'write boundary'):
                require_role(self.auth, 'release')
        with patch('rsi_vla.qualification_import._readonly', return_value=False):
            with self.assertRaisesRegex(ValueError, 'boundary missing'):
                require_role(self.auth, 'evaluation')
        with patch('rsi_vla.qualification_import._readonly',
                   side_effect=lambda p: Path(p) not in {writable, writable.parent}):
            with self.assertRaisesRegex(ValueError, 'parent permits replacement'):
                require_role(self.auth, 'evaluation')


if __name__ == '__main__':
    unittest.main()
