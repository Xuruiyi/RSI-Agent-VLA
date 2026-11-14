"""Static release scope checks use isolated synthetic assets, never a Runtime."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from rsi_vla.storage import atomic_json


class ReleaseScopeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repository = Path(__file__).resolve().parents[2]
        self.protocol = json.loads((self.repository / 'config/pilot_5a_release06.json').read_text())
        self.profile = json.loads((self.repository / 'config/run_profile_5a_release06.json').read_text())
        self.protocol.update(
            protocol_id='rsi-libero-pro-5a-release07-four-seed3',
            authorization_scope='repair05-fresh-release07-eight-slots-only',
            anchor_bundle_id='initial-medium-5a-release07', authorization_ref='authorization.json')
        self.protocol['runtime_config']['profile_reference'] = 'profile.json'
        self.protocol['development_derivation']['source_candidate_id'] = 'development-5a-repair05-experiment-finish-audit-v4-candidate'
        self.profile['profile_id'] = 'matched-5a-release07-20261113'
        self.authorization = {
            'authorized': True, 'scope': self.protocol['authorization_scope'],
            'protocol_id': self.protocol['protocol_id'], 'max_research_dispatches': 0,
            'maximum_starts': 8, 'automatic_retries': 0, 'release_authorized': True,
            'protocol_budget': copy.deepcopy(self.protocol['budget']['authorized_now'])}
        assets = self.root / 'RPent/.venv/lib/python3.11/site-packages/liberopro/liberopro'
        mapping = {}
        base_bddl = ('(:language synthetic task)\n'
                     '(On salad_dressing_1 floor_target_object_region)\n'
                     '(On tomato_sauce_1 floor_other_object_region_4)')
        swap_bddl = base_bddl.replace(
            '(On salad_dressing_1 floor_target_object_region)',
            '(On salad_dressing_1 floor_other_object_region_4)').replace(
            '(On tomato_sauce_1 floor_other_object_region_4)',
            '(On tomato_sauce_1 floor_target_object_region)')
        for task in self.protocol['tasks']:
            suite = task['suite'].removesuffix('_swap')
            names = mapping.setdefault(suite, ['fixture'] * 10)
            names[task['task_id']] = task['task_name']
            task['language'] = 'synthetic task'
            bddl = assets / task['bddl_relative_path']
            bddl.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(swap_bddl if task['task_key'] == 'position_swap_salad' else base_bddl)
            init = assets / task['init_relative_path']
            init.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(init, 'w') as archive:
                archive.writestr('archive/data.pkl', b'(K\x01K' + bytes([50]) + b'K' + bytes([task['init_shape_static'][1]]) + b'\x86.')
        benchmark = assets / 'benchmark'
        benchmark.mkdir(parents=True, exist_ok=True)
        (benchmark / 'libero_suite_task_map.py').write_text(
            'libero_task_map = ' + repr(mapping) + '\n'
            '# libero_task_map[f"{_base}_{_perturb}"] = libero_task_map[_base]\n')
        (benchmark / '__init__.py').write_text('task_orders = ' + repr([list(range(10))]))

    def check(self, protocol, profile=None, authorization=None):
        atomic_json(self.root / 'protocol.json', protocol)
        atomic_json(self.root / 'profile.json', self.profile if profile is None else profile)
        atomic_json(self.root / 'authorization.json', self.authorization if authorization is None else authorization)
        return subprocess.run(
            [sys.executable, str(self.repository / 'scripts/check_pilot.py'),
             '--root', str(self.root), '--protocol', str(self.root / 'protocol.json')],
            text=True, capture_output=True, check=False)

    def test_release07_and_previous_eight_slot_scopes(self):
        result = self.check(self.protocol)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS split identity', result.stdout)
        for scope in ('repair05-first-release-eight-slots-only', 'repair05-fresh-release06-eight-slots-only'):
            with self.subTest(scope=scope):
                protocol = copy.deepcopy(self.protocol)
                protocol.update(authorization_scope=scope, protocol_id='previous-release',
                                anchor_bundle_id='previous-base')
                profile = {**self.profile, 'profile_id': 'previous-profile'}
                authorization = {**self.authorization, 'scope': scope}
                result = self.check(protocol, profile=profile, authorization=authorization)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_release07_identity_cannot_use_legacy_budget_fallback(self):
        protocol = copy.deepcopy(self.protocol)
        protocol.update(authorization_scope='repair05-fresh-release07-typo')
        protocol['budget']['authorized_now'].update(simulation_episodes=16, gpu_seconds=14400, llm_requests=320)
        protocol['next_step_proposal']['episodes'] = 1
        authorization = {**self.authorization, 'scope': protocol['authorization_scope']}
        result = self.check(protocol, authorization=authorization)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('AssertionError', result.stderr)

    def test_release07_rejects_budget_split_and_identity_changes(self):
        cases = [
            ('budget', 'protocol', lambda p: p['budget']['authorized_now'].update(simulation_episodes=16, gpu_seconds=14400, llm_requests=320)),
            ('development split', 'protocol', lambda p: p['task_sets']['development'].update(authorized=True)),
            ('runtime split', 'protocol', lambda p: p.update(allowed_runtime_splits=['development', 'release'])),
            ('release seed', 'protocol', lambda p: p['task_sets']['release'].update(seeds=[4], init_indices=[4])),
            ('protocol identity', 'protocol', lambda p: p.update(protocol_id='old-release06')),
            ('anchor identity', 'protocol', lambda p: p.update(anchor_bundle_id='initial-medium-5a-release06')),
            ('source Candidate identity', 'protocol', lambda p: p['development_derivation'].update(source_candidate_id='old-candidate')),
            ('profile identity', 'profile', lambda p: p.update(profile_id='matched-5a-release06-20261113')),
            ('authorization identity', 'authorization', lambda p: p.update(protocol_id='old-release06')),
            ('authorization starts', 'authorization', lambda p: p.update(maximum_starts=9)),
            ('automatic retries', 'authorization', lambda p: p.update(automatic_retries=1)),
        ]
        for label, target, mutate in cases:
            with self.subTest(change=label):
                values = dict(protocol=copy.deepcopy(self.protocol), profile=copy.deepcopy(self.profile),
                              authorization=copy.deepcopy(self.authorization))
                mutate(values[target])
                result = self.check(**values)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('AssertionError', result.stderr)


class Release08ScopeTest(ReleaseScopeTest):
    def release08(self):
        p, profile, auth = copy.deepcopy(self.protocol), copy.deepcopy(self.profile), copy.deepcopy(self.authorization)
        p.update(protocol_id='rsi-libero-pro-5a-release08-four-seed3',
                 authorization_scope='repair05-fresh-release08-eight-slots-two-recoveries-only',
                 anchor_bundle_id='initial-medium-5a-release08', comparison_only=True,
                 deployment_authorized=False, qualification_establishment_authorized=False)
        profile['profile_id'] = 'matched-5a-release08-20261113'
        budget = dict(simulation_episodes=10, gpu_seconds=9000, llm_requests=200,
                      training_steps=0, downloads_bytes=0, hardware_actions=0)
        p['budget']['authorized_now'] = budget
        slot = dict(model_requests=20, tool_calls=60, simulation_episodes=1,
                    environment_steps=2000, wall_seconds=900, gpu_seconds=900, storage_bytes=1073741824)
        auth.update(scope=p['authorization_scope'], protocol_id=p['protocol_id'],
                    protocol_budget=budget, maximum_starts=10, formal_assignments=8,
                    max_technical_recoveries=2, max_recoveries_per_slot=1,
                    comparison_only=True, deployment_authorized=False,
                    qualification_establishment_authorized=False, inheritance_authorized=False,
                    per_slot_envelope=slot, inheritance_reserve=slot,
                    total_dynamic_envelope={k:v*10 for k,v in slot.items()},
                    offline_reserve=dict(wall_seconds=600, storage_bytes=268435456),
                    admission_envelope=dict(model_requests=220, tool_calls=660,
                        simulation_episodes=11, environment_steps=22000, wall_seconds=10500,
                        gpu_seconds=9900, storage_bytes=12079595520))
        return dict(protocol=p, profile=profile, authorization=auth)

    def test_release08_exact_contract(self):
        values = self.release08()
        result = self.check(**values)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_release08_rejects_wrong_identity_budget_recovery_and_fallback(self):
        cases = [
            ('protocol', lambda p:p.update(authorization_scope='unknown-release08')),
            ('protocol', lambda p:p.update(protocol_id='release07')),
            ('protocol', lambda p:p.update(comparison_only=False)),
            ('protocol', lambda p:p.update(deployment_authorized=True)),
            ('protocol', lambda p:p.update(allowed_runtime_splits=['release','development'])),
            ('protocol', lambda p:p['budget']['authorized_now'].update(simulation_episodes=16)),
            ('profile', lambda p:p.update(profile_id='release07')),
            ('authorization', lambda p:p.update(maximum_starts=8)),
            ('authorization', lambda p:p.update(formal_assignments=10)),
            ('authorization', lambda p:p.update(max_technical_recoveries=3)),
            ('authorization', lambda p:p.update(max_recoveries_per_slot=2)),
            ('authorization', lambda p:p.update(qualification_establishment_authorized=True)),
            ('authorization', lambda p:p['admission_envelope'].update(wall_seconds=9900)),
        ]
        for target, mutate in cases:
            with self.subTest(target=target, mutation=mutate):
                values = self.release08()
                mutate(values[target])
                result = self.check(**values)
                self.assertNotEqual(result.returncode, 0, result.stdout)
