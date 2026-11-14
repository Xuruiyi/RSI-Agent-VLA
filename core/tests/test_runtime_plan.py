"""Offline acceptance of the shared launcher, configuration binding and queue."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from rsi_vla.__main__ import main as core_main
from rsi_vla.runtime import dynamic_job_lock, execute_runtime, prepare_runtime
from rsi_vla import ResearchStore, RunBundle


ROOT = Path(__file__).resolve().parents[2]


class RuntimePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.artifacts = Path(self.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            core_main(['init', '--artifacts', str(self.artifacts), '--profile', str(ROOT / 'config/run_profile_v2.json'),
                       '--protocol', str(ROOT / 'config/pilot_v2.json')])

    def plan(self, **kwargs):
        args = dict(protocol_path=ROOT / 'config/pilot_v2.json', run_id='bounded-medium',
                    bundle_id='initial-medium', task_key='position_swap_salad', seed=0,
                    split='smoke', output_dir=self.artifacts / 'run')
        args.update(kwargs)
        return prepare_runtime(ROOT, self.artifacts, **args)

    def test_medium_snapshot_and_no_silent_execution(self):
        plan = self.plan()
        command = plan['command']
        self.assertEqual(command[command.index('--reasoning-effort') + 1], 'medium')
        self.assertEqual(plan['context']['bundle_id'], 'initial-medium')
        binding = plan['context']['execution_binding']
        store = ResearchStore(self.artifacts / 'research')
        self.assertEqual(binding['bundle'], store.get('bundles', 'initial-medium'))
        self.assertEqual(binding['profile']['planner'], plan['context']['planner'])
        self.assertEqual(binding['candidates'], [])
        self.assertEqual(binding['references']['phi_ref'], binding['profile']['control'])
        self.assertIsNone(plan['model_configuration']['effective'])
        self.assertFalse((self.artifacts / 'run').exists())
        with self.assertRaisesRegex(RuntimeError, 'authorization missing'):
            execute_runtime(plan)
        self.assertFalse((self.artifacts / 'run').exists())
        frozen = json.loads((self.artifacts / 'research/profiles/initial-medium.json').read_text())
        self.assertEqual(frozen['research_llm']['reasoning_effort'], 'medium')
        self.assertEqual(json.loads((ROOT / 'config/run_profile_v1.json').read_text())['planner']['reasoning_effort'], 'low')

    def test_sealed_and_wrong_split_seed_refused(self):
        with self.assertRaisesRegex(ValueError, 'sealed'):
            self.plan(split='sealed', seed=5)
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            self.plan(split='release', seed=0)

    def test_runtime_profile_binding_and_stale_authorization(self):
        path = self.artifacts / 'research/profiles/initial-medium.json'
        profile = json.loads(path.read_text())
        profile['planner']['reasoning_effort'] = 'low'
        path.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, 'planner model/effort does not match the supported role settings'):
            self.plan()

    def test_shared_dynamic_slot_is_exclusive(self):
        with dynamic_job_lock():
            with self.assertRaisesRegex(RuntimeError, 'shared resource slot'):
                with dynamic_job_lock():
                    self.fail('overlapping job admitted')

    def test_training_handoff_uses_new_theta_and_keeps_runtime_rules(self):
        from rsi_vla.adapters.pi05_training import ACTION_CONTRACT, training_bundle_proposal
        store = ResearchStore(self.artifacts / 'research')
        base = RunBundle(**store.get('bundles', 'initial-medium'))
        descriptor = self.artifacts / 'trained-model.json'
        descriptor.write_text(json.dumps(dict(kind='pi05_libero_native_full_weights',
            model_path=str(self.artifacts / 'new-checkpoint'), norm_stats_path=str(self.artifacts / 'norm_stats.json'),
            config_name='pi05_libero', action_contract=ACTION_CONTRACT, actual_load_verified=False)))
        candidate = training_bundle_proposal(base, bundle_id='training-trial', theta_ref=str(descriptor),
            processor_ref=str(self.artifacts / 'norm_stats.json'), diff_ref='training-difference.json')
        store.save_bundle(candidate)
        plan = self.plan(bundle_id=candidate.bundle_id)
        self.assertEqual(plan['environment']['PI05_CHECKPOINT_PATH'], str(self.artifacts / 'new-checkpoint'))
        self.assertEqual(plan['environment']['PI05_NORM_STATS_PATH'], str(self.artifacts / 'norm_stats.json'))
        self.assertEqual((candidate.phi_ref, candidate.rho_ref, candidate.psi_ref),
                         (base.phi_ref, base.rho_ref, base.psi_ref))

    def test_changed_chunk_cannot_use_the_old_profile_baseline(self):
        store = ResearchStore(self.artifacts / 'research')
        base = RunBundle(**store.get('bundles', 'initial-medium'))
        rule = self.artifacts / 'rules.json'
        rule.write_text(json.dumps(dict(schema='rsi-runtime-rules-v1', chunk_steps=2, baseline_chunk_steps=5)))
        store.save_bundle(replace(base, bundle_id='rule-trial', parent_bundle_id=base.bundle_id, phi_ref=str(rule)))
        with self.assertRaisesRegex(ValueError, 'new matched protocol'):
            self.plan(bundle_id='rule-trial')

    def test_training_and_runtime_share_the_same_resource_slot(self):
        from rsi_vla.training import dynamic_lock
        with dynamic_job_lock():
            with self.assertRaisesRegex(RuntimeError, 'shared lock'):
                with dynamic_lock():
                    self.fail('training bypassed runtime resource queue')

    def test_physical_gpu_binding_survives_service_cli_overrides(self):
        plan = self.plan(cuda_device=1)
        self.assertEqual(plan['context']['gpu_devices'], [1])
        self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(plan['command'][plan['command'].index('--cuda-device') + 1], '1')


if __name__ == '__main__':
    unittest.main()


class Tool60PlanTests(unittest.TestCase):
    def test_v7_cap_reaches_runtime_and_lower_permit_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with contextlib.redirect_stdout(io.StringIO()):
                core_main(['init','--artifacts',d,'--profile',str(ROOT/'config/run_profile_v6.json'),
                           '--protocol',str(ROOT/'config/pilot_v7.json')])
            protocol=json.loads((ROOT/'config/pilot_v7.json').read_text())
            kwargs=dict(protocol_path=ROOT/'config/pilot_v7.json',run_id='tool60',bundle_id=protocol['anchor_bundle_id'],
                        task_key='position_swap_salad',seed=1,split='development',output_dir=root/'out')
            plan=prepare_runtime(ROOT,d,**kwargs)
            self.assertEqual(plan['context']['limits']['tool_calls'],60)
            permit={key:plan['context'][key] for key in ('run_id','bundle_id','protocol_id','gpu_devices')}
            permit.update(authorized=True,task_key='position_swap_salad',seed=1,simulation_episodes=1,
                          limits={**plan['context']['limits'],'tool_calls':17,'environment_steps':100})
            authorization=root/'permit.json'
            authorization.write_text(json.dumps(permit))
            bounded=prepare_runtime(ROOT,d,authorization_path=authorization,**kwargs)
            self.assertEqual(bounded['context']['limits']['tool_calls'],17)
            self.assertEqual(bounded['command'][bounded['command'].index('--max-episode-steps')+1],'100')
