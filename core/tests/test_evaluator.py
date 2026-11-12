"""File evidence, assignment identity and missing-result regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.evaluator import evaluate_matrix, native_flag


ROOT = Path(__file__).resolve().parents[2]


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / 'config/pilot_v2.json').read_text())
        self.profile = json.loads((ROOT / 'config/run_profile_v2.json').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def evaluate(self, manifest):
        return evaluate_matrix(self.protocol, self.profile, manifest, result_id='evidence-fixture',
                               bundle_id='initial-medium')

    def native(self, **updates):
        record = dict(run_id='r1', episode_id='base-s3', bundle_id='initial-medium',
                      protocol_id=self.protocol['protocol_id'],
                      task=dict(suite='libero_object', task=2, seed=3),
                      planner=self.profile['planner'], benchmark_success=True,
                      terminated=True, truncated=False, native_result_source=dict(terminated='native _check_success()'))
        record.update(updates)
        path = Path(self.temp.name) / 'result.json'
        path.write_text(json.dumps(record))
        return [dict(task_key='base_salad', seed=3, run_id='r1', episode_id='base-s3', result_path=str(path))]

    def test_unstarted_matrix_retains_all_unknown_slots(self):
        result = self.evaluate([])
        self.assertEqual((result.assigned_count, result.unknown_count), (8, 8))
        self.assertTrue(all(value is None for value in result.hard_constraints.values()))

    def test_low_runtime_cannot_establish_medium_baseline(self):
        planner = {**self.profile['planner'], 'reasoning_effort': 'low'}
        result = self.evaluate(self.native(planner=planner))
        self.assertEqual(result.invalid_count, 1)
        self.assertEqual(result.metrics['success_base'], 0)

    def test_native_success_kept_with_incomplete_action_cost_evidence(self):
        result = self.evaluate(self.native())
        self.assertEqual(result.unknown_count, 7)
        self.assertEqual(result.metrics['success_base'], 0.5)
        self.assertIsNone(result.hard_constraints['reference_coverage'])
        self.assertIsNone(result.hard_constraints['resource_limits'])

    def test_planner_success_and_cross_bundle_evidence_do_not_fill_outcome(self):
        result = self.evaluate(self.native(benchmark_success=None, terminated=False,
                                           planner_finish=dict(status='success')))
        self.assertEqual(result.unknown_count, 8)
        result = self.evaluate(self.native(bundle_id='other'))
        self.assertEqual((result.unknown_count, result.invalid_count), (8, 1))

    def test_duplicate_or_sealed_assignments_are_refused(self):
        manifest = self.native()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.evaluate(manifest * 2)
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.evaluate([{**manifest[0], 'seed': 5}])

    def test_scalar_and_chunk_flags_match_native_any_semantics(self):
        self.assertTrue(native_flag([False, [False, True]]))
        self.assertFalse(native_flag([[False], [False]]))
        self.assertIsNone(native_flag([False, None]))

    def test_partial_runtime_rule_observation_does_not_complete_release_constraints(self):
        manifest=self.native()
        (Path(manifest[0]['result_path']).parent/'runtime_context.json').write_text('{}')
        report={'passed':True,'costs':{},'notes':[], 'checks':{},
                'effective_rules':{'finish_requires_native_success':False,'planner_finish_reserve_calls':1}}
        with patch('rsi_vla.evaluator.inspect_attempt',return_value=report):
            result=self.evaluate(manifest)
        self.assertEqual(result.intervention['planner_finish_reserve_calls'],1)
        self.assertEqual(result.intervention['runtime_rule_observation_scope'],'executed_slots_only')
        self.assertEqual(result.unknown_count,7)
        self.assertIsNone(result.hard_constraints['reference_coverage'])
        self.assertIsNone(result.hard_constraints['resource_limits'])


if __name__ == '__main__':
    unittest.main()
