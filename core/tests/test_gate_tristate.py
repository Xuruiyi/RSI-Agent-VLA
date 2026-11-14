"""Fixed-score Gate must distinguish missing ability evidence from regression."""
import copy
import json
from pathlib import Path
import unittest

from rsi_vla.contracts import Candidate, ExperimentResult
from rsi_vla.evaluation import Gate, critical_preserved


ROOT = Path(__file__).resolve().parents[2]


class GateTristateTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT/'config/pilot_5a_release05.json').read_text())
        self.p['dynamic_authorized'] = True
        self.candidate = Candidate('candidate', self.p['anchor_bundle_id'], 'candidate-bundle',
            self.p['protocol_id'], self.p['benchmark_id'], {}, [], [], {}, {}, 'simulation')

    def result(self, bundle, outcomes=None, *, incomplete=False):
        outcomes = outcomes or {}
        rows = [dict(task_key=task['task_key'], seed=seed,
                     outcome='unknown' if incomplete else outcomes.get(task['task_key'], 'success'),
                     validity='valid', evidence_ref=None if incomplete else 'independent-result',
                     outcome_source=None if incomplete else 'native_benchmark')
                for task in self.p['tasks'] for seed in self.p['task_sets']['release']['seeds']]
        groups = {task['task_key']: task['group'] for task in self.p['tasks']}
        metrics = {'success_'+group: sum(row['outcome']=='success' for row in rows
                   if groups[row['task_key']]==group) / sum(groups[row['task_key']]==group for row in rows)
                   for group in ('base', 'controlled', 'old')}
        return ExperimentResult(bundle+'-result', bundle, self.p['protocol_id'], self.p['benchmark_id'],
            'release', 'simulation', {}, [], 'criteria', metrics,
            dict.fromkeys(self.p['hard_constraints'], None if incomplete else True), {},
            candidate_id='candidate' if bundle=='candidate-bundle' else None,
            assigned_count=len(rows), unknown_count=len(rows) if incomplete else 0, assignments=rows)

    def gate(self, candidate, baseline=None, anchor=None):
        return Gate(self.p).evaluate('gate', self.candidate,
            current_bundle_id=self.candidate.base_bundle_id, candidate_result=candidate,
            baseline_result=baseline, anchor_result=anchor)

    def test_all_unstarted_and_partial_native_results_are_inconclusive(self):
        candidate = self.result('candidate-bundle', incomplete=True)
        baseline = self.result(self.candidate.base_bundle_id, incomplete=True)
        self.assertIsNone(critical_preserved(self.p, candidate, baseline, baseline))
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, 'inconclusive')
        baseline.assignments[0].update(outcome='success', validity='invalid',
                                       evidence_ref='native', outcome_source='native_benchmark')
        object.__setattr__(baseline, 'unknown_count', len(baseline.assignments)-1)
        object.__setattr__(baseline, 'invalid_count', 1)
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, 'inconclusive')

    def test_confirmed_violation_rejects_even_with_missing_comparators(self):
        candidate = self.result('candidate-bundle', incomplete=True)
        candidate.hard_constraints['safety'] = False
        self.assertEqual(self.gate(candidate).verdict, 'reject')
        other = copy.deepcopy(candidate)
        object.__setattr__(other, 'candidate_id', 'other')
        self.assertEqual(self.gate(other).verdict, 'inconclusive')

    def test_complete_comparable_fixed_scores_accept_and_reject(self):
        baseline = self.result(self.candidate.base_bundle_id, {'position_swap_salad': 'failure'})
        candidate = self.result('candidate-bundle')
        accepted = self.gate(candidate, baseline, baseline)
        self.assertEqual(accepted.verdict, 'accept')
        self.assertAlmostEqual(accepted.scores['candidate'], 1)
        self.assertAlmostEqual(accepted.scores['baseline'], .65)
        self.assertEqual(self.gate(candidate, candidate.__class__(**{
            **candidate.__dict__, 'bundle_id': baseline.bundle_id, 'candidate_id': None}),
            baseline).verdict, 'reject')
        next(row for row in candidate.assignments if row['task_key']=='old_milk')['outcome'] = 'failure'
        candidate.metrics['success_old'] = .5
        self.assertEqual(self.gate(candidate, baseline, baseline).verdict, 'reject')
        self.assertEqual(self.p['min_gain'], .02)
        self.assertEqual(self.p['anchor_max_drop'], .02)


if __name__ == '__main__':
    unittest.main()
