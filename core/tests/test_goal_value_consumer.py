"""Independent goal-value review uses registered inputs and evaluation storage."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_task_completion
from rsi_vla.protocol10_roles import run_role
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskCompletionBackend

ROOT = Path(__file__).resolve().parents[2]


class GoalValueConsumerTests(unittest.TestCase):
    def setUp(self):
        # The role's private /tmp must not hide the fixture's source records.
        make_temp = tempfile.TemporaryDirectory
        with patch('test_task_completion.tempfile.TemporaryDirectory',
                   lambda: make_temp(dir='/var/tmp', prefix='rsi-goal-value-')):
            test_task_completion.ConsumerAdmissionTests.setUp(self)
        self.config['isolation_task_root'] = str(self.root)
        self.config['repository_root'] = str(ROOT)
        self.config['offline_fixture'] = True
        self.config['research_store_ref'] = str(self.root / 'state/research')
        self.config['evaluation_store_ref'] = str(self.root / 'state/evaluation')
        self.backend = TaskCompletionBackend(self.config)
        self.policy_ref = self.root / 'operator/goal-policy.json'
        self.measurement_ref = self.backend.evaluations.root / 'goal-measurement.json'
        source = self.backend.evaluations.root / 'domain-measurement.txt'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('Offline fixture measurement, not a real domain result.\n')
        self.target = dict(domain='tabletop', kind='unlisted_goal', description='fixture goal',
                           value_criterion_ref='criterion-v1', independent_value_test_ref='test-v1')
        self.policy = dict(policy_ref=str(self.policy_ref), domain='tabletop',
                           allowed_goal_kinds=['unlisted_goal'], value_criterion_ref='criterion-v1',
                           independent_value_test_ref='test-v1', metric='domain_value', minimum=1)
        self.opportunity = dict(opportunity_id='goal', protocol_id=self.backend.protocol['protocol_id'],
                                source='new_goal', target=self.target,
                                goal_review=dict(policy_ref=str(self.policy_ref), value_status='unverified'))
        self.evidence = dict(schema='rsi-goal-value-evidence-v1',
                            protocol_id=self.backend.protocol['protocol_id'], opportunity_id='goal',
                            target=self.target, criterion_ref='criterion-v1', test_ref='test-v1',
                            information_scope='development', validity='valid',
                            metrics=dict(domain_value=1), evidence_refs=[str(source)])
        atomic_json(self.policy_ref, self.policy)
        atomic_json(self.measurement_ref, self.evidence)
        self.backend.store.put('opportunities', 'goal', self.opportunity)
        self.op = dict(operation_id='review-goal', kind='goal_value_review', dynamic=False,
                       payload=dict(opportunity_id='goal', policy_ref=str(self.policy_ref),
                                    evidence_ref=str(self.measurement_ref), review_id='goal-value'))
        self.config['operations'] = [self.op]

    def test_real_isolated_evaluation_role_records_review_without_changing_goal(self):
        result = run_role(self.config, dict(action='execute', operation=self.op, inputs={}))
        self.assertNotIn('error', result, result)
        self.assertEqual(result['role'], 'evaluation')
        value = result['value']['review']
        self.assertEqual(value['value_status'], 'confirmed')
        self.assertIsNone(value['capability_gain'])
        self.assertFalse(value['release_eligible'])
        self.assertEqual(self.backend.store.get('opportunities', 'goal'), self.opportunity)
        self.assertTrue(Path(result['value']['review_ref']).is_relative_to(self.backend.evaluations.root))

    def test_research_role_cannot_review_its_own_goal(self):
        with patch.dict('os.environ', RSI_ISOLATED_ROLE='research'):
            with self.assertRaisesRegex(PermissionError, 'independent evaluation role'):
                self.backend.execute(self.op, {})

    def test_research_measurement_and_unregistered_inputs_are_rejected(self):
        research_ref = self.backend.store.root / 'untrusted-measurement.json'
        atomic_json(research_ref, self.evidence)
        changed = copy.deepcopy(self.op)
        changed['payload']['evidence_ref'] = str(research_ref)
        with patch.dict('os.environ', RSI_ISOLATED_ROLE='evaluation'):
            with self.assertRaisesRegex(ValueError, 'registered fixed'):
                self.backend.execute(changed, {})
            self.config['operations'] = [changed]
            with self.assertRaisesRegex(PermissionError, 'independently evaluated'):
                self.backend.execute(changed, {})

    def test_missing_measurement_source_is_not_confirmed(self):
        Path(self.evidence['evidence_refs'][0]).unlink()
        with patch.dict('os.environ', RSI_ISOLATED_ROLE='evaluation'):
            with self.assertRaisesRegex(ValueError, 'source is missing'):
                self.backend.execute(self.op, {})


class TaskOpportunitySelectionTests(unittest.TestCase):
    def setUp(self):
        import test_task_policy
        from rsi_vla.task_completion import TaskResearchAdapter

        test_task_policy.SC3OpportunityTests.setUp(self)
        self.cfg = dict(request_id="normal", reserved_bundle_id="next-trial",
                        development_slots=[dict(task_key="salad", seed=2)])
        self.adapter = TaskResearchAdapter(self.cfg)

    def test_budget_defer_does_not_leave_dispatchable_development_slots(self):
        import test_task_policy
        response = test_task_policy.SC3OpportunityTests.prepare_response(self)
        response["development_slots"] = self.cfg["development_slots"]
        result = self.adapter.apply_response(self.tools, response, evidence_level="offline",
            cost_limit={"offline_calls": self.job.state["limits"]["offline_calls"] + 1},
            synthetic=True)
        self.assertEqual(result["action"], "defer")
        selected = self.store.get("task_completion_selections", "normal")
        self.assertEqual(selected["development_slots"], [])
        self.assertEqual(self.store.all("rule_trials"), [])
        self.assertEqual(self.store.all("candidates"), [])

    def test_goal_proposal_cannot_schedule_development_before_value_review(self):
        import test_task_policy
        response = test_task_policy.SC3OpportunityTests.goal(self)
        response["development_slots"] = self.cfg["development_slots"]
        with self.assertRaisesRegex(ValueError, "deferred Research"):
            self.adapter.apply_response(self.tools, response, evidence_level="offline",
                cost_limit={"offline_calls": 1}, synthetic=True)
        self.assertEqual(self.store.all("opportunities"), [])
        self.assertEqual(self.store.all("task_completion_selections"), [])

    result = __import__('test_task_policy').SC3OpportunityTests.result
    prepare_response = __import__('test_task_policy').SC3OpportunityTests.prepare_response


if __name__ == '__main__':
    unittest.main()
