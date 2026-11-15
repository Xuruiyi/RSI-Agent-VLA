"""Context-specific research preferences preserve host prerequisites and finite exploration."""

from dataclasses import asdict, replace
from pathlib import Path
import unittest

import test_relation_learning
from rsi_vla.cooperation import fixed_research_step
from rsi_vla.effective_psi import consume_effective_psi, freeze_effective_psi
from rsi_vla.relation_learning import (contextual_research_decision, contextual_research_policy,
                                      learn_relations, research_context, version_signature)
from rsi_vla.storage import atomic_json


class ContextualResearchTests(unittest.TestCase):
    setUp = test_relation_learning.RelationLearningTest.setUp
    make_tools = test_relation_learning.RelationLearningTest.make_tools

    def configure(self):
        rho = self.source.get('rho', 'rho')
        self.legacy = {'type': 'rsi-relation-feedback-v1',
                       'values': {'task': {'bounded_experiment': {'count': 2, 'total': 0}}}}
        rho['rhoC'] = contextual_research_policy(self.legacy, max_explorations=1,
                                               data_context={'dataset_ref': 'no_training_required'})
        self.source.put('rho', 'contextual', rho)
        self.base = replace(self.base, bundle_id='context-base', rho_ref='contextual')
        self.source.save_bundle(self.base)
        self.tools = self.make_tools(self.source, self.base, 'context-job')
        self.protocol['task_policy'] = {'schema': 'rsi-task-policy-v1'}
        self.policy = rho['rhoC']

    def decision(self, key, **kwargs):
        return self.tools.research_relation_admission(['result'], legal_repairs=['planner_task_hint'],
            selection_id=key, technical_admitted=kwargs.get('technical_admitted', True),
            data_admitted=kwargs.get('data_admitted', True))

    def test_legacy_rejection_and_new_context_bounded_reevaluation_are_distinct(self):
        self.configure()
        historical = fixed_research_step({'rhoC': self.legacy}, self.tools.job, state='task',
            evidence_available=True, inspected=True, legal_repairs=['planner_task_hint'], experiment_permitted=True)
        self.assertEqual(historical, 'defer')
        first = self.decision('first')
        self.assertEqual(first['action'], 'prepare')
        self.assertEqual(first['decisions'][0]['reason'], 'bounded_context_reevaluation')
        self.assertIsNone(first['decisions'][0]['unexecuted_task_gain'])
        self.assertEqual(self.decision('first'), first)
        second = self.decision('second')
        self.assertEqual(second['action'], 'defer')
        self.assertEqual(second['decisions'][0]['reason'], 'registered_exploration_exhausted')
        self.assertEqual(self.policy['legacy_baseline'], self.legacy)
        self.assertEqual(self.policy['values'], {})
        self.assertEqual(self.tools.job.state['charges'], [])

    def test_exploration_never_bypasses_technical_data_or_exhausted_job(self):
        self.configure()
        for key, kwargs in [('technical', dict(technical_admitted=False)), ('data', dict(data_admitted=False))]:
            self.assertEqual(self.decision(key, **kwargs)['action'], 'defer')
        self.assertEqual(self.decision('good')['action'], 'prepare')
        self.tools.job.consume({'input_tokens': 1000}, 'fixture exhaustion')
        self.assertEqual(self.decision('after-exhaustion')['action'], 'defer')

    def test_feedback_keeps_unknown_gain_and_costs_and_is_consumed_by_new_policy(self):
        self.configure()
        selection = self.decision('observed')
        ref = self.root / 'transport-receipt.json'
        atomic_json(ref, {'status': 'completed', 'origin': 'offline fixture'})
        feedback = self.tools.record_contextual_research_feedback('feedback', selection_id='observed',
            technical_valid=True, screening_pass=None, source_refs=[str(ref)])
        self.assertIsNone(feedback['feedback']['task_gain'])
        self.assertIsNone(feedback['feedback']['costs'])
        self.assertEqual(feedback, self.tools.record_contextual_research_feedback('feedback', selection_id='observed',
            technical_valid=True, screening_pass=None, source_refs=[str(ref)]))
        with self.assertRaisesRegex(ValueError, 'trigger evidence'):
            self.tools.record_contextual_research_feedback('bad', selection_id='observed', result_id='result')
        state = learn_relations(self.tools, self.derived, 'context-learning', ['result'], relations=['rhoC'])
        context = selection['decisions'][0]['context']
        values = state['snapshot']['rhoC']['values'][context['key']]
        self.assertEqual(values['technical_valid'], {'count': 1, 'total': 1.0})
        self.assertNotIn('screening_pass', values)
        self.assertEqual(len(state['feedback']), 1)
        self.assertEqual(state['snapshot']['rhoC']['legacy_baseline'], self.legacy)
        self.assertEqual(state, learn_relations(self.tools, self.derived, 'context-learning', ['result'], relations=['rhoC']))

    def test_technical_failure_does_not_become_contextual_task_screening_reward(self):
        self.configure()
        self.decision('technical-selection')
        receipt = self.root / 'failed-transport.json'
        atomic_json(receipt, {'status': 'technical_failure'})
        self.tools.record_contextual_research_feedback('technical-feedback', selection_id='technical-selection',
            technical_valid=False, screening_pass=False, source_refs=[str(receipt)])
        state = learn_relations(self.tools, self.derived, 'technical-learning', ['result'], relations=['rhoC'])
        context = next(iter(state['snapshot']['rhoC']['values'].values()))
        self.assertEqual(context['technical_valid'], {'count': 1, 'total': 0.0})
        self.assertNotIn('screening_pass', context)
        self.assertFalse(state['feedback'][0]['feedback']['screening_pass'])

    def test_freezing_policy_carries_exploration_reservations_into_a_derived_store(self):
        self.configure()
        self.decision('initial-reservation')
        ref = self.root / 'receipt.json'
        atomic_json(ref, {'status': 'completed'})
        self.tools.record_contextual_research_feedback('observed', selection_id='initial-reservation',
            technical_valid=True, source_refs=[str(ref)])
        learn_relations(self.tools, self.derived, 'learning', ['result'], relations=['rhoC'])
        from rsi_vla.relation_learning import freeze_relations
        target = freeze_relations(self.tools, self.derived, 'learning', rho_ref='inherited-rho', bundle_id='inherited')
        self.derived.put('development_results', 'result', self.source.get('development_results', 'result'))
        self.derived.put('evidence_views', 'view', self.source.get('evidence_views', 'view'))
        inherited = self.make_tools(self.derived, target, 'inherited-job')
        decision = inherited.research_relation_admission(['result'], legal_repairs=['planner_task_hint'],
            selection_id='new-job-cannot-reset', technical_admitted=True, data_admitted=True)
        self.assertEqual(decision['action'], 'defer')
        self.assertEqual(decision['decisions'][0]['reason'], 'registered_exploration_exhausted')

    def test_operator_evidence_versions_and_data_change_the_context(self):
        self.configure()
        args = dict(task_key='task', operator='planner_task_hint', evidence_category='development_l0',
                    signature=version_signature(self.base), data_context=self.policy['data_context'])
        original = research_context(**args)
        for changes in (dict(operator='relation_feedback_update'), dict(evidence_category='technical_invalid_development'),
                        dict(signature={**args['signature'], 'theta_ref': 'new-theta'}),
                        dict(data_context={'dataset_ref': 'teacher-v2'})):
            self.assertNotEqual(original['key'], research_context(**{**args, **changes})['key'])

    def test_effective_psi_separates_facts_and_blocks_a_fake_later_policy(self):
        self.configure()
        program = self.root / 'policy.py'
        program.write_text('def policy(): return "bounded"\n')
        args = dict(program_refs=[program], prompt_rules={'mode': 'readable_v1'},
                    retrieval_rules={'scope': 'legal_development_events_only'},
                    dependencies={'python': 'fixture'}, development_costs={'model_requests': 0, 'programming_tokens': None},
                    development_families=['task'], reserved_families=['held_out_task'],
                    development_mechanisms=['prompt'], reserved_mechanisms=['training'],
                    result_ids=['result'], origin_ref='engineering-task')
        snapshot = freeze_effective_psi(self.tools, 'initial', **args)
        consumed = consume_effective_psi(self.tools, 'initial')
        self.assertEqual(consumed['components']['rhoC']['policy'], self.policy)
        self.assertNotIn('development_facts', snapshot['components'])
        facts = self.source.get('fact_memory_snapshots', 'initial')
        self.assertEqual(facts['development_facts'][0]['result_id'], 'result')
        with self.assertRaisesRegex(ValueError, 'no policy change'):
            freeze_effective_psi(self.tools, 'fake-later', parent_snapshot_id='initial', **args)
        program.write_text('def policy(): return "changed"\n')
        with self.assertRaisesRegex(ValueError, 'program changed'):
            consume_effective_psi(self.tools, 'initial')
        later = freeze_effective_psi(self.tools, 'later', parent_snapshot_id='initial',
                                     **{**args, 'origin_kind': 'program', 'origin_ref': 'feedback-receipt'})
        self.assertEqual(later['origin']['kind'], 'program')
        self.assertIsNone(later['policy_improvement'])


if __name__ == '__main__':
    unittest.main()
