"""Bounded request projection must retain the learned evidence selection."""

import unittest

import test_relation_learning
from rsi_vla.research_tools import ResearchLLMAdapter
from rsi_vla.research_transport import select_input


class RelationInputProjectionTests(unittest.TestCase):
    setUp = test_relation_learning.RelationLearningTest.setUp
    make_tools = test_relation_learning.RelationLearningTest.make_tools
    learn_and_freeze = test_relation_learning.RelationLearningTest.learn_and_freeze
    inherited_tools = test_relation_learning.RelationLearningTest.inherited_tools

    def test_request_projection_preserves_rhoA_and_complete_source_facts(self):
        _, target = self.learn_and_freeze()
        tools = self.inherited_tools(target)
        tools.protocol['task_policy'] = dict(schema='rsi-task-policy-v1', evidence_input=dict(
            mode='selected_l0', reason='old frozen input', event_ids={'old-request-result': []}))
        tools.domain_policy['evidence_input'] = dict(mode='selected_l0', reason='registered current request',
                                                    event_ids={'result': ['e0', 'e2']})
        raw = ResearchLLMAdapter().prepare_request(tools, request_id='request', result_ids=['result'],
                                                   max_input_tokens=100, max_output_tokens=10)
        projected = select_input(tools, raw)
        view = projected['evidence_views'][0]
        self.assertEqual(view['policy_ref'], target.rho_ref)
        self.assertEqual(view['selected_media'], raw['evidence_views'][0]['selected_media'])
        self.assertEqual(len(view['selected_media']), 2)
        self.assertEqual([event['event_id'] for event in view['l0']], ['e0', 'e2'])
        self.assertEqual(len(tools.store.get('evidence_views', 'view')['l0']), 4)
        tools.protocol.pop('task_policy')
        with self.assertRaisesRegex(ValueError, 'registered new-task'):
            select_input(tools, raw)


if __name__ == '__main__':
    unittest.main()
