"""Actual relation consumption requires real matching consumer sources."""

from copy import deepcopy
from dataclasses import asdict
import unittest

import test_relation_evidence
from rsi_vla.relation_consumption import evaluate_relation_consumption
from rsi_vla.storage import atomic_json


class RelationConsumptionTests(unittest.TestCase):
    setUp = test_relation_evidence.SkillFeedbackTests.setUp
    write_streams = test_relation_evidence.SkillFeedbackTests.write_streams

    def configure(self):
        rho = self.store.get('rho', self.bundle.rho_ref)
        self.context['execution_binding'] = dict(bundle=asdict(self.bundle), records=dict(rho_ref=rho))
        self.context['relation_dispatch_admission'] = dict(rho_ref=self.bundle.rho_ref, actual_dispatch_gate=True,
            source_result_ids=['result'], job_state_before_dispatch=dict(status='running'),
            decisions=[dict(state='task', action='bounded_experiment')], action='prepare')
        atomic_json(self.output/'runtime_context.json', self.context)
        for name in ['task_completion.py', 'cooperation.py']:
            path = self.output/'code_snapshot/core/src/rsi_vla'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture-only consumer source')
        evidence = self.root/'actual-transport.json'
        evidence.write_text('{}')
        view = self.store.get('evidence_views', self.result.intervention['evidence_view_id'])
        request = dict(bundle_id=self.bundle.bundle_id,
                       evidence_views=[{**view, 'policy_ref': self.bundle.rho_ref}])
        self.store.put('llm_requests', 'request', request)
        self.store.put('bounded_requests', 'request', request)
        from rsi_vla.research_transport import pack
        self.store.put('bounded_transports', 'request', pack(request))
        self.store.put('llm_executions', 'request', dict(base_bundle_id=self.bundle.bundle_id,
            transport_facts=dict(effective_model='gpt-6.1-sol', effective_reasoning_effort='high', evidence_ref=str(evidence))))

    def test_all_consumers_are_verified_without_using_native_success(self):
        self.configure()
        report = evaluate_relation_consumption(self.store, self.output, research_request_ids=['request'])
        self.assertTrue(all(row['value'] is True for row in report['relations'].values()))
        self.assertFalse(report['binding_alone_sufficient'])

    def test_same_candidate_other_run_cannot_supply_this_slots_rhoA_receipt(self):
        self.configure()
        request = self.store.get('llm_requests', 'request')
        for view in request['evidence_views']:
            for event in view['l0']: event['run_id'] = 'other-run'
        self.store.put('llm_requests', 'other-run', request)
        self.store.put('bounded_requests', 'other-run', request)
        from rsi_vla.research_transport import pack
        self.store.put('bounded_transports', 'other-run', pack(request))
        self.store.put('llm_executions', 'other-run', self.store.get('llm_executions', 'request'))
        with self.assertRaisesRegex(ValueError, 'does not cover this development'):
            evaluate_relation_consumption(self.store, self.output, research_request_ids=['other-run'])

    def test_runtime_binding_cannot_substitute_for_actual_research(self):
        self.configure()
        report = evaluate_relation_consumption(self.store, self.output)
        self.assertIsNone(report['relations']['rhoA']['value'])
        self.assertTrue(report['relations']['rhoB']['value'])
        self.assertTrue(report['relations']['rhoC']['value'])

    def test_contextual_dispatch_is_recomputed_from_its_saved_selection(self):
        self.configure()
        from rsi_vla.relation_learning import contextual_research_policy
        from rsi_vla.research import ResearchJob
        from rsi_vla.research_tools import ResearchTools
        rho = self.store.get('rho', self.bundle.rho_ref)
        rho['rhoC'] = contextual_research_policy(rho['rhoC'], max_explorations=1,
                                               data_context={'dataset_ref': 'not_training'})
        atomic_json(self.store.path('rho', self.bundle.rho_ref), rho)
        tools = ResearchTools(self.store, ResearchJob(self.store, 'context-job', self.bundle.bundle_id,
            {'input_tokens': 10}), {'protocol_id': 'protocol', 'scope': 'offline', 'benchmark_id': 'bench',
                'task_policy': {'schema': 'rsi-task-policy-v1'},
                'task_sets': {'development': {'task_keys': ['task'], 'seeds': [1]}}}, domain_policy={})
        selection = tools.research_relation_admission(['result'], legal_repairs=['planner_task_hint'],
            selection_id='selection', technical_admitted=True, data_admitted=True)
        self.context['execution_binding']['records']['rho_ref'] = rho
        self.context['relation_dispatch_admission'] = {**selection, 'actual_dispatch_gate': True}
        atomic_json(self.output / 'runtime_context.json', self.context)
        report = evaluate_relation_consumption(self.store, self.output)
        self.assertTrue(report['relations']['rhoC']['value'])
        self.context['relation_dispatch_admission']['decisions'][0]['exploration_used_before'] = 50
        atomic_json(self.output / 'runtime_context.json', self.context)
        with self.assertRaisesRegex(ValueError, 'reserved selection'):
            evaluate_relation_consumption(self.store, self.output)

    def test_readable_rhoA_reports_reference_payload_and_unknown_quality_separately(self):
        self.configure()
        from rsi_vla.research_transport import pack, sdk_input
        request = self.store.get('bounded_requests', 'request')
        request.update(request_id='request', job_id='job', readable_evidence={'schema': 'rsi-readable-development-v1'})
        transport = pack(request)
        atomic_json(self.store.path('bounded_requests', 'request'), request)
        atomic_json(self.store.path('bounded_transports', 'request'), transport)
        actual = sdk_input(transport, {'type': 'object'})
        self.store.put('research_sdk_inputs', 'request', actual)
        self.store.put('research_sdk_consumption', 'request', {'request_id': 'request',
            'sdk_turn_returned': True, 'image_bytes_sent': False})
        report = evaluate_relation_consumption(self.store, self.output, research_request_ids=['request'])
        measurement = report['relations']['rhoA']['measurement']
        self.assertEqual(measurement['media_mode'], 'text_only')
        self.assertEqual(measurement['sdk_prompt_bytes'], len(actual['prompt'].encode('utf-8')))
        self.assertIsNone(measurement['diagnostic_quality'])
        self.assertIsNone(measurement['persisted_media_bytes'])

    def test_foreign_research_candidate_and_changed_dispatch_are_rejected(self):
        self.configure()
        self.store.put('llm_requests', 'foreign', dict(bundle_id='other', evidence_views=[]))
        self.store.put('llm_executions', 'foreign', self.store.get('llm_executions', 'request'))
        with self.assertRaisesRegex(ValueError, 'another relation candidate'):
            evaluate_relation_consumption(self.store, self.output, research_request_ids=['foreign'])
        self.context['relation_dispatch_admission']['decisions'][0]['action'] = 'defer'
        atomic_json(self.output/'runtime_context.json', self.context)
        with self.assertRaisesRegex(ValueError, 'independently recomputed'):
            evaluate_relation_consumption(self.store, self.output)


if __name__ == '__main__':
    unittest.main()
