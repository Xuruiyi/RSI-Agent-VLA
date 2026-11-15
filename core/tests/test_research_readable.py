"""Readable legal facts, host supplementation and exact SDK call arguments."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import test_relation_research
from rsi_vla.bounded_runner import child
from rsi_vla.research_transport import model_prompt, pack, sdk_input, unpack
from rsi_vla.storage import atomic_json


class ReadableResearchTests(unittest.TestCase):
    setUp = test_relation_research.RelationResearchTests.setUp
    make_tools = test_relation_research.RelationResearchTests.make_tools
    response = test_relation_research.RelationResearchTests.response
    apply = test_relation_research.RelationResearchTests.apply

    def configure(self):
        self.protocol['task_policy']['research_input'] = dict(mode='readable_v1', max_read_requests=2,
                                                             max_events_per_read=2)
        self.tools.domain_policy['evidence_input'] = dict(mode='selected_l0', reason='bounded diagnosis',
                                                        event_ids={'result': ['e0']})

    def test_readable_prompt_preserves_original_payload_omissions_and_unknown_media(self):
        self.configure()
        self.response()
        request = self.source.get('bounded_requests', 'rho-request')
        prompt = model_prompt(pack(request))
        decoded = json.loads(prompt.split('\n', 1)[1])
        self.assertEqual(decoded, request)
        self.assertEqual(decoded['evidence_views'][0]['l0'][0], self.source.get('evidence_views', 'view')['l0'][0])
        manifest = request['readable_evidence']['views'][0]
        self.assertEqual(manifest['selected_event_ids'], ['e0'])
        self.assertEqual(manifest['omitted_event_ids'], ['e1', 'e2', 'e3'])
        self.assertIsNone(request['readable_evidence']['diagnostic_quality'])
        self.assertEqual(manifest['media_consumption'], dict(mode='text_only', image_bytes_sent=False,
                                                           visual_content='unknown'))
        self.assertIn('"event_id": "e0"', prompt)
        self.assertNotIn('Complete reversible evidence:', prompt)

    def test_defer_consumes_exact_legal_read_and_prepares_separately_identified_followup(self):
        self.configure()
        response = self.response()
        response.update(decision='defer', intervention=None, missing_evidence=['handoff event missing'],
                        evidence_requests=[dict(result_id='result', event_ids=['e1'], reason='check recorded handoff')])
        value = self.apply(response)
        self.assertEqual(value['next_action'], 'prepare_evidence_followup')
        receipt = self.source.get('research_evidence_supplements', 'rho-request')
        self.assertEqual(receipt['reads'][0]['events'][0], self.source.get('evidence_views', 'view')['l0'][1])
        followup = unpack(self.adapter.prepare_evidence_followup(self.tools, parent_request_id='rho-request',
            request_id='followup', max_input_tokens=100, max_output_tokens=10))
        self.assertEqual([e['event_id'] for e in followup['evidence_views'][0]['l0']], ['e0', 'e1'])
        self.assertEqual(followup['evidence_followup']['parent_request_id'], 'rho-request')
        self.assertEqual(len(self.source.get('evidence_views', 'view')['l0']), 4)
        self.assertFalse(self.source.all('llm_executions'))
        self.assertFalse(self.source.all('rule_trials'))
        self.assertFalse(self.source.all('candidates'))
        # The original request and decision keep the same immutable identity on replay.
        self.assertEqual(value, self.apply(response))

    def test_read_requests_reject_paths_outside_results_unknown_ids_and_unbounded_reads(self):
        self.configure()
        self.response()
        for row in (dict(result_id='sealed', event_ids=['e1'], reason='invalid'),
                    dict(result_id='result', event_ids=['/etc/passwd'], reason='invalid'),
                    dict(result_id='result', event_ids=['e1', 'e2', 'e3'], reason='too many'),
                    dict(result_id='result', event_ids=['e1', 'e1'], reason='duplicate'),
                    dict(result_id='result', event_ids=['e1'], reason='bad', path='/etc/passwd')):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.tools.consume_evidence_requests('rho-request', [row])
        self.assertFalse(self.source.all('research_evidence_supplements'))

    def test_actual_sdk_consumer_receives_saved_prompt_without_media(self):
        self.configure()
        self.response()
        transport = self.source.get('bounded_transports', 'rho-request')
        output = self.root / 'sdk'
        output.mkdir()
        permit = output / 'permit.json'
        atomic_json(permit, dict(authorized=True, input=transport))
        atomic_json(output / 'runtime_context.json', dict(authorization_ref=str(permit)))
        atomic_json(output / 'request.json', transport)
        schema = self.adapter.response_schema('rho-request', tools=self.tools)
        atomic_json(output / 'response-schema.json', schema)
        atomic_json(output / 'dispatch.json', {'request_id': 'rho-request'})
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        thread = client.thread_start.return_value
        thread.turn.return_value.run.return_value = Mock(status='completed', final_response='{}')
        with patch('openai_codex.Codex', return_value=client), patch('rsi_vla.bounded_runner._research_codex_config'):
            child(output)
        saved = json.loads((output / 'sdk-input.json').read_text())
        self.assertEqual(saved, sdk_input(transport, schema))
        self.assertEqual(thread.turn.call_args.args, (saved['prompt'],))
        self.assertEqual(thread.turn.call_args.kwargs['output_schema'], saved['output_schema'])
        self.assertFalse(saved['image_bytes_sent'])
        self.assertTrue(json.loads((output / 'sdk-input-consumption.json').read_text())['sdk_turn_returned'])
        with self.assertRaises(FileExistsError):
            child(output)


if __name__ == '__main__':
    unittest.main()
