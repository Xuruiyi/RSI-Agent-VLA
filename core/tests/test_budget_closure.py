"""A budget stop is independently closed; arbitrary interruptions stay unknown."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.adapters.budget_closure import verified_budget_result


class BudgetClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('runtime_context.json', {'limits':{'tool_calls':1},
            'run_id':'r','episode_id':'e','bundle_id':'b','protocol_id':'p'})
        self.raw = {'benchmark_success':None, 'planner':{'max_turns':1},
                    'attempt_outcome_source':{'native_success':'worker'}}
        self.write('result.json', self.raw)
        self.write('environment_service/runtime_costs.json',
                   {'last_native_success':False,'last_feedback_sequence':7})
        events = [{'kind':'tool_call','sequence':0},
                  {'kind':'tool_rejected','sequence':1,'reason':'tool_calls exhausted','tool':'finish'},
                  {'kind':'planner_usage','stats':{'planner_turn_status':'interrupted','turns_used':1}}]
        (self.root/'action_events.jsonl').write_text('\n'.join(map(json.dumps,events)))
        requests = [{'kind':'request_admitted'}, {'kind':'response_terminal','event_type':'response.completed'},
                    {'kind':'request_returned','http_status':200}]
        (self.root/'model_requests.jsonl').write_text('\n'.join(map(json.dumps, requests)))
        (self.root/'codex_fixture.stream.jsonl').write_text(json.dumps(
            {'method':'turn/completed','payload':{'turn':{'status':'interrupted'}}}))
        self.patcher = patch('rsi_vla.adapters.runtime_acceptance.inspect_attempt',
                             return_value={'checks':{'identity':True,'independent_result':False}})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_complete_negative_budget_stop_derives_failure_preserving_original(self):
        result = verified_budget_result(self.root)
        self.assertIs(result['benchmark_success'],False)
        self.assertEqual(result['termination_reason'],'verified_budget_exhaustion')
        self.assertEqual(json.loads((self.root/'result.json').read_text()),self.raw)

    def test_upstream_error_or_missing_negative_native_prevents_closure(self):
        path = self.root/'model_requests.jsonl'
        path.write_text(path.read_text()+'\n'+json.dumps({'kind':'transport_error'}))
        with self.assertRaises(ValueError):
            verified_budget_result(self.root)
        self.write('environment_service/runtime_costs.json',{'last_native_success':None})
        with self.assertRaises(ValueError):
            verified_budget_result(self.root)

    def test_incomplete_independent_cleanup_cannot_be_closed(self):
        with patch('rsi_vla.adapters.runtime_acceptance.inspect_attempt',
                   return_value={'checks':{'clean_completion':False}}):
            with self.assertRaises(ValueError):
                verified_budget_result(self.root)
