"""Automatic launcher closure over CPU-only, nonforwarding fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_vla.adapters.runtime_closure import finalize_runtime


class RuntimeClosureTests(unittest.TestCase):
    def test_all_terminal_paths_and_partial_usage(self):
        for case in ('success','failure','budget','cancel','timeout','transport','missing','wrong_identity'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                root=Path(d);identity=dict(run_id='r',episode_id='e',bundle_id='b',protocol_id='p')
                context={**identity,'output_dir':d,'task':{'suite':'s','task':2,'seed':1}}
                def write(name,value):
                    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'identity':identity,**value}))
                raw={**identity,'benchmark_success':case=='success','benchmark_status':'success' if case=='success' else 'failure',
                    'task':context['task'],'attempt_closed':True,'terminated':case=='success','truncated':case!='success','termination_reason':'verified_budget_exhaustion' if case=='budget' else 'normal_runner_completion','error':None}
                if case!='missing':write('result.json',raw)
                if case=='wrong_identity':write('result.json',{**raw,'identity':{**identity,'bundle_id':'wrong'},'bundle_id':'wrong'})
                original=(root/'result.json').read_bytes() if (root/'result.json').exists() else None
                write('supervisor.json',{'cleanup_confirmed':True,'exit_code':1 if case=='transport' else 0,
                    'stop_reason':'external_interruption' if case=='cancel' else 'wall_seconds' if case=='timeout' else None,'process_group_cleanup_seconds':.1})
                write('launcher_costs.json',{'whole_job_wall_seconds':5})
                write('environment_service/runtime_costs.json',{'service_closed':True,'pending_action':False,'environment_steps_unknown':0,'environment_steps_confirmed':30,'last_native_success':case=='success','last_feedback_sequence':5,'native_success_source':'native'})
                write('runtime_costs.json',{'tool_calls':2})
                write('request_gate.json',{'active_requests_at_close':0})
                write('model_load.json',{'model_load_seconds':1})
                if case=='budget':write('budget_stop.json',{'reason':'tool_calls','limit':2,'observed':2})
                events=[{'kind':'request_admitted','request_number':1}, {'kind':'response_terminal','request_number':1,'event_type':'response.completed','usage':{'input_tokens':10,'input_tokens_details':{'cached_tokens':6},'output_tokens':2}}]
                if case=='transport':events += [{'kind':'request_admitted','request_number':2},{'kind':'transport_error','request_number':2,'elapsed_seconds':1}]
                (root/'model_requests.jsonl').write_text('\n'.join(map(json.dumps,events)))
                (root/'action_events.jsonl').write_text('\n'.join(json.dumps({**identity,'sequence':i,**event}) for i,event in enumerate([
                    {'kind':'inference_start'},
                    {'kind':'inference_end','completed':True,'elapsed_seconds':.3},
                    {'kind':'chunk_execution_end','executed_steps':5,'elapsed_seconds':.2},
                    {'kind':'tool_end','tool':'vla_execute','elapsed_seconds':.8} ])))
                result=finalize_runtime(context)
                self.assertEqual(result['outcome'],'success' if case=='success' else 'failure' if case in {'failure','budget'} else 'unknown')
                if original:self.assertEqual((root/'result.json').read_bytes(),original)
                costs=json.loads((root/'cost_breakdown.json').read_text())
                self.assertEqual(costs['input_tokens'],None if case=='transport' else 10)
                self.assertEqual(costs['cached_input_tokens'],None if case=='transport' else 6)
                self.assertEqual(costs['observed_input_tokens'],10)
                if case=='cancel':self.assertEqual(result['runtime_status'],'cancelled')
                if case=='timeout':self.assertEqual(result['runtime_status'],'timeout')
                if case=='transport':self.assertEqual(result['runtime_status'],'technical_error')
                self.assertTrue(result['runtime_stop_reason'])
                self.assertEqual(costs['vla_predictions_started'],1)
                self.assertEqual(costs['vla_predictions_completed'],1)
                self.assertEqual(costs['vla_action_steps_confirmed'],5)
                self.assertEqual(costs['timings']['vla_inference_seconds'],.3)
                self.assertEqual(costs['timings']['vla_action_rpc_seconds'],.2)
                self.assertEqual(costs['timings']['completed_action_tool_seconds'],.8)
                if case=='success':self.assertEqual(result['runtime_status'],'completed')

    def test_launcher_saves_independent_unknown_when_backend_never_writes_result(self):
        from unittest.mock import patch
        from rsi_vla.runtime import execute_runtime
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'output'
            identity=dict(run_id='r',episode_id='e',bundle_id='b',protocol_id='p')
            context={**identity,'output_dir':str(output),'gpu_count':1,
                     'limits':{'wall_seconds':900},'task':{'suite':'s','task':2,'seed':1}}
            root=Path(__file__).resolve().parents[2]
            plan=dict(blockers=[],context=context,cwd=str(root/'RPent'),command=['never-executed'],environment={})
            def fake_supervise(command, *, context_path):
                self.assertEqual(command,['never-executed'])
                report={'identity':identity,'cleanup_confirmed':True,'exit_code':1,'stop_reason':None}
                (output/'supervisor.json').write_text(json.dumps(report))
                return report
            with patch.dict('os.environ'), patch('rsi_vla.adapters.runtime_supervisor.supervise',side_effect=fake_supervise):
                execute_runtime(plan)
            self.assertTrue((output/'launcher_costs.json').is_file())
            result=json.loads((output/'independent_result.json').read_text())
            self.assertEqual(result['outcome'],'unknown')
            self.assertEqual(result['runtime_status'],'technical_error')
            self.assertTrue((output/'cost_breakdown.json').is_file())

    def test_usage_requires_each_admitted_request_and_cached_field(self):
        from rsi_vla.adapters.runtime_closure import request_usage
        events=[{'kind':'request_admitted','request_number':1},
                {'kind':'response_terminal','request_number':1,'usage':{'input_tokens':10,'output_tokens':2}}]
        usage,missing=request_usage(events,expected_requests=1)
        self.assertEqual(usage['input_tokens'],10)
        self.assertIsNone(usage['cached_input_tokens'])
        self.assertIn('cached_input_tokens',missing)
        usage,missing=request_usage(events,expected_requests=2)
        self.assertIsNone(usage['input_tokens'])
        self.assertEqual(usage['observed_input_tokens'],10)
