"""Automatic launcher closure over CPU-only, nonforwarding fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_vla.adapters.runtime_closure import finalize_runtime


class LocalStopClosureTests(unittest.TestCase):
    def fixture(self, root):
        identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
        def write(name, value):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        marker = dict(identity=identity, reason='native_success', sequence=0, step=1, monotonic_ns=1000000000)
        write('native_completion.json', marker)
        artifacts = ['agentview_policy.png', 'agentview.png', 'wrist.png']
        write('states.json', {'steps': [dict(step_idx=1, terminated=True, state={'pose': [0]}, artifacts=artifacts)]})
        for name in artifacts:
            path = root/name/('01' + Path(name).suffix)
            path.parent.mkdir()
            path.write_bytes(b'offline saved observation')
        error = {'codex_error_info': {'response_too_many_failed_attempts': {'http_status_code': 429}},
                 'message': 'exceeded retry limit, last status: 429 Too Many Requests'}
        stats = {'sdk_errors': [{'error': error, 'local_rejection_candidate': True}]}
        actions = [{**identity, 'sequence': 0, 'kind': 'native_completion', 'step': 1, 'reason': 'native_success'},
                   {**identity, 'sequence': 1, 'kind': 'planner_usage', 'stats': stats}]
        model = [dict(kind='request_admitted', request_number=1),
                 dict(kind='response_terminal', request_number=1, event_type='response.completed',
                      usage={'input_tokens': 8, 'input_tokens_details': {'cached_tokens': 0}, 'output_tokens': 2}),
                 dict(kind='request_returned', request_number=1, http_status=200),
                 dict(kind='rejected', reason='native task completed', forwarded=False, monotonic_seconds=2)]
        model = [{'identity': identity, 'sequence': i, **event} for i, event in enumerate(model)]
        receipt = dict(schema_version='1.0', identity=identity, reason='native_success', forwarded=False,
                       rejection_sequence=3, marker_ref=str(root/'native_completion.json'), marker_sequence=0,
                       model_requests_ref=str(root/'model_requests.jsonl'))
        write('local_model_stop.json', receipt)
        (root/'codex_fixture.stream.jsonl').write_text('')
        write('result.json', {**identity, 'benchmark_success': True, 'terminated': True,
                              'attempt_closed': True, 'task': {}, 'error': None, 'validity': 'valid'})
        write('supervisor.json', dict(identity=identity, cleanup_confirmed=True, exit_code=0))
        write('request_gate.json', dict(identity=identity, active_requests_at_close=0, cleanup_complete=True, requests_forwarded=1))
        write('environment_service/runtime_costs.json', dict(identity=identity, service_closed=True,
            pending_action=False, environment_steps_unknown=0, last_native_success=True,
            last_feedback_sequence=1, native_success_source='native'))
        for name in ['runtime_costs.json', 'launcher_costs.json', 'model_load.json', 'runner_cleanup.json']:
            write(name, {'identity': identity})
        return identity, model, actions, receipt, write

    def test_local_stop_requires_every_boundary_and_retains_raw_error(self):
        from rsi_vla.adapters.runtime_closure import local_stop_audit
        for case in ['valid', 'receipt', 'wrong_identity', 'marker', 'missing_media', 'unfinished_tool',
                     'usage', 'upstream429', 'transport', 'other_sdk_error', 'cleanup', 'diagnostic_conflict']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                identity, model, actions, receipt, write = self.fixture(root)
                if case == 'receipt': (root/'local_model_stop.json').unlink()
                elif case == 'wrong_identity':
                    write('local_model_stop.json', {**receipt, 'identity': {**identity, 'run_id': 'wrong'}})
                elif case == 'marker': (root/'native_completion.json').unlink()
                elif case == 'missing_media': (root/'wrist.png/01.png').unlink()
                elif case == 'unfinished_tool':
                    actions.append({**identity, 'sequence': len(actions), 'kind': 'tool_call', 'tool': 'vla_execute'})
                elif case == 'usage': del model[1]['usage']['input_tokens_details']
                elif case == 'upstream429': model[2]['http_status'] = 429
                elif case == 'transport':
                    model.append(dict(identity=identity, sequence=len(model), kind='transport_error'))
                elif case == 'other_sdk_error':
                    actions[1]['stats']['sdk_errors'].append({'error': {'message': 'real RPC failure'}})
                elif case == 'cleanup':
                    write('request_gate.json', dict(identity=identity, cleanup_complete=False, active_requests_at_close=1, requests_forwarded=1))
                elif case == 'diagnostic_conflict':
                    raw = json.loads((root/'result.json').read_text())
                    write('result.json', {**raw, 'planner_diagnostics': {'sdk_errors': []}})
                (root/'model_requests.jsonl').write_text('\n'.join(map(json.dumps, model)))
                (root/'action_events.jsonl').write_text('\n'.join(map(json.dumps, actions)))
                original = (root/'result.json').read_bytes()
                result = finalize_runtime({**identity, 'output_dir': tmp, 'task': {}})
                self.assertEqual(result['outcome'], 'success' if case == 'valid' else 'unknown')
                self.assertEqual((root/'result.json').read_bytes(), original)
                self.assertEqual(result['local_stop_audit']['sdk_errors'][0]['error']['message'],
                                 'exceeded retry limit, last status: 429 Too Many Requests')
                if case not in {'valid', 'cleanup'}:
                    self.assertTrue(result['local_stop_audit']['issues'])
                else:
                    self.assertEqual(local_stop_audit(root, identity, model, actions)['issues'], [])


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
        import test_runtime_plan
        fixture = test_runtime_plan.RuntimePlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'output'
            plan=fixture.plan(run_id='r', output_dir=output)
            context=plan['context']
            identity={key:context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
            # Keep real frozen inputs; only the external process boundary is fake.
            context.pop('task_input', None)
            plan.update(blockers=[], command=['never-executed'], environment={})
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

    def test_verified_native_consumer_close_is_distinct_from_upstream_failure(self):
        from rsi_vla.adapters.runtime_closure import transport_issues, request_usage
        from rsi_vla.storage import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
            marker = dict(identity=identity, reason='native_success', sequence=7, monotonic_ns=1000000000)
            atomic_json(root/'native_completion.json', marker)
            atomic_json(root/'result.json', {**identity, 'benchmark_success':True, 'terminated':True})
            atomic_json(root/'environment_service/runtime_costs.json', dict(identity=identity,
                last_native_success=True, service_closed=True, pending_action=False, environment_steps_unknown=0))
            events = [dict(kind='request_admitted', request_number=1),
                dict(kind='response_terminal', request_number=1, event_type='response.completed',
                     usage={'input_tokens':10, 'input_tokens_details':{'cached_tokens':6}, 'output_tokens':2}),
                dict(kind='downstream_closed', request_number=1, reason='native_success',
                     native_completion_sequence=7, native_completion_ref=str(root/'native_completion.json'), monotonic_seconds=2),
                dict(kind='request_returned', request_number=1, http_status=200)]
            self.assertEqual(transport_issues(root, identity, events), [])
            self.assertEqual(request_usage(events, expected_requests=1)[1], [])
            for change in [{'reason':'client_disconnected'}, {'monotonic_seconds':.5},
                           {'native_completion_sequence':8}, {'native_completion_ref':str(root/'other.json')}]:
                bad = [*events[:2], {**events[2], **change}, events[3]]
                self.assertTrue(transport_issues(root, identity, bad))
            self.assertTrue(transport_issues(root, identity, events+[{'kind':'transport_error'}]))
            self.assertTrue(transport_issues(root, identity, events+[{'kind':'request_incomplete'}]))
            missing = [events[0], events[2]]
            self.assertTrue(transport_issues(root, identity, missing))
            self.assertIsNone(request_usage(missing, expected_requests=1)[0]['input_tokens'])
            atomic_json(root/'native_completion.json', {**marker, 'identity':{**identity, 'bundle_id':'wrong'}})
            self.assertTrue(transport_issues(root, identity, events))

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
