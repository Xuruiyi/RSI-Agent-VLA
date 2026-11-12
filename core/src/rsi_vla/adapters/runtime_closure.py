"""Launcher-owned final records after backend and transport cleanup; no model calls."""
import json
from pathlib import Path

from ..storage import atomic_json
from .rpent import closed_native_failure, execution_health

IDENTITY = ('run_id', 'episode_id', 'bundle_id', 'protocol_id')
ACTION_TOOLS = {'vla_execute','pi0_pick','pi0_doubled','move_to','move_pose','release','set_gripper',
                'rotate_wrist','rotate_pitch','execute_action','wam_act'}


def request_usage(model_events, *, expected_requests=None):
    """Keep returned usage as a lower bound when any admitted request is missing."""
    admitted = [event for event in model_events if event['kind']=='request_admitted']
    terminals = {event.get('request_number'):event for event in model_events if event['kind']=='response_terminal'}
    missing=[]
    token_values = {}
    for name,key in [('input_tokens','input_tokens'),('cached_input_tokens','cached_tokens'),('output_tokens','output_tokens')]:
        values=[]
        for event in admitted:
            usage=terminals.get(event['request_number'],{}).get('usage') or {}
            value=(usage.get('input_tokens_details') or {}).get(key) if name=='cached_input_tokens' else usage.get(key)
            if type(value) is int:values.append(value)
        token_values[name] = sum(values) if len(values)==len(admitted) else None
        token_values['observed_'+name] = sum(values) if values else None
        if len(values)!=len(admitted):missing.append(name)
    if expected_requests is not None and expected_requests!=len(admitted):
        for name in ('input_tokens','cached_input_tokens','output_tokens'):
            token_values[name]=None
            if name not in missing:missing.append(name)
    return token_values, missing


def finalize_runtime(context):
    """Keep raw records; absent or interrupted evidence stays unknown."""
    root = Path(context['output_dir'])
    identity = {key:context[key] for key in IDENTITY}
    missing = []
    def read(name):
        try:
            value = json.loads((root/name).read_text())
            if any(value.get('identity', value).get(k) != v for k,v in identity.items()):
                raise ValueError('identity mismatch')
            return value
        except (OSError, ValueError):
            missing.append(name)
            return {}
    def events(name):
        values = []
        try:
            for line in (root/name).read_text().splitlines():
                try: values.append(json.loads(line))
                except ValueError: missing.append(name+':incomplete_line')
        except OSError: missing.append(name)
        return values
    raw = read('result.json')
    supervisor = read('supervisor.json')
    launcher = read('launcher_costs.json')
    service = read('environment_service/runtime_costs.json')
    client = read('runtime_costs.json')
    gate = read('request_gate.json')
    model = read('model_load.json')
    runner_cleanup = read('runner_cleanup.json')
    action_events = events('action_events.jsonl')
    model_events = events('model_requests.jsonl')
    runtime_stop = root/'runtime_stop.json'
    budget_stop = root/'budget_stop.json'
    reason = supervisor.get('stop_reason')
    if runtime_stop.exists():
        reason = read('runtime_stop.json').get('reason') or reason
    budget = read('budget_stop.json') if budget_stop.exists() else {}
    bad_transport = any(e['kind']=='transport_error' or
        e['kind']=='request_returned' and e.get('http_status') != 200 or
        e['kind']=='response_terminal' and e.get('event_type') != 'response.completed'
        for e in model_events)
    health = execution_health(action_events, identity, source_ref=str(root/'action_events.jsonl'))
    if any(name.startswith('action_events.jsonl') for name in missing):
        health['status'] = 'unconfirmed'
        health['missing'].append('execution_events_unavailable_or_incomplete')
    execution_fault = health['status'] != 'clear' or (raw.get('execution_health') or {}).get('status', 'clear') != 'clear'
    technical = bool(execution_fault or raw.get('validity') == 'technical_invalid' or bad_transport or raw.get('error') or supervisor.get('error') or
        (root/'launcher_error.json').exists() or supervisor.get('exit_code') not in (0, None))
    cancelled = reason == 'external_interruption' or any(e.get('stop_reason') == 'cancelled' for e in health['stops']) or raw.get('interrupted') is True and not technical and not budget
    clean = supervisor.get('cleanup_confirmed') is True
    native_complete = bool(service.get('service_closed') is True and
        service.get('pending_action') is False and service.get('environment_steps_unknown') == 0
        and service.get('last_native_success') is raw.get('benchmark_success')
        and service.get('last_feedback_sequence') is not None and service.get('native_success_source'))
    native_consistent = (raw.get('benchmark_success') is True and raw.get('terminated') is True or
        raw.get('benchmark_success') is False and raw.get('terminated') is False and
        (raw.get('truncated') is True or closed_native_failure(raw)))
    task_matches = all(raw.get('task',{}).get(key)==value for key,value in context.get('task',{}).items())
    status = ('cancelled' if cancelled else
              'timeout' if reason in {'wall_seconds','gpu_seconds_envelope'} else
              'technical_error' if technical else
              'budget_exhausted' if budget or reason=='storage_bytes' else
              'completed' if clean and raw.get('attempt_closed') else 'unknown')
    # Native terminal evidence is consumed only after a clean, nontechnical closure.
    accepted = bool(raw and clean and native_complete and native_consistent and task_matches and
                    gate.get('active_requests_at_close') == 0 and not technical and not cancelled and
                    not reason and raw.get('attempt_closed') is True and
                    type(raw.get('benchmark_success')) is bool)
    native_task_result = {key: raw.get(key) for key in
                          ('benchmark_success', 'benchmark_status', 'terminated', 'truncated', 'attempt_closed', 'termination_reason')}
    native_task_result.update(last_native_success=service.get('last_native_success'),
                              source_ref=str(root/'result.json'), service_ref=str(root/'environment_service/runtime_costs.json'))
    independent = {**raw, **identity, 'schema_version':'1.0', 'task':context.get('task',{}),
        'benchmark_success':raw.get('benchmark_success') if accepted else None,
        'benchmark_status':raw.get('benchmark_status') if accepted else 'unknown',
        'attempt_closed':accepted, 'validity':'valid' if accepted else 'technical_invalid',
        'outcome':'success' if accepted and raw['benchmark_success'] else 'failure' if accepted else 'unknown',
        'runtime_status':status, 'runtime_stop_reason':reason or ('execution_technical_error' if execution_fault else None) or ('vla_cancelled' if cancelled else None) or ('model_transport_error' if bad_transport else None)
            or ('backend_exit_'+str(supervisor['exit_code']) if supervisor.get('exit_code') not in (0,None) else None)
            or budget.get('reason') or raw.get('termination_reason') or 'result_unavailable_or_unconfirmed',
        'execution_health':health, 'native_task_result':native_task_result,
        'raw_result_ref':str(root/'result.json') if raw else None,
        'closure_source':'launcher after process and transport cleanup', 'missing':list(missing)}
    admitted = [e for e in model_events if e['kind']=='request_admitted']
    terminals = {e.get('request_number'):e for e in model_events if e['kind']=='response_terminal'}
    token_values, token_missing = request_usage(model_events, expected_requests=gate.get('requests_forwarded'))
    if not model_events and gate.get('requests_forwarded') is None:
        token_values={name:None for name in token_values}
        token_missing.extend(['input_tokens','cached_input_tokens','output_tokens'])
    missing.extend(token_missing)
    inference_starts = [e for e in action_events if e['kind']=='inference_start']
    inference_ends = [e for e in action_events if e['kind']=='inference_end']
    inference_complete = bool(action_events) and len(inference_starts)==len(inference_ends)
    timings = {
        'whole_job_wall_seconds':launcher.get('whole_job_wall_seconds'),
        'model_load_seconds':model.get('model_load_seconds'),
        'auxiliary_model_load_seconds':None,
        'daemon_cleanup_seconds':runner_cleanup.get('daemon_cleanup_seconds'),
        'model_request_seconds':(sum(e['elapsed_seconds'] for e in model_events if e['kind'] in {'request_returned','transport_error'})
            if all('elapsed_seconds' in e for e in model_events if e['kind'] in {'request_returned','transport_error'})
            and len([e for e in model_events if e['kind'] in {'request_returned','transport_error'}])==len(admitted) else None),
        'completed_tool_seconds':sum(e.get('elapsed_seconds',0) for e in action_events if e['kind']=='tool_end'),
        'completed_action_tool_seconds':sum(e.get('elapsed_seconds',0) for e in action_events if e['kind']=='tool_end' and e.get('tool') in ACTION_TOOLS),
        'completed_state_media_seconds':sum(e.get('elapsed_seconds',0) for e in action_events if e['kind']=='media_state_end'),
        'cleanup_media_seconds':sum(e.get('elapsed_seconds',0) for e in action_events if e['kind']=='media_cleanup_end') if any(e['kind']=='media_cleanup_end' for e in action_events) else None,
        'process_group_cleanup_seconds':supervisor.get('process_group_cleanup_seconds'),
        'vla_inference_seconds':sum(e['elapsed_seconds'] for e in inference_ends) if inference_complete and all('elapsed_seconds' in e for e in inference_ends) else None,
        'vla_action_rpc_seconds':sum(e.get('elapsed_seconds',0) for e in action_events if e['kind']=='chunk_execution_end') if any(e['kind']=='chunk_execution_end' for e in action_events) else None,
        'reset_seconds':service.get('reset_wall_seconds'),
    }
    costs={'identity':identity, **token_values, 'cached_input_is_subset':True,
        'actual_model_requests':gate.get('requests_forwarded', len(admitted) if model_events else None), 'tool_calls':client.get('tool_calls'),
        'vla_predictions_started':len(inference_starts) if action_events else None,
        'vla_predictions_completed':sum(e.get('completed') is True for e in inference_ends) if inference_complete else None,
        'vla_action_steps_confirmed':sum(e.get('executed_steps',0) for e in action_events if e['kind']=='chunk_execution_end'),
        'environment_steps':service.get('environment_steps_confirmed'),
        'environment_steps_reserved':service.get('environment_steps_reserved'),
        'timings':timings,
        'timing_relationships':'whole_job includes startup through final cleanup; model load/request/tool/media/cleanup are nested or overlapping diagnostics, never summed as total; completed action_tool and state_media are subsets of completed tool spans; readonly calls may overlap; incomplete tools have no full-span estimate; cleanup_media is outside tool; process_group_cleanup includes daemon drain and may overlap cleanup_media',
        'missing':missing + [name for name,value in timings.items() if value is None],
        'supervisor_ref':str(root/'supervisor.json'), 'launcher_ref':str(root/'launcher_costs.json')}
    atomic_json(root/'independent_result.json',independent)
    atomic_json(root/'cost_breakdown.json',costs)
    return independent
