"""Independently inspect one actual Runtime attempt and its full producer chain."""
import json
from pathlib import Path

from .rpent import closed_native_failure, execution_health
from .runtime_closure import request_usage


def read(path):
    return json.loads(Path(path).read_text())


def inspect_attempt(output, *, require_vla=True, result_path=None):
    root = Path(output)
    context = read(root/'runtime_context.json')
    identity = {key: context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
    checks, notes = {}, []
    token_usage=None
    def bound(document):
        value = document.get('identity',document)
        return all(value.get(k)==v for k,v in identity.items())
    try:
        selected = Path(result_path) if result_path else root/('independent_result.json' if (root/'independent_result.json').exists() else 'result.json')
        result=read(selected); service=read(root/'environment_service/runtime_costs.json')
        client=read(root/'runtime_costs.json'); launcher=read(root/'launcher_costs.json')
        supervisor=read(root/'supervisor.json'); gate=read(root/'request_gate.json'); model=read(root/'model_load.json')
        checks['identity']=all(bound(d) for d in (result,service,client,launcher,supervisor,gate,model))
        if context.get('code_snapshot_ref'):
            snapshot=read(context['code_snapshot_ref'])
            checks['runtime_source_snapshot']=(bound(snapshot) and bool(snapshot['files'])
                and all((root/'code_snapshot'/name).is_file() for name in snapshot['files']))
        events=[json.loads(s) for s in (root/'action_events.jsonl').read_text().splitlines()]
        health=execution_health(events, identity, source_ref=str(root/'action_events.jsonl'))
        checks['execution_health']=health['status']=='clear' and (result.get('execution_health') or {}).get('status','clear')=='clear'
        native=[json.loads(s) for s in (root/'environment_service/action_events.jsonl').read_text().splitlines()]
        model_events=[json.loads(s) for s in (root/'model_requests.jsonl').read_text().splitlines()]
        sdk_tools=[]
        for raw_path in root.glob('codex_*.stream.jsonl'):
            for line in raw_path.read_text().splitlines():
                event=json.loads(line)
                item=(event.get('payload') or {}).get('item',{})
                if event.get('method')=='item/completed' and isinstance(item,dict) and item.get('type') in {'mcpToolCall','commandExecution','fileChange','dynamicToolCall'}:
                    sdk_tools.append(item)
        outside_tools=[item for item in sdk_tools if item.get('type')!='mcpToolCall' or item.get('server')!='rpent']
        checks['all_tools_use_counted_dispatcher']=not outside_tools
        checks['ordered_l0']=all(bound(e) and e['sequence']==i for stream in (events,native) for i,e in enumerate(stream))
        checks['setup_coverage']=(service['construction_complete'] is True and service['reset_calls']>0
            and service['reset_calls']==service['reset_completed'] and service['setup_steps_confirmed']>0
            and service['service_closed'] is True and service['pending_action'] is False
            and service['environment_steps_unknown']==0)
        sends={e['sequence']:e for e in events if e['kind']=='action_send'}
        ends={e['rpc_evidence']['client_send_sequence']:e for e in native if e['kind']=='rpc_dispatch_end'}
        checks['action_mapping']=(bool(sends) and set(sends)==set(ends)
            and all(e['execution_mask'] is not None and not e['unknown_proposal_indices'] for e in ends.values()))
        checks['step_budget']=(service['environment_steps_confirmed']<=context['limits']['environment_steps']
            and service['setup_steps_confirmed']+service['policy_steps_confirmed']==service['environment_steps_confirmed']
            and service['policy_steps_confirmed']>0)
        first_policy=next((e['sequence'] for e in native if e.get('source')=='rlinf.policy' and e['kind']=='action_send'),None)
        checks['no_mid_attempt_reset']=(first_policy is not None and not any(e['kind']=='reset_start' and e['sequence']>first_policy for e in native))
        checks['tool_budget']=client['tool_calls']+len(outside_tools)<=context['limits']['tool_calls']
        checks['independent_result']=(result.get('benchmark_success') is True and result.get('terminated') is True
            or result.get('benchmark_success') is False and (result.get('truncated') is True or closed_native_failure(result)))
        checks['clean_completion']=(not result.get('error') and result.get('validity')!='technical_invalid' and supervisor['status']=='completed'
            and supervisor['exit_code']==0 and supervisor['cleanup_confirmed'] is True)
        admitted=[e for e in model_events if e['kind']=='request_admitted']
        terminals=[e for e in model_events if e['kind']=='response_terminal']
        checks['actual_model_configuration']=(bool(admitted) and all(e['model']=='gpt-6.1-sol'
            and e['reasoning_effort']=='medium' for e in admitted))
        checks['request_budget']=(gate['hard_request_cap_enforced'] is True and 0<gate['requests_forwarded']<=context['limits']['model_requests']
            and gate['requests_forwarded']==len(admitted) and gate['active_requests_at_close']==0)
        stop_path=Path(context.get('budget_stop_ref', root/'budget_stop.json'))
        if stop_path.exists():
            stop=read(stop_path)
            checks['budget_stop_propagation']=(bound(stop) and not any(
                e['monotonic_seconds']*1e9>stop['monotonic_ns'] for e in admitted))
        elif result.get('closure_evidence',{}).get('stop_propagation_gap'):
            checks['budget_stop_propagation']=False
        if context.get('task_input'):
            task_input=read(root/'task_input.json')
            checks['startup_input_identity']=(bound(task_input) and task_input.get('version')==context['task_input']['version']
                and task_input.get('task')==context.get('task') and task_input.get('project_learning') is False)
        checks['vla_actual_loading_and_prediction']=(model['loaded'] is True
            and model['embodiment']=='libero' and model['action_chunk']==5
            and (not require_vla and model['prediction_count']==0 or model['prediction_count']>0
                 and model['last_actions_finite'] is True and model['last_action_shape'][-1]==7))
        plan=read(root/'launch_plan.json')
        checks['model_reference']=(Path(model['model_path']).resolve()==Path(plan['environment']['PI05_CHECKPOINT_PATH']).resolve()
            and Path(model['norm_stats_path']).resolve()==Path(plan['environment']['PI05_NORM_STATS_PATH']).resolve()
            and model['effective_config'].get('pi05') is True
            and model['effective_config']['openpi']['config_name']=='pi05_libero')
        token_usage,token_missing=request_usage(model_events, expected_requests=gate['requests_forwarded'])
        checks['complete_usage']=not token_missing and len(admitted)==gate['requests_forwarded']
        if token_missing:notes.append('Token usage incomplete; observed fields are lower bounds: '+', '.join(token_missing))
        actual_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        checks['wall_gpu_storage_budget']=(launcher['whole_job_wall_seconds']<=context['limits']['wall_seconds']
            and launcher['gpu_device_seconds_envelope']<=context['limits']['gpu_seconds']
            and actual_bytes<=context['limits']['storage_bytes'])
        costs=dict(whole_job_wall_seconds=launcher['whole_job_wall_seconds'],gpu_device_seconds_envelope=launcher['gpu_device_seconds_envelope'],
            environment_steps=service['environment_steps_confirmed'],setup_environment_steps=service['setup_steps_confirmed'],
            policy_environment_steps=service['policy_steps_confirmed'],reset_seconds=service['reset_wall_seconds'],
            model_requests=gate['requests_forwarded'],tool_calls=client['tool_calls']+len(outside_tools),
            rpent_tool_calls=client['tool_calls'],model_visible_tool_calls=len(sdk_tools),uncounted_external_tools=len(outside_tools),artifact_bytes=actual_bytes,
            training_seconds=launcher['training_seconds'],manual_seconds=launcher['manual_seconds'])
        for key,name in [('input_tokens','llm_input_tokens'),('cached_input_tokens','llm_cached_input_tokens_subset'),('output_tokens','llm_output_tokens')]:
            if token_usage[key] is not None:costs[name]=token_usage[key]
            elif token_usage['observed_'+key] is not None:costs['observed_'+name]=token_usage['observed_'+key]
        outcome='success' if result.get('benchmark_success') is True else 'failure' if result.get('benchmark_success') is False else 'unknown'
        if not checks['execution_health']:
            outcome='unknown'
            notes.append('Unresolved or unconfirmed execution fault; normal Planner completion is not recovery evidence')
        effective_rules=next((e.get('effective_rules') for e in events if e['kind']=='run_start'),None)
    except (OSError,ValueError,KeyError,TypeError) as error:
        checks['required_artifacts_complete']=False;notes.append(f'{type(error).__name__}: {error}')
        costs={};outcome='unknown';effective_rules=None
    return dict(identity=identity,checks=checks,passed=bool(checks) and all(v is True for v in checks.values()),
        outcome=outcome,costs=costs,token_usage=token_usage,effective_rules=effective_rules,notes=notes,
        scope='runtime infrastructure and independently observed outcome; not formal release or capability improvement',
        explicit_unknown=['controller applied action','internal physics ticks','causal recovery benefit'])
