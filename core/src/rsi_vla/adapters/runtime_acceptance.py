"""Independently inspect one actual Runtime attempt and its full producer chain."""
import json
from pathlib import Path

from .rpent import closed_native_failure, execution_health
from .runtime_closure import request_usage, transport_issues


def read(path):
    return json.loads(Path(path).read_text())


def audit_sdk_tools(output, identity, events=None):
    """Reconcile unique SDK item IDs with the identity-bound dispatcher log.

    SDK delivery can stop after a tool has closed in L0. Missing SDK completion
    is retained as a diagnostic; it does not erase the call or its L0 terminal.
    """
    root = Path(output)
    issues, tools, sdk_terminal_missing = [], {}, []
    paths = sorted(root.glob('codex_*.stream.jsonl'))
    if not paths:
        issues.append('SDK tool stream missing')
    for path in paths:
        try:
            for line in path.read_text().splitlines():
                event = json.loads(line)
                payload = event.get('payload') or {}
                item = payload.get('item') or {}
                if not isinstance(item, dict) or item.get('type') not in {
                    'mcpToolCall', 'commandExecution', 'fileChange', 'dynamicToolCall'
                }:
                    continue
                item_id = item.get('id')
                if not item_id:
                    issues.append('SDK tool item ID missing')
                    item_id = f'{path.name}:line:{len(tools)}'
                record = tools.setdefault(item_id, {'id': item_id, 'sdk_started': False,
                                                   'sdk_completed': False})
                for key in ('type', 'server', 'tool', 'started_at_ms', 'completed_at_ms', 'status'):
                    if key in item:
                        if key in ('type', 'server', 'tool') and key in record and record[key] != item[key]:
                            issues.append(f'{item_id}: conflicting SDK {key}')
                        record[key] = item[key]
                if event.get('method') in {'item/failed', 'item/incomplete'}:
                    record['status'] = event['method'].split('/')[1]
                record['sdk_started'] |= event.get('method') == 'item/started'
                record['sdk_completed'] |= event.get('method') == 'item/completed'
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f'{path.name}: {exc}')
    if events is None:
        try:
            events = [json.loads(line) for line in (root/'action_events.jsonl').read_text().splitlines()]
        except (OSError, ValueError) as exc:
            events = []
            issues.append(f'L0 tool events unavailable: {exc}')
    starts, spans, active, rejected = [], [], {}, []
    for event in events:
        kind = event.get('kind')
        if kind not in {'tool_call', 'tool_end', 'tool_error'}:
            continue
        if any(event.get(key) != value for key, value in identity.items()):
            issues.append('L0 tool identity mismatch')
            continue
        name = event.get('tool')
        if kind == 'tool_end' and event.get('status') == 'rejected':
            # Toolkit emits this in finally even when admission raises before
            # tool_call. It is a refusal, never an executed tool span.
            rejected.append(event)
            spans.append((event, event))
        elif kind == 'tool_call':
            starts.append(event)
            if active.get(name):
                issues.append(f'{name}: overlapping L0 calls cannot be uniquely associated')
            active.setdefault(name, []).append(event)
        elif active.get(name):
            start = active[name].pop(0)
            spans.append((start, event))
            if kind != 'tool_end' or event.get('status') != 'completed':
                issues.append(f'{name}: L0 tool did not complete successfully')
        else:
            issues.append(f'{name}: L0 terminal without start')
    for name, pending in active.items():
        if pending:
            issues.append(f'{name}: L0 tool terminal missing')
    outside = [item for item in tools.values()
               if item.get('type') != 'mcpToolCall' or item.get('server') != 'rpent']
    used, matched = set(), set()
    for item_id, item in tools.items():
        if not item['sdk_completed']:
            sdk_terminal_missing.append(item_id)
        if item.get('status') in {'failed', 'incomplete', 'cancelled', 'interrupted'}:
            issues.append(f'{item_id}: SDK tool status {item["status"]}')
        if item in outside:
            issues.append(f'{item_id}: tool outside counted rpent dispatcher')
            continue
        start_ms, end_ms = item.get('started_at_ms'), item.get('completed_at_ms')
        candidates = []
        for index, (start, end) in enumerate(spans):
            if start.get('tool') != item.get('tool'):
                continue
            # L0 does not expose SDK IDs. Its UTC span must be contained in the
            # SDK span; multiple eligible spans remain unconfirmed.
            utc_start, utc_end = start.get('utc_ns'), end.get('utc_ns')
            if (type(start_ms) in (int, float) and type(utc_start) is int
                    and type(utc_end) is int and start_ms * 1e6 <= utc_start <= utc_end
                    and type(start.get('sequence')) is int and type(end.get('sequence')) is int
                    and (start['sequence'] < end['sequence'] or end.get('status') == 'rejected')
                    and (end_ms is None or utc_end <= (end_ms + 1) * 1e6)):
                candidates.append(index)
        if len(candidates) != 1 or candidates[0] in used:
            issues.append(f'{item_id}: SDK tool lacks unique L0 association')
            continue
        index = candidates[0]
        used.add(index)
        matched.add(item_id)
        start, end = spans[index]
        admitted = end.get('status') != 'rejected'
        item.update(l0_start_sequence=start['sequence'] if admitted else None,
                    l0_terminal_sequence=end['sequence'], l0_status=end.get('status'), l0_admitted=admitted)
        failed_issue = f'{item_id}: SDK tool status {item.get("status")}'
        if not admitted and failed_issue in issues:
            issues.remove(failed_issue)
    if len(used) != len(spans):
        issues.append('L0 tools not covered by SDK stream')
    # Unassociated records remain in the union. For a confirmed budget
    # violation, allow every still-possible same-tool/time association: missing
    # or ambiguous terminal evidence cannot by itself prove two executions.
    attempts = starts + rejected
    possible = {}
    for item_id, item in tools.items():
        possible[item_id] = []
        if item.get('type') != 'mcpToolCall' or item.get('server') != 'rpent':
            continue
        begin, end = item.get('started_at_ms'), item.get('completed_at_ms')
        for index, attempt in enumerate(attempts):
            timestamp = attempt.get('utc_ns')
            if (item.get('tool') == attempt.get('tool') and type(timestamp) is int
                    and type(begin) in (int, float) and timestamp >= begin * 1e6
                    and (end is None or timestamp <= (end + 1) * 1e6)):
                possible[item_id].append(index)
    associations = {}
    def associate(item_id, visited):
        for index in possible[item_id]:
            if index in visited:
                continue
            visited.add(index)
            if index not in associations or associate(associations[index], visited):
                associations[index] = item_id
                return True
        return False
    for item_id in tools:
        associate(item_id, set())
    return dict(tools=list(tools.values()), issues=issues, outside_tools=outside,
                counted_tool_calls=len(attempts) + len(tools) - len(matched),
                minimum_tool_calls=len(attempts) + len(tools) - len(associations),
                l0_tool_calls=len(starts), l0_rejected_tool_calls=len(rejected), sdk_tool_calls=len(tools),
                sdk_terminal_missing=sdk_terminal_missing)


def audit_native_persistence(output, identity, events=None):
    """Check the native success record and its required saved observation files."""
    root = Path(output)
    issues, refs, step = [], [], None
    try:
        marker = read(root/'native_completion.json')
        refs.extend([str(root/'native_completion.json'), str(root/'states.json')])
        if any(marker.get('identity', {}).get(k) != v for k, v in identity.items()):
            raise ValueError('native marker identity mismatch')
        if events is None:
            events = [json.loads(line) for line in (root/'action_events.jsonl').read_text().splitlines()]
        matches = [event for event in events if event.get('sequence') == marker.get('sequence')
                   and event.get('kind') == 'native_completion'
                   and all(event.get(k) == v for k, v in identity.items())]
        step = marker.get('step')
        if (marker.get('reason') != 'native_success' or type(step) is not int or len(matches) != 1
                or matches[0].get('step') != step or matches[0].get('reason') != 'native_success'):
            raise ValueError('native marker lacks matching L0 completion')
        states = read(root/'states.json')
        records = [record for record in states['steps'] if record.get('step_idx') == step]
        if len(records) != 1 or records[0].get('terminated') is not True or not records[0].get('state'):
            raise ValueError('native completion state missing or not terminated')
        artifacts = records[0].get('artifacts', [])
        if not {'agentview_policy.png', 'agentview.png', 'wrist.png'} <= set(artifacts):
            raise ValueError('required native observation images missing from state manifest')
        for name in artifacts:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('invalid state artifact name')
            path = root/name/f'{step:02d}{Path(name).suffix}'
            refs.append(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f'native observation artifact missing/empty: {name}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return dict(issues=issues, evidence_refs=refs, step=step)


def inspect_attempt(output, *, require_vla=True, result_path=None):
    root = Path(output)
    context = read(root/'runtime_context.json')
    identity = {key: context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
    checks, notes = {}, []
    confirmed_violations = []
    tool_audit = audit_sdk_tools(root, identity)
    sdk_tools = tool_audit['tools']
    outside_tools = tool_audit['outside_tools']
    checks['all_tools_use_counted_dispatcher'] = (False if outside_tools else
        None if tool_audit['issues'] else True)
    checks['tool_l0_coverage'] = not tool_audit['issues']
    notes.extend(tool_audit['issues'])
    if outside_tools:
        confirmed_violations.append('safety')
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
        checks['model_transport'] = (not transport_issues(root, identity, model_events)
                                     and gate.get('cleanup_complete', True) is True)
        checks['tool_accounting'] = client['tool_calls'] == tool_audit['l0_tool_calls']
        checks['ordered_l0']=all(bound(e) and e['sequence']==i for stream in (events,native) for i,e in enumerate(stream))
        checks['setup_coverage']=(service['construction_complete'] is True and service['reset_calls']>0
            and service['reset_calls']==service['reset_completed'] and service['setup_steps_confirmed']>0
            and service['service_closed'] is True and service['pending_action'] is False
            and service['environment_steps_unknown']==0)
        sends={e['sequence']:e for e in events if e['kind']=='action_send'}
        ends={e['rpc_evidence']['client_send_sequence']:e for e in native if e['kind']=='rpc_dispatch_end'}
        checks['action_mapping']=(bool(sends) and set(sends)==set(ends)
            and all(e['execution_mask'] is not None and not e['unknown_proposal_indices'] for e in ends.values()))
        if bound(service) and service['environment_steps_confirmed'] > context['limits']['environment_steps']:
            confirmed_violations.extend(['safety', 'resource_limits'])
        checks['step_budget']=(service['environment_steps_confirmed']<=context['limits']['environment_steps']
            and service['setup_steps_confirmed']+service['policy_steps_confirmed']==service['environment_steps_confirmed']
            and service['policy_steps_confirmed']>0)
        first_policy=next((e['sequence'] for e in native if e.get('source')=='rlinf.policy' and e['kind']=='action_send'),None)
        checks['no_mid_attempt_reset']=(first_policy is not None and not any(e['kind']=='reset_start' and e['sequence']>first_policy for e in native))
        tool_calls = max(client['tool_calls'] + len(outside_tools), tool_audit['counted_tool_calls'])
        known_tool_calls = max(client['tool_calls'] + len(outside_tools), tool_audit['minimum_tool_calls'])
        checks['tool_budget'] = (False if known_tool_calls > context['limits']['tool_calls'] else
            True if tool_calls <= context['limits']['tool_calls'] else None)
        if bound(client) and checks['tool_budget'] is False:
            confirmed_violations.append('resource_limits')
        if first_policy is not None and all(bound(e) for e in native) and any(e['kind']=='reset_start' and e['sequence']>first_policy for e in native):
            confirmed_violations.append('safety')
        checks['independent_result']=(result.get('benchmark_success') is True and result.get('terminated') is True
            or result.get('benchmark_success') is False and (result.get('truncated') is True or closed_native_failure(result)))
        checks['launcher_wrapper_completion'] = not (root/'launcher_failure.json').exists()
        checks['clean_completion']=(not result.get('error') and result.get('validity')!='technical_invalid' and supervisor['status']=='completed'
            and supervisor['exit_code']==0 and supervisor['cleanup_confirmed'] is True)
        admitted=[e for e in model_events if e['kind']=='request_admitted']
        checks['actual_model_configuration']=(bool(admitted) and all(e['model']=='gpt-6.1-sol'
            and e['reasoning_effort']=='medium' for e in admitted))
        if bound(gate) and gate['requests_forwarded'] > context['limits']['model_requests']:
            confirmed_violations.append('resource_limits')
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
        config = model.get('effective_config') or {}
        openpi = config.get('openpi') or {}
        expected_paths = [('model_path', 'PI05_CHECKPOINT_PATH'), ('norm_stats_path', 'PI05_NORM_STATS_PATH')]
        path_matches = []
        for actual, expected in expected_paths:
            actual_path, expected_path = model.get(actual), plan['environment'].get(expected)
            path_matches.append(Path(actual_path).resolve() == Path(expected_path).resolve()
                if isinstance(actual_path, str) and isinstance(expected_path, str) else None)
        config_matches = [config['pi05'] is True if type(config.get('pi05')) is bool else None,
            openpi['config_name'] == 'pi05_libero' if isinstance(openpi.get('config_name'), str) else None]
        values = path_matches + config_matches
        checks['model_reference'] = (False if False in values else True if all(v is True for v in values) else None)
        if bound(model) and bound(plan['context']) and False in values:
            confirmed_violations.append('compatibility')
        token_usage,token_missing=request_usage(model_events, expected_requests=gate['requests_forwarded'])
        checks['complete_usage']=not token_missing and len(admitted)==gate['requests_forwarded']
        if token_missing:notes.append('Token usage incomplete; observed fields are lower bounds: '+', '.join(token_missing))
        actual_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        from ..process_accounting import accounted_storage_bytes
        actual_bytes = accounted_storage_bytes(root)
        checks['wall_gpu_storage_budget']=(launcher['whole_job_wall_seconds']<=context['limits']['wall_seconds']
            and launcher['gpu_device_seconds_envelope']<=context['limits']['gpu_seconds']
            and actual_bytes<=context['limits']['storage_bytes'])
        if bound(launcher) and checks['wall_gpu_storage_budget'] is False:
            confirmed_violations.append('resource_limits')
        costs=dict(whole_job_wall_seconds=launcher['whole_job_wall_seconds'],gpu_device_seconds_envelope=launcher['gpu_device_seconds_envelope'],
            environment_steps=service['environment_steps_confirmed'],setup_environment_steps=service['setup_steps_confirmed'],
            policy_environment_steps=service['policy_steps_confirmed'],reset_seconds=service['reset_wall_seconds'],
            model_requests=gate['requests_forwarded'],
            tool_calls_lower_bound=known_tool_calls,tool_calls_upper_bound=tool_calls,
            rpent_tool_calls=client['tool_calls'],model_visible_tool_calls=len(sdk_tools),uncounted_external_tools=len(outside_tools),artifact_bytes=actual_bytes,
            training_seconds=launcher['training_seconds'],manual_seconds=launcher['manual_seconds'])
        if tool_calls == known_tool_calls:
            costs['tool_calls'] = tool_calls
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
        outcome=outcome,costs=costs,token_usage=token_usage,tool_audit=tool_audit,
        confirmed_violations=sorted(set(confirmed_violations)),effective_rules=effective_rules,notes=notes,
        scope='runtime infrastructure and independently observed outcome; not formal release or capability improvement',
        explicit_unknown=['controller applied action','internal physics ticks','causal recovery benefit'])
