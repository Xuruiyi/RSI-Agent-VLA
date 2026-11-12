"""Independent closure of a recorded local budget stop; raw files stay intact."""
import json
from pathlib import Path


def verified_budget_result(output):
    """Derive failure only from exhausted admission and complete negative feedback.

    Legacy SDK interrupted has no explicit cause field. This narrow recovery
    requires both a rejected finish and the installed local max-turn controller;
    it does not interpret arbitrary interruption or provider errors as failure.
    """
    from .runtime_acceptance import inspect_attempt
    root = Path(output).resolve()
    def read(name):
        return json.loads((root/name).read_text())
    context, raw = read('runtime_context.json'), read('result.json')
    audit = inspect_attempt(root, require_vla=False, result_path=root/'result.json')
    if any(value is not True for key, value in audit['checks'].items()
           if key != 'independent_result'):
        raise ValueError('incomplete independent action/identity/cost/cleanup evidence')
    identity = {key: context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
    events = [json.loads(line) for line in (root/'action_events.jsonl').read_text().splitlines()]
    requests = [json.loads(line) for line in (root/'model_requests.jsonl').read_text().splitlines()]
    service = read('environment_service/runtime_costs.json')
    stats = next(e['stats'] for e in reversed(events) if e['kind']=='planner_usage')
    admitted = [e for e in events if e['kind']=='tool_call']
    rejected = [e for e in events if e['kind']=='tool_rejected' and e.get('reason')=='tool_calls exhausted']
    if (raw.get('benchmark_success') is not None or raw.get('error') or
        service.get('last_native_success') is not False or
        len(admitted) != context['limits']['tool_calls'] or not rejected or
        not any(e.get('tool')=='finish' for e in rejected) or
        any(e['kind']=='action_send' and e['sequence']>rejected[0]['sequence'] for e in events)):
        raise ValueError('no verified exhausted-tool negative terminal state')
    if (stats.get('planner_turn_status') != 'interrupted' or
        stats.get('turns_used') != raw['planner']['max_turns'] or
        len([e for e in requests if e['kind']=='request_admitted']) != raw['planner']['max_turns'] or
        any(e['kind'] in {'rejected','transport_error'} for e in requests) or
        any(e['kind']=='response_terminal' and e.get('event_type')!='response.completed' for e in requests) or
        any(e['kind']=='request_returned' and e.get('http_status')!=200 for e in requests)):
        raise ValueError('provider failure or unverified local turn-limit stop')
    streams = list(root.glob('codex_*.stream.jsonl'))
    if len(streams)!=1:
        raise ValueError('one bound SDK stream required')
    sdk = [json.loads(line) for line in streams[0].read_text().splitlines()]
    if (not sdk or sdk[-1].get('method')!='turn/completed' or
        sdk[-1].get('payload',{}).get('turn',{}).get('status')!='interrupted' or
        any(e.get('type') in {'timeout','error'} for e in sdk)):
        raise ValueError('SDK terminal evidence is incomplete')
    source = dict(raw['attempt_outcome_source'])
    source['runner_completion'] = 'independent verified budget closure after owned service and process cleanup'
    late = sum(e['kind']=='request_admitted' and e.get('monotonic_seconds',0)*1e9>
               rejected[0].get('monotonic_ns',float('inf')) for e in requests)
    return {**raw, 'benchmark_success':False, 'benchmark_status':'failure',
        'attempt_closed':True, 'termination_reason':'verified_budget_exhaustion',
        'attempt_outcome_source':source, 'derived_from':str(root/'result.json'),
        'closure_evidence':{'identity':identity, 'tool_admissions':len(admitted),
            'first_rejected_sequence':rejected[0]['sequence'],
            'finish_rejected_sequences':[e['sequence'] for e in rejected if e.get('tool')=='finish'],
            'native_last_feedback_sequence':service['last_feedback_sequence'],
            'cause_evidence':'derived from recorded cap/usage/terminal and actual Codex local max-turn controller; original stream has no explicit budget-stop cause',
            'source_refs':[str(root/name) for name in ('runtime_context.json','action_events.jsonl',
                'environment_service/action_events.jsonl','environment_service/runtime_costs.json',
                'model_requests.jsonl','request_gate.json','supervisor.json','launch_plan.json')]+[str(streams[0])],
            'stop_propagation_gap':f'{late} model forwards after first rejected tool; no later environment action sent' if late else None,
            'scope':'negative outcome closure only; does not repair historical stop propagation'}}
