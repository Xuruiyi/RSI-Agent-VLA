"""Independent file consumer for the frozen all-slot task matrix.

Run this in the evaluator's write domain. No research-selected success labels
or hard-constraint booleans are accepted. Missing checks remain inconclusive.
"""
import argparse
import json
from pathlib import Path

from .contracts import ExperimentResult
from .evaluation import EvaluationStore, load_protocol
from .adapters.rpent import closed_native_failure
from .critical_progress import inspect_progress
from .adapters.runtime_acceptance import inspect_attempt


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def native_flag(value):
    """Match LiberoEnvClient's any/OR semantics for scalar and chunk flags."""
    if type(value) is bool:
        return value
    if isinstance(value, list) and value:
        values = [native_flag(item) for item in value]
        if all(type(item) is bool for item in values):
            return any(values)
    return None


def evaluate_matrix(protocol, profile, manifest, *, result_id, bundle_id,
                    candidate_id=None, role='release'):
    if role not in ('development', 'release'):
        raise ValueError('sealed evidence is not accepted by this candidate feedback consumer')
    split = protocol['task_sets'][role]
    expected = {(key, seed) for key in split['task_keys'] for seed in split['seeds']}
    records = {(row['task_key'], row['seed']): row for row in manifest}
    if len(records) != len(manifest) or not set(records) <= expected:
        raise ValueError('duplicate or unexpected task assignment')
    tasks = {task['task_key']: task for task in protocol['tasks']}
    assignments, notes, refs, costs = [], [], [], {}
    all_identity = all_l0 = all_costs = True
    runtime_reports = []
    progress_reports = []
    for task_key, seed in sorted(expected):
        entry = records.get((task_key, seed))
        row = dict(task_key=task_key, seed=seed, outcome='unknown', validity='valid',
                   evidence_ref=None, outcome_source=None)
        assignments.append(row)
        if entry is None:
            notes.append(f'{task_key}/{seed}: unstarted/missing slot')
            all_identity = all_l0 = all_costs = False
            continue
        try:
            result_path = Path(entry['result_path'])
            if result_path.name == 'result.json' and (result_path.parent/'independent_result.json').exists():
                result_path = result_path.parent/'independent_result.json'
            result = read_json(result_path)
        except (OSError, ValueError, KeyError) as exc:
            notes.append(f'{task_key}/{seed}: missing native result ({exc})')
            all_identity = all_l0 = all_costs = False
            continue
        refs.append(str(result_path))
        identity = dict(run_id=entry['run_id'], episode_id=entry['episode_id'],
                        bundle_id=bundle_id, protocol_id=protocol['protocol_id'])
        task = tasks[task_key]
        task_matches = all(result.get('task', {}).get(key) == value for key, value in
                           dict(suite=task['suite'], task=task['task_id'], seed=seed).items())
        if not task_matches or any(result.get(key) != value for key, value in identity.items()):
            row['validity'] = 'invalid'
            all_identity = all_l0 = all_costs = False
            notes.append(f'{task_key}/{seed}: native result identity mismatch')
            continue
        planner = result.get('planner', {})
        if any(planner.get(key) != profile['planner'][key] for key in ('backend', 'model', 'reasoning_effort')):
            row['validity'] = 'invalid'
            all_identity = False
            notes.append(f'{task_key}/{seed}: runtime model configuration mismatch')
        native = result.get('benchmark_success')
        terminated, truncated = result.get('terminated'), result.get('truncated')
        consistent = (native is True and terminated is True or
                      native is False and terminated is False and (truncated is True or closed_native_failure(result)))
        sources = result.get('native_result_source', {})
        if (type(native) is bool and consistent and sources.get('terminated')
                and (native is True or sources.get('truncated') or closed_native_failure(result))):
            row.update(outcome='success' if native else 'failure', evidence_ref=str(result_path),
                       outcome_source='native_benchmark')
        else:
            notes.append(f'{task_key}/{seed}: no consistent independent terminal result')
        if result.get('error') or result.get('validity') == 'technical_invalid':
            row['validity'] = 'invalid'
        run_root=Path(entry['result_path']).parent
        progress = inspect_progress(run_root, identity)
        progress_reports.append(progress)
        refs.extend(progress['evidence_refs'])
        if (run_root/'runtime_context.json').exists():
            audited=inspect_attempt(run_root,require_vla=False,result_path=result_path)
            runtime_reports.append(audited)
            if audited['checks'].get('execution_health') is False:
                row['validity'] = 'invalid'
                row['outcome'] = 'unknown'
            all_l0 &= audited['passed']
            all_costs &= audited['checks'].get('complete_usage') is True and audited['checks'].get('wall_gpu_storage_budget') is True
            for key,value in audited['costs'].items():
                if type(value) in (int,float):costs[key]=costs.get(key,0)+value
            refs.append(str(run_root/'independent-audit.json'))
            notes.extend(audited['notes'])
            continue
        try:
            events = [json.loads(line) for line in Path(entry['action_events_path']).read_text().splitlines() if line]
            if not events or any(any(event.get(key) != value for key, value in identity.items())
                                 or event.get('sequence') != index for index, event in enumerate(events)):
                raise ValueError('action identity/sequence mismatch')
            required_gaps = {'service_initialization_actions', 'initial_reset_settling_steps'}
            if any(required_gaps.intersection(event.get('missing', [])) for event in events):
                all_l0 = False
                notes.append(f'{task_key}/{seed}: setup/reset action coverage remains incomplete')
            sends = {event['sequence']: event for event in events if event['kind'] == 'action_send'}
            feedback = {event['send_sequence']: event for event in events if event['kind'] == 'action_feedback'}
            if not sends or set(sends) != set(feedback) or any(event.get('execution_mask') is None for event in feedback.values()):
                raise ValueError('action sends lack confirmed feedback')
            flags = {key: [native_flag(event.get('feedback', {}).get(key)) for event in feedback.values()]
                     for key in ('terminated', 'truncated')}
            if any(any(value is None for value in values) or any(values) != result.get(key)
                   for key, values in flags.items()):
                raise ValueError('native outcome conflicts with final action feedback')
            runtime = read_json(entry['costs_path'])
            supervisor = read_json(entry['supervisor_path'])
            for document in (runtime, supervisor):
                if any(document.get('identity', {}).get(key) != value for key, value in identity.items()):
                    raise ValueError('cost/supervisor identity mismatch')
            required = ('tool_calls', 'input_tokens', 'cached_input_tokens', 'output_tokens',
                        'environment_steps_confirmed')
            complete_cost = (all(type(runtime.get(key)) in (int, float) for key in required)
                             and runtime.get('environment_steps_unknown') == 0
                             and supervisor.get('status') == 'completed'
                             and supervisor.get('exit_code') == 0
                             and supervisor.get('cleanup_confirmed') is True
                             and not runtime.get('missing'))
            all_costs &= complete_cost
            for key in required:
                if type(runtime.get(key)) in (int, float):
                    costs[key] = costs.get(key, 0) + runtime[key]
            for key in ('wall_seconds', 'gpu_device_seconds_envelope', 'storage_bytes'):
                if type(supervisor.get(key)) in (int, float):
                    costs[key] = costs.get(key, 0) + supervisor[key]
            refs.extend(str(entry[key]) for key in ('action_events_path', 'costs_path', 'supervisor_path'))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            all_l0 = all_costs = False
            notes.append(f'{task_key}/{seed}: incomplete required L0/costs ({exc})')
    metrics = {}
    for group in ('base', 'controlled', 'old'):
        keys = [key for key in split['task_keys'] if tasks[key]['group'] == group]
        metrics['success_' + group] = sum(
            sum(row['outcome'] == 'success' and row['validity'] == 'valid' for row in assignments
                if row['task_key'] == key) / len(split['seeds']) for key in keys) / len(keys)
    unknown = sum(row['outcome'] == 'unknown' for row in assignments)
    invalid = sum(row['validity'] == 'invalid' for row in assignments)
    # Cost and action coverage do not establish critical progression, safety or
    # model/processor compatibility. Their independent tests are still required.
    constraints = dict.fromkeys(protocol['hard_constraints'], None)
    if all_identity and all_l0 and not unknown and not invalid:
        constraints['reference_coverage'] = True
    if 'critical_progress' in constraints:
        values = [r['value'] for r in progress_reports]
        constraints['critical_progress'] = (False if False in values else True
            if len(values) == len(expected) and all(v is True for v in values) else None)
    actual_intervention={'information_scope':role, 'critical_progress_evidence': progress_reports}
    if len(runtime_reports)==len(expected):
        def verified(keys):
            values=[r['checks'].get(key) for r in runtime_reports for key in keys]
            return True if all(v is True for v in values) else False if any(v is False for v in values) else None
        constraints['compatibility']=verified(['identity','model_reference','vla_actual_loading_and_prediction'])
        constraints['resource_limits']=verified(['tool_budget','request_budget','step_budget','wall_gpu_storage_budget','all_tools_use_counted_dispatcher'])
        constraints['safety']=verified(['all_tools_use_counted_dispatcher','no_mid_attempt_reset','step_budget','clean_completion'])
        old_keys=[key for key in split['task_keys'] if tasks[key]['group']=='old']
        constraints['critical_old_preserved']=all(sum(r['outcome']=='success' and r['validity']=='valid'
            for r in assignments if r['task_key']==key)>=1 for key in old_keys)
    if runtime_reports:
        for key in ('finish_requires_native_success','planner_finish_reserve_calls'):
            values={(r.get('effective_rules') or {}).get(key) for r in runtime_reports}
            if len(values)==1 and None not in values:actual_intervention[key]=values.pop()
        actual_intervention['runtime_rule_observation_scope']='executed_slots_only'
    if not all_costs:
        notes.append('complete full-cost evidence unavailable; resource_limits remains unknown')
    notes.append('safety, compatibility, critical_progress and actual authorized resource limits require independent checks')
    outcome = 'unknown' if unknown else 'failure' if any(row['outcome'] == 'failure' for row in assignments) else 'success'
    return ExperimentResult(result_id, bundle_id, protocol['protocol_id'], split['benchmark_id'], role,
                            protocol['scope'], actual_intervention, refs, protocol['protocol_id'], metrics, constraints, costs,
                            candidate_id=candidate_id, assigned_count=len(expected), unknown_count=unknown,
                            invalid_count=invalid, outcome=outcome, assignments=assignments, notes=notes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'profile', 'manifest', 'output', 'result', 'bundle'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate')
    args = parser.parse_args(argv)
    result = evaluate_matrix(load_protocol(args.protocol), read_json(args.profile), read_json(args.manifest),
                             result_id=args.result, bundle_id=args.bundle, candidate_id=args.candidate)
    EvaluationStore(args.output).save_result(result)
    print(json.dumps(dict(result_id=result.result_id, unknown_count=result.unknown_count,
                          invalid_count=result.invalid_count, assigned_count=result.assigned_count)))


if __name__ == '__main__':
    main()
