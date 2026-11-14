#!/usr/bin/env python3
"""Execute one fresh authorized four-initial plus four-candidate release matrix."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core/src'))

from rsi_vla.adapters.runtime_acceptance import inspect_attempt
from rsi_vla.critical_progress import inspect_progress
from rsi_vla.contracts import Candidate
from rsi_vla.evaluation import EvaluationStore, Gate, score, apply_comparison_attempt_evidence
from rsi_vla.evaluator import evaluate_matrix
from rsi_vla.release import inspect_initial_result, inspect_initial_comparison_result
from rsi_vla.research_cycle import _phase_charge
from rsi_vla.runtime import prepare_runtime, execute_runtime
from rsi_vla.storage import ResearchStore, atomic_json, identifier
from rsi_vla.process_accounting import accounted_storage_bytes


def read(path):
    return json.loads(Path(path).read_text())


def gpu_admission(*, details=False, policy=None):
    """Read every physical card once; optionally report compatibility/load facts."""
    cards = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True).stdout
    jobs = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True).stdout
    memory, processes, missing = {}, [], set()
    for line in jobs.splitlines():
        if not line.strip():
            continue
        uuid, pid, value = [s.strip() for s in line.split(',')]
        if not value.isdigit():
            missing.add(uuid)
        else:
            memory[uuid] = memory.get(uuid, 0) + int(value)
        processes.append({'gpu_uuid': uuid, 'pid': int(pid), 'memory_mib': int(value) if value.isdigit() else None})
    eligible, rows = [], []
    detail_rows = {}
    if details or policy is not None:
        info = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_cap', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
        for line in info.splitlines():
            index, uuid, name, total, free, utilization, capability = [s.strip() for s in line.split(',')]
            detail_rows[int(index)] = {'uuid': uuid, 'name': name, 'total_memory_mib': int(total), 'free_memory_mib': int(free), 'utilization_percent': int(utilization) if utilization.isdigit() else None, 'compute_capability': capability}
    for line in cards.splitlines():
        index, uuid = [s.strip() for s in line.split(',')]
        used = memory.get(uuid, 0) if uuid not in missing else None
        rows.append({'physical_gpu': int(index), 'uuid': uuid, 'other_jobs_memory_mib': used, **detail_rows.get(int(index), {})})
        if used is not None and used <= 12288:
            eligible.append((used, int(index)))
    if policy is not None:
        from rsi_vla.task_policy import select_gpu
        selected = select_gpu(rows, policy)
    else:
        selected = min(eligible)[1] if eligible else None
    return {'cards': rows, 'processes': processes, 'selected': selected,
            'threshold_mib': 12288 if policy is None else None, 'policy': policy,
            'queries': 1, 'detail_queries': int(details or policy is not None), 'other_jobs_signalled': False}


def check_budget(plan, phase, *, remaining_slots, remaining_recoveries=0):
    if phase.get('unresolved_costs'):
        raise ValueError('phase actual costs unresolved')
    reserve = dict(plan.get('inheritance_reserve', plan['slot_envelope']))
    if comparison_only(plan):
        remaining_slots += remaining_recoveries
        for key, value in offline_remaining(plan, phase).items():
            reserve[key] += value
    missing = {k: v * remaining_slots + reserve[k] - phase['remaining'].get(k, 0)
               for k, v in plan['slot_envelope'].items()
               if v * remaining_slots + reserve[k] > phase['remaining'].get(k, 0)}
    if missing:
        raise ValueError('remaining complete assigned-slot envelopes plus inheritance reserve do not fit phase: ' + str(missing))
    if comparison_only(plan):
        spent = [a for s in plan['slots'] for a in s.get('attempts', []) if a.get('status') != 'unstarted']
        dynamic = {'simulation_episodes': len(spent)}
        for dimension, field in COST_FIELDS.items():
            values = [a.get('costs', {}).get(field) for a in spent]
            if not all(type(v) in (int, float) and v >= 0 for v in values):
                raise ValueError('attempt costs unresolved before next start')
            dynamic[dimension] = sum(values)
        if any(dynamic[k] + v * remaining_slots > plan['dynamic_envelope'][k]
               for k, v in plan['slot_envelope'].items()):
            raise ValueError('remaining attempts do not fit dynamic envelope')


def book_offline(plan, root, key, seconds):
    """Separate CPU evaluation and newly derived logical storage from raw runs."""
    phase_path = Path(plan['phase_snapshot_ref'])
    phase = read(phase_path)
    key = plan.get('campaign_id', root.name) + ':' + key
    entries = phase.setdefault('release_offline_charges', [])
    if not any(e['key'] == key for e in entries):
        entries.append({'key': key, 'costs': {'wall_seconds': seconds}, 'evidence_root': str(root)})
        phase['used']['wall_seconds'] += seconds
        phase['remaining']['wall_seconds'] = phase['limits']['wall_seconds'] - phase['used']['wall_seconds']
    # Raw outputs are already in their per-slot charge; new source and derived
    # evidence copies are billed here. Ledger snapshots remain metadata.
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()
               and p.relative_to(root).parts[0] not in {'outputs', 'phase-before.json', 'phase-after.json'})
    records = phase.setdefault('release_derived_storage_charges', [])
    previous = next((e for e in records if e['root'] == str(root)), None)
    booked = previous['storage_bytes'] if previous else 0
    delta = max(0, size-booked)
    if previous:
        previous['storage_bytes'] = max(booked, size)
    else:
        records.append({'root': str(root), 'storage_bytes': size})
    phase['used']['storage_bytes'] += delta
    phase['remaining']['storage_bytes'] = phase['limits']['storage_bytes'] - phase['used']['storage_bytes']
    atomic_json(phase_path, phase)
    if comparison_only(plan):
        offline_remaining(plan, phase)
    return {'offline_seconds': seconds, 'new_derived_bytes': delta}


def select_comparison_attempts(plan, root, protocol, profile):
    """Reinspect all attempts in launch order; retain original expense records."""
    reports = []
    plan['final_attempt_violations'] = []
    for slot in plan['slots']:
        chosen = None
        for attempt in slot.get('attempts', []):
            entry = {'task_key': slot['task_key'], 'seed': slot['seed'], 'run_id': attempt['run_id'],
                     'episode_id': attempt.get('episode_id'),
                     'result_path': str(Path(attempt['output']) / 'independent_result.json')}
            try:
                evaluated = evaluate_matrix(protocol, profile, [entry],
                            result_id=attempt['run_id'] + '-attempt-final', bundle_id=slot['bundle_id'])
                row = next(r for r in evaluated.assignments if r['task_key'] == slot['task_key'] and r['seed'] == slot['seed'])
                audited = inspect_attempt(attempt['output'], require_vla=False)
                progress = inspect_progress(attempt['output'], audited['identity'])
                valid = (audited['passed'] and progress['value'] is True
                         and row['validity'] == 'valid' and row['outcome'] in ('success', 'failure'))
                identity = {'run_id': attempt['run_id'], 'episode_id': attempt.get('episode_id'),
                            'bundle_id': slot['bundle_id'], 'protocol_id': protocol.get('protocol_id')}
                if (audited.get('checks', {}).get('identity') is True
                        and all(audited.get('identity', {}).get(k) == v for k, v in identity.items())):
                    confirmed = set(audited.get('confirmed_violations', []))
                    if progress['value'] is False:
                        confirmed.add('critical_progress')
                    for constraint in sorted(confirmed):
                        if evaluated.hard_constraints.get(constraint) is False:
                            plan['final_attempt_violations'].append({'side': slot['side'],
                                'run_id': attempt['run_id'], 'constraint': constraint, 'identity': identity})
                report = {'run_id': attempt['run_id'], 'valid': valid, 'assignment': row,
                          'technical_audit': audited, 'critical_progress': progress}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                valid = False
                report = {'run_id': attempt['run_id'], 'valid': False, 'error': str(exc)}
            reports.append(report)
            if valid and chosen is None:
                chosen = {**attempt, 'outcome': row['outcome'], 'validity': 'valid', 'status': 'completed'}
        slot['final_selected_attempt'] = chosen
    atomic_json(root / 'attempt-final-evaluation.json', {'attempts': reports,
                'selection': 'first technical-valid attempt in launch order, regardless of success',
                'charges_recomputed': False, 'confirmed_violations': plan['final_attempt_violations']})


def evaluate(plan, root, *, final=False):
    """Independent file evaluation only; no qualification or activation writer."""
    protocol, profile = read(plan['protocol']), read(plan['profile'])
    store = ResearchStore(Path(plan['artifacts']) / 'research')
    candidate = Candidate(**store.get('candidates', plan['candidate_id']))
    campaign_id = plan.get('campaign_id', root.name)
    version = identifier(plan.get('evaluation_version', 'read-default-sdk-closure-v2'))
    revision_root = root / 'evaluation-revisions' / version
    results = {}
    if comparison_only(plan):
        select_comparison_attempts(plan, revision_root, protocol, profile)
    for side, bundle in [('initial', candidate.base_bundle_id), ('candidate', candidate.bundle_id)]:
        manifest = []
        for slot in plan['slots']:
            if slot['side'] != side or slot['status'] == 'unstarted':
                continue
            selected = slot.get('final_selected_attempt') if comparison_only(plan) else slot
            if selected is None:
                continue
            manifest.append({'task_key': slot['task_key'], 'seed': slot['seed'],
                             'run_id': selected['run_id'], 'episode_id': selected['episode_id'],
                             'result_path': str(Path(selected['output']) / 'independent_result.json')})
        results[side] = evaluate_matrix(protocol, profile, manifest, result_id=campaign_id + '-' + version + '-' + side + '-matrix',
                                        bundle_id=bundle, candidate_id=candidate.candidate_id if side == 'candidate' else None)
        if comparison_only(plan):
            apply_comparison_attempt_evidence(plan, side, results[side])
    bundle = store.get('bundles', candidate.base_bundle_id)
    qualification = inspect_initial_result(protocol, bundle, results['initial'])
    atomic_json(revision_root / 'initial-qualification-check.json', qualification)
    initial = qualification
    if comparison_only(plan):
        initial = inspect_initial_comparison_result(protocol, bundle, results['initial'])
        atomic_json(revision_root / 'initial-comparison-check.json', initial)
    if final:
        evaluation_store = EvaluationStore(revision_root / 'evaluation')
        evaluation_store.freeze_protocol(plan['protocol'])
        for result in results.values():
            evaluation_store.save_result(result)
        decision = Gate(protocol).evaluate(campaign_id + '-' + version + '-independent-gate', candidate,
                    current_bundle_id=candidate.base_bundle_id, candidate_result=results['candidate'],
                    baseline_result=results['initial'], anchor_result=results['initial'])
        value = {'decision': asdict(decision), 'current_and_anchor_same_result_id': results['initial'].result_id,
                 'initial_qualification_check': qualification,
                 **({'initial_comparison_check': initial} if comparison_only(plan) else {}),
                 'qualification_established': False, 'OS_permission_isolation_accepted': False,
                 'activation_performed': False, 'inheritance_performed': False,
                 'baseline_J': score(protocol, results['initial']),
                 'positive_gain_impossible_at_baseline_one': score(protocol, results['initial']) == 1,
                 'efficiency_diagnostic_only': True, 'ten_percent_exception_enabled': False}
        atomic_json(revision_root / 'independent-gate.json', value)
        atomic_json(root / 'current-evaluation.json', {
            'version': version, 'gate_ref': str(revision_root / 'independent-gate.json'),
            'evaluation_store_ref': str(revision_root / 'evaluation'),
            'execution_source_manifest_ref': plan.get('source_manifest_ref'),
            'control_version': plan.get('control_version', 'receipt-resume-v1')})
    return initial


def execute_historical_matrix(plan_path):
    plan_path = Path(plan_path).resolve()
    root, plan = plan_path.parent, read(plan_path)
    if (root / 'execution-record.json').exists() or any(s['status'] != 'unstarted' for s in plan['slots']):
        raise ValueError('this matrix already entered execution; do not restore or retry')
    tasks = ['base_salad', 'position_swap_salad', 'old_milk', 'old_drawer']
    if ([s['side'] for s in plan['slots']] != ['initial'] * 4 + ['candidate'] * 4
            or [s['task_key'] for s in plan['slots']] != tasks * 2
            or any(s['seed'] != 3 for s in plan['slots'])
            or len({s['run_id'] for s in plan['slots']}) != 8
            or len({str(Path(s['output']).resolve()) for s in plan['slots']}) != 8):
        raise ValueError('matrix must be four fixed tasks initial then candidate, seed3, with fresh distinct runs/outputs')
    check_budget(plan, read(plan['phase_snapshot_ref']), remaining_slots=8)
    atomic_json(root / 'execution-record.json', {'status': 'running', 'runtime_start_attempts': 0, 'automatic_retries': 0})
    stop = None
    starts = 0
    root_repo = Path(__file__).resolve().parents[1]
    for number, slot in enumerate(plan['slots'], 1):
        try:
            phase = read(plan['phase_snapshot_ref'])
            check_budget(plan, phase, remaining_slots=9-number)
            admission = gpu_admission()
            slot_root = root / 'slots' / slot['run_id']
            atomic_json(slot_root / 'gpu-admission.json', admission)
            if admission['selected'] is None:
                stop = 'no authorized GPU has other jobs memory <=12288MiB'
                break
            gpu = admission['selected']
            kwargs = dict(protocol_path=plan['protocol'], run_id=slot['run_id'], bundle_id=slot['bundle_id'],
                          task_key=slot['task_key'], seed=3, split='release', output_dir=slot['output'], cuda_device=gpu)
            draft = prepare_runtime(root_repo, plan['artifacts'], **kwargs)
            permit = {'authorized': True, 'run_id': slot['run_id'], 'bundle_id': slot['bundle_id'],
                      'protocol_id': draft['context']['protocol_id'], 'task_key': slot['task_key'], 'seed': 3,
                      'split': 'release', 'simulation_episodes': 1, 'gpu_devices': [gpu],
                      'limits': draft['context']['limits'], 'collection': draft['context']['collection'],
                      'output_dir': slot['output'], 'code_ref': draft['context']['code_ref'],
                      'source_manifest_ref': plan['source_manifest_ref'], 'phase_snapshot_ref': plan['phase_snapshot_ref'],
                      'profile_ref': plan['profile'], 'profile': read(plan['profile']), 'protocol_ref': plan['protocol'],
                      'candidate_id': plan['candidate_id'], 'source_candidate_id': plan['source_candidate_id'],
                      'task_input': draft['context'].get('task_input'), 'model_and_processor': {k: draft['context'][k] for k in ('theta_ref', 'processor_ref', 'memory_ref')},
                      'round_authorization_ref': str(root / 'authorization.json'),
                      'external_destination': 'https://chatgpt.com/backend-api/codex/responses',
                      'Planner': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium'},
                      'gpu_admission_ref': str(slot_root / 'gpu-admission.json'),
                      'max_runtime_start_attempts': 1, 'automatic_retries': 0,
                      'planner_timeout_seconds': 600, 'planner_max_turns': 20,
                      'local_recovery_limit': 1, 'reset_retries': 0, 'token_policy': 'observed_only',
                      'runtime_rules_ref': draft['context'].get('runtime_rules_ref')}
            permit_path = slot_root / 'permit.json'
            atomic_json(permit_path, permit)
            prepared = prepare_runtime(root_repo, plan['artifacts'], **kwargs, authorization_path=permit_path)
            atomic_json(slot_root / 'dry-run.json', prepared)
            if prepared['blockers']:
                stop = 'dry-run blocked: ' + str(prepared['blockers'])
                break
            if Path(slot['output']).exists():
                raise ValueError('fresh release output already exists')
            starts += 1
            slot.update(status='started', runtime_start_attempts=1, physical_gpu=gpu, permit_ref=str(permit_path))
            atomic_json(plan_path, plan)
            atomic_json(root / 'execution-record.json', {'status': 'running', 'runtime_start_attempts': starts, 'automatic_retries': 0})
            print(json.dumps({'slot': number, 'run_id': slot['run_id'], 'status': 'starting', 'GPU': gpu}), flush=True)
            launcher_error = None
            attempt_started = time.perf_counter()
            try:
                execute_runtime(prepared)
            except Exception as exc:
                launcher_error = type(exc).__name__ + ': ' + str(exc)
                if Path(slot['output']).is_dir():
                    atomic_json(Path(slot['output']) / 'launcher_failure.json', {'identity': {k: prepared['context'][k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}, 'error': launcher_error})
            attempt_wall = time.perf_counter() - attempt_started
            cpu_started = time.perf_counter()
            try:
                audited = inspect_attempt(slot['output'], require_vla=False)
            except (OSError, ValueError, KeyError) as exc:
                audited = {'identity': {k: prepared['context'][k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}, 'passed': False, 'outcome': 'unknown', 'checks': {}, 'costs': {'whole_job_wall_seconds': attempt_wall, 'gpu_device_seconds_envelope': attempt_wall}, 'notes': ['incomplete startup evidence: ' + str(exc)]}
            if Path(slot['output']).is_dir():
                atomic_json(Path(slot['output']) / 'independent-audit.json', audited)
            identity = audited['identity']
            progress = inspect_progress(slot['output'], identity)
            atomic_json(slot_root / 'critical-progress.json', progress)
            costs = dict(audited['costs'])
            for field in ('whole_job_wall_seconds', 'gpu_device_seconds_envelope'):
                if type(costs.get(field)) not in (int, float):
                    costs[field] = attempt_wall
                    audited.setdefault('notes', []).append(
                        field + ' retained from measured single-device launcher invocation; other missing dimensions remain unresolved')
            audited['costs'] = costs
            costs['artifact_bytes'] = sum(p.stat().st_size for p in Path(slot['output']).rglob('*') if p.is_file() and not p.is_symlink())
            _phase_charge({**permit, 'permit_ref': str(permit_path)}, slot['run_id'], costs, slot['output'], charge_kind='release_trial_charges')
            slot.update(status='completed' if not launcher_error and audited['passed'] else 'technical_stop',
                        outcome=audited['outcome'], validity='valid' if audited['passed'] and not launcher_error else 'invalid',
                        critical_progress=progress['value'], costs=costs, launcher_error=launcher_error)
            # Keep all dynamically spent dimensions within both caps. The eight
            # complete maxima were admitted before slot 1; no retries are added.
            actual = read(plan['phase_snapshot_ref'])
            phase_before = read(root / 'phase-before.json')
            delta = {k: actual['used'][k]-phase_before['used'][k] for k in plan['dynamic_envelope']}
            fields = {'wall_seconds': 'whole_job_wall_seconds', 'gpu_seconds': 'gpu_device_seconds_envelope',
                      'model_requests': 'model_requests', 'tool_calls': 'tool_calls',
                      'environment_steps': 'environment_steps', 'storage_bytes': 'artifact_bytes'}
            spent = [s['costs'] for s in plan['slots'] if s.get('runtime_start_attempts')]
            dynamic = {'simulation_episodes': starts}
            for dimension, field in fields.items():
                values = [cost.get(field) for cost in spent]
                dynamic[dimension] = sum(values) if all(type(v) in (int, float) for v in values) else None
            breached = [k for k, v in dynamic.items()
                        if v is not None and v > plan['dynamic_envelope'][k]]
            atomic_json(slot_root / 'slot-report.json', {'slot': slot, 'technical_audit': audited, 'critical_progress': progress,
                        'cumulative_delta_including_offline': delta, 'cumulative_dynamic_only': dynamic, 'envelope_breaches': breached})
            atomic_json(plan_path, plan)
            if launcher_error or not audited['passed'] or progress['value'] is not True or (Path(slot['output']) / 'budget_stop.json').exists() or breached or actual.get('unresolved_costs'):
                stop = 'technical/error/missing-evidence/budget stop at ' + slot['run_id']
            if number == 4 and not stop:
                initial = evaluate(plan, root)
                if not initial['passed']:
                    stop = 'initial four-slot qualification check failed: ' + initial['reason']
            book_offline(plan, root, slot['run_id'] + '-independent-audit', time.perf_counter()-cpu_started)
            print(json.dumps({'slot': number, 'run_id': slot['run_id'], 'outcome': slot['outcome'], 'validity': slot['validity'], 'progress': progress['value'], 'stop': stop}), flush=True)
            if stop:
                break
        except Exception as exc:
            stop = type(exc).__name__ + ': ' + str(exc)
            break
    plan.update(status='stopped_dynamic_paused' if stop else 'completed_dynamic_paused', stop_reason=stop)
    atomic_json(plan_path, plan)
    started = time.perf_counter()
    evaluate(plan, root, final=True)
    book_offline(plan, root, 'final-independent-matrix-evaluation', time.perf_counter()-started)
    atomic_json(root / 'phase-after.json', read(plan['phase_snapshot_ref']))
    atomic_json(root / 'execution-record.json', {'status': plan['status'], 'runtime_start_attempts': starts,
                'automatic_retries': 0, 'stop_reason': stop, 'dynamic_after_round': 'paused'})
    print(json.dumps({'status': plan['status'], 'starts': starts, 'stop': stop, 'gate_ref': str(root / 'independent-gate.json')}), flush=True)


RELEASE08_SCOPE = 'repair05-fresh-release08-eight-slots-two-recoveries-only'
COST_FIELDS = {'wall_seconds': 'whole_job_wall_seconds', 'gpu_seconds': 'gpu_device_seconds_envelope',
               'model_requests': 'model_requests', 'tool_calls': 'tool_calls',
               'environment_steps': 'environment_steps', 'storage_bytes': 'artifact_bytes'}


def comparison_only(plan):
    """New comparison behavior is explicitly scoped; old plans retain their path."""
    return plan.get('comparison_only') is True


def cleaned_temporary_exception(plan, phase):
    """Apply only the authorized, charged and removed historical baseline."""
    reference = plan.get('budget_exception_ref')
    if not reference:
        return 0
    exception = read(reference)
    flags = ('authorized', 'all_costs_retained', 'temporary_baseline_removed',
             'dynamic_continuation_authorized')
    charges = [entry for entry in phase.get('derived_storage_charges', [])
               if entry.get('key') == 'release08-temporary-historical-baseline']
    if (not comparison_only(plan) or exception.get('authorization_scope') != RELEASE08_SCOPE
            or exception.get('exception_kind') != 'cleaned_temporary_historical_baseline'
            or exception.get('actual_bytes') != 756257057
            or exception.get('phase_snapshot_ref') != plan['phase_snapshot_ref']
            or exception.get('phase_charge_key') != 'release08-temporary-historical-baseline'
            or len(charges) != 1 or charges[0].get('storage_bytes') != 756257057
            or any(exception.get(flag) is not True for flag in flags)
            or plan.get('offline_actual_storage_bytes', 0) < 756257057
            or phase['used']['storage_bytes'] < 756257057):
        raise ValueError('invalid cleaned temporary baseline budget exception')
    return 756257057


def offline_remaining(plan, phase):
    reserve = plan['offline_reserve']
    prefix = plan.get('campaign_id', Path(plan['phase_snapshot_ref']).parent.name) + ':'
    wall = sum(e['costs']['wall_seconds'] for e in phase.get('release_offline_charges', [])
               if e['key'].startswith(prefix))
    storage = sum(e['storage_bytes'] for e in phase.get('release_derived_storage_charges', [])
                  if e['root'] == str(Path(plan['phase_snapshot_ref']).resolve().parent))
    # The evidence root may differ from the shared phase ledger directory.
    if plan.get('evidence_root'):
        storage = sum(e['storage_bytes'] for e in phase.get('release_derived_storage_charges', [])
                      if e['root'] == plan['evidence_root'])
    wall = max(wall, plan.get('offline_actual_wall_seconds', 0))
    # External charges exclude the driver-owned derived-storage record.
    storage += plan.get('offline_actual_storage_bytes', 0)
    storage -= cleaned_temporary_exception(plan, phase)
    if wall > reserve['wall_seconds'] or storage > reserve['storage_bytes']:
        raise ValueError('release comparison offline reserve exceeded')
    return {'wall_seconds': reserve['wall_seconds'] - wall,
            'storage_bytes': reserve['storage_bytes'] - storage}


def release08_gpu_admission(plan, root, key):
    """Wait at most 300 cumulative seconds, charging every wait as offline wall."""
    phase = read(plan['phase_snapshot_ref'])
    prefix = plan.get('campaign_id', root.name) + ':gpu-wait-'
    waited = sum(e['costs']['wall_seconds'] for e in phase.get('release_offline_charges', [])
                 if e['key'].startswith(prefix))
    checks = []
    while True:
        started = time.perf_counter()
        admission = gpu_admission()
        checks.append(admission)
        if admission['selected'] is not None or waited >= 300:
            query_wall = time.perf_counter() - started
            book_offline(plan, root, 'gpu-wait-' + key + '-' + str(len(checks)), query_wall)
            return {**admission, 'checks': checks, 'cumulative_wait_seconds': waited}
        phase = read(plan['phase_snapshot_ref'])
        available = offline_remaining(plan, phase)['wall_seconds']
        delay = min(30, 300 - waited, max(0, available - 1))
        if delay <= 0:
            return {**admission, 'checks': checks, 'cumulative_wait_seconds': waited}
        time.sleep(delay)
        elapsed = time.perf_counter() - started
        waited += elapsed
        book_offline(plan, root, 'gpu-wait-' + key + '-' + str(len(checks)), elapsed)


def recovery_reason(attempt, audited, output):
    """Recover only a closed, fully accounted local technical failure."""
    if audited.get('confirmed_violations') or (Path(output) / 'budget_stop.json').exists():
        return 'confirmed constraint or budget stop'
    costs = attempt['costs']
    if any(type(costs.get(field)) not in (int, float) or costs[field] < 0
           for field in COST_FIELDS.values()):
        return 'actual costs unresolved'
    checks = audited.get('checks', {})
    shared = ('identity', 'runtime_source_snapshot', 'startup_input_identity',
              'actual_model_configuration', 'model_reference', 'vla_actual_loading_and_prediction')
    if any(checks.get(key) is not True for key in shared):
        return 'shared identity/source/configuration/model fault'
    error = (attempt.get('launcher_error') or '').lower()
    if any(word in error for word in ('modulenotfound', 'importerror', 'permissionerror', 'no such file',
                                      'source manifest', 'configuration', 'checkpoint')):
        return 'shared launch dependency or configuration fault'
    try:
        identity = audited['identity']
        supervisor = read(Path(output) / 'supervisor.json')
        gate = read(Path(output) / 'request_gate.json')
        for record in (supervisor, gate):
            bound = record.get('identity', record)
            if any(bound.get(k) != v for k, v in identity.items()):
                return 'owned cleanup identity unconfirmed'
        if supervisor.get('cleanup_confirmed') is not True or gate.get('cleanup_complete') is not True or gate.get('active_requests_at_close') != 0:
            return 'owned processes or requests not confirmed closed'
    except (OSError, ValueError, KeyError, TypeError):
        return 'owned cleanup evidence unavailable'
    local_checks = ('execution_health', 'tool_l0_coverage', 'clean_completion',
                    'launcher_wrapper_completion', 'independent_result')
    prerequisites = ('model_transport', 'tool_budget', 'request_budget', 'step_budget',
                     'wall_gpu_storage_budget', 'all_tools_use_counted_dispatcher')
    if not all(checks.get(key) is True for key in prerequisites):
        return 'transport or resource fault not proven local'
    if not any(checks.get(key) is False for key in local_checks):
        return 'technical failure has no independently localized cause'
    return None


def release08_attempt(plan, root, slot, attempt, gpu, before_launch, checkpoint):
    """Launch one fresh permit and preserve independent audit plus actual charge."""
    root_repo = Path(__file__).resolve().parents[1]
    slot_root = root / 'slots' / attempt['run_id']
    kwargs = dict(protocol_path=plan['protocol'], run_id=attempt['run_id'], bundle_id=slot['bundle_id'],
                  task_key=slot['task_key'], seed=3, split='release', output_dir=attempt['output'], cuda_device=gpu)
    if Path(attempt['output']).exists() or (slot_root / 'permit.json').exists():
        raise ValueError('fresh release attempt output or permit already exists')
    draft = prepare_runtime(root_repo, plan['artifacts'], **kwargs)
    permit = {'authorized': True, 'run_id': attempt['run_id'], 'bundle_id': slot['bundle_id'],
              'protocol_id': draft['context']['protocol_id'], 'task_key': slot['task_key'], 'seed': 3,
              'split': 'release', 'simulation_episodes': 1, 'gpu_devices': [gpu],
              'limits': draft['context']['limits'], 'collection': draft['context']['collection'],
              'output_dir': attempt['output'], 'code_ref': draft['context']['code_ref'],
              'source_manifest_ref': plan['source_manifest_ref'], 'phase_snapshot_ref': plan['phase_snapshot_ref'],
              'profile_ref': plan['profile'], 'profile': read(plan['profile']), 'protocol_ref': plan['protocol'],
              'candidate_id': plan['candidate_id'], 'source_candidate_id': plan['source_candidate_id'],
              'task_input': draft['context'].get('task_input'),
              'model_and_processor': {k: draft['context'][k] for k in ('theta_ref', 'processor_ref', 'memory_ref')},
              'round_authorization_ref': str(root / 'authorization.json'),
              'external_destination': 'https://chatgpt.com/backend-api/codex/responses',
              'Planner': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium'},
              'gpu_admission_ref': str(slot_root / 'gpu-admission.json'),
              'max_runtime_start_attempts': 1, 'automatic_retries': 0,
              'planner_timeout_seconds': 600, 'planner_max_turns': 20, 'local_recovery_limit': 1,
              'reset_retries': 0, 'token_policy': 'observed_only',
              'runtime_rules_ref': draft['context'].get('runtime_rules_ref'),
              'comparison_only': True, 'recovery_of_run_id': slot['run_id'] if slot.get('attempts') and len(slot['attempts']) > 1 else None,
              'formal_slot_task_key': slot['task_key']}
    permit_path = slot_root / 'permit.json'
    atomic_json(permit_path, permit)
    prepared = prepare_runtime(root_repo, plan['artifacts'], **kwargs, authorization_path=permit_path)
    atomic_json(slot_root / 'dry-run.json', prepared)
    if prepared['blockers']:
        raise ValueError('dry-run blocked: ' + str(prepared['blockers']))
    attempt.update(status='started', physical_gpu=gpu, permit_ref=str(permit_path),
                   episode_id=prepared['context']['episode_id'])
    before_launch()
    started = time.perf_counter()
    launcher_error = None
    try:
        execute_runtime(prepared)
    except Exception as exc:
        launcher_error = type(exc).__name__ + ': ' + str(exc)
        if Path(attempt['output']).is_dir():
            atomic_json(Path(attempt['output']) / 'launcher_failure.json',
                        {'identity': {k: prepared['context'][k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')},
                         'error': launcher_error})
    wall = time.perf_counter() - started
    receipt = {'identity': {k: prepared['context'][k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')},
               'wall_seconds': wall, 'launcher_error': launcher_error, 'output': attempt['output']}
    atomic_json(slot_root / 'runtime-returned.json', receipt)
    attempt['status'] = 'runtime_returned'
    checkpoint()
    return finish_release08_attempt(plan, root, slot, attempt, checkpoint)


def _finish_release08_attempt(plan, root, slot, attempt, checkpoint):
    # Reconcile a returned execution without submitting it a second time.
    slot_root = root / 'slots' / attempt['run_id']
    receipt = read(slot_root / 'runtime-returned.json')
    permit_path = Path(attempt['permit_ref'])
    permit = read(permit_path)
    identity = {'run_id': attempt['run_id'], 'episode_id': attempt['episode_id'],
                'bundle_id': slot['bundle_id'], 'protocol_id': permit['protocol_id']}
    if (receipt.get('identity') != identity or receipt.get('output') != attempt['output']
            or any(permit.get(k) != v for k, v in identity.items() if k != 'episode_id')
            or permit.get('output_dir') != attempt['output']):
        raise ValueError('runtime receipt identity differs from checkpoint/permit')
    wall, launcher_error = receipt['wall_seconds'], receipt['launcher_error']
    try:
        audited = inspect_attempt(attempt['output'], require_vla=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audited = {'identity': identity, 'passed': False, 'outcome': 'unknown',
                   'checks': {}, 'costs': {}, 'notes': [str(exc)]}
    costs = {k: v for k, v in audited['costs'].items() if v is not None}
    for field in ('whole_job_wall_seconds', 'gpu_device_seconds_envelope'):
        if type(costs.get(field)) not in (int, float):
            costs[field] = wall
    audited['costs'] = costs
    costs['artifact_bytes'] = max(costs.get('artifact_bytes', 0), accounted_storage_bytes(attempt['output']))
    # Account execution before evaluating progress: evaluator failure must not
    # erase a returned job or its actual expense. Reuse the exact charge on resume.
    phase = read(plan['phase_snapshot_ref'])
    booked = [entry for entry in phase.get('release_trial_charges', []) if entry['run_id'] == attempt['run_id']]
    if booked:
        if len(booked) != 1 or booked[0]['permit_ref'] != str(permit_path) or booked[0]['output_ref'] != attempt['output']:
            raise ValueError('existing trial charge differs from receipt/permit')
        attempt['costs'] = {field: booked[0]['costs'][dimension] for dimension, field in COST_FIELDS.items()
                            if dimension in booked[0]['costs']}
    if 'costs' not in attempt:
        if Path(attempt['output']).is_dir():
            atomic_json(Path(attempt['output']) / 'independent-audit.json', audited)
        costs['artifact_bytes'] = max(costs.get('artifact_bytes', 0), accounted_storage_bytes(attempt['output']))
        attempt['costs'] = costs
    costs = attempt['costs']
    _phase_charge({**permit, 'permit_ref': str(permit_path)}, attempt['run_id'], costs,
                  attempt['output'], charge_kind='release_trial_charges')
    attempt['status'] = 'evaluation_paused'
    checkpoint()
    progress = inspect_progress(attempt['output'], audited['identity'])
    if audited['passed'] and progress['value'] is None:
        raise ValueError('critical-progress evaluation incomplete; execution retained')
    valid = (not launcher_error and audited['passed'] and progress['value'] is True
             and audited['outcome'] in ('success', 'failure') and not audited.get('confirmed_violations')
             and not (Path(attempt['output']) / 'budget_stop.json').exists())
    attempt.update(status='completed' if valid else 'technical_stop', outcome=audited['outcome'],
                   validity='valid' if valid else 'invalid', critical_progress=progress['value'],
                   launcher_error=launcher_error)
    refusal = recovery_reason(attempt, audited, attempt['output']) if not valid else None
    attempt['fault_scope'] = None if valid else 'localized' if refusal is None else 'unresolved_or_shared'
    atomic_json(slot_root / 'recovery-eligibility.json', {'eligible': not valid and refusal is None,
                'fault_scope': attempt['fault_scope'], 'refusal': refusal,
                'diagnosed_failed_checks': [k for k, v in audited.get('checks', {}).items() if v is False],
                'validated_checks': [k for k, v in audited.get('checks', {}).items() if v is True],
                'cleanup_validated': not valid and refusal is None,
                'costs_complete': all(type(costs.get(f)) in (int, float) for f in COST_FIELDS.values())})
    atomic_json(slot_root / 'slot-report.json', {'attempt': attempt, 'technical_audit': audited,
                'critical_progress': progress, 'recovery_refusal': refusal})
    return valid, refusal


def finish_release08_attempt(plan, root, slot, attempt, checkpoint):
    attempt['evaluation_attempts'] = attempt.get('evaluation_attempts', 0) + 1
    checkpoint()
    started = time.perf_counter()
    try:
        return _finish_release08_attempt(plan, root, slot, attempt, checkpoint)
    finally:
        book_offline(plan, root, attempt['run_id'] + '-audit-' + str(attempt['evaluation_attempts']),
                     time.perf_counter() - started)


def evaluate_checkpoint(plan, root, checkpoint, *, final=False):
    plan['matrix_evaluation_attempts'] = plan.get('matrix_evaluation_attempts', 0) + 1
    checkpoint()
    started = time.perf_counter()
    try:
        return evaluate(plan, root, final=final)
    finally:
        book_offline(plan, root, 'matrix-evaluation-' + str(plan['matrix_evaluation_attempts']),
                     time.perf_counter() - started)


def select_completed_attempt(slot, attempt):
    slot['selected_attempt'] = dict(attempt)
    slot.update({k: attempt[k] for k in ('run_id', 'output', 'episode_id', 'status', 'outcome',
                                       'validity', 'critical_progress', 'costs', 'permit_ref')})
    slot['runtime_start_attempts'] = len(slot['attempts'])


def execute_release08(plan_path, *, resume=False):
    plan_path = Path(plan_path).resolve()
    root, plan = plan_path.parent, read(plan_path)
    phase = read(plan['phase_snapshot_ref'])
    exception_bytes = cleaned_temporary_exception(plan, phase)
    record_path = root / 'execution-record.json'
    record = read(record_path) if record_path.exists() else None
    zero_start = (record is not None and type(record.get('runtime_start_attempts')) is int
                  and record['runtime_start_attempts'] == 0)
    if (plan.get('status') == 'stopped_dynamic_paused'
            or (root / 'offline-budget-stop.json').exists()):
        if not (exception_bytes and zero_start):
            raise ValueError('stopped release campaign cannot restore or retry')
    protocol = read(plan['protocol'])
    if (protocol.get('comparison_only') is not True
            or protocol.get('authorization_scope') != RELEASE08_SCOPE
            or protocol.get('protocol_id') != 'rsi-libero-pro-5a-release08-four-seed3'
            or protocol.get('anchor_bundle_id') != 'initial-medium-5a-release08'
            or read(plan['profile']).get('profile_id') != 'matched-5a-release08-20261113'):
        raise ValueError('comparison-only plan requires release08 scoped protocol and identity')
    resuming = resume and record is not None and plan.get('status') in ('evaluation_paused', 'running')
    if resume and not resuming:
        raise ValueError('only an open evaluation checkpoint can resume; closed campaigns cannot restore or retry')
    if not resuming and (record is not None and not (exception_bytes and zero_start)
            or any(s['status'] != 'unstarted' or any(a.get('status') != 'unstarted' for a in s.get('attempts', []))
                   for s in plan['slots'])):
        raise ValueError('this matrix already entered execution; do not restore or retry')
    tasks = ['base_salad', 'position_swap_salad', 'old_milk', 'old_drawer']
    if ([s['side'] for s in plan['slots']] != ['initial'] * 4 + ['candidate'] * 4
            or [s['task_key'] for s in plan['slots']] != tasks * 2
            or any(s['seed'] != 3 for s in plan['slots'])
            or len({s['run_id'] for s in plan['slots']}) != 8
            or len({str(Path(s['output']).resolve()) for s in plan['slots']}) != 8):
        raise ValueError('matrix must be eight fixed tasks, seed3, with fresh distinct runs/outputs')
    expected = {'simulation_episodes': 10, 'model_requests': 200, 'tool_calls': 600,
                'environment_steps': 20000, 'wall_seconds': 9000, 'gpu_seconds': 9000,
                'storage_bytes': 10737418240}
    slot_expected = {key: value / 10 for key, value in expected.items()}
    if (plan['dynamic_envelope'] != expected or plan['slot_envelope'] != slot_expected
            or plan.get('inheritance_reserve') != slot_expected
            or plan.get('offline_reserve') != {'wall_seconds': 600, 'storage_bytes': 268435456}
            or plan.get('max_runtime_start_attempts') != 10
            or plan.get('max_technical_recoveries') != 2 or plan.get('max_recoveries_per_slot') != 1):
        raise ValueError('release08 requires fixed ten-attempt caps, two recoveries, inheritance and offline reserves')
    if any(s['bundle_id'] != protocol['anchor_bundle_id'] for s in plan['slots'][:4]):
        raise ValueError('initial formal slots must use release08 anchor bundle')
    plan['evidence_root'] = str(root)
    plan.setdefault('evaluation_version', 'read-default-sdk-closure-v2')
    plan.setdefault('control_version', 'receipt-resume-v1')
    attempts = [a for slot in plan['slots'] for a in slot.get('attempts', [])]
    starts = sum(a.get('status') != 'unstarted' for a in attempts)
    recoveries = sum(max(0, len(slot.get('attempts', [])) - 1) for slot in plan['slots'])
    if resuming and (record.get('runtime_start_attempts') != starts
                     or record.get('technical_recoveries') != recoveries):
        raise ValueError('checkpoint counters disagree; submission status unknown')
    for attempt in attempts:
        if attempt.get('status') == 'started' and (root / 'slots' / attempt['run_id'] / 'runtime-returned.json').is_file():
            attempt['status'] = 'runtime_returned'
        elif attempt.get('status') in ('started', 'unstarted'):
            raise ValueError('submission status unknown; inspect receipt before resuming')
    if not resuming:
        check_budget(plan, read(plan['phase_snapshot_ref']), remaining_slots=8, remaining_recoveries=2)
    stop = plan.get('dynamic_stop_reason')
    plan['status'] = 'running'
    def persist():
        atomic_json(plan_path, plan)
        atomic_json(root / 'execution-record.json', {'status': plan['status'], 'runtime_start_attempts': starts,
                    'technical_recoveries': recoveries, 'automatic_retries': 0})
    persist()
    def pause_evaluation(exc):
        plan.update(status='evaluation_paused', stop_reason=type(exc).__name__ + ': ' + str(exc))
        persist()

    for number, slot in enumerate([] if plan.get('final_evaluation_pending') else plan['slots']):
        if number == 4 and not plan.get('initial_comparison_evaluated'):
            try:
                initial = evaluate_checkpoint(plan, root, persist)
                if not initial['passed'] and not any(s.get('localized_failure_retained') for s in plan['slots'][:4]):
                    raise ValueError('initial technical comparison check incomplete: ' + initial['reason'])
            except Exception as exc:
                pause_evaluation(exc)
                return
            plan['initial_comparison_evaluated'] = True
            persist()
        slot.setdefault('formal_run_id', slot['run_id'])
        slot.setdefault('formal_output', slot['output'])
        slot.setdefault('attempts', [])
        if slot.get('selected_attempt') or slot.get('localized_failure_retained'):
            continue
        pending = slot['attempts'][-1] if slot['attempts'] else None
        if pending and pending['status'] == 'completed' and pending.get('validity') == 'valid':
            select_completed_attempt(slot, pending)
            persist()
            continue
        if pending and pending['status'] == 'technical_stop':
            if pending.get('fault_scope') != 'localized':
                stop = 'technical stop: prior attempt has unresolved or shared fault'
                break
            if len(slot['attempts']) >= 2 or recoveries >= 2:
                slot['localized_failure_retained'] = True
                persist()
                continue
        first = len(slot['attempts']) - 1 if pending and pending['status'] in ('runtime_returned', 'evaluation_paused') else len(slot['attempts'])
        for local_attempt in range(first, 2):
            try:
                reconciling = pending is not None and pending['status'] in ('runtime_returned', 'evaluation_paused')
                if not reconciling and (starts >= 10 or local_attempt and recoveries >= 2):
                    raise ValueError('technical recovery or total start limit reached')
                if reconciling:
                    attempt = pending
                    valid, refusal = finish_release08_attempt(plan, root, slot, attempt, persist)
                    pending = None
                else:
                    remaining_slots = 8 - number - (1 if local_attempt else 0)
                    check_budget(plan, read(plan['phase_snapshot_ref']), remaining_slots=remaining_slots,
                                 remaining_recoveries=2 - recoveries)
                    run_id = slot['formal_run_id'] if local_attempt == 0 else slot['formal_run_id'] + '-recovery1'
                    output = slot['formal_output'] if local_attempt == 0 else slot['formal_output'] + '-recovery1'
                    if local_attempt and any(run_id == s['run_id'] or Path(output).resolve() == Path(s['output']).resolve() for s in plan['slots']):
                        raise ValueError('recovery identity collides with a formal slot')
                    if any(run_id == a['run_id'] or Path(output).resolve() == Path(a['output']).resolve()
                           for s in plan['slots'] for a in s.get('attempts', [])):
                        raise ValueError('attempt identity reused')
                    admission = release08_gpu_admission(plan, root, run_id)
                    atomic_json(root / 'slots' / run_id / 'gpu-admission.json', admission)
                    if admission['selected'] is None:
                        raise ValueError('no authorized GPU after bounded cumulative wait')
                    check_budget(plan, read(plan['phase_snapshot_ref']), remaining_slots=remaining_slots,
                                 remaining_recoveries=2 - recoveries)
                    attempt = {'run_id': run_id, 'output': output, 'status': 'unstarted'}
                    slot['attempts'].append(attempt)
                    # Start identity and authorization are persisted before launch.
                    def before_launch():
                        nonlocal starts, recoveries
                        starts += 1
                        recoveries += local_attempt
                        slot['status'] = 'started'
                        persist()
                    valid, refusal = release08_attempt(plan, root, slot, attempt, admission['selected'], before_launch, persist)
                persist()
                phase = read(plan['phase_snapshot_ref'])
                spent = [a['costs'] for s in plan['slots'] for a in s.get('attempts', []) if 'costs' in a]
                dynamic = {'simulation_episodes': starts}
                for dimension, field in COST_FIELDS.items():
                    values = [a.get(field) for a in spent]
                    dynamic[dimension] = sum(values) if all(type(v) in (int, float) for v in values) else None
                atomic_json(root / 'dynamic-costs.json', {'attempts': dynamic})
                if phase.get('unresolved_costs') or any(v is not None and v > expected[k] for k, v in dynamic.items()):
                    raise ValueError('actual cost unresolved or dynamic envelope exceeded')
                if valid:
                    select_completed_attempt(slot, attempt)
                    break
                slot.update(status='technical_stop', outcome='unknown', validity='invalid')
                if refusal:
                    raise ValueError('technical stop: ' + refusal)
                if local_attempt or recoveries >= 2:
                    slot['localized_failure_retained'] = True
                    break
            except Exception as exc:
                if slot['attempts'] and slot['attempts'][-1]['status'] in ('runtime_returned', 'evaluation_paused'):
                    pause_evaluation(exc)
                    return
                stop = type(exc).__name__ + ': ' + str(exc)
                break
        persist()
        if stop:
            break
    plan.update(status='running', stop_reason=stop, dynamic_stop_reason=stop, final_evaluation_pending=True)
    atomic_json(plan_path, plan)
    try:
        evaluate_checkpoint(plan, root, persist, final=True)
    except Exception as exc:
        pause_evaluation(exc)
        return
    plan.update(status='stopped_dynamic_paused' if stop else 'completed_dynamic_paused',
                stop_reason=stop, final_evaluation_pending=False)
    atomic_json(plan_path, plan)
    atomic_json(root / 'phase-after.json', read(plan['phase_snapshot_ref']))
    atomic_json(root / 'execution-record.json', {'status': plan['status'], 'runtime_start_attempts': starts,
                'technical_recoveries': recoveries, 'automatic_retries': 0, 'stop_reason': stop,
                'dynamic_after_round': 'paused'})


def _execute_matrix(plan_path, *, resume=False):
    plan = read(plan_path)
    protocol = read(plan['protocol']) if comparison_only(plan) or str(plan.get('protocol', '')).endswith('.json') else {}
    if protocol.get('authorization_scope') == RELEASE08_SCOPE or protocol.get('protocol_id') == 'rsi-libero-pro-5a-release08-four-seed3':
        if not comparison_only(plan):
            raise ValueError('release08 protocol requires comparison-only plan')
        return execute_release08(plan_path, resume=resume)
    if comparison_only(plan):
        return execute_release08(plan_path, resume=resume)
    if resume:
        raise ValueError("historical matrix does not support resume")
    return execute_historical_matrix(plan_path)


def execute_matrix(plan_path, *, resume=False):
    # Serialize checkpoint consumption so concurrent continue commands cannot launch twice.
    with Path(plan_path).with_suffix('.execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('matrix execution already active') from exc
        return _execute_matrix(plan_path, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue an open evaluation checkpoint without replaying completed attempts')
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.execute:
        execute_matrix(args.plan, resume=args.resume)
    else:
        p = read(args.plan)
        check_budget(p, read(p['phase_snapshot_ref']), remaining_slots=8, remaining_recoveries=2 if comparison_only(p) else 0)
        print(json.dumps({'mode': 'dry-run', 'dispatch': False, 'slots': len(p['slots']), 'phase_budget_fits': True}))


if __name__ == '__main__':
    main()
