"""Versioned, read-only Runtime reparse and deterministic development screening."""

import time
from dataclasses import asdict, replace
from pathlib import Path

from .critical_progress import AUDIT_VERSION
from .research_cycle import _collect, read
from .storage import atomic_json, identifier


def _overlaps(first, second):
    return first == second or first in second.parents or second in first.parents


def _reserve_audit_root(tools, key, request):
    """Bind a derived directory before any audit or snapshot can be replaced."""
    root = Path(request['audit_root'])
    for previous in tools.store.all('trial_reparse_inputs'):
        previous_root = Path(previous['audit_root']).resolve()
        previous_key = previous['experiment_id'] + '-' + previous['revision_id']
        if previous_key != key and _overlaps(root, previous_root):
            raise ValueError('audit_root overlaps another revision: ' + previous_key)
    binding = {'revision_key': key, 'store_ref': str(tools.store.root.resolve()),
               'request': request}
    marker = root / 'revision-input.json'
    if marker.exists():
        if read(marker) != binding:
            raise ValueError('audit_root belongs to another revision or store')
    else:
        # Source provenance may be prepared here, but derived outputs cannot be
        # adopted from an unbound revision directory.
        for name in ('independent-audit.json', 'critical-progress.json',
                     'collection-check.json', 'candidate-cost-snapshot.json'):
            if (root / name).exists():
                raise ValueError('audit_root contains unbound revision output: ' + name)
        for parent in root.parents:
            if (parent / 'revision-input.json').exists():
                raise ValueError('audit_root is inside another revision directory')
        if root.exists() and any(root.rglob('revision-input.json')):
            raise ValueError('audit_root contains another revision directory')
        atomic_json(marker, binding)


def reparse_trial(tools, experiment_id, *, output, permit_path, revision_id,
                  audit_root, read_source_ref=None):
    """Consume existing dispatched output; never reopen a job or call an executor."""
    identifier(revision_id)
    key = experiment_id + '-' + revision_id
    output, audit_root = Path(output).resolve(), Path(audit_root).resolve()
    if _overlaps(audit_root, output):
        raise ValueError('revised audit must be outside immutable Runtime output')
    experiment = tools._owned('research_experiments', experiment_id)
    trial = tools._owned('rule_trials', experiment_id)
    original = tools._owned('research_cycles', experiment_id + '-cycle')
    finding = tools._owned('research_findings', experiment_id)
    receipt = tools._owned('trial_collections', experiment_id)
    dispatch = tools._owned('trial_dispatches', experiment_id + '-dispatch')
    permit, context = read(permit_path), read(output / 'runtime_context.json')
    identity = {k: context[k] for k in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
    if (permit.get('experiment_id') != experiment_id or permit.get('job_id') != tools.job.state['job_id']
            or permit.get('run_id') != context['run_id'] or context['bundle_id'] != trial['bundle_id']
            or context['protocol_id'] != tools.protocol['protocol_id']
            or any(Path(p).resolve() != output for p in (dispatch['output'], permit['output_dir']))
            or original['result_id'] != finding['result_id']
            or receipt['result']['costs'] != finding['costs']):
        raise ValueError('reparse differs from original dispatch/permit/receipt/finding')
    request = {'experiment_id': experiment_id, 'revision_id': revision_id,
               'audit_version': AUDIT_VERSION, 'identity': identity,
               'output': str(output), 'permit_ref': str(Path(permit_path).resolve()),
               'audit_root': str(audit_root),
               'read_source_ref': str(Path(read_source_ref).resolve()) if read_source_ref else None,
               'source_provenance': read(read_source_ref) if read_source_ref else None}
    previous_input = tools.store.path('trial_reparse_inputs', key)
    if previous_input.exists() and tools._owned('trial_reparse_inputs', key) != {
            'job_id': tools.job.state['job_id'], 'base_bundle_id': tools.base.bundle_id,
            'protocol_id': tools.protocol['protocol_id'], **request}:
        raise ValueError('conflicting reparse revision input')
    if tools.store.path('research_cycle_revisions', key).exists():
        if not previous_input.exists():
            raise ValueError('revision cycle lacks its immutable input')
        return tools._owned('research_cycle_revisions', key)
    started = time.perf_counter()
    _reserve_audit_root(tools, key, request)
    from .artifact_accounting import begin, finish
    begin(tools, key)
    tools._save('trial_reparse_inputs', key, request)
    result, checks = _collect(tools, experiment_id, output, revision_id=revision_id,
                             derived_root=audit_root, read_source_ref=read_source_ref)
    # Recomputed technical facts must agree with the historical charge; current
    # file sizes and new CPU work are distinct from the original Runtime cost.
    for name, value in finding['costs'].items():
        if name != 'artifact_bytes' and result.costs.get(name) != value:
            raise ValueError('historical Runtime cost changed during offline reparse: ' + name)
    result = replace(result, costs=finding['costs'])
    tools._validate_result(result, {experiment['bundle_id']})
    if (result.criteria_ref != experiment['criterion_ref'] or result.evidence_level != experiment['evidence_level']
            or any(result.intervention.get(k) != v for k, v in experiment['intervention'].items())):
        raise ValueError('reparse changes original intervention/criterion/source')
    if tools._hard_costs(experiment, result.costs) != finding['hard_costs']:
        raise ValueError('reparse does not preserve original cost receipt')
    phase_path = Path(permit['phase_snapshot_ref'])
    phase = read(phase_path)
    # Re-evaluate the registered receipt and limits, rather than copying the old
    # screening boolean. No historical cost is consumed a second time.
    hard_costs = tools._hard_costs(experiment, result.costs)
    checks['resource_accounting_complete'] = (
        not phase.get('unresolved_costs')
        and all(v <= experiment['cost_limit'].get(
            k, tools.job.state['limits'].get(k, -1) if tools._observed_only(experiment) else -1)
                for k, v in hard_costs.items()))
    if checks['collection_complete'] is True:
        from rpent.flywheel.episode import validate_episode
        from robots.libero.flywheel import LIBERO_SPEC
        manifest = read(output / 'dataset_manifest.json')
        try:
            for episode in manifest['episodes']:
                validate_episode(episode['path'], spec=LIBERO_SPEC)
        except (OSError, ValueError, KeyError) as exc:
            checks['collection_complete'] = False
            atomic_json(audit_root / 'collection-check.json', {'passed': False, 'reason': str(exc)})
        else:
            atomic_json(audit_root / 'collection-check.json', {'passed': True, 'manifest_ref': str(output / 'dataset_manifest.json'),
                        'persisted_steps': sum(e['persisted_steps'] for e in manifest['episodes']), 'training_admission': 'not_admitted'})
    tools.store.save_development_result(result)
    source_refs = {kind: str(tools.store.path(kind, experiment_id + '-cycle' if kind == 'research_cycles' else experiment_id))
                   for kind in ('research_findings', 'trial_collections', 'research_cycles')}
    revised = tools._save('research_finding_revisions', key, {
        'experiment_id': experiment_id, 'revision_id': revision_id,
        'result_id': result.result_id, 'outcome': result.outcome, 'evidence_level': result.evidence_level,
        'costs': finding['costs'], 'hard_costs': finding['hard_costs'],
        'original_charge_ref': source_refs['research_findings'], 'cost_booking': 'historical_reuse_no_charge',
        'source_refs': source_refs, 'audit_version': AUDIT_VERSION,
        'audit_ref': str(audit_root / 'independent-audit.json'),
        'critical_progress_ref': str(audit_root / 'critical-progress.json'),
        'screening_checks': checks,
        'check_diff': {k: {'before': original['screening_checks'].get(k), 'after': v}
                       for k, v in checks.items() if original['screening_checks'].get(k) != v},
        'hypotheses_resolved': False})
    tools._save('trial_collection_revisions', key, {'experiment_id': experiment_id, 'revision_id': revision_id,
                'result': asdict(result), 'checks': checks, 'source_refs': source_refs,
                'audit_version': AUDIT_VERSION})
    seconds = time.perf_counter() - started
    # A stopped job may record additional observed offline costs, never resume.
    reason = 'offline audit revision ' + key
    if not any(c['reason'] == reason for c in tools.job.state['charges']):
        tools.job.state['charges'].append({'reason': reason, 'costs': {'whole_job_wall_seconds': seconds}})
        tools.job.state['used']['whole_job_wall_seconds'] += seconds
        tools.job._save()
    phase = read(phase_path)
    entries = phase.setdefault('offline_revision_charges', [])
    if not any(e['revision_key'] == key for e in entries):
        entries.append({'revision_key': key, 'costs': {'wall_seconds': seconds}, 'audit_root': str(audit_root)})
        phase['used']['wall_seconds'] += seconds
        phase['remaining']['wall_seconds'] = phase['limits']['wall_seconds'] - phase['used']['wall_seconds']
        atomic_json(phase_path, phase)
    finish(tools, key, phase_path=phase_path, output=audit_root, billed_output_bytes=0)
    phase = read(phase_path)
    snapshot_path = audit_root / 'candidate-cost-snapshot.json'
    snapshot = {'revision_key': key, 'identity': identity,
                'phase_snapshot_ref': str(phase_path), 'phase': phase,
                'basis': 'entire authoritative phase history as of submission; includes previous failures and original Research; later derived/evaluation charges remain in phase ledger',
                'token_policy': 'observed_only; per-request missing usage retained in source evidence'}
    if not snapshot_path.exists():
        atomic_json(snapshot_path, snapshot)
    else:
        saved_snapshot = read(snapshot_path)
        if (saved_snapshot.get('revision_key') != key or saved_snapshot.get('identity') != identity
                or saved_snapshot.get('phase_snapshot_ref') != str(phase_path)):
            raise ValueError('candidate cost snapshot belongs to another revision')

    if all(v is True for v in checks.values()):
        candidate = tools.submit_candidate(key + '-candidate', experiment_id=experiment_id,
                                          finding_revision_id=key, cost_snapshot_ref=snapshot_path)
        action = 'candidate_for_independent_evaluation'
    else:
        candidate, action = None, 'stop'
        tools.defer(key + '-stop', opportunity_id=experiment['opportunity_id'],
                    reason='registered deterministic development screening incomplete or unsatisfied',
                    missing_evidence=[k for k, v in checks.items() if v is not True])
    cycle = tools._save('research_cycle_revisions', key, {
        'experiment_id': experiment_id, 'revision_id': revision_id, 'action': action,
        'result_id': result.result_id, 'finding_ref': key, 'candidate': candidate,
        'screening_checks': checks, 'source_refs': source_refs, 'audit_version': AUDIT_VERSION,
        'audit_ref': revised['audit_ref'], 'model_reanalysis': False, 'runtime_reexecution': False,
        'hypotheses_resolved': False, 'criterion': original['criterion'],
        'costs': finding['costs'], 'offline_audit_seconds': seconds,
        'cost_booking': 'original Runtime reused once; offline audit CPU/storage incremental'})
    finish(tools, key, phase_path=phase_path, output=audit_root, billed_output_bytes=0)
    return cycle
