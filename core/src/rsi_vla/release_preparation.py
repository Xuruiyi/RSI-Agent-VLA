"""Derive release identities from screened development without relabeling a run."""

import json
import copy
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import Candidate, RunBundle
from .rule_updates import read_phi_chunk, rule_update


def derive_initial_deployment(source_bundle, source_protocol, source_profile, *,
                              bundle_id, protocol_id, profile_id, backend_ref, code_ref,
                              authorization_ref, compatibility_ref, evidence_import_id,
                              deployment_scope):
    """Prepare a new deployment identity; never rewrite the comparison evidence."""
    if (source_bundle['bundle_id'] != source_protocol['anchor_bundle_id']
            or source_bundle['protocol_id'] != source_protocol['protocol_id']
            or source_protocol.get('comparison_only') is not True
            or source_protocol.get('status') != 'frozen'
            or bundle_id == source_bundle['bundle_id']
            or protocol_id == source_protocol['protocol_id']):
        raise ValueError('deployment derivation requires a distinct tested initial identity')
    if read_phi_chunk(source_bundle['phi_ref']).get('planner_segment_handoff_hint', False) is not False:
        raise ValueError('only the tested hint=false initial version can be imported')
    profile = copy.deepcopy(source_profile)
    profile.update(profile_id=profile_id, code_ref=code_ref, dynamic_authorized=True,
                   release_authorized=False, deployment_authorized=True,
                   authorization_reference=str(authorization_ref),
                   changed_configuration='Explicit initial deployment control derivation; behavior preserved')
    protocol = copy.deepcopy(source_protocol)
    derivation = dict(kind='initial_evidence_import', source_bundle_id=source_bundle['bundle_id'],
                      source_protocol_id=source_protocol['protocol_id'],
                      evidence_import_id=evidence_import_id, authorization_ref=str(authorization_ref),
                      compatibility_ref=str(compatibility_ref), source_runs_relabelled=False,
                      historical_os_isolation_claimed=False)
    protocol.update(protocol_id=protocol_id, anchor_bundle_id=bundle_id, comparison_only=False,
                    dynamic_authorized=True, deployment_authorized=True,
                    qualification_establishment_authorized=True, release_authorized=False,
                    allowed_runtime_splits=['deployment'], authorization_scope=deployment_scope,
                    authorization_ref=str(authorization_ref), deployment_derivation=derivation,
                    governance='Initial publication from imported independent evidence; no candidate gain claim')
    for split in ('development', 'release', 'sealed'):
        protocol['task_sets'][split]['authorized'] = False
    protocol['runtime_config']['profile_reference'] = str(backend_ref)
    protocol['runtime_config']['authority'] = 'explicit deployment derivation'
    protocol['next_step_proposal']['authorized'] = False
    protocol['budget']['authorized_now'] = {key: 0 for key in protocol['budget']['authorized_now']}
    protocol['budget']['first_release_cycle_proposal']['authorized'] = False
    protocol['budget']['authorization_scope'] = deployment_scope
    bundle = replace(RunBundle(**source_bundle), bundle_id=bundle_id, protocol_id=protocol_id,
                     backend_ref=str(backend_ref), code_ref=code_ref,
                     parent_bundle_id=source_bundle['bundle_id'])
    return dict(bundle=asdict(bundle), protocol=protocol, profile=profile, derivation=derivation)


def validate_initial_deployment_derivation(source_bundle, source_protocol, source_profile,
                                           bundle, protocol, profile):
    """Reject changes outside the explicit deployment control derivation."""
    derivation = protocol['deployment_derivation']
    expected = derive_initial_deployment(source_bundle, source_protocol, source_profile,
        bundle_id=bundle['bundle_id'], protocol_id=protocol['protocol_id'],
        profile_id=profile['profile_id'], backend_ref=bundle['backend_ref'], code_ref=bundle['code_ref'],
        authorization_ref=derivation['authorization_ref'],
        compatibility_ref=derivation['compatibility_ref'],
        evidence_import_id=derivation['evidence_import_id'],
        deployment_scope=protocol['authorization_scope'])
    if (expected['bundle'] != bundle or expected['protocol'] != protocol
            or expected['profile'] != profile):
        raise ValueError('deployment derivation changes evaluated behavior or criterion')


def derive_release_candidate(store, source_cycle_id, protocol, profile, *, base_id,
                             trial_id, candidate_id, backend_ref, code_ref,
                             release_admission_snapshot_ref=None):
    """Bind release artifacts; inherit source submission costs without a new charge.

    An optional frozen admission snapshot records the later phase state separately.
    It never refreshes Candidate costs or replaces the authoritative phase ledger.
    """
    cycle = store.get('research_cycle_revisions', source_cycle_id)
    if (cycle['action'] != 'candidate_for_independent_evaluation' or not cycle.get('screening_checks')
            or not all(v is True for v in cycle['screening_checks'].values()) or not cycle.get('candidate')):
        raise ValueError('release needs a complete program-screened development Candidate')
    source = Candidate(**store.get('candidates', cycle['candidate']['candidate_id']))
    if asdict(source) != cycle['candidate']:
        raise ValueError('source cycle/Candidate mismatch')
    source_results = [store.get('development_results', key) for key in source.development_results]
    if any(result['protocol_id'] != source.protocol_id or result['bundle_id'] != source.bundle_id
           or result['role'] != 'development' for result in source_results):
        raise ValueError('release source development result identity mismatch')
    development_sources = [
        {'result_id': result['result_id'], 'protocol_id': result['protocol_id'],
         'bundle_id': result['bundle_id'], 'identity': result['intervention'].get('identity'),
         'result_ref': str(store.path('development_results', result['result_id']).resolve())}
        for result in source_results]
    source_snapshot_ref = source.compatibility.get('cost_snapshot_ref')
    cost_provenance = {
        'costs_basis': 'source_candidate_at_submission',
        'source_candidate_ref': str(store.path('candidates', source.candidate_id).resolve()),
        'source_cost_snapshot_ref': source_snapshot_ref,
        'inherited_costs': source.costs,
        'cost_booking': 'historical_reuse_no_charge',
        'later_costs_basis': 'authoritative_phase_ledger',
        'release_admission_snapshot_ref': None,
    }
    if release_admission_snapshot_ref is not None:
        admission_ref = Path(release_admission_snapshot_ref).resolve()
        admission = json.loads(admission_ref.read_text())
        if not source_snapshot_ref:
            raise ValueError('release admission requires source submission cost snapshot')
        submission = json.loads(Path(source_snapshot_ref).read_text())
        mapping = source.compatibility.get('cost_dimension_mapping', {})
        if {mapping.get(k, k): v for k, v in submission['phase']['used'].items()} != source.costs:
            raise ValueError('source submission cost snapshot differs from Candidate costs')
        if (not admission.get('phase_snapshot_ref')
                or admission['phase_snapshot_ref'] != submission.get('phase_snapshot_ref')):
            raise ValueError('release admission uses a different authoritative phase ledger')
        if admission['phase'].get('unresolved_costs'):
            raise ValueError('release admission phase has unresolved actual costs')
        if any(admission['phase']['used'].get(k, -1) < v
               for k, v in submission['phase']['used'].items()):
            raise ValueError('release admission predates source submission costs')
        cost_provenance['release_admission_snapshot_ref'] = str(admission_ref)
    oldbase = RunBundle(**store.get('bundles', source.base_bundle_id))
    oldtrial = RunBundle(**store.get('bundles', source.bundle_id))
    origin = protocol.get('development_derivation', {})
    if origin.get('source_protocol_id') != source.protocol_id or origin.get('source_candidate_id') != source.candidate_id:
        raise ValueError('release protocol lacks exact development derivation')
    previous = json.loads(Path(origin['source_protocol_ref']).read_text())
    if previous['protocol_id'] != source.protocol_id:
        raise ValueError('release source protocol identity mismatch')
    for key in ('scope', 'benchmark_id', 'tasks', 'asset_binding', 'metrics', 'hard_constraints',
                'min_gain', 'anchor_max_drop', 'cost_weights', 'initial_requirements', 'recovery', 'outcomes', 'state_evaluation', 'draft_comparison'):
        if protocol[key] != previous[key]:
            raise ValueError('release derivation changes frozen behavior/criterion: ' + key)
    for split in ('development', 'release', 'sealed'):
        original = {k: v for k, v in previous['task_sets'][split].items() if k != 'authorized'}
        derived = {k: v for k, v in protocol['task_sets'][split].items() if k != 'authorized'}
        if original != derived:
            raise ValueError('release derivation changes assignment split: ' + split)
    if (protocol.get('allowed_runtime_splits') != ['release'] or protocol.get('release_authorized') is not True
            or protocol['task_sets']['release'].get('authorized') is not True
            or protocol['task_sets']['development'].get('authorized') is not False
            or protocol['task_sets']['sealed'].get('authorized') is not False):
        raise ValueError('release requires its explicit release-only authorization')
    oldprofile = json.loads(Path(oldbase.backend_ref).read_text())
    excluded = {'profile_id', 'dynamic_authorized', 'authorization_reference', 'changed_configuration', 'release_authorized'}
    if {k: v for k, v in profile.items() if k not in excluded} != {k: v for k, v in oldprofile.items() if k not in excluded}:
        raise ValueError('release profile changes runtime behavior; new development required')
    ignored_runtime_metadata = {'profile_reference', 'authority'}
    if {k: v for k, v in protocol['runtime_config'].items() if k not in ignored_runtime_metadata} != {k: v for k, v in previous['runtime_config'].items() if k not in ignored_runtime_metadata}:
        raise ValueError('release changes common runtime configuration')
    if protocol['budget']['per_attempt_proposal'] != previous['budget']['per_attempt_proposal']:
        raise ValueError('release changes per-attempt limits')
    before, after = read_phi_chunk(oldbase.phi_ref), read_phi_chunk(oldtrial.phi_ref)
    expected, _ = rule_update(before, planner_segment_handoff_hint=True)
    if after != expected or before.get("planner_segment_handoff_hint", False) is not False:
        raise ValueError('release difference is not solely handoff hint false -> true')
    base = replace(oldbase, bundle_id=base_id, protocol_id=protocol['protocol_id'],
                   backend_ref=str(backend_ref), code_ref=code_ref, parent_bundle_id=oldbase.bundle_id)
    trial = replace(oldtrial, bundle_id=trial_id, protocol_id=protocol['protocol_id'],
                    backend_ref=str(backend_ref), code_ref=code_ref, parent_bundle_id=base_id)
    if protocol['anchor_bundle_id'] != base_id:
        raise ValueError('derived release anchor identity mismatch')
    store.save_bundle(base)
    store.save_bundle(trial)
    candidate = replace(source, candidate_id=candidate_id, base_bundle_id=base_id,
                        bundle_id=trial_id, protocol_id=protocol['protocol_id'],
                        compatibility={**source.compatibility, 'source_candidate_id': source.candidate_id,
                                       'source_cycle_revision_id': source_cycle_id,
                                       'source_development_protocol_id': source.protocol_id,
                                       'development_results_relabelled': False,
                                       'source_development_results': development_sources,
                                       'common_code_ref': code_ref,
                                       'cost_provenance': cost_provenance})
    store.save_candidate(candidate)
    record = {'source_cycle_revision_id': source_cycle_id, 'source_candidate_id': source.candidate_id,
              'release_candidate_id': candidate_id, 'source_protocol_id': source.protocol_id,
              'release_protocol_id': protocol['protocol_id'], 'source_base_id': oldbase.bundle_id,
              'source_trial_id': oldtrial.bundle_id, 'release_base_id': base_id, 'release_trial_id': trial_id,
              'cost_provenance': cost_provenance,
              'development_results': source.development_results,
              'source_development_results': development_sources, 'new_development_runs': 0,
              'behavior_diff': {'planner_segment_handoff_hint': {'before': False, 'after': True}}}
    store.put('release_derivations', candidate_id, record)
    return asdict(candidate), record
