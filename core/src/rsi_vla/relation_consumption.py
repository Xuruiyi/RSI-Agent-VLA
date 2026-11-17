"""Independent verification of actual relation consumers, separate from binding."""

import json
from pathlib import Path
from types import SimpleNamespace

from .cooperation import fixed_research_step
from .relation_evidence import review_skill_feedback
from .relation_learning import select_media


def evaluate_relation_consumption(store, output, *, research_request_ids=()):
    """Verify real Research input, dispatch gate and skill execution receipts.

    A bound rho in a Runtime context is insufficient. Each relation must reach
    its corresponding consumer; missing consumer evidence remains unknown.
    The caller owns the independent evaluation role and output persistence.
    """
    root = Path(output).resolve()
    context_ref = root / 'runtime_context.json'
    context = json.loads(context_ref.read_text())
    identity = {key: context[key] for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
    binding = context['execution_binding']
    bundle = binding['bundle']
    rho_ref = bundle['rho_ref']
    rho = binding['records']['rho_ref']
    if context.get('information_scope') != 'development' or context.get('rho_ref') != rho_ref:
        raise ValueError('relation consumption requires the bound development execution')
    relations = {name: dict(value=None, source_refs=[], missing=[]) for name in ('rhoA', 'rhoB', 'rhoC')}
    for request_id in research_request_ids:
        request = store.get('llm_requests', request_id)
        execution = store.get('llm_executions', request_id)
        if request.get('bundle_id') != bundle['bundle_id'] or execution.get('base_bundle_id') != bundle['bundle_id']:
            raise ValueError('Research consumer belongs to another relation candidate')
        facts = execution['transport_facts']
        if (facts['effective_model'], facts['effective_reasoning_effort']) != ('gpt-6.1-sol', 'high'):
            raise ValueError('relation Research consumption needs actual high transport facts')
        views = request.get('evidence_views', [])
        if not views or any(view.get('policy_ref') != rho_ref
                            or view.get('selected_media') != select_media(rho['rhoA'], view['l0'])
                            for view in views):
            raise ValueError('actual Research input did not consume the candidate rhoA')
        from .research_transport import unpack
        actual = store.get('bounded_requests', request_id)
        transport = store.get('bounded_transports', request_id)
        if unpack(transport) != actual or actual.get('bundle_id') != bundle['bundle_id']:
            raise ValueError('actual Research transport differs from its bound request')
        originals = {view['view_id']: view for view in views}
        actual_views = actual.get('evidence_views', [])
        if not actual_views or any(view['view_id'] not in originals
                or view.get('policy_ref') != rho_ref
                or view.get('selected_media') != originals[view['view_id']]['selected_media']
                for view in actual_views):
            raise ValueError('actual Research transport discarded the candidate rhoA selection')
        covered = [view for view in views if any(
            event.get('run_id') == identity['run_id']
            and event.get('episode_id') == identity['episode_id']
            and event.get('active_bundle') == bundle['bundle_id'] for event in view['l0'])]
        actual_ids = {view['view_id'] for view in actual_views}
        if not covered or not any(view['view_id'] in actual_ids for view in covered):
            raise ValueError('actual rhoA transport does not cover this development execution')
        refs = [store.path('llm_requests', request_id), store.path('llm_executions', request_id),
                store.path('bounded_requests', request_id), store.path('bounded_transports', request_id),
                Path(facts['evidence_ref'])]
        if not all(ref.is_file() for ref in refs):
            raise ValueError('actual Research consumer source is missing')
        selection_report = dict(objective='recorded_event_coverage_minus_reference_cost',
            selected_reference_count=sum(len(view.get('selected_media', [])) for view in actual_views),
            sdk_prompt_bytes=None, persisted_media_bytes=None, diagnostic_quality=None,
            media_mode='unknown', image_bytes_sent=None)
        if actual.get('readable_evidence') is not None:
            from .research_transport import sdk_input
            sdk = store.get('research_sdk_inputs', request_id)
            receipt = store.get('research_sdk_consumption', request_id)
            if sdk != sdk_input(transport, sdk['output_schema']) or receipt.get('request_id') != request_id:
                raise ValueError('actual SDK input differs from its readable business request')
            if receipt.get('sdk_turn_returned') is not True or receipt.get('image_bytes_sent') is not False:
                raise ValueError('readable Research input lacks its text-only SDK consumer receipt')
            refs.extend([store.path('research_sdk_inputs', request_id),
                         store.path('research_sdk_consumption', request_id)])
            selection_report.update(sdk_prompt_bytes=len(sdk['prompt'].encode('utf-8')),
                                    media_mode='text_only', image_bytes_sent=False)
        relations['rhoA'].update(value=True, source_refs=[str(ref) for ref in refs],
                                  measurement=selection_report)
    if context.get('l1_media_policy') is not None:
        persistence = review_media_persistence(root, context)
        relations['rhoA']['persistence'] = persistence
        if persistence['value'] is True:
            relations['rhoA']['value'] = True
            relations['rhoA']['source_refs'].extend(persistence['source_refs'])
    if relations['rhoA']['value'] is None:
        relations['rhoA']['missing'].append('No actual Research request/execution with candidate evidence selection')
    gate = context.get('relation_dispatch_admission')
    if gate is not None:
        if gate.get('rho_ref') != rho_ref or gate.get('actual_dispatch_gate') is not True:
            raise ValueError('actual dispatch gate has another policy binding')
        evidence = [store.get('development_results', key) for key in gate['source_result_ids']]
        if any(row.get('role') != 'development' or row['intervention'].get('information_scope') != 'development'
               for row in evidence):
            raise ValueError('dispatch gate cannot consume formal or sealed feedback')
        from .relation_learning import CONTEXT_SCHEMA, contextual_research_decision, research_context, version_signature
        if rho['rhoC'].get('type') == CONTEXT_SCHEMA:
            from .contracts import RunBundle
            saved = store.get('research_relation_selections', gate['selection_id'])
            for key in ('rho_ref', 'source_result_ids', 'decisions', 'action', 'legal_repairs',
                        'technical_admitted', 'data_admitted', 'job_state_before_dispatch'):
                if gate.get(key) != saved.get(key):
                    raise ValueError('actual contextual dispatch differs from its reserved selection')
            decisions = []
            for row in gate['decisions']:
                category = ('technical_invalid_development' if any(r['invalid_count'] for r in evidence)
                            else 'development_l0')
                expected_context = research_context(task_key=row['state'], operator=gate['legal_repairs'][0],
                    evidence_category=category, signature=version_signature(RunBundle(**bundle)),
                    data_context=rho['rhoC']['data_context'])
                if row['context'] != expected_context:
                    raise ValueError('actual contextual dispatch changed operator/evidence/version/data context')
                decision = contextual_research_decision(rho['rhoC'],
                    SimpleNamespace(state=gate['job_state_before_dispatch']), context=expected_context,
                    evidence_available=bool(evidence), inspected=True, legal_repairs=gate['legal_repairs'],
                    experiment_permitted=True, technical_admitted=gate['technical_admitted'],
                    data_admitted=gate['data_admitted'], exploration_used=row['exploration_used_before'])
                decisions.append({'state': row['state'], **decision})
        else:
            decisions = [dict(state=row['state'], action=fixed_research_step(
                rho, SimpleNamespace(state=gate['job_state_before_dispatch']), evidence_available=bool(evidence),
                inspected=True, legal_repairs=['relation_feedback_update'], experiment_permitted=True,
                state=row['state'])) for row in gate['decisions']]
        if decisions != gate['decisions'] or gate['action'] != 'prepare':
            raise ValueError('actual dispatch differs from independently recomputed rhoC')
        relations['rhoC'].update(value=bool(decisions) and all(row['action'] == 'bounded_experiment' for row in decisions),
                                  source_refs=[str(context_ref), str(root/'code_snapshot/core/src/rsi_vla/task_completion.py'),
                                               str(root/'code_snapshot/core/src/rsi_vla/cooperation.py')])
    else:
        relations['rhoC']['missing'].append('No actual candidate-policy dispatch gate')
    if context.get('relation_control', {}).get('routes'):
        report = review_skill_feedback(root)
        known = [row for row in report['reviews'] if row.get('client_route_sequence') is not None]
        relations['rhoB'].update(value=True if known and all(not row['missing'] for row in known) else None,
                                 source_refs=report['source_refs'], review=report)
        if relations['rhoB']['value'] is None:
            relations['rhoB']['missing'].append('No complete independently linked skill consumer')
    else:
        relations['rhoB']['missing'].append('No registered skill consumer for this task')
    for name, value in relations.items():
        if value['value'] is True and not all(Path(ref).is_file() for ref in value['source_refs']):
            raise ValueError(name+' consumer references missing evidence')
    return dict(schema='rsi-relation-consumption-v1', identity=identity, rho_ref=rho_ref,
                relations=relations, role='evaluation', binding_alone_sufficient=False)


def review_media_persistence(output, context=None):
    """Read the actual L1 outlet receipts independently of reference selection."""
    from .process_accounting import accounted_storage_bytes, snapshot

    root = Path(output).resolve()
    context = context or json.loads((root / "runtime_context.json").read_text())
    config = context["l1_media_policy"]
    binding = context["execution_binding"]
    rho_ref = binding["bundle"]["rho_ref"]
    if config.get("rho_ref") != rho_ref:
        raise ValueError("L1 outlet receipt policy binding mismatch")
    reference = root / "action_events.jsonl"
    rows = [json.loads(line) for line in reference.read_text().splitlines()]
    identity = {key: context[key] for key in ("run_id", "episode_id", "bundle_id", "protocol_id")}
    if any(any(row.get(key) != value for key, value in identity.items()) for row in rows):
        raise ValueError("L1 outlet receipt identity mismatch")
    selections = [row for row in rows if row["kind"] == "media_retention_selection"]
    admissions = [row for row in rows if row["kind"] == "media_persistence_admission"]
    written = [row for row in rows if row["kind"] == "media_materialized"]
    optional = set(config["optional_artifacts"]) - set(config.get("protected_artifacts", []))
    protected_run = not (context.get("information_scope") == "development"
        and context.get("execution_arm", {}).get("arm") in {"D", "S", "R"}
        and context.get("planner", {}).get("backend") == "flash" and not context.get("relation_control"))
    policy = binding["records"]["rho_ref"]["rhoA"]
    chosen = select_media(policy, [{"media": sorted(optional)}])
    states_path = root / "states.json"
    states = json.loads(states_path.read_text()).get("steps", []) if states_path.exists() else []
    protected_steps = set(config.get("protected_steps", [])) | {
        state["step_idx"] for state in states if state.get("terminated") or state.get("truncated")
        or state.get("extras", {}).get("audit_required")}
    missing = []
    for row in selections:
        protected = protected_run or row["artifact"] not in optional or row["step"] in protected_steps
        expected = protected or row["artifact"] in chosen
        if (row.get("rho_ref") != rho_ref or row.get("protected") != protected
                or row.get("selected") != expected or row.get("probability") != float(expected)):
            raise ValueError("actual L1 outlet differs from its bound rhoA or protected control evidence")
    for row in admissions:
        if (row.get("rho_ref") != rho_ref
                or row.get("max_persisted_bytes") != config["max_persisted_bytes"]
                or row.get("encoded_bytes", -1) < 0):
            raise ValueError("L1 byte admission receipt differs from frozen budget")
    for row in written:
        name = row["artifact"]
        step = row["step"]
        expected_path = root / name if step is None else root / name / f"{step:02d}{Path(name).suffix}"
        if Path(name).name != name or Path(row["path"]) != expected_path or row.get("rho_ref") != rho_ref:
            raise ValueError("L1 materialization receipt escaped its owned outlet")
        if not expected_path.is_file() or expected_path.stat().st_size != row["persisted_bytes"]:
            missing.append(str(expected_path))
    for row in selections:
        if row["selected"] and not any((item["artifact"], item["step"]) == (row["artifact"], row["step"])
                for item in written) and not any((item["artifact"], item["step"]) == (row["artifact"], row["step"])
                and item.get("admitted") is False for item in admissions):
            missing.append("materialization_or_budget_receipt:" + row["artifact"] + ":" + str(row["step"]))
    if not selections:
        missing.append("actual_L1_outlet_selection")
    return {"schema": "rsi-l1-persistence-consumption-v1",
        "value": True if selections and (written or admissions) and not missing else None,
        "rho_ref": rho_ref, "identity": identity,
        "selection_target": "actual_media_persistence", "selections": selections,
        "byte_admissions": admissions, "materializations": written,
        "persistent_output_bytes": sum(snapshot([root]).values()),
        "cumulative_new_bytes_lower_bound": accounted_storage_bytes(root),
        "exact_cumulative_bytes": None, "diagnostic_quality": None,
        "missing": missing, "source_refs": [str(reference), str(root / "runtime_context.json")],
        "deletion_refunds_cumulative_cost": False}
