"""F06 real persistence, full facts, interruption and independent-consumer evidence."""
import json
from dataclasses import replace

import numpy as np
import pytest

from rsi_vla import EventLedger, ExecutionEvent, ResearchStore, RunBundle, initialize
from rsi_vla.collection import register_collection
from rsi_vla.relation_consumption import review_media_persistence
from rsi_vla.process_accounting import StorageMeter
from rpent.session import EnvState
from rpent.session.media import configure_media_policy
from rpent.session.runtime import RuntimeJournal


def test_reference_selection_keeps_full_l0_and_reports_gaps_unknown_probability(tmp_path):
    store = ResearchStore(tmp_path / 'research')
    bundle = RunBundle('b', 't', 'p', 'f', 'r', 's', 'm', 'protocol', 'backend', 'code')
    initialize(store, bundle, psi={'learned': False}, memory={'provenance': 'prior', 'legal_information': ['public']})
    ledger = EventLedger(store)
    first = ExecutionEvent('first', 'run', 'episode', 0, 'b', 'action_proposal', 'runtime', proposal=[1], sent=[1], feedback={'observed': False}, missing=['applied'], flags={'action_source': 'vla'}, media=['a.png', 'b.png'])
    last = replace(first, event_id='last', sequence=2, kind='task_result', outcome='unknown', media=[], flags={'costs': {'artifact_bytes': 10}})
    ledger.append_many([first, last])
    view = ledger.view('view', 'b', selected_media=[])
    assert len(view.l0) == 2 and view.selected_media == []
    assert view.l0[0]['proposal'] == view.l0[0]['sent'] == [1]
    assert view.l0[0]['feedback'] == {'observed': False}
    assert view.l0[-1]['flags']['costs']['artifact_bytes'] == 10
    assert view.l0[0]['probabilities']['outcome_label'] is None
    assert view.selection['probabilities']['a.png'] is None
    assert view.selection['zero_coverage'] == ['a.png', 'b.png']
    assert view.selection['l0_integrity']['sequence_gaps'][0]['first_missing_sequence'] == 1
    assert view.selection['persisted_bytes_saved'] is None
    assert store.get('events', 'first')['missing'] == ['applied']


def runtime(root, *, feedback=False, limit=100000, rho=None, bundle=None):
    policy = {'type': 'fixed', 'retain_l0': 'all', 'media': 'all_available'}
    if feedback:
        policy = {'type': 'rsi-relation-feedback-v1', 'values': {'media': {'all_available': {'count': 1, 'total': .2}, 'one_per_event': {'count': 1, 'total': .9}}}}
    context = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p', output_dir=str(root),
        information_scope='development', planner={'backend': 'flash'}, execution_arm={'arm': 'D'},
        execution_binding={'bundle': {'rho_ref': 'rho'}, 'records': {'rho_ref': {'rhoA': policy}}},
        l1_media_policy={'schema': 'rsi-l1-media-policy-v1', 'rho_ref': 'rho', 'optional_artifacts': ['agentview_world_high.npz', 'wrist_world_high.npz'], 'max_persisted_bytes': limit})
    if bundle is not None:
        context['bundle_id'] = bundle.bundle_id
        context['execution_binding'] = {'bundle': asdict(bundle), 'records': {'rho_ref': rho}}
        context['l1_media_policy']['rho_ref'] = bundle.rho_ref
    journal = RuntimeJournal(context)
    (root/'runtime_context.json').write_text(json.dumps(context))
    state = EnvState(root)
    configure_media_policy(state, journal)
    return state, journal, context


@pytest.mark.parametrize('feedback,limit', [(False, 100000), (True, 100000), (False, 0)])
def test_independent_consumer_checks_actual_files_and_temporary_cost(tmp_path, feedback, limit):
    state, journal, context = runtime(tmp_path, feedback=feedback, limit=limit)
    meter = StorageMeter([tmp_path])
    with state.record_step(state={}):
        for name in context['l1_media_policy']['optional_artifacts']:
            state.save(name, np.ones((4, 4, 3)))
        state.save('audit.json', {'outcome': 'unknown'})
    storage = meter.sample()
    report = review_media_persistence(tmp_path)
    assert report['value'] is True
    assert report['persistent_output_bytes'] > 0
    assert report['cumulative_new_bytes_lower_bound'] >= report['persistent_output_bytes']
    assert storage['cumulative_new_bytes_lower_bound'] >= storage['current_new_bytes']
    assert report['diagnostic_quality'] is None
    if limit == 0:
        assert all(not row['admitted'] for row in report['byte_admissions'])
        assert report['byte_admissions'][-1]['cumulative_new_bytes_lower_bound'] > 0
    if feedback:
        assert not (tmp_path/'wrist_world_high.npz/00.npz').exists()
    rows = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
    rows[1]['rho_ref'] = 'forged'
    journal.events_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='bound rhoA'):
        review_media_persistence(tmp_path)


@pytest.mark.parametrize('scope,source', [('development', 'autonomous_execution'), ('release', 'independent_evaluation')])
def test_partial_collection_keeps_pending_dispatch_and_separate_source(tmp_path, scope, source):
    identity = dict(run_id='r', episode_id='e', bundle_id='b', protocol_id='p')
    (tmp_path/'independent_result.json').write_text(json.dumps(identity))
    episode = tmp_path/'data/episode_interrupted.partial'
    episode.mkdir(parents=True)
    pending = {'dispatch_id': 'sent-without-feedback', 'count': 2}
    (episode/'collection.json').write_text(json.dumps({'runtime_identity': identity, 'episode_id': 'local', 'batches': [{'start': 0, 'file': 'not-persisted.npz', 'count': 2}], 'persisted_steps': 0, 'pending_dispatch': pending, 'unflushed_observed_steps': 1}))
    context = dict(identity, output_dir=str(tmp_path), information_scope=scope, task={'task_key': 'task'}, collection={'root': str(tmp_path/'data')})
    manifest = register_collection(context)
    assert manifest['source_kind'] == source
    assert manifest['episodes'][0]['pending_dispatch'] == pending
    assert manifest['episodes'][0]['unflushed_observed_steps'] == 1
    assert manifest['alignment_complete'] is False
    assert manifest['selection_mechanisms']['outcome_label']['probability'] is None
    assert manifest['artifact_storage']['persistent_bytes'] > 0
    assert manifest['training_admission'] == 'not_admitted'


import unittest
import test_relation_learning
from dataclasses import asdict
from rsi_vla.relation_learning import learn_relations, freeze_relations


class LearnedPersistenceTests(unittest.TestCase):
    setUp = test_relation_learning.RelationLearningTest.setUp
    make_tools = test_relation_learning.RelationLearningTest.make_tools
    inherited_tools = test_relation_learning.RelationLearningTest.inherited_tools

    def test_legal_feedback_freeze_and_new_job_consume_actual_l1_outlet(self):
        original = self.source.get('rho', self.base.rho_ref)
        learning = learn_relations(self.tools, self.derived, 'l1-learning', ['result'], relations=['rhoA'])
        target = freeze_relations(self.tools, self.derived, 'l1-learning', rho_ref='rho-l1', bundle_id='l1-bundle')
        tools = self.inherited_tools(target)
        learned = self.derived.get('rho', target.rho_ref)
        self.assertGreater(learned['rhoA']['values']['media']['one_per_event']['total'], learned['rhoA']['values']['media']['all_available']['total'])
        for name, bundle, rho in [('fixed', self.base, original), ('learned', target, learned), ('inherited-job', tools.base, learned)]:
            state, journal, context = runtime(self.root/name, rho=rho, bundle=bundle)
            with state.record_step(state={}):
                state.save('agentview_world_high.npz', np.ones((4, 4, 3)))
                state.save('wrist_world_high.npz', np.ones((4, 4, 3)))
            self.assertEqual((journal.output_dir/'wrist_world_high.npz/00.npz').exists(), name == 'fixed')
            self.assertTrue(review_media_persistence(journal.output_dir)['value'])
            rows = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
            self.assertTrue(all(row['rho_ref'] == bundle.rho_ref for row in rows if row['kind'] == 'media_materialized'))
        self.assertEqual(self.source.get('rho', self.base.rho_ref), original)
        self.assertEqual(learning['relations'], ['rhoA'])
        self.assertEqual(tools.job.state['used'], {'input_tokens': 0, 'output_tokens': 0})
