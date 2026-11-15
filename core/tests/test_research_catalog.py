"""Lossless compact catalog facts in opt-in Research SDK input."""
import copy
import json
import unittest
from rsi_vla.research_transport import _sdk_projection


def request():
    prefix='scope.run.episode.'
    index=[{'event_id':prefix+'step-0','sequence':0,'kind':'observation'},
           {'event_id':prefix+'action-0','sequence':1,'kind':'tool_start'},
           {'event_id':prefix+'result','sequence':2,'kind':'task_result'}]
    selected=[index[-1]['event_id']];omitted=[x['event_id']for x in index[:-1]]
    return {'evidence_input':{'sdk_projection':'terminal_history_v1','views':[{
        'view_id':'v','selected_event_ids':selected,'omitted_event_ids':omitted,'source_view_ref':'original.json'}]},
        'evidence_views':[{'view_id':'v','l0':[{'event_id':selected[0],'kind':'task_result','flags':{
            'independent_result':{'episode_execution':{'decisions':[{'actual':1}],'segments':[{'actual':2}]}}}}]}],
        'readable_evidence':{'views':[{'view_id':'v','event_index':index,'selected_event_ids':selected,'omitted_event_ids':omitted}]}}


class CatalogTests(unittest.TestCase):
    def test_catalog_restores_all_original_facts_without_mutating_source(self):
        original=request();original['evidence_input']['sdk_projection']='terminal_history_catalog_v2';before=copy.deepcopy(original)
        projected=_sdk_projection(original);idx=projected['readable_evidence']['views'][0]['event_index']
        restored=[{'event_id':idx['event_id_prefix']+row[0],'sequence':row[1],'kind':row[2]}for row in json.loads(idx['rows_json'])]
        self.assertEqual(restored,original['readable_evidence']['views'][0]['event_index'])
        self.assertEqual(original,before)
        legacy=copy.deepcopy(original);legacy['evidence_input']['sdk_projection']='terminal_history_v1'
        self.assertEqual(projected['evidence_views'],_sdk_projection(legacy)['evidence_views'])

    def test_actual_bounded_request_preparation_accepts_catalog_mode(self):
        import test_research_readable as readable
        from rsi_vla.research_transport import model_prompt, pack
        fixture = readable.ReadableResearchTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.configure()
        fixture.tools.domain_policy["evidence_input"].update(
            media_refs="selected_events", sdk_projection="terminal_history_catalog_v2"
        )
        fixture.response("catalog-entry")
        business = fixture.source.get("bounded_requests", "catalog-entry")
        self.assertEqual(business["evidence_input"]["sdk_projection"], "terminal_history_catalog_v2")
        shown = json.loads(model_prompt(pack(business)).split("\n", 1)[1])
        self.assertTrue(shown["sdk_catalog_encoding"]["lossless"])
        self.assertFalse(fixture.source.all("llm_executions"))

    def test_followup_keeps_complete_requested_terminal_histories(self):
        original=request();original['evidence_input']['sdk_projection']='terminal_history_catalog_v2';original['evidence_followup']={'requested':3}
        projected=_sdk_projection(original)
        self.assertEqual(projected['evidence_views'],original['evidence_views'])
        self.assertNotIn('sdk_projection',projected)
        self.assertTrue(projected['sdk_catalog_encoding']['lossless'])

    def test_actual_followup_keeps_lossless_catalog_and_full_requested_payload(self):
        from test_research_media_selection import SelectedMediaTests
        from rsi_vla.research_transport import model_prompt, pack, unpack
        fixture = SelectedMediaTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        f = fixture.f
        f.tools.domain_policy["evidence_input"].update(media_refs="selected_events", sdk_projection="terminal_history_catalog_v2")
        response = f.response("catalog-first")
        response.update(decision="defer", intervention=None, missing_evidence=["full event needed"], evidence_requests=[{"result_id":"result", "event_ids":["e0","e1"], "reason":"read complete terminal"}])
        f.apply(response)
        followup = unpack(f.adapter.prepare_evidence_followup(f.tools, parent_request_id="catalog-first", request_id="catalog-followup", max_input_tokens=100, max_output_tokens=10))
        self.assertEqual(followup["evidence_input"]["sdk_projection"], "terminal_history_catalog_v2")
        shown = json.loads(model_prompt(pack(followup)).split("\n",1)[1])
        self.assertEqual(shown["evidence_views"][0]["l0"][0], f.source.get("evidence_views", "view")["l0"][0])
        self.assertTrue(shown["sdk_catalog_encoding"]["lossless"])
        self.assertFalse(f.source.all("llm_executions"))

    def test_unknown_index_fields_are_not_silently_removed(self):
        original=request();original['evidence_input']['sdk_projection']='terminal_history_catalog_v2';original['readable_evidence']['views'][0]['event_index'][0]['new_fact']='keep'
        with self.assertRaisesRegex(ValueError,'complete known index schema'):_sdk_projection(original)

if __name__=='__main__':unittest.main()
