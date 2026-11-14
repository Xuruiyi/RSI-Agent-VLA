"""Observable relation learning, information boundaries and offline inheritance."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from rsi_vla.contracts import ExecutionEvent, ExperimentResult, RunBundle
from rsi_vla.evidence import EventLedger
from rsi_vla.relation_learning import freeze_relations, learn_relations, version_signature
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchLLMAdapter, ResearchTools
from rsi_vla.storage import ResearchStore, initialize


class RelationLearningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = ResearchStore(self.root / "source")
        self.derived = ResearchStore(self.root / "derived")
        self.base = RunBundle("base", "theta", "processor", "phi", "rho", "psi", "memory",
                              "protocol", "backend", "code")
        initialize(self.source, self.base, psi={"learned": False},
                   memory={"provenance": "fixture", "legal_information": ["development"]})
        self.protocol = {"protocol_id": "protocol", "benchmark_id": "bench", "scope": "offline",
                         "task_sets": {"development": {"task_keys": ["task"], "seeds": [1]}}}
        self.tools = self.make_tools(self.source, self.base, "job")
        signature = version_signature(self.base)
        events = [ExecutionEvent("e0", "run", "episode", 0, "base", "observation", "fixture",
                                 media=["a.png", "a.depth.npz"]),
                  ExecutionEvent("e1", "run", "episode", 1, "base", "observation_update", "fixture",
                                 media=["b.png", "b.depth.npz"])]
        for index, worker, success in ((2, "worker-a", False), (3, "worker-b", True)):
            events.append(ExecutionEvent("e" + str(index), "run", "episode", index, "base",
                "handoff_review", "fixture-independent-evaluator", flags={"route_feedback": {
                    "evidence_kind": "independent_criterion", "criterion_ref": "independent-handoff-v1",
                    "state": "task", "worker": worker, "success": success,
                    "version_signature": signature, "start_event_id": "e0", "handoff_event_id": "e1"}}))
        ledger = EventLedger(self.source)
        for event in events:
            ledger.append(event)
        self.source.put("evidence_views", "view", ledger.view("view", "base"))
        self.result = ExperimentResult("result", "base", "protocol", "bench", "development", "offline",
            {"information_scope": "development", "evidence_view_id": "view", "runtime_loaded": True,
             "identity": {"run_id": "run", "episode_id": "episode"},
             "screening_checks": {"technical_valid": True, "native_success": False}},
            ["fixture-only"], "screening", {}, {}, {}, assigned_count=1, outcome="failure",
            assignments=[{"task_key": "task", "seed": 1, "outcome": "failure", "validity": "valid"}])
        self.source.save_development_result(self.result)

    def make_tools(self, store, bundle, job_id):
        job = ResearchJob(store, job_id, bundle.bundle_id, {"input_tokens": 1000, "output_tokens": 100})
        return ResearchTools(store, job, self.protocol,
            domain_policy={"historical_development_bundle_ids": ["base"]})

    def learn_and_freeze(self):
        state = learn_relations(self.tools, self.derived, "learning", ["result"])
        target = freeze_relations(self.tools, self.derived, "learning", rho_ref="rho-next", bundle_id="next")
        return state, target

    def inherited_tools(self, target):
        for kind, key in (("development_results", "result"), ("evidence_views", "view")):
            self.derived.put(kind, key, self.source.get(kind, key))
        return self.make_tools(self.derived, target, "next-job")

    def test_each_relation_changes_observable_consumer_without_activation(self):
        before = self.source.get("rho", "rho")
        state, target = self.learn_and_freeze()
        tools = self.inherited_tools(target)
        step = tools.prepare_relation_step(["result"], requested_worker="worker-a",
            allowed_workers=["worker-a", "worker-b"], state="task", inspected=True,
            legal_repairs=["phi"], experiment_permitted=True)
        self.assertEqual(step["worker"], "worker-b")
        self.assertEqual(step["research_step"], "defer")
        self.assertEqual(len(step["evidence_views"][0]["selected_media"]), 2)
        self.assertEqual(step["evidence_views"][0]["l0"], self.source.get("evidence_views", "view")["l0"])
        self.assertEqual(self.source.get("rho", "rho"), before)
        self.assertEqual(target.phi_ref, self.base.phi_ref)
        self.assertEqual(target.psi_ref, self.base.psi_ref)
        self.assertFalse((self.derived.root / "active.json").exists())
        self.assertEqual({f["relation"] for f in state["feedback"]}, {"rhoA", "rhoB", "rhoC"})
        self.assertEqual(tools.job.state["used"], {"input_tokens": 0, "output_tokens": 0})

    def test_learned_rhoA_reaches_actual_request_builder_and_ledger(self):
        _, target = self.learn_and_freeze()
        tools = self.inherited_tools(target)
        request = ResearchLLMAdapter().prepare_request(tools, request_id="offline-request",
            result_ids=["result"], max_input_tokens=100, max_output_tokens=10)
        self.assertEqual(len(request["evidence_views"][0]["selected_media"]), 2)
        event = ExecutionEvent("next-event", "run2", "ep2", 0, target.bundle_id,
                               "observation", "fixture", media=["first.png", "second.png"])
        ledger = EventLedger(self.derived)
        ledger.append(event)
        view = ledger.view("next-view", target.bundle_id)
        self.assertEqual(view.selected_media, ["first.png"])
        self.assertEqual(view.policy_ref, "rho-next")
        self.assertEqual(len(view.l0), 1)

    def test_permissions_version_and_budget_survive_learning(self):
        _, target = self.learn_and_freeze()
        tools = self.inherited_tools(target)
        args = dict(state="task", inspected=True, legal_repairs=["phi"], experiment_permitted=True)
        step = tools.prepare_relation_step(["result"], requested_worker="worker-a",
                                          allowed_workers=["worker-a"], **args)
        self.assertEqual(step["worker"], "worker-a")
        changed = replace(target, bundle_id="changed-code", code_ref="other-code")
        self.derived.save_bundle(changed)
        incompatible = self.make_tools(self.derived, changed, "changed-job")
        step = incompatible.prepare_relation_step(["result"], requested_worker="worker-a",
            allowed_workers=["worker-a", "worker-b"], **args)
        self.assertEqual(step["worker"], "worker-a")
        step = tools.prepare_relation_step(["result"], requested_worker="not-authorized",
            allowed_workers=[], inspected=True, legal_repairs=["phi"], experiment_permitted=False)
        self.assertEqual((step["worker"], step["research_step"]), ("defer", "defer"))
        tools.job.consume({"input_tokens": 1000}, "fixture budget exhaustion")
        self.assertEqual(tools.prepare_relation_step(["result"], requested_worker="worker-a",
            allowed_workers=["worker-a"], **args)["research_step"], "defer")

    def test_no_worker_supervision_from_native_terminal_outcome(self):
        view = self.source.get("evidence_views", "view")
        for event in view["l0"]:
            event["flags"] = {}
        self.source.put("evidence_views", "no-routes", {**view, "view_id": "no-routes"})
        result = replace(self.result, result_id="no-routes", outcome="success",
                         intervention={**self.result.intervention, "evidence_view_id": "no-routes"})
        self.source.save_development_result(result)
        state = learn_relations(self.tools, self.derived, "no-routes", ["no-routes"])
        self.assertEqual(state["snapshot"]["rhoB"]["values"], {})
        self.assertTrue(any(row.get("relation") == "rhoB" for row in state["missing"]))

    def test_replay_is_idempotent_and_inheritance_preserves_denominator(self):
        state = learn_relations(self.tools, self.derived, "learning", ["result"])
        replay = learn_relations(self.tools, self.derived, "learning", ["result"])
        self.assertEqual(state, replay)
        target = freeze_relations(self.tools, self.derived, "learning", rho_ref="rho-next", bundle_id="next")
        tools = self.inherited_tools(target)
        third = ResearchStore(self.root / "third")
        inherited = learn_relations(tools, third, "second-round", ["result"])
        self.assertEqual(inherited["feedback"], state["feedback"])
        self.assertEqual(inherited["snapshot"]["rhoA"]["values"], state["snapshot"]["rhoA"]["values"])
        self.assertEqual(freeze_relations(self.tools, self.derived, "learning",
            rho_ref="rho-next", bundle_id="next"), target)
        with self.assertRaisesRegex(ValueError, "frozen"):
            learn_relations(self.tools, self.derived, "learning", ["result"])
        with self.assertRaisesRegex(ValueError, "identity"):
            freeze_relations(self.tools, self.derived, "learning", rho_ref="changed", bundle_id="changed")

    def test_feedback_bound_and_formal_or_cross_execution_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "bound exhausted"):
            learn_relations(self.tools, self.derived, "small", ["result"], max_feedback=1)
        self.assertFalse(self.derived.path("relation_learning", "small").exists())
        for scope in ("release", "sealed"):
            bad = replace(self.result, result_id=scope,
                intervention={**self.result.intervention, "information_scope": scope})
            self.source.save_development_result(bad)
            with self.assertRaisesRegex(ValueError, "development"):
                learn_relations(self.tools, self.derived, scope, [scope])
        with self.assertRaisesRegex(ValueError, "separate"):
            learn_relations(self.tools, self.source, "same-store", ["result"])
        view = self.source.get("evidence_views", "view")
        view["l0"][0]["run_id"] = "other-execution"
        self.source.put("evidence_views", "cross-execution", {**view, "view_id": "cross-execution"})
        self.source.save_development_result(replace(self.result, result_id="cross-execution",
            intervention={**self.result.intervention, "evidence_view_id": "cross-execution"}))
        with self.assertRaisesRegex(ValueError, "crosses"):
            learn_relations(self.tools, self.derived, "cross-execution", ["cross-execution"])

    def test_mismatched_route_feedback_version_is_rejected(self):
        view = self.source.get("evidence_views", "view")
        view["l0"][2]["flags"]["route_feedback"]["version_signature"]["phi_ref"] = "unobserved-phi"
        self.source.put("evidence_views", "wrong-version", {**view, "view_id": "wrong-version"})
        self.source.save_development_result(replace(self.result, result_id="wrong-version",
            intervention={**self.result.intervention, "evidence_view_id": "wrong-version"}))
        with self.assertRaisesRegex(ValueError, "matching version"):
            learn_relations(self.tools, self.derived, "wrong-version", ["wrong-version"])


if __name__ == "__main__":
    unittest.main()
