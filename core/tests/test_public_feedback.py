"""Research reads only fixed evaluator publications and repeats within its Job."""

import unittest
from copy import deepcopy
from dataclasses import replace

import test_research_tools
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchToolkit, ResearchTools
from rsi_vla.storage import atomic_json
from test_objective_evaluation import fixture


class PublicFeedbackConsumerTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def test_published_aggregate_is_consumed_without_reading_formal_result(self):
        _, _, result = fixture()
        self.protocol.update(status="frozen", feedback_policy={"release": "aggregate"})
        evaluations = EvaluationStore(self.root / "evaluation")
        native = replace(
            result(True, (True, True)), protocol_id=self.protocol["protocol_id"]
        )
        native.intervention["information_scope"] = "release"
        evaluations.save_result(native)
        published = evaluations.public_feedback(
            native.result_id, protocol=self.protocol
        )
        tools = ResearchTools(
            self.store,
            self.job,
            self.protocol,
            domain_policy=self.tools.domain_policy,
            public_feedback_root=evaluations.root / "public_feedback",
        )
        # The formal file can be completely hidden. Public query follows no ref
        # into it, and the toolkit calls the actual publication consumer.
        evaluations.path("results", native.result_id).unlink()
        feedback = ResearchToolkit(tools).call(
            "query_public_feedback", {"result_ids": [native.result_id]}
        )
        self.assertEqual(feedback, [published["feedback"]])
        self.assertNotIn("assignments", feedback[0]["feedback"])
        value = deepcopy(published["feedback"])
        value["feedback"]["data_refs"] = ["private"]
        atomic_json(evaluations.path("public_feedback", native.result_id), value)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            tools.query_public_feedback([native.result_id])
        with self.assertRaises(ValueError):
            tools.query_public_feedback(["../results/native"])

    def test_sealed_details_and_research_owned_fake_publication_are_refused(self):
        self.protocol["feedback_policy"] = {"meta_exam": "aggregate"}
        public = self.root / "evaluation/public_feedback"
        public.mkdir(parents=True)
        value = {
            "schema": "rsi-public-evaluation-feedback-v1",
            "result_id": "sealed",
            "protocol_id": self.protocol["protocol_id"],
            "bundle_id": "bound",
            "evaluator_source": {
                "role": "evaluation",
                "result_id": "sealed",
                "protocol_id": self.protocol["protocol_id"],
            },
            "feedback": {
                "schema": "rsi-evaluation-feedback-v1",
                "information_scope": "meta_exam",
                "status": "withheld",
            },
        }
        atomic_json(public / "sealed.json", value)
        tools = ResearchTools(
            self.store,
            self.job,
            self.protocol,
            domain_policy=self.tools.domain_policy,
            public_feedback_root=public,
        )
        self.assertEqual(
            tools.query_public_feedback(["sealed"])[0]["feedback"]["status"], "withheld"
        )
        value["feedback"]["metrics"] = {"hidden_score": 1}
        atomic_json(public / "sealed.json", value)
        with self.assertRaisesRegex(ValueError, "withheld"):
            tools.query_public_feedback(["sealed"])
        tools.public_feedback_root = self.store.root / "public_feedback"
        with self.assertRaises(PermissionError):
            tools.query_public_feedback(["sealed"])


class IndependentJobRepetitionTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result
    opportunity = test_research_tools.ResearchToolsTest.opportunity

    def test_default_development_query_includes_own_trials_and_excludes_other_jobs(
        self,
    ):
        trial = test_research_tools.ResearchToolsTest.prepare(self)
        self.store.save_development_result(self.result("own-trial", trial["bundle_id"]))
        self.assertEqual(
            self.tools.query_development(["own-trial"])[0]["bundle_id"],
            trial["bundle_id"],
        )
        job = ResearchJob(
            self.store, "other-job", self.base.bundle_id, {"offline_calls": 10}
        )
        tools = ResearchTools(
            self.store, job, self.protocol, domain_policy=self.tools.domain_policy
        )
        with self.assertRaisesRegex(ValueError, "matching development"):
            tools.query_development(["own-trial"])
        with self.assertRaisesRegex(ValueError, "outside this job"):
            tools.query_development(["own-trial"], bundle_ids=[trial["bundle_id"]])

    def test_registered_jobs_repeat_same_facts_but_one_job_cannot_reprocess(self):
        self.protocol["task_policy"] = {"schema": "rsi-task-policy-v1"}
        self.opportunity()

        def plan(tools, name, opportunity):
            return tools.plan_rule_experiment(
                name,
                opportunity_id=opportunity,
                bundle_id=name + "-bundle",
                chunk_steps=2,
                criterion_ref="criterion",
                distinguishing_prediction="same registered comparison",
                cost_limit={"offline_calls": 1},
                evidence_level="offline",
            )

        plan(self.tools, "first", "opportunity")
        self.store.put("research_findings", "first", {"experiment_id": "first"})
        job = ResearchJob(
            self.store, "independent-repeat", self.base.bundle_id, {"offline_calls": 10}
        )
        tools = ResearchTools(
            self.store, job, self.protocol, domain_policy=self.tools.domain_policy
        )
        tools.discover_opportunity(
            "second-opportunity",
            source="normal",
            result_ids=["base-result"],
            external_value="repeat registered process",
            competing_explanations=self.explanations,
            expected_cost={"offline_calls": 1},
        )
        self.assertEqual(
            plan(tools, "second", "second-opportunity")["job_id"], "independent-repeat"
        )
        self.store.put("research_findings", "second", {"experiment_id": "second"})
        with self.assertRaisesRegex(ValueError, "same processed evidence"):
            plan(tools, "third", "second-opportunity")


if __name__ == "__main__":
    unittest.main()
