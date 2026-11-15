"""Research feedback routing reads materialized stores and actual closure files."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import test_research_tools
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskCompletionBackend


class ResearchFeedbackBoundaryTests(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def test_routes_follow_materialized_artifacts_and_preserve_unknown_feedback(self):
        self.protocol.update(
            status="frozen",
            anchor_bundle_id=self.base.bundle_id,
            metrics={"success": {"direction": "max", "min": 0, "max": 1, "weight": 1}},
            hard_constraints=["resource_limits"],
            min_gain=0.1,
            anchor_max_drop=0,
            task_policy={"schema": "rsi-task-policy-v1"},
        )
        protocol_ref, policy_ref = (
            self.root / "protocol.json",
            self.root / "domain-policy.json",
        )
        atomic_json(protocol_ref, self.protocol)
        atomic_json(policy_ref, self.tools.domain_policy)
        config = {
            "schema": "rsi-task-completion-v1",
            "protocol_ref": str(protocol_ref),
            "base_bundle_id": self.base.bundle_id,
            "research_store_ref": str(self.store.root),
            "evaluation_store_ref": str(self.root / "evaluation"),
            "research_requests": [],
            "operations": [],
        }
        backend = TaskCompletionBackend(config)
        costs = {
            "model_requests": 1,
            "wall_seconds": 2.5,
            "input_tokens": 13,
            "output_tokens": 2,
        }
        expected = {
            "defer": "defer",
            "trial": "development",
            "supplement": "prepare_evidence_followup",
            "empty_supplement": "defer",
            "sdk_incomplete": "defer",
            "cleanup_missing": "defer",
            "development": "experiment_completed",
        }
        original_recorder = ResearchTools.record_contextual_research_feedback
        # Spy only on the final feedback recorder, retaining its actual store
        # derivation. Tools, files, operations and successor persistence are real.
        with patch.object(
            ResearchTools,
            "record_contextual_research_feedback",
            autospec=True,
            side_effect=original_recorder,
        ) as recorder:
            for case, next_action in expected.items():
                with self.subTest(case=case):
                    request_id, feedback_id = "request-" + case, "feedback-" + case
                    output = self.root / request_id
                    cfg = {
                        "request_id": request_id,
                        "job_id": "job",
                        "output_ref": str(output),
                        "domain_policy_ref": str(policy_ref),
                    }
                    config["research_requests"].append(cfg)
                    payload = {"request_id": request_id, "feedback_id": feedback_id}
                    self.tools._save(
                        "research_relation_selections",
                        request_id,
                        {
                            "action": "prepare",
                            "source_result_ids": ["base-result"],
                            "states": ["salad"],
                            "legal_repairs": ["planner_finish_reserve_calls"],
                            "rho_ref": self.base.rho_ref,
                            "decisions": [
                                {
                                    "state": "salad",
                                    "context": {
                                        "operator": "planner_finish_reserve_calls"
                                    },
                                }
                            ],
                        },
                    )
                    self.tools._save(
                        "llm_executions",
                        request_id,
                        {
                            "request_id": request_id,
                            "transport_facts": {
                                "effective_model": "gpt-6.1-sol",
                                "effective_reasoning_effort": "high",
                            },
                            "costs": costs,
                            # A model's claimed route does not materialize a trial.
                            "response": {
                                "next_action": "development",
                                "decision": "prepare_rule_trial",
                            },
                        },
                    )
                    atomic_json(
                        output / "supervisor.json",
                        {
                            "status": "completed",
                            "exit_code": 0,
                            "cleanup_confirmed": case != "cleanup_missing",
                        },
                    )
                    atomic_json(
                        output / "sdk-result.json",
                        {
                            "status": "TurnStatus.incomplete"
                            if case == "sdk_incomplete"
                            else "TurnStatus.completed"
                        },
                    )
                    (output / "model_requests.jsonl").write_text("")
                    if case == "trial":
                        self.tools._save(
                            "rule_trials",
                            request_id + "-experiment",
                            {
                                "experiment_id": request_id + "-experiment",
                                "bundle_id": self.base.bundle_id,
                            },
                        )
                    elif case in {"supplement", "empty_supplement"}:
                        source = self.root / (case + "-event.json")
                        atomic_json(
                            source,
                            {
                                "event_id": "development-event",
                                "information_scope": "development",
                            },
                        )
                        self.tools._save(
                            "research_evidence_supplements",
                            request_id,
                            {
                                "reads": [{"source_ref": str(source)}]
                                if case == "supplement"
                                else []
                            },
                        )
                    elif case == "development":
                        result_id = "new-development-result"
                        self.store.save_development_result(
                            replace(
                                self.result(
                                    result_id,
                                    self.base.bundle_id,
                                    outcome="failure",
                                    costs={"model_requests": 3},
                                ),
                                intervention={
                                    "information_scope": "development",
                                    "planner_finish_reserve_calls": 1,
                                    "screening_checks": {"bounded_trial": False},
                                },
                            )
                        )
                        self.tools._save(
                            "research_experiments",
                            request_id + "-experiment",
                            {
                                "experiment_id": request_id + "-experiment",
                                "bundle_id": self.base.bundle_id,
                            },
                        )
                        payload["result_id"] = result_id
                    op = {
                        "operation_id": feedback_id,
                        "kind": "research_feedback",
                        "payload": payload,
                    }
                    config["operations"].append(op)
                    response = backend.execute(op, {})
                    self.assertEqual(response["next_action"], next_action)
                    saved = self.store.get("research_feedback_routes", feedback_id)
                    self.assertEqual(
                        saved,
                        {
                            "request_id": request_id,
                            "feedback_id": feedback_id,
                            "next_action": next_action,
                            "result_id": payload.get("result_id"),
                        },
                    )
                    call = recorder.call_args.kwargs
                    self.assertEqual(call["selection_id"], request_id)
                    self.assertTrue(
                        all(Path(ref).is_file() for ref in call["source_refs"])
                    )
                    feedback = response["feedback"]
                    self.assertIsNone(feedback["task_gain"])
                    self.assertIsNone(feedback["old_capability"])
                    if case == "development":
                        self.assertIsNone(call["technical_valid"])
                        self.assertIsNone(call["costs"])
                        self.assertIs(feedback["technical_valid"], True)
                        self.assertIs(feedback["screening_pass"], False)
                        self.assertEqual(feedback["costs"], {"model_requests": 3})
                    else:
                        self.assertEqual(call["costs"], costs)
                        self.assertEqual(feedback["costs"], costs)
                        self.assertIs(
                            feedback["technical_valid"],
                            case not in {"sdk_incomplete", "cleanup_missing"},
                        )
                        self.assertIsNone(feedback["screening_pass"])
                    if next_action == "defer":
                        self.assertFalse(
                            self.store.path(
                                "rule_trials", request_id + "-experiment"
                            ).exists()
                        )
            self.assertEqual(recorder.call_count, len(expected))


if __name__ == "__main__":
    unittest.main()
