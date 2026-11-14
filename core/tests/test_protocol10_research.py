"""Protocol10 response selection and multi-slot Candidate evidence stay immutable."""

import copy
import json
import unittest
from dataclasses import replace

import test_research_tools
from rsi_vla.improvement_protocol import load_frozen_protocol, runtime_projection
from rsi_vla.research_cycle import _phase_charge, reconcile_trial
from rsi_vla.research_tools import Protocol10ResearchAdapter
from rsi_vla.storage import atomic_json


class Protocol10ResearchTest(unittest.TestCase):
    setUp = test_research_tools.ResearchToolsTest.setUp
    result = test_research_tools.ResearchToolsTest.result

    def prepare(self):
        self.tools.domain_policy["rule_operators"] = ["chunk_steps"]
        self.adapter = Protocol10ResearchAdapter()
        self.adapter.prepare_request(
            self.tools,
            request_id="p10-request",
            result_ids=["base-result"],
            max_input_tokens=100,
            max_output_tokens=10,
        )
        return {
            "schema": "rsi-bounded-research-response-v2",
            "request_id": "p10-request",
            "decision": "prepare_rule_trial",
            "source": "normal",
            "result_ids": ["base-result"],
            "reason": "reduce observed overhead",
            "external_value": "preserve success with fewer actions",
            "competing_explanations": self.explanations,
            "intervention": {"chunk_steps": 2},
            "distinguishing_prediction": "registered matched timing fixture",
            "stop_condition": "one candidate and two fixed development slots",
            "missing_evidence": ["matched timing evidence"],
            "development_slots": [
                {"task_key": "base_salad", "seed": 2},
                {"task_key": "old_milk", "seed": 2},
            ],
        }

    def apply(self, response):
        return self.adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={"offline_calls": 1},
            synthetic=True,
        )

    def test_selection_before_trial_and_equal_replay_conflict(self):
        response = self.prepare()
        trial = self.apply(response)
        self.assertEqual(trial["bundle_id"], "candidate-protocol10-01")
        selection = self.store.get("protocol10_selections", "p10-request")
        self.assertEqual(selection["response"], response)
        self.assertEqual(selection["development_slots"], response["development_slots"])
        self.assertEqual(self.apply(response), trial)
        self.assertEqual(self.job.state["charges"], [])
        changed = copy.deepcopy(response)
        changed["development_slots"].reverse()
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.apply(changed)
        self.assertEqual(len(self.store.all("rule_trials")), 1)

    def test_illegal_slots_hint_or_timing_do_not_generate_trial(self):
        response = self.prepare()
        for slots in (
            [],
            [{"task_key": "base_salad", "seed": 0}],
            [{"task_key": "base_salad", "seed": 2}] * 2,
            [{"task_key": "arbitrary", "seed": 2}],
        ):
            with self.subTest(slots=slots):
                bad = {**response, "development_slots": slots}
                with self.assertRaises(ValueError):
                    self.apply(bad)
        with self.assertRaisesRegex(ValueError, "rejected hint"):
            self.apply(
                {**response, "intervention": {"planner_segment_handoff_hint": True}}
            )
        with self.assertRaisesRegex(ValueError, "matched baseline"):
            self.adapter.apply_response(
                self.tools,
                response,
                evidence_level="simulation",
                cost_limit={},
                synthetic=False,
            )
        self.assertEqual(self.store.all("rule_trials"), [])

    def test_stop_has_no_candidate_or_development(self):
        response = self.prepare()
        response.update(decision="defer", intervention=None, development_slots=[])
        result = self.apply(response)
        self.assertEqual(result["action"], "defer")
        self.assertEqual(self.store.all("rule_trials"), [])
        self.assertEqual(self.store.all("candidates"), [])

    def test_all_registered_findings_and_failure_costs_enter_one_candidate(self):
        response = self.prepare()
        trial = self.apply(response)
        experiment_id = "p10-request-experiment"
        extra_id = experiment_id + "-slot02"
        for kind in ("research_experiments", "rule_trials"):
            original = self.tools._owned(kind, experiment_id)
            self.tools._save(kind, extra_id, {**original, "experiment_id": extra_id})
        for index, key in enumerate((experiment_id, extra_id)):
            result = replace(
                self.result(
                    "development-" + str(index),
                    trial["bundle_id"],
                    outcome="failure" if index == 0 else "success",
                    seed=index + 1,
                    chunk_steps=2,
                ),
                criteria_ref=self.protocol["protocol_id"],
            )
            self.tools.record_experiment_result(key, result)
        candidate = self.tools.submit_candidate(
            "candidate-protocol10-01",
            experiment_id=experiment_id,
            additional_experiment_ids=[extra_id],
        )
        self.assertEqual(
            candidate["development_results"], ["development-0", "development-1"]
        )
        self.assertEqual(candidate["costs"]["offline_calls"], 2)
        self.assertEqual(len(self.job.state["charges"]), 2)
        self.assertEqual(
            self.tools.submit_candidate(
                "candidate-protocol10-01",
                experiment_id=experiment_id,
                additional_experiment_ids=[extra_id],
            ),
            candidate,
        )
        self.assertEqual(len(self.job.state["charges"]), 2)
        with self.assertRaisesRegex(ValueError, "same candidate"):
            self.tools.submit_candidate(
                "bad",
                experiment_id=experiment_id,
                additional_experiment_ids=[experiment_id],
            )
        self.assertEqual(len(self.store.all("candidates")), 1)

    def test_returned_development_costs_survive_collector_error(self):
        response = self.prepare()
        costs = {
            "whole_job_wall_seconds": 1,
            "gpu_device_seconds_envelope": 1,
            "model_requests": 1,
            "tool_calls": 2,
            "environment_steps": 30,
            "artifact_bytes": 10000,
        }
        for key in costs:
            self.job.state["limits"][key] = 1000000
            self.job.state["used"][key] = 0
        self.job._save()
        trial = self.adapter.apply_response(
            self.tools,
            response,
            evidence_level="offline",
            cost_limit={**dict.fromkeys(costs, 1000000), "offline_calls": 1},
            synthetic=True,
        )
        experiment = "p10-request-experiment"
        output = self.root / "returned"
        output.mkdir()
        context = {
            "run_id": "returned-runtime",
            "episode_id": "e",
            "bundle_id": trial["bundle_id"],
            "protocol_id": self.protocol["protocol_id"],
        }
        atomic_json(output / "runtime_context.json", context)
        self.tools._save(
            "trial_dispatches",
            experiment + "-dispatch",
            {
                "experiment_id": experiment,
                "output": str(output),
                "status": "dispatched",
            },
        )
        phase = self.root / "phase.json"
        caps = {
            "wall_seconds": 100,
            "gpu_seconds": 100,
            "model_requests": 100,
            "tool_calls": 100,
            "environment_steps": 1000,
            "storage_bytes": 1000000,
            "simulation_episodes": 10,
        }
        atomic_json(
            phase, {"limits": caps, "used": dict.fromkeys(caps, 0), "remaining": caps}
        )
        permit = self.root / "permit.json"
        atomic_json(
            permit,
            {
                "experiment_id": experiment,
                "job_id": "job",
                "run_id": context["run_id"],
                "output_dir": str(output),
                "phase_snapshot_ref": str(phase),
            },
        )

        def observer(*args, **kwargs):
            return {"identity": context, "costs": costs}

        def broken(*args):
            raise ValueError("repair development parser")

        with self.assertRaisesRegex(ValueError, "repair development parser"):
            reconcile_trial(
                self.tools,
                experiment,
                output=output,
                permit_path=permit,
                collector=broken,
                auto_submit=False,
                settle_before_collect=True,
                cost_observer=observer,
            )
        before = json.loads(phase.read_text())
        self.assertEqual(before["used"]["simulation_episodes"], 1)
        self.assertEqual(before["used"]["model_requests"], 1)

        def repaired(*args):
            result = replace(
                self.result("repaired-result", trial["bundle_id"], chunk_steps=2),
                criteria_ref=self.protocol["protocol_id"],
            )
            return result, {"technical_valid": True}

        cycle = reconcile_trial(
            self.tools,
            experiment,
            output=output,
            permit_path=permit,
            collector=repaired,
            auto_submit=False,
            settle_before_collect=True,
            cost_observer=lambda *a, **kw: self.fail("measurement must be reused"),
        )
        self.assertEqual(
            cycle["action"], "development_complete_for_registered_candidate"
        )
        self.assertIsNone(cycle["candidate"])
        self.assertEqual(self.store.all("candidates"), [])
        after = json.loads(phase.read_text())
        self.assertEqual(after["used"]["model_requests"], 1)
        self.assertEqual(len(after["development_trial_charges"]), 1)

    def test_missing_cost_reconciles_only_new_measurements_once(self):
        phase = self.root / "phase.json"
        caps = {
            "wall_seconds": 100,
            "gpu_seconds": 100,
            "model_requests": 100,
            "tool_calls": 100,
            "environment_steps": 1000,
            "storage_bytes": 1000000,
            "simulation_episodes": 10,
        }
        atomic_json(
            phase, {"limits": caps, "used": dict.fromkeys(caps, 0), "remaining": caps}
        )
        permit = {"phase_snapshot_ref": str(phase), "permit_ref": "same-permit"}
        costs = {
            "whole_job_wall_seconds": 2,
            "gpu_device_seconds_envelope": 2,
            "artifact_bytes": 3,
            "model_requests": 1,
            "tool_calls": 1,
        }
        _phase_charge(permit, "r", costs, "same-output")
        self.assertTrue(json.loads(phase.read_text())["unresolved_costs"])
        complete = {**costs, "environment_steps": 30}
        _phase_charge(permit, "r", complete, "same-output")
        _phase_charge(permit, "r", complete, "same-output")
        ledger = json.loads(phase.read_text())
        self.assertEqual(ledger["unresolved_costs"], [])
        self.assertEqual(ledger["used"]["simulation_episodes"], 1)
        self.assertEqual(ledger["used"]["environment_steps"], 30)
        self.assertEqual(len(ledger["trial_cost_reconciliations"]), 1)
        self.assertNotIn(
            "environment_steps",
            ledger["trial_cost_reconciliations"][0]["previous_costs"],
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _phase_charge(permit, "r", {**complete, "model_requests": 2}, "same-output")

    def test_runtime_projection_consumes_only_explicit_split_authority(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        frozen = root / "config/pilot_5a_protocol10.json"
        protocol = load_frozen_protocol(frozen, frozen)
        authority = self.root / "synthetic-authority.json"
        value = {
            "authorized": True,
            "scope": protocol["scope_id"],
            "protocol_id": protocol["protocol_id"],
            "frozen_protocol_ref": str(frozen.resolve()),
            "allowed_runtime_splits": ["release", "development"],
        }
        atomic_json(authority, value)
        projected = runtime_projection(
            protocol, frozen_ref=frozen, authorization_ref=authority
        )
        self.assertEqual(
            projected["allowed_runtime_splits"], ["release", "development"]
        )
        self.assertEqual(projected["task_sets"]["development"]["seeds"], [2])
        self.assertFalse(protocol["dynamic_authorized"])
        atomic_json(
            authority, {**value, "allowed_runtime_splits": ["release", "sealed"]}
        )
        with self.assertRaisesRegex(ValueError, "administrative"):
            runtime_projection(protocol, frozen_ref=frozen, authorization_ref=authority)


if __name__ == "__main__":
    unittest.main()
