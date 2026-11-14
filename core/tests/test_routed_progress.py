"""Actual routed pick control-chain audits preserve raw Planner request identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.critical_progress import audit_skill_routes, inspect_progress
from test_cooperation_runtime import CRITERION, ROUTE, SIGNATURE


class RoutedProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(run_id="r", episode_id="e", bundle_id="b", protocol_id="p")
        source_root = Path(__file__).resolve().parents[2] / "RPent/robots/libero"
        target_root = self.root / "code_snapshot/RPent/robots/libero"
        target_root.mkdir(parents=True)
        for name in ("tools.py", "toolkit.py", "skill_routing.py", "env_client.py", "env_server.py", "action_spec.py"):
            (target_root / name).write_bytes((source_root / name).read_bytes())
        route = {**copy.deepcopy(ROUTE), "selected_worker": "pi0_pick"}
        self.write("runtime_context.json", {**self.identity, **SIGNATURE,
            "relation_control": {"schema": "rsi-skill-routing-v1", "rho_ref": "rho", "version_signature": SIGNATURE, "routes": [route]}})
        binding = dict(rho_ref="rho", route_id=ROUTE["route_id"], state=ROUTE["state"],
                       requested_worker="vla_execute", trigger_worker="vla_execute",
                       selected_worker="pi0_pick", criterion_ref=CRITERION["criterion_ref"])
        actual = {"prompt": ROUTE["instruction"], "max_chunks": 2}
        actions = [[0.] * 7 for _ in range(5)]
        raw = [{"kind": "run_start"},
               {"kind": "relation_policy_loaded", "relation": "rhoB", "rho_ref": "rho"},
               {"kind": "observation_update"}, {"kind": "tool_call", "tool": "vla_execute"},
               {"kind": "relation_route_decision", **binding},
               {"kind": "skill_route_start", **binding, "client_route_sequence": 4, "actual_arguments": actual,
                "environment_observation": {"source_sequence": 0}},
               {"kind": "inference_start", "source": "vla", "instruction": ROUTE["instruction"],
                "segment_sequence": None, "observation_sequence": 2, "observation_mode": "latest"},
               {"kind": "inference_end", "inference_sequence": 6, "completed": True},
               {"kind": "action_proposal", "source": "vla", "proposal": actions,
                "inference_sequence": 6, "segment_sequence": None, "instruction": ROUTE["instruction"]},
               {"kind": "action_send", "source": "vla", "sent": actions,
                "proposal_sequence": 8, "proposal_indices": list(range(5))},
               {"kind": "chunk_execution_end", "inference_sequence": 6, "predictions_started": 1,
                "predictions_completed": 1, "unknown_steps": 0, "executed_steps": 5},
               {"kind": "observation_update"},
               {"kind": "skill_route_handoff", **binding, "client_route_sequence": 4, "actual_arguments": actual,
                "environment_observation": {"source_sequence": 1}, "chunks_used": 1, "execution_error": None},
               {"kind": "tool_end", "tool": "vla_execute", "status": "completed"}]
        self.events = [{**self.identity, "sequence": i, **event} for i, event in enumerate(raw)]
        self.states = {"steps": [{"command": {"action": "vla_execute", "instruction": ROUTE["instruction"], "max_chunks": 2},
                                  "result": {"relation_route": {**binding, "decision_sequence": 4}}}]}
        private = [{**self.identity, "sequence": i, "kind": "skill_route_observation", "phase": phase,
                    "route_id": ROUTE["route_id"], "worker": "pi0_pick", "client_route_sequence": 4,
                    "criterion_ref": CRITERION["criterion_ref"], "version_signature": SIGNATURE}
                   for i, phase in enumerate(("start", "handoff"))]
        self.write("environment_service/action_events.jsonl", private, lines=True)

    def write(self, name, value, lines=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in value) if lines else json.dumps(value))

    def test_routed_pick_reuses_control_mapping_without_fabricating_vla_segment(self):
        self.write("states.json", self.states)
        self.write("action_events.jsonl", self.events, lines=True)
        checks = {name: True for name in ("ordered_l0", "execution_health", "action_mapping",
                  "setup_coverage", "no_mid_attempt_reset", "vla_actual_loading_and_prediction")}
        with patch("rsi_vla.adapters.runtime_acceptance.inspect_attempt", return_value={"checks": checks}):
            report = inspect_progress(self.root, self.identity)
        self.assertIs(report["value"], True)
        self.assertTrue(report["checks"]["proposal_to_native_send"])
        self.assertEqual(self.states["steps"][0]["command"]["action"], "vla_execute")
        self.assertFalse(any(event["kind"] == "vla_segment_start" for event in self.events))
        self.assertNotIn("holding_or_settled", report["not_applicable"])

    def test_argument_source_handoff_and_completed_inference_gaps_stay_unknown(self):
        changes = [(5, "actual_arguments", {"prompt": "place another object", "max_chunks": 2}),
                   (12, "environment_observation", {"source_sequence": 0}),
                   (7, "completed", False), (10, "unknown_steps", 5),
                   (4, "selected_worker", "vla_execute")]
        for index, field, value in changes:
            with self.subTest(field=field):
                events = copy.deepcopy(self.events)
                events[index][field] = value
                self.assertIsNone(audit_skill_routes(self.root, self.identity, events, self.states)["value"])
        (self.root / "code_snapshot/RPent/robots/libero/skill_routing.py").unlink()
        self.assertIsNone(audit_skill_routes(self.root, self.identity, self.events, self.states)["value"])

    def test_foreign_none_segment_inference_cannot_use_a_pick_exception(self):
        self.write("states.json", self.states)
        events = copy.deepcopy(self.events)
        events[6]["instruction"] = "unregistered instruction"
        self.write("action_events.jsonl", events, lines=True)
        report = inspect_progress(self.root, self.identity)
        self.assertIsNone(report["value"])
        self.assertIn("goal binding", " ".join(report["missing"]))
