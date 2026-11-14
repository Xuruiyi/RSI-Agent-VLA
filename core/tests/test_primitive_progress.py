"""Primitive progress uses current observations and exact independently saved facts."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rsi_vla.primitive_progress as primitive

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "RPent").exists():
    # The draft runs outside the checkout; the existing test module owns the
    # repository path without adding a machine-specific path to product code.
    import test_research_tools

    ROOT = Path(test_research_tools.__file__).resolve().parents[2]


class PrimitiveProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = {
            "run_id": "run",
            "episode_id": "episode",
            "bundle_id": "bundle",
            "protocol_id": "protocol",
        }
        self.events = []

        def emit(kind, **extra):
            index = len(self.events)
            self.events.append(
                {
                    **self.identity,
                    "sequence": index,
                    "utc_ns": 1_000_000_000_000 + index * 1_000_000_000,
                    "kind": kind,
                    **extra,
                }
            )

        emit("run_start")
        emit("observation_update")
        emit("tool_call", tool="vla_execute")
        emit(
            "action_proposal",
            source="vla",
            proposal=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
        )
        emit(
            "action_send",
            source="vla",
            sent=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
            proposal_sequence=3,
        )
        emit(
            "action_feedback",
            source="vla",
            send_sequence=4,
            execution_mask=[True],
            feedback={"observed_state": [0.0, 0.0, 1.0]},
        )
        emit("observation_update")
        emit("tool_end", tool="vla_execute", status="completed")
        emit("tool_call", tool="back_project_batch")
        emit("tool_end", tool="back_project_batch", status="completed")
        emit("tool_call", tool="move_to")
        action = np.zeros(7, dtype=np.float32)
        action[0] = np.clip(np.float32(0.1), -0.012, 0.012) / 0.05
        action[6] = -1
        emit("action_proposal", source="move_to", proposal=action.tolist())
        emit(
            "action_send",
            source="move_to",
            sent=action.tolist(),
            proposal_sequence=11,
            proposal_indices=[0],
        )
        emit(
            "action_feedback",
            source="move_to",
            send_sequence=12,
            proposal_sequence=11,
            execution_mask=[True],
            feedback={"observed_state": [0.01, 0.0, 1.0]},
        )
        emit("observation_update")
        emit("tool_end", tool="move_to", status="completed")
        self.move = {
            "xyz": [0.1, 0.0, 1.0],
            "gripper": -1,
            "max_steps": 1,
            "step_clip": 0.012,
        }
        vla = {"instruction": "move", "max_chunks": 1}
        self.states = {
            "steps": [
                {
                    "step_idx": 0,
                    "state": {"robot0_eef_pos": [0.0, 0.0, 1.0]},
                    "artifacts": [],
                },
                {
                    "step_idx": 1,
                    "state": {"robot0_eef_pos": [0.0, 0.0, 1.0]},
                    "artifacts": ["agentview_world_high.npz"],
                    "command": {"action": "vla_execute", **vla},
                },
                {
                    "step_idx": 2,
                    "state": {"robot0_eef_pos": [0.01, 0.0, 1.0]},
                    "artifacts": [],
                    "command": {"action": "move_to", **self.move},
                    "result": {"steps_used": 1},
                },
            ]
        }
        self.sdk = []

        def sdk(name, args, first, last, returned):
            base = {
                "id": name,
                "type": "mcpToolCall",
                "server": "rpent",
                "tool": name,
                "arguments": args,
                "started_at_ms": self.events[first]["utc_ns"] / 1e6 - 1,
                "completed_at_ms": self.events[last]["utc_ns"] / 1e6 + 1,
                "status": "completed",
            }
            self.sdk.append({"method": "item/started", "payload": {"item": base}})
            self.sdk.append(
                {
                    "method": "item/completed",
                    "payload": {
                        "item": {
                            **base,
                            "result": {
                                "content": [
                                    {"type": "text", "text": json.dumps(returned)}
                                ]
                            },
                        }
                    },
                }
            )

        sdk(
            "vla_execute",
            vla,
            2,
            7,
            {
                "step": 1,
                "state": self.states["steps"][1]["state"],
                "log": {"command": {"action": "vla_execute", **vla}},
            },
        )
        sdk(
            "back_project_batch",
            {
                "step": 1,
                "camera": "agentview",
                "resolution": "high",
                "points": [[0, 0]],
            },
            8,
            9,
            {
                "step": 1,
                "results": [
                    {
                        "index": 0,
                        "pixel": [0, 0],
                        "camera": "agentview",
                        "resolution": "high",
                        "world_xyz": [1.0, 2.0, 3.0],
                        "step": 1,
                        "image_size": [2, 2],
                        "source_artifact": "agentview_world_high.npz",
                    }
                ],
            },
        )
        sdk(
            "move_to",
            self.move,
            10,
            15,
            {
                "step": 2,
                "state": self.states["steps"][2]["state"],
                "log": {
                    "command": {"action": "move_to", **self.move},
                    "result": self.states["steps"][2]["result"],
                },
            },
        )
        world = np.ones((2, 2, 3), dtype=np.float32)
        world[0, 0] = [1.0, 2.0, 3.0]
        artifact = self.root / "agentview_world_high.npz/01.npz"
        artifact.parent.mkdir()
        np.savez(artifact, array=world)
        for name in (
            "robots/libero/tools.py",
            "robots/libero/toolkit.py",
            "rpent/tools/toolkit.py",
            "rpent/session/base.py",
        ):
            destination = self.root / "code_snapshot/RPent" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "RPent" / name, destination)
        evidence = {
            **self.identity,
            "client_send_sequence": 12,
            "proposal_sequence": 11,
            "proposal_indices": [0],
            "proposal_length": 1,
        }
        self.native = [
            {
                **self.identity,
                "sequence": 0,
                "kind": "rpc_dispatch_start",
                "rpc_evidence": evidence,
            },
            {
                **self.identity,
                "sequence": 1,
                "kind": "rpc_dispatch_end",
                "rpc_evidence": evidence,
                "execution_mask": [True],
                "unknown_proposal_indices": [],
                "error": None,
            },
        ]
        (self.root / "environment_service").mkdir()

    def inspect(self):
        (self.root / "codex_fixture.stream.jsonl").write_text(
            "\n".join(map(json.dumps, self.sdk))
        )
        (self.root / "environment_service/action_events.jsonl").write_text(
            "\n".join(map(json.dumps, self.native))
        )
        with patch.object(
            primitive, "__file__", str(ROOT / "core/src/rsi_vla/primitive_progress.py")
        ):
            return primitive.audit_primitives(
                self.root, self.identity, self.events, self.states
            )

    def test_current_projection_and_literal_target_live_servo_are_audited(self):
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        # The projected XYZ differs from the literal target; no invented
        # proximity tolerance or model-internal derivation is necessary.
        self.assertIn(
            "no equality/tolerance",
            report["not_applicable"]["projection_target_causality"],
        )

    def test_stale_projection_is_a_known_freshness_violation(self):
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] == "back_project_batch":
                item["arguments"]["step"] = 0
        report = self.inspect()
        self.assertIs(report["value"], False)
        self.assertIs(report["checks"]["projection_uses_latest_step"], False)

    def test_wrong_identity_keeps_prerequisite_evidence_unknown(self):
        self.events[13]["run_id"] = "other"
        report = self.inspect()
        self.assertIsNone(report["value"])
        self.assertTrue(any("identity" in missing for missing in report["missing"]))

    def test_unclosed_client_feedback_is_not_accepted(self):
        self.events[13]["kind"] = "diagnostic"
        report = self.inspect()
        self.assertIsNone(report["value"])
        self.assertTrue(
            any("closure incomplete" in missing for missing in report["missing"])
        )

    def test_unclosed_native_rpc_is_not_accepted(self):
        self.native.pop()
        report = self.inspect()
        self.assertIs(report["value"], False)
        self.assertIs(report["checks"]["primitive_native_rpc_closure"], False)

    def test_modified_action_or_saved_projection_cannot_pass(self):
        self.events[12]["sent"][0] = 0.7
        report = self.inspect()
        self.assertIs(report["value"], False)
        self.assertIs(report["checks"]["move_to_exact_live_actions"], False)

    def test_changed_saved_pixel_does_not_match_the_recorded_projection(self):
        np.savez(
            self.root / "agentview_world_high.npz/01.npz",
            array=np.zeros((2, 2, 3), dtype=np.float32),
        )
        report = self.inspect()
        self.assertIs(report["value"], False)
        self.assertIs(report["checks"]["projection_matches_saved_pixels"], False)

    def test_closed_gripper_is_literal_control_without_a_holding_claim(self):
        self.move["gripper"] = 1
        self.states["steps"][2]["command"]["gripper"] = 1
        self.events[11]["proposal"][6] = self.events[12]["sent"][6] = 1
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] == "move_to" and "result" in item:
                data = json.loads(item["result"]["content"][0]["text"])
                data["log"]["command"]["gripper"] = 1
                item["result"]["content"][0]["text"] = json.dumps(data)
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIn(
            "no physical holding", report["not_applicable"]["holding_or_goal_commit"]
        )

    def test_projection_only_path_does_not_require_a_primitive_action(self):
        self.events = self.events[:10]
        self.sdk = [
            record
            for record in self.sdk
            if record["payload"]["item"]["tool"] != "move_to"
        ]
        self.states["steps"].pop()
        self.native = []
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertNotIn("move_to_exact_live_actions", report["checks"])

    def release_fixture(self):
        self.events[5]["feedback"]["observed_state"] = [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.04,
            -0.04,
        ]
        self.events[13]["feedback"]["observed_state"] = [
            0.01,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.04,
            -0.04,
        ]
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        for index in (10, 15):
            self.events[index]["tool"] = "release"
        for index in (11, 12, 13):
            self.events[index]["source"] = "release"
        self.events[11]["proposal"] = self.events[12]["sent"] = action
        command = {"action": "release", "max_steps": 1}
        result = {
            "steps_used": 1,
            "start_gripper_opening": 0.08,
            "peak_gripper_opening": 0.08,
            "final_gripper_opening": 0.08,
        }
        self.states["steps"][2].update(command=command, result=result)
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] == "move_to":
                item["tool"] = "release"
                item["arguments"] = {"max_steps": 1}
                if "result" in item:
                    data = json.loads(item["result"]["content"][0]["text"])
                    data["log"] = {"command": command, "result": result}
                    item["result"]["content"][0]["text"] = json.dumps(data)

    def test_release_zero_pose_open_command_and_observed_openings_are_audited(self):
        self.release_fixture()
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["release_exact_live_actions"], True)
        self.assertIs(report["checks"]["release_actual_opening_observations"], True)

    def test_release_cannot_hide_a_nonzero_pose_command(self):
        self.release_fixture()
        self.events[11]["proposal"][0] = 0.1
        report = self.inspect()
        self.assertIs(report["value"], False)
        self.assertIs(report["checks"]["release_exact_live_actions"], False)

    def test_original_move_tolerance_counts_the_final_non_actuating_check(self):
        self.move["max_steps"] = 2
        self.states["steps"][2]["command"]["max_steps"] = 2
        self.events[13]["feedback"]["observed_state"][0] = 0.095
        self.states["steps"][2]["state"]["robot0_eef_pos"][0] = 0.095
        self.states["steps"][2]["result"]["steps_used"] = 2
        for record in self.sdk:
            item = record["payload"]["item"]
            if item["tool"] == "move_to" and "result" in item:
                data = json.loads(item["result"]["content"][0]["text"])
                data.update(state=self.states["steps"][2]["state"])
                data["log"] = {
                    "command": self.states["steps"][2]["command"],
                    "result": self.states["steps"][2]["result"],
                }
                item["result"]["content"][0]["text"] = json.dumps(data)
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["move_to_existing_convergence"], True)
        self.events[13]["feedback"]["observed_state"][0] = 0.01
        self.assertIs(self.inspect()["checks"]["move_to_existing_convergence"], False)


if __name__ == "__main__":
    unittest.main()
