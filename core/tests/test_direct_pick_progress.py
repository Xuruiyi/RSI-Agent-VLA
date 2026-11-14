"""Direct pick failures keep independent loop closure separate from task success."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rsi_vla.critical_progress import inspect_progress
from rsi_vla.primitive_progress import audit_direct_picks, audit_primitives
import test_research_tools

ROOT = Path(test_research_tools.__file__).resolve().parents[2]


class DirectPickProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(
            run_id="fixture",
            episode_id="episode",
            bundle_id="bundle",
            protocol_id="protocol",
        )
        for name in [
            "robots/libero/tools.py",
            "robots/libero/toolkit.py",
            "rpent/tools/toolkit.py",
            "rpent/session/base.py",
            "robots/libero/env_client.py",
            "robots/libero/env_server.py",
            "robots/libero/action_spec.py",
        ]:
            target = self.root / "code_snapshot/RPent" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "RPent" / name, target)
        self.initial = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04, -0.04]
        self.observed = [0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 0.04, -0.04]
        self.args = dict(prompt="attempt the observed handle grasp", max_chunks=1)
        self.result = self.result_from_feedback()
        state0 = dict(
            robot0_eef_pos=self.initial[:3], robot0_gripper_qpos=self.initial[6:8]
        )
        state1 = dict(
            robot0_eef_pos=self.observed[:3], robot0_gripper_qpos=self.observed[6:8]
        )
        self.states = {
            "steps": [
                dict(
                    step_idx=0,
                    state=state0,
                    terminated=False,
                    truncated=False,
                    artifacts=["agentview_world_high.npz"],
                ),
                dict(
                    step_idx=1,
                    state=state1,
                    terminated=False,
                    truncated=False,
                    artifacts=[],
                    command={"action": "pi0_pick", **self.args},
                    result=self.result,
                ),
            ]
        }
        actions = [[0.0] * 7 for _ in range(5)]
        raw = [
            dict(kind="run_start"),
            dict(kind="observation_update"),
            dict(
                kind="action_feedback",
                source="setup",
                feedback={"observed_state": self.initial},
            ),
            dict(kind="tool_call", tool="back_project_batch"),
            dict(kind="tool_end", tool="back_project_batch", status="completed"),
            dict(kind="tool_call", tool="pi0_pick"),
            dict(
                kind="inference_start",
                source="vla",
                instruction=self.args["prompt"],
                segment_sequence=None,
                observation_sequence=1,
                observation_mode="latest",
            ),
            dict(
                kind="inference_end",
                source="vla",
                inference_sequence=6,
                segment_sequence=None,
                completed=True,
            ),
            dict(
                kind="action_proposal",
                source="vla",
                inference_sequence=6,
                segment_sequence=None,
                instruction=self.args["prompt"],
                proposal=actions,
            ),
            dict(
                kind="action_send",
                source="vla",
                proposal_sequence=8,
                proposal_indices=list(range(5)),
                sent=actions,
            ),
            dict(
                kind="action_feedback",
                source="vla",
                send_sequence=9,
                proposal_sequence=8,
                execution_mask=[True] * 5,
                feedback={
                    "observed_state": self.observed,
                    "terminated": [False] * 5,
                    "truncated": [False] * 5,
                },
            ),
            dict(
                kind="chunk_execution_end",
                source="vla",
                inference_sequence=6,
                predictions_started=1,
                predictions_completed=1,
                executed_steps=5,
                unknown_steps=0,
            ),
            dict(kind="observation_update"),
            dict(kind="tool_end", tool="pi0_pick", status="completed"),
        ]
        self.events = [
            {
                **self.identity,
                "sequence": i,
                "utc_ns": 1_000_000_000 + i * 1_000_000,
                **item,
            }
            for i, item in enumerate(raw)
        ]
        self.sdk = []
        self.sdk_call(
            "projection",
            "back_project_batch",
            dict(points=[[0, 0]], step=0, camera="agentview", resolution="high"),
            3,
            4,
            dict(
                step=0,
                results=[
                    dict(
                        index=0,
                        pixel=[0, 0],
                        camera="agentview",
                        resolution="high",
                        world_xyz=[1.0, 2.0, 3.0],
                        step=0,
                        image_size=[2, 2],
                        source_artifact="agentview_world_high.npz",
                    )
                ],
            ),
        )
        self.sdk_call(
            "pick",
            "pi0_pick",
            self.args,
            5,
            13,
            dict(
                step=1,
                state=state1,
                terminated=False,
                truncated=False,
                log=dict(
                    command={"action": "pi0_pick", **self.args}, result=self.result
                ),
            ),
        )
        world = np.ones((2, 2, 3), dtype=np.float32)
        world[0, 0] = [1.0, 2.0, 3.0]
        target = self.root / "agentview_world_high.npz/00.npz"
        target.parent.mkdir()
        np.savez(target, array=world)
        (self.root / "environment_service").mkdir()
        (self.root / "environment_service/action_events.jsonl").write_text("")
        (self.root / "runtime_context.json").write_text(json.dumps(self.identity))

    def result_from_feedback(self):
        initial = np.asarray(self.initial, dtype=np.float32)
        last = np.asarray(self.observed, dtype=np.float32)
        z0, z1 = float(initial[2]), float(last[2])
        grip = float(abs(last[6]) + abs(last[7]))
        return dict(
            name="pick",
            instruction=self.args["prompt"],
            success=False,
            chunks_used=1,
            max_chunks=1,
            peak_lift_m=z1 - z0,
            min_gripper_opening=grip,
            final_gripper_opening=grip,
            terminated=False,
            truncated=False,
            diagnostics=dict(
                start_eef_z=round(z0, 4),
                peak_eef_z=round(z1, 4),
                min_eef_z=round(z0, 4),
                post_min_peak_z=round(z1, 4),
                descent_m=0.0,
                post_min_ascent_m=round(z1 - z0, 4),
                descent_done=False,
                lift_thresh=0.05,
                gripper_closed_thresh=0.06,
                gripper_open_thresh=0.0,
                descent_thresh=0.1,
            ),
        )

    def sdk_call(self, identity, name, args, first, last, value):
        item = dict(
            id=identity,
            type="mcpToolCall",
            server="rpent",
            tool=name,
            arguments=deepcopy(args),
            status="inProgress",
            started_at_ms=self.events[first]["utc_ns"] / 1e6 - 1,
        )
        self.sdk.append(dict(method="item/started", payload={"item": item}))
        self.sdk.append(
            dict(
                method="item/completed",
                payload={
                    "item": {
                        **deepcopy(item),
                        "status": "completed",
                        "completed_at_ms": self.events[last]["utc_ns"] / 1e6 + 1,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(value)}]
                        },
                    }
                },
            )
        )

    def write(self):
        (self.root / "codex_fixture.stream.jsonl").write_text(
            "\n".join(map(json.dumps, self.sdk))
        )
        (self.root / "action_events.jsonl").write_text(
            "\n".join(map(json.dumps, self.events))
        )
        (self.root / "states.json").write_text(json.dumps(self.states))

    def inspect(self):
        self.write()
        return audit_direct_picks(self.root, self.identity, self.events, self.states)

    def test_failed_pick_is_closed_without_route_or_B_feedback(self):
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIs(self.result["success"], False)
        self.assertIs(self.result["terminated"], False)
        self.assertEqual(
            report["pick_spans"],
            [dict(start=5, end=13, instruction=self.args["prompt"], step_idx=1)],
        )
        self.assertIn(
            "no registered route", report["not_applicable"]["route_and_B_feedback"]
        )
        self.assertFalse(
            any(event["kind"] == "relation_route_decision" for event in self.events)
        )

    def test_projection_state_index_advances_over_a_direct_pick(self):
        self.write()
        report = audit_primitives(self.root, self.identity, self.events, self.states)
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["complete_state_trace"], True)
        self.assertIs(self.result["success"], False)

    def test_critical_progress_uses_verified_pick_span_without_fabricated_segment(self):
        self.write()
        checks = {
            name: True
            for name in [
                "ordered_l0",
                "execution_health",
                "action_mapping",
                "setup_coverage",
                "no_mid_attempt_reset",
                "vla_actual_loading_and_prediction",
            ]
        }
        with patch(
            "rsi_vla.adapters.runtime_acceptance.inspect_attempt",
            return_value={"checks": checks},
        ):
            report = inspect_progress(self.root, self.identity)
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["actual_direct_pick_audit"], True)
        self.assertIs(report["checks"]["proposal_to_native_send"], True)
        self.assertFalse(
            any(event["kind"] == "vla_segment_start" for event in self.events)
        )
        self.assertIs(self.result["success"], False)

    def test_missing_sdk_terminal_or_chunk_feedback_stays_unknown(self):
        original = deepcopy(self.sdk)
        self.sdk.pop()
        self.assertIsNone(self.inspect()["value"])
        self.sdk = original
        self.events[10]["kind"] = "diagnostic"
        report = self.inspect()
        self.assertIsNone(report["value"])
        self.assertTrue(report["missing"])

    def test_missing_initial_feedback_or_changed_consumer_stays_unknown(self):
        self.events[2]["kind"] = "diagnostic"
        self.assertIsNone(self.inspect()["value"])
        self.events[2]["kind"] = "action_feedback"
        path = self.root / "code_snapshot/RPent/robots/libero/tools.py"
        path.write_text(path.read_text() + "\n# changed\n")
        self.assertIsNone(self.inspect()["value"])

    def test_tampered_heuristic_cannot_be_self_confirmed_by_matching_sdk_state(self):
        self.result["success"] = True
        item = self.sdk[-1]["payload"]["item"]
        data = json.loads(item["result"]["content"][0]["text"])
        data["log"]["result"]["success"] = True
        item["result"]["content"][0]["text"] = json.dumps(data)
        report = self.inspect()
        self.assertIs(report["value"], False, report)
        self.assertIs(report["checks"]["direct_pick_recorded_heuristic_result"], False)

    def test_changed_inference_goal_is_a_known_argument_violation(self):
        self.events[6]["instruction"] = "another goal"
        report = self.inspect()
        self.assertIs(report["value"], False, report)
        self.assertIs(report["checks"]["direct_pick_count_and_arguments"], False)

    def test_chunk_flag_any_matches_client_scalar_state(self):
        self.events[10]["feedback"]["truncated"] = [False, True, False, False, False]
        self.states["steps"][1]["truncated"] = self.result["truncated"] = True
        item = self.sdk[-1]["payload"]["item"]
        data = json.loads(item["result"]["content"][0]["text"])
        data["truncated"] = data["log"]["result"]["truncated"] = True
        item["result"]["content"][0]["text"] = json.dumps(data)
        report = self.inspect()
        self.assertIs(report["value"], True, report)
        self.assertIs(report["checks"]["direct_pick_scalar_native_state"], True)
        self.assertIs(self.result["success"], False)

    def test_missing_wrong_length_or_non_boolean_chunk_flags_stay_unknown(self):
        original = deepcopy(self.events)
        for value in (
            None,
            [],
            [False] * 4,
            [False, False, 0, False, False],
            [False, False, None, False, False],
            False,
        ):
            with self.subTest(flags=value):
                self.events = deepcopy(original)
                self.events[10]["feedback"]["terminated"] = value
                self.assertIsNone(self.inspect()["value"])
        self.events = deepcopy(original)
        del self.events[10]["feedback"]["terminated"]
        self.assertIsNone(self.inspect()["value"])
        self.events = deepcopy(original)
        self.events[10]["execution_mask"] = [True] * 4
        self.assertIsNone(self.inspect()["value"])

    def test_sdk_state_disagreement_remains_unknown(self):
        self.states["steps"][1]["command"]["prompt"] = "different tool arguments"
        self.assertIsNone(self.inspect()["value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
