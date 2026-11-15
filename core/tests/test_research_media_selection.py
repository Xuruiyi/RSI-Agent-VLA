# Copyright 2026 RSI-Agent-VLA contributors.

"""Explicit selected-event media and SDK projection preserve full legal reads."""

import json
import unittest
from unittest.mock import Mock, patch

import test_research_readable as readable
from rsi_vla.midterm_host import MidtermHost
from rsi_vla.research_transport import model_prompt, pack, unpack
from rsi_vla.storage import atomic_json


class SelectedMediaTests(unittest.TestCase):
    def setUp(self):
        self.f = readable.ReadableResearchTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.configure()
        view = self.f.source.get("evidence_views", "view")
        view["l0"][0]["kind"] = "task_result"
        view["l0"][0]["flags"]["independent_result"] = {
            "run_id": "run",
            "benchmark_success": False,
            "costs": {"model_requests": 0},
            "missing": ["actual_image_attention"],
            "episode_execution": {
                "status": "budget_exhausted",
                "native_success": None,
                "spec": {"arm": "D"},
                "decisions": [{"native_fact": "original-history"}],
                "segments": [{"executed_steps": 5}],
            },
        }
        atomic_json(self.f.source.path("evidence_views", "view"), view)

    def test_old_default_is_full_and_opt_in_only_projects_the_SDK_copy(self):
        f = self.f
        archive = f.source.path("evidence_views", "view").read_bytes()
        f.response("old-default")
        old = f.source.get("bounded_requests", "old-default")
        self.assertEqual(json.loads(model_prompt(pack(old)).split("\n", 1)[1]), old)
        self.assertEqual(len(old["evidence_views"][0]["selected_media"]), 4)
        domain = {
            **f.tools.domain_policy,
            "evidence_selection": {
                "kinds": ["task_result"],
                "first_matches": 1,
                "last_events": 0,
                "media_refs": "selected_events",
                "sdk_projection": "terminal_history_v1",
                "unrelated": "not an input policy field",
            },
        }
        domain_ref = f.root / "selected-domain.json"
        atomic_json(domain_ref, domain)
        backend = Mock()
        backend.tools.return_value = (
            f.tools,
            {"domain_policy_ref": str(domain_ref), "result_ids": ["result"]},
        )
        host = MidtermHost.__new__(MidtermHost)
        host.core_config = {}
        with patch(
            "rsi_vla.task_completion.TaskCompletionBackend", return_value=backend
        ):
            host.select_research_input("selected")
        f.tools.domain_policy = json.loads(domain_ref.read_text())
        self.assertNotIn("unrelated", f.tools.domain_policy["evidence_input"])
        f.response("selected")
        business = f.source.get("bounded_requests", "selected")
        projected = json.loads(model_prompt(pack(business)).split("\n", 1)[1])
        self.assertEqual(
            business["evidence_views"][0]["selected_media"], ["a.depth.npz", "a.png"]
        )
        full = business["evidence_views"][0]["l0"][0]["flags"]["independent_result"]
        shown = projected["evidence_views"][0]["l0"][0]["flags"]["independent_result"]
        self.assertIn("decisions", full["episode_execution"])
        self.assertNotIn("decisions", shown["episode_execution"])
        for name in ("run_id", "benchmark_success", "costs", "missing"):
            self.assertEqual(shown[name], full[name])
        self.assertEqual(shown["episode_execution"]["native_success"], None)
        self.assertEqual(
            len(projected["readable_evidence"]["views"][0]["event_index"]), 4
        )
        self.assertNotIn(
            "omitted_event_ids", projected["readable_evidence"]["views"][0]
        )
        self.assertEqual(
            projected["readable_evidence"]["views"][0]["omitted_event_count"], 3
        )
        counts = business["evidence_input"]["views"][0]["media_selection"]
        self.assertEqual(
            (
                counts["input_media_count_before"],
                counts["selected_media_count"],
                counts["omitted_media_count"],
            ),
            (4, 2, 2),
        )
        self.assertFalse(counts["image_bytes_sent"])
        self.assertEqual(f.source.path("evidence_views", "view").read_bytes(), archive)

    def test_requested_full_event_survives_followup_and_media_stays_bounded(self):
        f = self.f
        f.tools.domain_policy["evidence_input"].update(
            media_refs="selected_events", sdk_projection="terminal_history_v1"
        )
        response = f.response("first")
        response.update(
            decision="defer",
            intervention=None,
            missing_evidence=["full terminal history needed"],
            evidence_requests=[
                {
                    "result_id": "result",
                    "event_ids": ["e0", "e1"],
                    "reason": "read full projected terminal plus next observation",
                }
            ],
        )
        f.apply(response)
        followup = unpack(
            f.adapter.prepare_evidence_followup(
                f.tools,
                parent_request_id="first",
                request_id="followup",
                max_input_tokens=100,
                max_output_tokens=10,
            )
        )
        shown = json.loads(model_prompt(pack(followup)).split("\n", 1)[1])
        self.assertNotIn("sdk_projection", followup["evidence_input"])
        self.assertNotIn("sdk_projection", shown)
        self.assertEqual(followup["evidence_input"]["media_refs"], "selected_events")
        self.assertEqual(len(shown["evidence_views"][0]["selected_media"]), 4)
        self.assertEqual(
            shown["evidence_views"][0]["l0"][0],
            f.source.get("evidence_views", "view")["l0"][0],
        )
        self.assertFalse(f.source.all("llm_executions"))
        self.assertFalse(f.source.all("candidates"))


if __name__ == "__main__":
    unittest.main()
