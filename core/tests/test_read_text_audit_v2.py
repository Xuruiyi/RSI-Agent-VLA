"""Isolated consumer regressions using synthetic SDK/L0 files only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.read_text_audit import audit_read, READ_PATH, READ_SOURCES, REVIEWED_REVISION

SOURCE = Path(__file__).resolve().parents[2]


class ReadAuditV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = dict(run_id="r", episode_id="e", bundle_id="b", protocol_id="p")
        self.sources = {name: (SOURCE / "RPent" / name).read_bytes() for name in READ_SOURCES}
        for name, data in self.sources.items():
            destination = self.root / "source-origin" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        for name in ("robot_spec.py", "toolkit.py"):
            destination = self.root / "code_snapshot/RPent/robots/libero" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((SOURCE / "RPent/robots/libero" / name).read_bytes())
        self.write("runtime_context.json", self.identity)
        self.write("runtime_costs.json", {"tool_calls": 1})
        repo = SOURCE / "RPent"
        self.write("launch_plan.json", {"cwd": str(repo), "environment": {"RPENT_REPO_ROOT": str(repo)}})
        self.write("before.json", {"RPent_HEAD_and_origin": REVIEWED_REVISION + "\n", "RPent": "## main\n"})
        self.ref = self.root / "provenance.json"
        self.write("provenance.json", dict(identity=self.identity, run_root=str(self.root), repository=str(repo),
            revision=REVIEWED_REVISION, git_state_before_ref=str(self.root / "before.json"), files=list(READ_SOURCES)))
        text = "---\nname: stop-that-shit\n---\nMeet the task requirements.\n"
        self.text = text + "x" * (8197 - len(text))
        args = {"path": READ_PATH}
        start = dict(id="read", type="mcpToolCall", server="rpent", tool="read_text_file", arguments=args,
            status="inProgress", started_at_ms=1)
        end = {**start, "status": "completed", "completed_at_ms": 4, "result": {"content": [{"type": "text",
            "text": json.dumps({"path": READ_PATH, "size": len(self.text), "content": self.text})}]}}
        self.sdk = [{"method": "item/started", "payload": {"item": start}},
                    {"method": "item/completed", "payload": {"item": end}}]
        self.events = [dict(self.identity, sequence=0, kind="tool_call", tool="read_text_file", utc_ns=2_000_000),
                       dict(self.identity, sequence=1, kind="tool_end", tool="read_text_file", utc_ns=3_000_000, status="completed")]

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def check(self):
        (self.root / "codex_fixture.stream.jsonl").write_text("\n".join(map(json.dumps, self.sdk)))
        (self.root / "action_events.jsonl").write_text("\n".join(map(json.dumps, self.events)))
        calls = [event for event in self.events if event["kind"] == "tool_call"]
        ends = [event for event in self.events if event["kind"] == "tool_end"]
        with patch("rsi_vla.read_text_audit.subprocess.run", side_effect=lambda args, **kw:
                SimpleNamespace(stdout=self.sources[args[-1].split(":", 1)[1]])):
            return audit_read(self.root, self.identity, calls, ends, source_ref=self.ref)

    def add_missing_nonread(self, local_stop=False):
        self.sdk.append({"method": "item/started", "payload": {"item": dict(id="vla", type="mcpToolCall",
            server="rpent", tool="vla_execute", arguments={}, status="inProgress", started_at_ms=5)}})
        self.events.extend([dict(self.identity, sequence=2, kind="tool_call", tool="vla_execute", utc_ns=6_000_000),
                            dict(self.identity, sequence=3, kind="tool_end", tool="vla_execute", utc_ns=7_000_000, status="completed")])
        self.write("runtime_costs.json", {"tool_calls": 2})
        self.write("result.json", {**self.identity, "benchmark_success": True, "terminated": True})
        model = [dict(kind="request_admitted", request_number=1),
                 dict(kind="response_terminal", request_number=1, event_type="response.completed",
                    usage={"input_tokens": 8, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 2}),
                 dict(kind="request_returned", request_number=1, http_status=200),
                 dict(kind="rejected", reason="native task completed", forwarded=False, monotonic_seconds=2)]
        model = [dict(identity=self.identity, sequence=i, **event) for i, event in enumerate(model)]
        (self.root / "model_requests.jsonl").write_text("\n".join(map(json.dumps, model)))
        self.write("request_gate.json", {"identity": self.identity, "requests_forwarded": 1})
        if local_stop:
            self.events.append(dict(self.identity, sequence=4, kind="native_completion", step=1, reason="native_success"))
            self.events.append(dict(self.identity, sequence=5, kind="planner_usage", stats={"sdk_errors": [
                {"error": {"http_status_code": 429}, "local_rejection_candidate": True}]}))
            self.write("native_completion.json", dict(identity=self.identity, sequence=4, step=1,
                reason="native_success", monotonic_ns=1_000_000_000))
            images = ["agentview_policy.png", "agentview.png", "wrist.png"]
            self.write("states.json", {"steps": [dict(step_idx=1, terminated=True, state={"synthetic": True}, artifacts=images)]})
            for image in images:
                path = self.root / image / "01.png"
                path.parent.mkdir()
                path.write_bytes(b"synthetic observation")
            self.write("local_model_stop.json", dict(identity=self.identity, reason="native_success", forwarded=False,
                rejection_sequence=3, marker_sequence=4, marker_ref=str(self.root / "native_completion.json"),
                model_requests_ref=str(self.root / "model_requests.jsonl")))

    def test_omitted_default_comes_from_reviewed_source_ast(self):
        report = self.check()
        self.assertIs(report["value"], True)
        self.assertEqual(report["source_basis"]["read_max_chars_default"], 40000)
        self.assertEqual(report["reads"][0]["effective_max_chars"], 40000)
        self.assertEqual(report["reads"][0]["returned_chars"], 8197)
        self.assertNotIn("max_chars", report["reads"][0]["arguments"])

    def test_explicit_positive_range_preserved(self):
        for event in self.sdk:
            event["payload"]["item"]["arguments"] = {"path": READ_PATH, "max_chars": 12000}
        self.assertIs(self.check()["value"], True)

    def test_null_and_invalid_ranges_rejected(self):
        original = copy.deepcopy(self.sdk)
        for value in (None, 0, -1, True, "40000", 40000.0):
            with self.subTest(value=value):
                self.sdk = copy.deepcopy(original)
                for event in self.sdk:
                    event["payload"]["item"]["arguments"] = {"path": READ_PATH, "max_chars": value}
                self.assertIs(self.check()["value"], False)

    def test_truncated_content_rejected(self):
        end = self.sdk[1]["payload"]["item"]
        end["result"]["content"][0]["text"] = json.dumps({"path": READ_PATH,
            "size": len(self.text), "content": self.text + "[TRUNCATED]"})
        self.assertIs(self.check()["value"], False)

    def test_wrong_path_and_identity_rejected(self):
        for event in self.sdk:
            event["payload"]["item"]["arguments"] = {"path": "/unreviewed/file"}
        self.assertIs(self.check()["value"], False)
        self.write("runtime_context.json", {**self.identity, "run_id": "wrong"})
        self.assertIs(self.check()["value"], False)

    def test_read_sdk_terminal_cannot_be_inferred_from_l0(self):
        self.sdk.pop(1)
        self.assertIs(self.check()["value"], False)

    def test_read_l0_terminal_cannot_be_inferred_from_sdk(self):
        self.events.pop()
        self.assertIs(self.check()["value"], False)

    def test_invalid_explicit_range_is_not_hidden_by_missing_source(self):
        for event in self.sdk:
            event["payload"]["item"]["arguments"] = {"path": READ_PATH, "max_chars": None}
        (self.root / "source-origin/rpent/tools/common.py").unlink()
        self.assertIs(self.check()["value"], False)

    def test_source_byte_provenance_still_required(self):
        path = self.root / "source-origin/rpent/tools/common.py"
        path.write_bytes(path.read_bytes() + b"\n# altered source\n")
        self.assertIs(self.check()["value"], False)

    def test_missing_nonread_terminal_requires_valid_local_stop(self):
        self.add_missing_nonread(local_stop=False)
        self.assertIs(self.check()["value"], False)

    def test_nonread_local_stop_and_l0_closure_do_not_fabricate_sdk_terminal(self):
        self.add_missing_nonread(local_stop=True)
        before = copy.deepcopy(self.sdk)
        report = self.check()
        self.assertIs(report["value"], True)
        self.assertEqual(report["tool_audit"]["sdk_terminal_missing"], ["vla"])
        self.assertTrue(report["local_stop_audit"]["applicable"])
        self.assertFalse(report["local_stop_audit"]["issues"])
        self.assertEqual(self.sdk, before)

    def test_local_stop_missing_read_terminal_still_rejected(self):
        self.add_missing_nonread(local_stop=True)
        self.sdk.pop(1)
        self.assertIs(self.check()["value"], False)

    def test_local_stop_missing_nonread_l0_terminal_still_rejected(self):
        self.add_missing_nonread(local_stop=True)
        self.events = [event for event in self.events if not (event["kind"] == "tool_end" and event["tool"] == "vla_execute")]
        self.assertIs(self.check()["value"], False)

    def test_local_stop_wrong_marker_cannot_close_nonread_delivery(self):
        self.add_missing_nonread(local_stop=True)
        marker = json.loads((self.root / "native_completion.json").read_text())
        self.write("native_completion.json", {**marker, "identity": {**self.identity, "run_id": "wrong"}})
        self.assertIs(self.check()["value"], False)


if __name__ == "__main__":
    unittest.main()
