"""Loopback fixtures proving pre-forward request admission, no real model."""

import json
import socketserver
import time
from unittest.mock import patch
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from rsi_vla.adapters.request_gate import RequestGate


class RequestGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        calls = self.calls
        release = self.release

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                calls.append(payload)
                if payload.get("input") == "drop before headers":
                    self.close_connection = True
                    return
                body = b'data: {"type":"response.completed","response":{"id":"offline","model":"gpt-6.1-sol","status":"completed","usage":{"input_tokens":2,"output_tokens":1}}}\n\n'
                if payload.get("input") in {"delayed terminal", "bad upstream body"}:
                    prefix = b'data: {"type":"response.created"}\n\n'
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(prefix) + len(body)))
                    self.end_headers()
                    self.wfile.write(prefix)
                    self.wfile.flush()
                    if payload["input"] == "bad upstream body":
                        self.close_connection = True
                        return
                    release.wait(2)
                    try:
                        self.wfile.write(body)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.server.daemon_threads = True
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        ).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.context = dict(
            output_dir=self.temp.name,
            planner=dict(model="gpt-6.1-sol", reasoning_effort="medium"),
            limits=dict(model_requests=1, wall_seconds=5),
            run_id="offline",
            episode_id="e",
            bundle_id="b",
            protocol_id="p",
        )

    def gate(self, **kwargs):
        return RequestGate(
            self.context,
            upstream_base=f"http://127.0.0.1:{self.server.server_port}",
            **kwargs,
        )

    def native_marker(self, *, early=False):
        identity = {
            key: self.context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        name = "planner_admission_closed.json" if early else "native_completion.json"
        path = Path(self.temp.name) / name
        path.write_text(
            json.dumps(
                dict(
                    identity=identity,
                    reason="native_success_observed" if early else "native_success",
                    sequence=3,
                    monotonic_ns=time.monotonic_ns(),
                )
            )
        )

    def payload(self, mode="offline fixture"):
        return dict(model="gpt-6.1-sol", reasoning=dict(effort="medium"), input=mode)

    def force_downstream_disconnect(self, gate, needle, observed):
        original = socketserver._SocketWriter.write

        def write(stream, data):
            if (
                stream._sock.getsockname()[1] == gate.server.server_port
                and needle in data
            ):
                observed.set()
                raise BrokenPipeError("offline downstream closure")
            return original(stream, data)

        return patch("socketserver._SocketWriter.write", write)

    def post_in_thread(self, gate, mode):
        result = {}

        def post():
            try:
                result["response"] = httpx.post(
                    gate.url + "/responses",
                    json=self.payload(mode),
                    trust_env=False,
                    timeout=3,
                )
            except httpx.HTTPError as error:
                result["error"] = type(error).__name__

        thread = threading.Thread(target=post, daemon=True)
        thread.start()
        self.addCleanup(lambda: thread.join(timeout=3))
        return thread, result

    def test_early_native_boundary_holds_request_locally_until_state_capture(self):
        self.native_marker(early=True)
        waiting = threading.Event()
        with self.gate() as gate:
            emit = gate.emit

            def observe(kind, **fields):
                emit(kind, **fields)
                if kind == "admission_wait":
                    waiting.set()

            gate.emit = observe
            thread, result = self.post_in_thread(gate, "offline fixture")
            self.assertTrue(waiting.wait(1))
            self.assertEqual(self.calls, [])
            self.assertEqual(gate.sent, 0)
            self.assertFalse((Path(self.temp.name) / "native_completion.json").exists())
            self.native_marker()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result["response"].status_code, 429)
            receipt = json.loads((Path(self.temp.name) / "local_model_stop.json").read_text())
            self.assertEqual(receipt["reason"], "native_success")
            self.assertIs(receipt["forwarded"], False)
            rejection = gate.events[receipt["rejection_sequence"]]
            self.assertEqual(rejection["identity"], receipt["identity"])
            self.assertEqual(rejection["kind"], "rejected")
            self.assertIs(rejection["forwarded"], False)
            self.assertEqual(gate.sent, 0)
        self.assertFalse((Path(self.temp.name) / "budget_stop.json").exists())

    def test_unfinished_state_capture_never_forwards_or_claims_native_completion(self):
        self.native_marker(early=True)
        self.context["limits"]["wall_seconds"] = 0.15
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses", json=self.payload(), trust_env=False, timeout=1
            )
            self.assertEqual(response.status_code, 429)
            self.assertIn("state capture incomplete", response.text)
        self.assertEqual(self.calls, [])
        self.assertFalse((Path(self.temp.name) / "native_completion.json").exists())
        self.assertFalse((Path(self.temp.name) / "local_model_stop.json").exists())

    def test_terminal_usage_recorded_before_client_write_fails(self):
        with self.gate() as gate:
            # Publish success only after admission, as a real completed action would.
            original_admit = gate.admit

            def admit(payload):
                number = original_admit(payload)
                self.native_marker()
                return number

            gate.admit = admit
            observed = threading.Event()
            with self.force_downstream_disconnect(
                gate, b"response.completed", observed
            ):
                response = httpx.post(
                    gate.url + "/responses",
                    json=self.payload(),
                    trust_env=False,
                    timeout=2,
                )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(observed.is_set())
        events = gate.events
        terminal = next(e for e in events if e["kind"] == "response_terminal")
        closed = next(e for e in events if e["kind"] == "downstream_closed")
        self.assertLess(terminal["sequence"], closed["sequence"])
        self.assertEqual(terminal["usage"]["input_tokens"], 2)
        self.assertEqual(closed["reason"], "native_success")
        self.assertFalse(any(e["kind"] == "transport_error" for e in events))
        self.assertEqual(len(self.calls), 1)

    def test_gate_exit_drains_existing_request_to_terminal_without_retry(self):
        observed = threading.Event()
        gate = self.gate(cleanup_timeout_s=0.8)
        with gate:
            original_admit = gate.admit

            def admit(payload):
                number = original_admit(payload)
                self.native_marker()
                return number

            gate.admit = admit
            with self.force_downstream_disconnect(gate, b"response.created", observed):
                thread, _ = self.post_in_thread(gate, "delayed terminal")
                self.assertTrue(observed.wait(1))
            timer = threading.Timer(0.05, self.release.set)
            timer.start()
        timer.join(timeout=1)
        thread.join(timeout=1)
        self.assertTrue(any(e["kind"] == "response_terminal" for e in gate.events))
        self.assertEqual(len(self.calls), 1)
        report = json.loads((Path(self.temp.name) / "request_gate.json").read_text())
        self.assertTrue(report["cleanup_complete"])
        self.assertEqual(report["active_requests_at_close"], 0)

    def test_hanging_upstream_is_cancelled_within_original_job_budget(self):
        observed = threading.Event()
        self.context["limits"]["wall_seconds"] = 0.5
        gate = self.gate(cleanup_timeout_s=0.3)
        started = time.monotonic()
        with gate:
            original_admit = gate.admit

            def admit(payload):
                number = original_admit(payload)
                self.native_marker()
                return number

            gate.admit = admit
            with self.force_downstream_disconnect(gate, b"response.created", observed):
                thread, _ = self.post_in_thread(gate, "delayed terminal")
                self.assertTrue(observed.wait(1))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, self.context["limits"]["wall_seconds"])
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(any(e["kind"] == "response_terminal" for e in gate.events))
        self.assertTrue(any(e["kind"] == "request_incomplete" for e in gate.events))
        self.release.set()
        thread.join(timeout=1)
        report = json.loads((Path(self.temp.name) / "request_gate.json").read_text())
        self.assertEqual(report["active_requests_at_close"], 0)

    def test_true_upstream_fault_after_native_success_remains_transport_error(self):
        with self.gate() as gate:
            original_admit = gate.admit

            def admit(payload):
                number = original_admit(payload)
                self.native_marker()
                return number

            gate.admit = admit
            thread, _ = self.post_in_thread(gate, "bad upstream body")
            thread.join(timeout=1)
        self.assertTrue(any(e["kind"] == "transport_error" for e in gate.events))
        self.assertEqual(len(self.calls), 1)

    def test_supervisor_cancel_blocks_forward_and_leaves_request_count_unchanged(self):
        identity = {
            key: self.context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        (Path(self.temp.name) / "runtime_stop.json").write_text(
            json.dumps(dict(identity=identity, reason="external_interruption"))
        )
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses",
                json=dict(model="gpt-6.1-sol", reasoning=dict(effort="medium")),
                trust_env=False,
            )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])

    def test_native_success_blocks_forward_without_budget_or_cancel(self):
        identity = {
            key: self.context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        (Path(self.temp.name) / "native_completion.json").write_text(
            json.dumps(dict(identity=identity, reason="native_success", step=1))
        )
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses",
                json=dict(model="gpt-6.1-sol", reasoning=dict(effort="medium")),
                trust_env=False,
            )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])
        self.assertFalse((Path(self.temp.name) / "budget_stop.json").exists())
        self.assertFalse((Path(self.temp.name) / "runtime_stop.json").exists())

    def test_one_request_admitted_and_second_not_forwarded(self):
        payload = dict(
            model="gpt-6.1-sol",
            reasoning=dict(effort="medium"),
            input="offline fixture",
        )
        with self.gate() as gate:
            first = httpx.post(gate.url + "/responses", json=payload, trust_env=False)
            second = httpx.post(gate.url + "/responses", json=payload, trust_env=False)
            self.assertEqual((first.status_code, second.status_code), (200, 429))
        self.assertEqual(len(self.calls), 1)
        marker = json.loads((Path(self.temp.name) / "budget_stop.json").read_text())
        self.assertEqual(marker["reason"], "model_requests")
        self.assertEqual(marker["limit"], marker["observed"])
        live = json.loads((Path(self.temp.name) / "request_gate_live.json").read_text())
        self.assertEqual(live["requests_forwarded"], 1)
        report = json.loads((Path(self.temp.name) / "request_gate.json").read_text())
        self.assertEqual(
            (report["requests_forwarded"], report["requests_rejected"]), (1, 1)
        )
        events = (Path(self.temp.name) / "model_requests.jsonl").read_text()
        self.assertNotIn("offline fixture", events)
        self.assertIn("response_terminal", events)

    def test_wrong_effective_effort_or_endpoint_never_reaches_upstream(self):
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses",
                json=dict(model="gpt-6.1-sol", reasoning=dict(effort="low")),
                trust_env=False,
            )
            compact = httpx.post(
                gate.url + "/responses/compact", json={}, trust_env=False
            )
            self.assertEqual((response.status_code, compact.status_code), (400, 403))
        self.assertEqual(self.calls, [])

    def test_runtime_budget_stop_blocks_next_forward_without_charging_request(self):
        identity = {
            key: self.context[key]
            for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        (Path(self.temp.name) / "budget_stop.json").write_text(
            json.dumps(dict(identity=identity, reason="tool_calls", sequence=42))
        )
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses",
                json=dict(model="gpt-6.1-sol", reasoning=dict(effort="medium")),
                trust_env=False,
            )
            self.assertEqual(response.status_code, 429)
        self.assertEqual(self.calls, [])
        report = json.loads((Path(self.temp.name) / "request_gate.json").read_text())
        self.assertEqual(
            (report["requests_forwarded"], report["requests_rejected"]), (0, 1)
        )

    def test_upstream_failure_returns_502_and_closes_downstream_without_retry(self):
        with self.gate() as gate:
            response = httpx.post(
                gate.url + "/responses",
                json=dict(
                    model="gpt-6.1-sol",
                    reasoning=dict(effort="medium"),
                    input="drop before headers",
                ),
                trust_env=False,
                timeout=2,
            )
            self.assertEqual(response.status_code, 502)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(gate.sent, 1)


if __name__ == "__main__":
    unittest.main()
