# Copyright 2026 RSI-Agent-VLA Authors.
# ruff: noqa: UP017 - Python 3.10 support
"""Loopback-only observed token stops share the original Host accounting."""

import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from rsi_vla.adapters.request_gate import RequestGate
from rsi_vla.storage import atomic_json


class ScopeRequestGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        calls = self.calls

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                calls.append(payload)
                mode = payload["input"]
                usage = {
                    "input_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 7,
                }
                if mode == "missing":
                    usage.pop("output_tokens_details")
                elif mode == "inconsistent":
                    usage["total_tokens"] = 9
                model = "wrong-backend" if mode == "wrong_model" else "gpt-6.1-sol"
                response = {
                    "type": "response.failed"
                    if mode == "failed"
                    else "response.completed",
                    "response": {
                        "id": "fixture-" + str(len(calls)),
                        "model": model,
                        "usage": usage,
                    },
                }
                body = ("data: " + json.dumps(response) + "\n\n").encode()
                if mode == "transport_failure":
                    body = b'{"error":"offline fixture failure"}'
                self.send_response(502 if mode == "transport_failure" else 200)
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
        self.scope_ref = self.root / "scope.json"
        self.accounting_ref = self.root / "accounting.json"
        limits = {
            "model_requests": 4,
            "research_requests": 1,
            "input_tokens": 100,
            "output_tokens": 100,
        }
        self.scope = {
            "schema": "rsi-midterm-scope-v1",
            "scope_id": "offline-scope",
            "owner_uid": os.getuid(),
            "authorized": True,
            "status": "active",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "outer_limits": limits,
            "token_policy": {
                "mode": "observed_cumulative_stop_threshold",
                "input_tokens_threshold": 100,
                "output_tokens_including_reasoning_threshold": 100,
            },
        }
        self.ledger = {
            "scope_id": "offline-scope",
            "limits": limits,
            "used": dict.fromkeys(limits, 0),
            "unresolved_costs": [],
            "reservations": {
                "runtime": {
                    "scope_id": "offline-scope",
                    "status": "active",
                    "budget": {
                        "model_requests": 4,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                }
            },
        }
        self.context = {
            "run_id": "offline-runtime",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
            "output_dir": str(self.root / "output"),
            "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
            "limits": {"model_requests": 4, "wall_seconds": 10},
            "request_scope": {
                "scope_authorization_ref": str(self.scope_ref),
                "accounting_ref": str(self.accounting_ref),
                "operation_id": "runtime",
            },
        }
        self.save()

    def save(self):
        atomic_json(self.scope_ref, self.scope)
        atomic_json(self.accounting_ref, self.ledger)

    def gate(self):
        return RequestGate(
            self.context, upstream_base=f"http://127.0.0.1:{self.server.server_port}"
        )

    def post(self, gate, mode="known"):
        return httpx.post(
            gate.url + "/responses",
            trust_env=False,
            timeout=3,
            json={
                "model": "gpt-6.1-sol",
                "reasoning": {"effort": self.context["planner"]["reasoning_effort"]},
                "input": mode,
            },
        )

    def test_one_inflight_may_cross_threshold_but_next_http_never_forwards(self):
        self.scope["outer_limits"].update(input_tokens=10, output_tokens=8)
        self.scope["token_policy"].update(
            input_tokens_threshold=10, output_tokens_including_reasoning_threshold=8
        )
        self.ledger["used"].update(input_tokens=8, output_tokens=7)
        self.save()
        original = self.accounting_ref.read_bytes()
        with self.gate() as gate:
            self.assertEqual(self.post(gate).status_code, 200)
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.accounting_ref.read_bytes(), original)
        self.assertIn(
            "stop threshold", (self.root / "output/model_requests.jsonl").read_text()
        )

    def test_missing_inconsistent_unattributed_or_transport_usage_pauses_next(self):
        for mode in ("missing", "inconsistent", "wrong_model", "transport_failure"):
            with self.subTest(mode=mode):
                self.context["output_dir"] = str(self.root / mode)
                before = len(self.calls)
                with self.gate() as gate:
                    first = self.post(gate, mode)
                    self.assertEqual(
                        first.status_code, 502 if mode == "transport_failure" else 200
                    )
                    self.assertEqual(self.post(gate).status_code, 429)
                    self.assertEqual(gate.sent, 1)
                self.assertEqual(len(self.calls) - before, 1)

    def test_failed_responses_and_retries_share_the_hard_four_http_cap(self):
        with self.gate() as gate:
            for _ in range(4):
                self.assertEqual(self.post(gate, "failed").status_code, 200)
            self.assertEqual(self.post(gate, "failed").status_code, 429)
            self.assertEqual(gate.sent, 4)
        self.assertEqual(len(self.calls), 4)

    def test_prior_scope_requests_and_tokens_are_not_reset_by_a_new_gate(self):
        self.ledger["used"]["model_requests"] = 3
        self.save()
        with self.gate() as gate:
            self.assertEqual(self.post(gate).status_code, 200)
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 1)

    def test_scope_research_count_refuses_before_forward(self):
        self.context["job_id"] = "research"
        self.context["planner"]["reasoning_effort"] = "high"
        self.ledger["reservations"]["runtime"]["budget"]["research_requests"] = 4
        self.ledger["used"]["research_requests"] = 1
        self.save()
        with self.gate() as gate:
            self.assertEqual(self.post(gate).status_code, 429)
            self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])

    def test_unresolved_or_already_charged_operation_never_starts_another_http(self):
        for case in ("unresolved", "already_charged"):
            with self.subTest(case=case):
                self.context["output_dir"] = str(self.root / case)
                if case == "unresolved":
                    self.ledger["unresolved_costs"] = [
                        {"operation_id": "prior", "missing": ["output_tokens"]}
                    ]
                else:
                    self.ledger["unresolved_costs"] = []
                    self.ledger["reservations"]["runtime"]["actual"] = {
                        "model_requests": 1
                    }
                self.save()
                with self.gate() as gate:
                    self.assertEqual(self.post(gate).status_code, 429)
                    self.assertEqual(gate.sent, 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
