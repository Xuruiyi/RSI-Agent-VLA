"""Loopback fixtures proving pre-forward request admission, no real model."""
import json
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
        calls = self.calls
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(payload)
                if payload.get('input') == 'drop before headers':
                    self.close_connection = True
                    return
                body = b'data: {"type":"response.completed","response":{"id":"offline","model":"gpt-6.1-sol","status":"completed","usage":{"input_tokens":2,"output_tokens":1}}}\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.01}, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.context = dict(output_dir=self.temp.name, planner=dict(model='gpt-6.1-sol', reasoning_effort='medium'),
                            limits=dict(model_requests=1, wall_seconds=5),run_id='offline',episode_id='e',bundle_id='b',protocol_id='p')

    def gate(self):
        return RequestGate(self.context, upstream_base=f'http://127.0.0.1:{self.server.server_port}')

    def test_supervisor_cancel_blocks_forward_and_leaves_request_count_unchanged(self):
        identity={key:self.context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
        (Path(self.temp.name)/'runtime_stop.json').write_text(json.dumps(
            dict(identity=identity, reason='external_interruption')))
        with self.gate() as gate:
            response=httpx.post(gate.url+'/responses',json=dict(model='gpt-6.1-sol',
                reasoning=dict(effort='medium')),trust_env=False)
            self.assertEqual(response.status_code,429)
            self.assertEqual(gate.sent,0)
        self.assertEqual(self.calls,[])

    def test_native_success_blocks_forward_without_budget_or_cancel(self):
        identity={key:self.context[key] for key in ('run_id','episode_id','bundle_id','protocol_id')}
        (Path(self.temp.name)/'native_completion.json').write_text(json.dumps(
            dict(identity=identity, reason='native_success', step=1)))
        with self.gate() as gate:
            response=httpx.post(gate.url+'/responses',json=dict(model='gpt-6.1-sol',
                reasoning=dict(effort='medium')),trust_env=False)
            self.assertEqual(response.status_code,429)
            self.assertEqual(gate.sent,0)
        self.assertEqual(self.calls,[])
        self.assertFalse((Path(self.temp.name)/'budget_stop.json').exists())
        self.assertFalse((Path(self.temp.name)/'runtime_stop.json').exists())

    def test_one_request_admitted_and_second_not_forwarded(self):
        payload = dict(model='gpt-6.1-sol', reasoning=dict(effort='medium'), input='offline fixture')
        with self.gate() as gate:
            first = httpx.post(gate.url+'/responses', json=payload, trust_env=False)
            second = httpx.post(gate.url+'/responses', json=payload, trust_env=False)
            self.assertEqual((first.status_code, second.status_code), (200, 429))
        self.assertEqual(len(self.calls), 1)
        marker=json.loads((Path(self.temp.name)/'budget_stop.json').read_text())
        self.assertEqual(marker['reason'],'model_requests')
        self.assertEqual(marker['limit'],marker['observed'])
        live=json.loads((Path(self.temp.name)/'request_gate_live.json').read_text())
        self.assertEqual(live['requests_forwarded'],1)
        report = json.loads((Path(self.temp.name)/'request_gate.json').read_text())
        self.assertEqual((report['requests_forwarded'], report['requests_rejected']), (1, 1))
        events = (Path(self.temp.name)/'model_requests.jsonl').read_text()
        self.assertNotIn('offline fixture', events)
        self.assertIn('response_terminal', events)

    def test_wrong_effective_effort_or_endpoint_never_reaches_upstream(self):
        with self.gate() as gate:
            response = httpx.post(gate.url+'/responses', json=dict(model='gpt-6.1-sol', reasoning=dict(effort='low')),trust_env=False)
            compact = httpx.post(gate.url+'/responses/compact', json={},trust_env=False)
            self.assertEqual((response.status_code,compact.status_code),(400,403))
        self.assertEqual(self.calls, [])

    def test_runtime_budget_stop_blocks_next_forward_without_charging_request(self):
        identity = {key: self.context[key] for key in
                    ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
        (Path(self.temp.name)/'budget_stop.json').write_text(json.dumps(
            dict(identity=identity, reason='tool_calls', sequence=42)))
        with self.gate() as gate:
            response = httpx.post(gate.url+'/responses', json=dict(
                model='gpt-6.1-sol', reasoning=dict(effort='medium')), trust_env=False)
            self.assertEqual(response.status_code, 429)
        self.assertEqual(self.calls, [])
        report = json.loads((Path(self.temp.name)/'request_gate.json').read_text())
        self.assertEqual((report['requests_forwarded'], report['requests_rejected']), (0, 1))

    def test_upstream_failure_returns_502_and_closes_downstream_without_retry(self):
        with self.gate() as gate:
            response = httpx.post(gate.url+'/responses',json=dict(model='gpt-6.1-sol',
                reasoning=dict(effort='medium'),input='drop before headers'),trust_env=False,timeout=2)
            self.assertEqual(response.status_code,502)
            self.assertEqual(len(self.calls),1)
            self.assertEqual(gate.sent,1)


if __name__ == '__main__':
    unittest.main()
