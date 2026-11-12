"""Count actual Codex HTTP model requests before forwarding them.

The gate keeps Codex's authentication and request body intact. Only request
identity, effective model options and returned usage are persisted; credentials,
prompts, images and model text are never written by this transport.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..storage import atomic_json


class RequestGate:
    def __init__(self, context, *, upstream_base='https://chatgpt.com/backend-api/codex'):
        self.context = context
        self.output = Path(context['output_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.limit = context['limits']['model_requests']
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError('model_requests must be a positive integer')
        self.upstream = upstream_base.rstrip('/')
        self.started = time.monotonic()
        self.deadline = context.get('job_started_monotonic', self.started) + context['limits']['wall_seconds']
        self.lock = threading.RLock()
        self.sent = 0
        self.rejected = 0
        self.events = []
        self.active = set()
        self.closing = False
        self.publish_live()

    def publish_live(self):
        atomic_json(self.output / 'request_gate_live.json', {
            'identity': {key: self.context.get(key) for key in ('run_id','episode_id','bundle_id','protocol_id')},
            'requests_forwarded': self.sent, 'request_limit': self.limit,
        })

    def stop(self, reason, limit, observed):
        import os
        import tempfile
        path = Path(self.context.get('budget_stop_ref', self.output/'budget_stop.json'))
        self.emit('budget_stop', reason=reason, limit=limit, observed=observed)
        record = dict(schema_version='1.0',
            identity={key: self.context.get(key) for key in ('run_id','episode_id','bundle_id','protocol_id','profile_ref','parent_decision_id')},
            reason=reason, limit=limit, observed=observed, stop_after_inflight=True,
            producer_events_ref=str(self.output/'model_requests.jsonl'), sequence=len(self.events)-1,
            monotonic_ns=time.monotonic_ns())
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try: os.link(temporary, path)
            except FileExistsError: pass
        finally:
            temporary.unlink()


    def emit(self, kind, **fields):
        with self.lock:
            event = {'sequence': len(self.events), 'kind': kind, 'monotonic_seconds': time.monotonic(), **fields}
            self.events.append(event)
            with (self.output / 'model_requests.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')

    def admit(self, payload):
        with self.lock:
            expected = self.context['planner']
            actual_model = payload.get('model')
            actual_effort = (payload.get('reasoning') or {}).get('effort')
            if (actual_model != expected['model'] or actual_effort != expected['reasoning_effort']):
                self.rejected += 1
                self.emit('rejected', reason='actual request model/effort mismatch',
                          model=actual_model, reasoning_effort=actual_effort)
                raise ValueError('actual model/effort must match the frozen profile')
            native_path = self.output/'native_completion.json'
            if native_path.exists():
                native = json.loads(native_path.read_text())
                if any(native.get('identity', {}).get(key) != self.context.get(key) for key in
                       ('run_id','episode_id','bundle_id','protocol_id')) or native.get('reason') != 'native_success':
                    raise ValueError('native completion identity/reason mismatch')
                self.rejected += 1
                self.emit('rejected', reason='native task completed', evidence_ref=str(native_path))
                raise PermissionError('native task completed; no further Planner request forwarded')
            runtime_stop = self.output/'runtime_stop.json'
            if runtime_stop.exists():
                stop = json.loads(runtime_stop.read_text())
                if any(stop.get('identity', {}).get(key) != self.context.get(key) for key in
                       ('run_id','episode_id','bundle_id','protocol_id')):
                    raise ValueError('runtime stop identity mismatch')
                self.rejected += 1
                self.emit('rejected', reason='runtime stopped', runtime_reason=stop.get('reason'))
                raise PermissionError('runtime stopped; no request forwarded')
            stop_path = Path(self.context.get('budget_stop_ref', self.output / 'budget_stop.json'))
            if stop_path.exists():
                stop = json.loads(stop_path.read_text())
                identity = stop.get('identity', {})
                if any(identity.get(key) != self.context.get(key) for key in
                       ('run_id', 'episode_id', 'bundle_id', 'protocol_id')):
                    raise ValueError('runtime budget stop identity mismatch')
                self.rejected += 1
                self.emit('rejected', reason='runtime budget exhausted',
                          runtime_reason=stop.get('reason'), evidence_ref=str(stop_path))
                raise PermissionError('runtime budget exhausted; no further model request forwarded')
            if self.closing or time.monotonic() >= self.deadline or self.sent >= self.limit:
                self.rejected += 1
                self.stop('model_requests' if self.sent >= self.limit else 'wall_seconds',
                          self.limit if self.sent >= self.limit else self.context['limits']['wall_seconds'],
                          self.sent if self.sent >= self.limit else time.monotonic()-self.started)
                self.emit('rejected', reason='model request count or job wall limit')
                raise PermissionError('model request budget exhausted; no request forwarded')
            self.sent += 1
            self.publish_live()
            self.emit('request_admitted', request_number=self.sent, model=actual_model,
                      reasoning_effort=actual_effort, configuration_source='actual HTTP JSON body',
                      tool_catalog=[{'type':tool.get('type'),'name':tool.get('name'),
                          'nested_names':[item.get('name') for item in tool.get('tools',[])]}
                          for tool in payload.get('tools',[])])
            return self.sent

    def observe_event(self, event, request_number):
        if event.get('type') not in {'response.completed', 'response.incomplete', 'response.failed'}:
            return
        response = event.get('response') or {}
        self.emit('response_terminal', request_number=request_number, event_type=event['type'],
                  response_id=response.get('id'), backend_model=response.get('model'),
                  usage=response.get('usage'), status=response.get('status'))

    def __enter__(self):
        gate = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def reject(self, code, message):
                body = json.dumps({'error': {'message': message, 'type': 'rsi_budget_gate'}}).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if urlsplit(self.path).path not in {'/responses', '/v1/responses'}:
                    self.reject(403, 'unregistered model endpoint; no request forwarded')
                    return
                try:
                    original = self.rfile.read(int(self.headers['Content-Length']))
                    decoded = original
                    if self.headers.get('Content-Encoding') == 'zstd':
                        import zstandard
                        decoded = zstandard.ZstdDecompressor().decompress(original)
                    elif self.headers.get('Content-Encoding') not in (None, 'identity'):
                        raise ValueError('unregistered request compression')
                    payload = json.loads(decoded)
                    request_number = gate.admit(payload)
                except (ValueError, KeyError, TypeError) as error:
                    self.reject(400, str(error))
                    return
                except PermissionError as error:
                    self.reject(429, str(error))
                    return
                import httpx
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in {'host', 'connection', 'content-length', 'transfer-encoding'}}
                remaining = max(.1, gate.deadline - time.monotonic())
                client = httpx.Client(follow_redirects=False, timeout=httpx.Timeout(min(300, remaining), connect=min(15,remaining)))
                request_started = time.monotonic()
                response = None
                headers_sent = False
                try:
                    request = client.build_request('POST', gate.upstream + '/responses', content=original, headers=headers)
                    response = client.send(request, stream=True)
                    with gate.lock:
                        gate.active.add(response)
                    self.send_response(response.status_code)
                    self.send_header('Content-Type', response.headers.get('content-type', 'application/json'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    headers_sent = True
                    self.close_connection = True
                    buffer = b''
                    for chunk in response.iter_bytes():
                        if gate.closing or time.monotonic() >= gate.deadline:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        buffer += chunk
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            if line.startswith(b'data:'):
                                try:
                                    gate.observe_event(json.loads(line[5:].strip()), request_number)
                                except (ValueError, TypeError):
                                    pass
                    gate.emit('request_returned', request_number=request_number, http_status=response.status_code, elapsed_seconds=time.monotonic()-request_started)
                except Exception as error:
                    # Do not log exception text that might contain authorization headers.
                    gate.emit('transport_error', request_number=request_number, error_type=type(error).__name__, elapsed_seconds=time.monotonic()-request_started)
                    if not headers_sent:
                        self.reject(502, 'upstream transport failed; charged request is not automatically retried')
                    self.close_connection = True
                finally:
                    if response is not None:
                        with gate.lock:
                            gate.active.discard(response)
                        response.close()
                    client.close()
                    with gate.lock:
                        if gate.sent >= gate.limit:
                            gate.stop('model_requests', gate.limit, gate.sent)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        return self

    def __exit__(self, *_args):
        self.closing = True
        with self.lock:
            for response in list(self.active):
                response.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        atomic_json(self.output / 'request_gate.json', {
            'identity': {key: self.context.get(key) for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')},
            'requests_forwarded': self.sent, 'requests_rejected': self.rejected,
            'request_limit': self.limit, 'hard_request_cap_enforced': True,
            'token_policy': 'observed_only', 'upstream': self.upstream,
            'active_requests_at_close': len(self.active),
            'model_options_source': 'actual model HTTP request body',
        })
