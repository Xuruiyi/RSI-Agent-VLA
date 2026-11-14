"""Count actual Codex HTTP model requests before forwarding them.

The gate keeps Codex's authentication and request body intact. Only request
identity, effective model options and returned usage are persisted; credentials,
prompts, images and model text are never written by this transport.
"""

import asyncio
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..storage import atomic_json


class _UnexpectedDownstreamClose(Exception):
    pass


class RequestGate:
    def __init__(
        self,
        context,
        *,
        upstream_base="https://chatgpt.com/backend-api/codex",
        cleanup_timeout_s=5.0,
    ):
        self.context = context
        self.output = Path(context["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.limit = context["limits"]["model_requests"]
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("model_requests must be a positive integer")
        self.upstream = upstream_base.rstrip("/")
        self.started = time.monotonic()
        self.deadline = (
            context.get("job_started_monotonic", self.started)
            + context["limits"]["wall_seconds"]
        )
        if context["limits"].get("gpu_seconds") is not None:
            self.deadline = min(
                self.deadline,
                context.get("job_started_monotonic", self.started)
                + context["limits"]["gpu_seconds"] / context["gpu_count"],
            )
        if not 0 < cleanup_timeout_s <= 5.0:
            raise ValueError(
                "transport cleanup must fit the existing five-second grace"
            )
        self.cleanup_timeout_s = cleanup_timeout_s
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.sent = 0
        self.rejected = 0
        self.events = []
        self.active = {}
        self.aborting = False
        self.closing = False
        self.publish_live()

    def publish_live(self):
        atomic_json(
            self.output / "request_gate_live.json",
            {
                "identity": {
                    key: self.context.get(key)
                    for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                },
                "requests_forwarded": self.sent,
                "request_limit": self.limit,
            },
        )

    def stop(self, reason, limit, observed):
        import os
        import tempfile

        path = Path(
            self.context.get("budget_stop_ref", self.output / "budget_stop.json")
        )
        self.emit("budget_stop", reason=reason, limit=limit, observed=observed)
        record = dict(
            schema_version="1.0",
            identity={
                key: self.context.get(key)
                for key in (
                    "run_id",
                    "episode_id",
                    "bundle_id",
                    "protocol_id",
                    "profile_ref",
                    "parent_decision_id",
                )
            },
            reason=reason,
            limit=limit,
            observed=observed,
            stop_after_inflight=True,
            producer_events_ref=str(self.output / "model_requests.jsonl"),
            sequence=len(self.events) - 1,
            monotonic_ns=time.monotonic_ns(),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()

    def emit(self, kind, **fields):
        with self.lock:
            event = {
                "identity": {key: self.context.get(key) for key in
                             ("run_id", "episode_id", "bundle_id", "protocol_id")},
                "sequence": len(self.events),
                "kind": kind,
                "monotonic_seconds": time.monotonic(),
                **fields,
            }
            if kind == "rejected":
                event["forwarded"] = False
            self.events.append(event)
            with (self.output / "model_requests.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def publish_local_stop(self, reason, marker_path):
        """Bind an unforwarded rejection to its persisted owner marker."""
        event = self.events[-1]
        marker = json.loads(Path(marker_path).read_text())
        atomic_json(self.output / "local_model_stop.json", {
            "schema_version": "1.0", "identity": event["identity"],
            "reason": reason, "forwarded": False,
            "rejection_sequence": event["sequence"],
            "monotonic_seconds": event["monotonic_seconds"],
            "model_requests_ref": str(self.output / "model_requests.jsonl"),
            "marker_ref": str(marker_path), "marker_sequence": marker.get("sequence"),
        })

    def native_completion(self):
        path = self.output / "native_completion.json"
        if not path.exists():
            return None
        native = json.loads(path.read_text())
        if (
            any(
                native.get("identity", {}).get(k) != self.context.get(k)
                for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
            )
            or native.get("reason") != "native_success"
        ):
            raise ValueError("native completion identity/reason mismatch")
        return native

    def wait_for_native_completion(self):
        """Hold an unforwarded request locally while already admitted tools persist state."""
        path = self.output / "planner_admission_closed.json"
        if not path.exists():
            return
        marker = json.loads(path.read_text())
        if marker.get("reason") != "native_success_observed" or any(
            marker.get("identity", {}).get(k) != self.context.get(k)
            for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
        ):
            raise ValueError("native admission boundary identity/reason mismatch")
        self.emit("admission_wait", evidence_ref=str(path))
        while (
            self.native_completion() is None
            and not self.closing
            and time.monotonic() < self.deadline
        ):
            if (self.output / "runtime_stop.json").exists() or Path(
                self.context.get("budget_stop_ref", self.output / "budget_stop.json")
            ).exists():
                break
            self.condition.wait(
                timeout=min(0.02, max(0, self.deadline - time.monotonic()))
            )
        self.rejected += 1
        completed = self.native_completion() is not None
        self.emit(
            "rejected",
            reason="native task completed"
            if completed
            else "native state capture incomplete",
            evidence_ref=str(path),
        )
        if completed:
            self.publish_local_stop("native_success", self.output / "native_completion.json")
            raise PermissionError(
                "native task completed; no further Planner request forwarded"
            )
        raise PermissionError(
            "native state capture incomplete; no further Planner request forwarded"
        )

    def admit(self, payload):
        with self.lock:
            expected = self.context["planner"]
            actual_model = payload.get("model")
            actual_effort = (payload.get("reasoning") or {}).get("effort")
            if (
                actual_model != expected["model"]
                or actual_effort != expected["reasoning_effort"]
            ):
                self.rejected += 1
                self.emit(
                    "rejected",
                    reason="actual request model/effort mismatch",
                    model=actual_model,
                    reasoning_effort=actual_effort,
                )
                raise ValueError("actual model/effort must match the frozen profile")
            self.wait_for_native_completion()
            native_path = self.output / "native_completion.json"
            if self.native_completion() is not None:
                self.rejected += 1
                self.emit(
                    "rejected",
                    reason="native task completed",
                    evidence_ref=str(native_path),
                )
                self.publish_local_stop("native_success", native_path)
                raise PermissionError(
                    "native task completed; no further Planner request forwarded"
                )
            runtime_stop = self.output / "runtime_stop.json"
            if runtime_stop.exists():
                stop = json.loads(runtime_stop.read_text())
                if any(
                    stop.get("identity", {}).get(key) != self.context.get(key)
                    for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                ):
                    raise ValueError("runtime stop identity mismatch")
                self.rejected += 1
                self.emit(
                    "rejected",
                    reason="runtime stopped",
                    runtime_reason=stop.get("reason"),
                )
                raise PermissionError("runtime stopped; no request forwarded")
            stop_path = Path(
                self.context.get("budget_stop_ref", self.output / "budget_stop.json")
            )
            if stop_path.exists():
                stop = json.loads(stop_path.read_text())
                identity = stop.get("identity", {})
                if any(
                    identity.get(key) != self.context.get(key)
                    for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                ):
                    raise ValueError("runtime budget stop identity mismatch")
                self.rejected += 1
                self.emit(
                    "rejected",
                    reason="runtime budget exhausted",
                    runtime_reason=stop.get("reason"),
                    evidence_ref=str(stop_path),
                )
                self.publish_local_stop("budget_exhausted", stop_path)
                raise PermissionError(
                    "runtime budget exhausted; no further model request forwarded"
                )
            if self.closing:
                self.rejected += 1
                self.emit("rejected", reason="request transport closing")
                raise PermissionError("request transport closing; no request forwarded")
            if time.monotonic() >= self.deadline or self.sent >= self.limit:
                self.rejected += 1
                self.stop(
                    "model_requests" if self.sent >= self.limit else "wall_seconds",
                    self.limit
                    if self.sent >= self.limit
                    else self.context["limits"]["wall_seconds"],
                    self.sent
                    if self.sent >= self.limit
                    else time.monotonic() - self.started,
                )
                self.emit("rejected", reason="model request count or job wall limit")
                if self.sent >= self.limit:
                    self.publish_local_stop("budget_exhausted", Path(
                        self.context.get("budget_stop_ref", self.output / "budget_stop.json")
                    ))
                raise PermissionError(
                    "model request budget exhausted; no request forwarded"
                )
            self.sent += 1
            self.publish_live()
            self.emit(
                "request_admitted",
                request_number=self.sent,
                model=actual_model,
                reasoning_effort=actual_effort,
                configuration_source="actual HTTP JSON body",
                tool_catalog=[
                    {
                        "type": tool.get("type"),
                        "name": tool.get("name"),
                        "nested_names": [
                            item.get("name") for item in tool.get("tools", [])
                        ],
                    }
                    for tool in payload.get("tools", [])
                ],
            )
            return self.sent

    def observe_event(self, event, request_number):
        if event.get("type") not in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            return
        response = event.get("response") or {}
        self.emit(
            "response_terminal",
            request_number=request_number,
            event_type=event["type"],
            response_id=response.get("id"),
            backend_model=response.get("model"),
            usage=response.get("usage"),
            status=response.get("status"),
        )
        return True

    def __enter__(self):
        gate = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def reject(self, code, message):
                self.close_connection = True
                body = json.dumps(
                    {"error": {"message": message, "type": "rsi_budget_gate"}}
                ).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def cancel_forward(self, reason):
                self.cancel_reason = reason
                if self.loop is not None and self.task is not None:
                    try:
                        self.loop.call_soon_threadsafe(self.task.cancel)
                    except RuntimeError:
                        # asyncio.run finished; the handler still owns its final accounting.
                        if not self.loop.is_closed():
                            raise
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            async def forward(self, original, request_number):
                import httpx

                self.loop = asyncio.get_running_loop()
                self.task = asyncio.current_task()
                request_started = time.monotonic()
                timer = None
                terminal_seen = False
                downstream_closed = False
                headers_sent = False

                def arm_timeout(seconds, reason):
                    nonlocal timer
                    if timer is not None:
                        timer.cancel()

                    def cancel():
                        self.cancel_reason = reason
                        self.task.cancel()

                    timer = self.loop.call_later(max(0, seconds), cancel)

                def client_write(callback):
                    nonlocal downstream_closed
                    if downstream_closed:
                        return
                    try:
                        callback()
                    except (BrokenPipeError, ConnectionResetError) as error:
                        native = gate.native_completion()
                        gate.emit(
                            "downstream_closed",
                            request_number=request_number,
                            error_type=type(error).__name__,
                            reason="native_success"
                            if native
                            else "client_disconnected",
                            native_completion_sequence=native.get("sequence")
                            if native
                            else None,
                            native_completion_ref=str(
                                gate.output / "native_completion.json"
                            )
                            if native
                            else None,
                        )
                        downstream_closed = True
                        self.close_connection = True
                        if native is None:
                            raise _UnexpectedDownstreamClose from None
                        arm_timeout(
                            min(
                                gate.cleanup_timeout_s, gate.deadline - time.monotonic()
                            ),
                            "native_drain_timeout",
                        )

                try:
                    if gate.aborting:
                        raise asyncio.CancelledError
                    remaining = max(0, gate.deadline - time.monotonic())
                    arm_timeout(remaining, "job_deadline")
                    headers = {
                        key: value
                        for key, value in self.headers.items()
                        if key.lower()
                        not in {
                            "host",
                            "connection",
                            "content-length",
                            "transfer-encoding",
                        }
                    }
                    async with httpx.AsyncClient(
                        follow_redirects=False,
                        timeout=httpx.Timeout(
                            max(0.01, min(300, remaining)),
                            connect=max(0.01, min(15, remaining)),
                        ),
                    ) as client:
                        async with client.stream(
                            "POST",
                            gate.upstream + "/responses",
                            content=original,
                            headers=headers,
                        ) as response:

                            def send_headers():
                                self.send_response(response.status_code)
                                self.send_header(
                                    "Content-Type",
                                    response.headers.get(
                                        "content-type", "application/json"
                                    ),
                                )
                                self.send_header("Connection", "close")
                                self.end_headers()

                            client_write(send_headers)
                            headers_sent = True
                            self.close_connection = True
                            buffer = b""
                            async for chunk in response.aiter_bytes():
                                # Record returned evidence before a local consumer can disconnect.
                                buffer += chunk
                                while b"\n" in buffer:
                                    line, buffer = buffer.split(b"\n", 1)
                                    if line.startswith(b"data:"):
                                        try:
                                            terminal_seen = (
                                                bool(
                                                    gate.observe_event(
                                                        json.loads(line[5:].strip()),
                                                        request_number,
                                                    )
                                                )
                                                or terminal_seen
                                            )
                                        except (ValueError, TypeError):
                                            pass

                                def send_chunk():
                                    self.wfile.write(chunk)
                                    self.wfile.flush()

                                client_write(send_chunk)
                            kind = (
                                "request_returned"
                                if terminal_seen
                                else "request_incomplete"
                            )
                            gate.emit(
                                kind,
                                request_number=request_number,
                                http_status=response.status_code,
                                reason=None
                                if terminal_seen
                                else "upstream_eof_without_terminal",
                                elapsed_seconds=time.monotonic() - request_started,
                            )
                except asyncio.CancelledError:
                    gate.emit(
                        "request_incomplete",
                        request_number=request_number,
                        reason=self.cancel_reason or "transport_cleanup",
                        elapsed_seconds=time.monotonic() - request_started,
                    )
                except _UnexpectedDownstreamClose:
                    gate.emit(
                        "request_incomplete",
                        request_number=request_number,
                        reason="unexpected_downstream_close",
                        elapsed_seconds=time.monotonic() - request_started,
                    )
                except Exception as error:
                    gate.emit(
                        "transport_error",
                        request_number=request_number,
                        error_type=type(error).__name__,
                        elapsed_seconds=time.monotonic() - request_started,
                    )
                    if not headers_sent:
                        self.reject(
                            502,
                            "upstream transport failed; charged request is not automatically retried",
                        )
                finally:
                    if timer is not None:
                        timer.cancel()
                    self.close_connection = True

            def do_POST(self):
                if urlsplit(self.path).path not in {"/responses", "/v1/responses"}:
                    self.reject(
                        403, "unregistered model endpoint; no request forwarded"
                    )
                    return
                try:
                    original = self.rfile.read(int(self.headers["Content-Length"]))
                    decoded = original
                    if self.headers.get("Content-Encoding") == "zstd":
                        import zstandard

                        decoded = zstandard.ZstdDecompressor().decompress(original)
                    elif self.headers.get("Content-Encoding") not in (None, "identity"):
                        raise ValueError("unregistered request compression")
                    payload = json.loads(decoded)
                    self.loop = self.task = None
                    self.cancel_reason = None
                    with gate.condition:
                        request_number = gate.admit(payload)
                        gate.active[request_number] = self
                except (ValueError, KeyError, TypeError) as error:
                    self.reject(400, str(error))
                    return
                except PermissionError as error:
                    self.reject(429, str(error))
                    return
                try:
                    asyncio.run(self.forward(original, request_number))
                finally:
                    with gate.condition:
                        gate.active.pop(request_number, None)
                        if gate.sent >= gate.limit:
                            gate.stop("model_requests", gate.limit, gate.sent)
                        gate.condition.notify_all()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *_args):
        # Stop admission first; existing forwards have one bounded cleanup window.
        started = time.monotonic()
        cleanup_deadline = min(self.deadline - 0.05, started + self.cleanup_timeout_s)
        drain_deadline = max(
            started, cleanup_deadline - min(1.0, self.cleanup_timeout_s / 2)
        )
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.server.shutdown()
        with self.condition:
            self.condition.wait_for(
                lambda: not self.active,
                timeout=max(0, drain_deadline - time.monotonic()),
            )
            self.aborting = True
            for handler in list(self.active.values()):
                handler.cancel_forward("transport_cleanup_timeout")
            self.condition.wait_for(
                lambda: not self.active,
                timeout=max(0, cleanup_deadline - time.monotonic()),
            )
            unfinished = sorted(self.active)
        self.server.server_close()
        self.thread.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        atomic_json(
            self.output / "request_gate.json",
            {
                "identity": {
                    key: self.context.get(key)
                    for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
                },
                "requests_forwarded": self.sent,
                "requests_rejected": self.rejected,
                "request_limit": self.limit,
                "hard_request_cap_enforced": True,
                "token_policy": "observed_only",
                "upstream": self.upstream,
                "active_requests_at_close": len(unfinished),
                "unfinished_request_numbers": unfinished,
                "cleanup_complete": not unfinished and not self.thread.is_alive(),
                "transport_cleanup_seconds": time.monotonic() - started,
                "model_options_source": "actual model HTTP request body",
            },
        )
