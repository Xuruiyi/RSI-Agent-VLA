"""Translate trusted Autoloop bindings into the fixed protocol10 CLI."""

import fcntl
import json
import subprocess
import sys
import time
import re
import urllib.request
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import atomic_json, identifier


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class AutoloopBridge:
    """Consume event data using local bindings; event prose has no authority."""

    def __init__(self, configuration, *, invoke=None):
        self.config = configuration
        self.root = Path(configuration["receipts_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        identifier(configuration["campaign_id"])
        self.invoke = invoke or self._invoke

    def _invoke(self, command, *, event=None, trigger_id=None):
        if self.config.get("transport_root"):
            transport = Path(self.config["transport_root"])
            identity = uuid4().hex
            request = {
                "request_id": identity,
                "action": "continue"
                if command in {"advance", "reconcile"}
                else command,
            }
            if event is not None:
                request["event"] = event
            if trigger_id is not None:
                request["trigger_id"] = trigger_id
            atomic_json(transport / "requests" / (identity + ".json"), request)
            response = transport / "responses" / (identity + ".json")
            deadline = time.monotonic() + self.config.get("rpc_timeout_seconds", 950)
            while not response.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "host receipt pending; reconcile the same core operation"
                    )
                time.sleep(0.02)
            result = _read(response)
            if result["request_id"] != identity:
                raise ValueError("host response identity mismatch")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result["result"]
        consumer = self.config.get("core_consumer", "protocol10")
        scripts = {"protocol10": "run_protocol10.py",
                   "task_continuation": "run_task_continuation.py"}
        if consumer not in scripts:
            raise ValueError("unsupported registered core consumer")
        argv = [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "scripts" / scripts[consumer]),
            command,
            "--root",
            self.config["core_root"],
            "--configuration",
            self.config["core_configuration_ref"],
        ]
        if self.config.get("cpu_boundary_factory"):
            argv += ["--cpu-boundary-factory", self.config["cpu_boundary_factory"]]
        if command in {"advance", "reconcile"}:
            argv += ["--inputs", self.config["inputs_ref"], "--trigger-id", trigger_id]
        if event is not None:
            path = self.root / (identifier(event["event_id"]) + "-signal.json")
            atomic_json(path, event)
            argv += ["--event", str(path)]
        returned = subprocess.run(argv, text=True, capture_output=True, check=False)
        try:
            result = json.loads(returned.stdout)
        except ValueError as exc:
            raise RuntimeError("core CLI did not return a JSON receipt") from exc
        if returned.returncode or result.get("error"):
            raise RuntimeError(result.get("error") or returned.stderr[-1000:])
        return result

    def _signal(self, event):
        expected = self.config["events"].get(event.get("id"))
        if expected is None or any(
            event.get(k) != expected["external"].get(k)
            for k in ("id", "source", "type", "subject")
        ):
            raise ValueError("unregistered external event identity")
        if expected.get("reference_ref"):
            record = _read(expected["reference_ref"])
            if any(
                record.get(k) != v for k, v in expected["reference_identity"].items()
            ):
                raise ValueError("external reference identity mismatch")
        return expected["signal"]

    def consume(self, context):
        """Return one terminal worker receipt, retaining core waits and receipts."""
        with (self.root / "bridge.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {
                    "summary": "协调入口正在处理，本轮结束。",
                    "status": "waiting",
                    "ask": [],
                }
            state = self.invoke("status")
            for event in context.get("events", []):
                # An answer is verified separately, never interpreted as external prose.
                if event.get("type") == "card.answered" and any(
                    event.get("id") == "card:" + str(answer.get("card"))
                    for answer in context.get("answers", [])
                ):
                    continue
                signal = self._signal(event)
                key = identifier(signal["event_id"])
                path = self.root / (key + "-event.json")
                if path.exists():
                    saved = _read(path)
                    if saved["external"] != event:
                        raise ValueError("conflicting external event identity")
                else:
                    atomic_json(path, {"external": event, "signal": signal})
                state = self.invoke("event", event=signal)
                trigger = "autoloop-" + key
                state = self._continue(state, trigger)
            wait_ref = self.root / "question.json"
            for answer in context.get("answers", []):
                card = answer.get("card")
                if type(card) is not int:
                    raise ValueError("invalid answer card")
                answer_ref = self.root / ("answer-" + str(card) + ".json")
                if answer_ref.exists():
                    saved_answer = _read(answer_ref)
                    if saved_answer["answer"] != answer:
                        raise ValueError("conflicting answer identity")
                    question = saved_answer["question"]
                else:
                    question = _read(wait_ref) if wait_ref.exists() else None
                if (
                    not question
                    or answer.get("about") != question["about"]
                    or answer.get("by") not in self.config.get("answerers", [])
                ):
                    raise ValueError("answer identity or question reference mismatch")
                card = answer.get("card")
                choice = answer.get("choice")
                if (
                    type(card) is not int
                    or type(choice) is not int
                    or choice not in {0, 1}
                ):
                    raise ValueError("invalid answer card or choice")
                signal = {
                    "event_id": "answer-" + str(card),
                    "type": "resume" if choice == 1 else "pause",
                    "operation_id": None,
                    "reference": question["about"],
                }
                atomic_json(answer_ref, {"answer": answer, "question": question})
                state = self.invoke("event", event=signal)
                if choice == 1:
                    state = self._continue(state, "autoloop-answer-" + str(card))
            asks = []
            if state["status"] in {"waiting", "paused"}:
                reason = state.get("wait", {}).get("reason", "pending_receipt")
                about = "protocol10:" + self.config["campaign_id"] + ":" + reason
                question = {
                    "title": "protocol10 等待处理",
                    "text": "当前等待："
                    + reason
                    + "。继续前会重新核对原操作回执与准入。",
                    "options": ["保持暂停", "继续核验"],
                    "recommend": 0,
                    "about": about,
                    "key": about,
                }
                atomic_json(wait_ref, question)
                asks = [question]
            result = {
                "summary": "protocol10：" + state["status"],
                "status": state["status"],
                "ask": asks,
                "core": state,
            }
            atomic_json(self.root / "latest.json", result)
            return result

    def _continue(self, state, trigger):
        if state["paused"] or state["status"] in {
            "completed",
            "terminated",
            "cancelled",
        }:
            return state
        pending = state.get("pending_receipt") or {}
        reconcile = state.get("pending") is not None and not (
            pending.get("status") == "not_submitted"
            and pending.get("submission_absent") is True
        )
        return self.invoke("reconcile" if reconcile else "advance", trigger_id=trigger)


def render(result):
    """Use a completed worker turn for core waiting, rejection and uncertainty."""
    return (
        result["summary"]
        + "\n```agent-autoloop\n"
        + json.dumps({"ask": result.get("ask", [])}, ensure_ascii=False)
        + "\n```"
    )


class AutoloopHost:
    """Own CLI launch authority outside the service's write and GPU domains."""

    def __init__(self, configuration):
        self.config = configuration
        self.transport = Path(configuration["transport_root"])
        self.private = Path(configuration["host_receipts_root"])
        self.private.mkdir(parents=True, exist_ok=True)
        for name in ("requests", "responses"):
            (self.transport / name).mkdir(parents=True, exist_ok=True)
        local = {k: v for k, v in configuration.items() if k != "transport_root"}
        self.bridge = AutoloopBridge(local)

    def _validate_signal(self, event):
        for value in self.config["events"].values():
            if event != value["signal"]:
                continue
            if value.get("reference_ref"):
                record = _read(value["reference_ref"])
                if any(
                    record.get(k) != v for k, v in value["reference_identity"].items()
                ):
                    raise ValueError("host signal reference identity mismatch")
            return
        key = event.get("event_id", "")
        if (
            set(event) != {"event_id", "type", "operation_id", "reference"}
            or not re.fullmatch(r"answer-[0-9]+", key)
            or event.get("type") not in {"pause", "resume"}
            or event.get("operation_id") is not None
        ):
            raise ValueError("unregistered host signal")
        card_id = int(key.split("-")[1])
        hub = _read(self.config["hub_configuration_ref"])
        request = urllib.request.Request(
            hub["url"] + "/v1/state",
            headers={"Authorization": "Bearer " + hub["owner"]},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            view = json.load(response)
            cards = view["cards"] + view.get("recentCards", [])
        card = next((card for card in cards if card["id"] == card_id), None)
        if (
            not card
            or card.get("state") not in {"decided", "done"}
            or card.get("by") not in self.config.get("answerers", [])
            or card.get("about") != event.get("reference")
            or card.get("choice") != (1 if event["type"] == "resume" else 0)
            or not event["reference"].startswith(
                "protocol10:" + self.config["campaign_id"] + ":"
            )
        ):
            raise ValueError("host answer card identity mismatch")

    def handle(self, request):
        identity = identifier(request["request_id"])
        action = request["action"]
        fields = {
            "status": {"request_id", "action"},
            "event": {"request_id", "action", "event"},
            "continue": {"request_id", "action", "trigger_id"},
        }
        if action not in fields or set(request) != fields[action]:
            raise ValueError("unsupported host request fields")
        intent = self.private / (identity + "-intent.json")
        receipt = self.private / (identity + "-receipt.json")
        if intent.exists() and _read(intent) != request:
            raise ValueError("conflicting host request identity")
        if receipt.exists():
            return _read(receipt)
        atomic_json(intent, request)
        if action == "status":
            result = self.bridge._invoke("status")
        elif action == "event":
            event = request["event"]
            self._validate_signal(event)
            result = self.bridge._invoke("event", event=event)
            atomic_json(
                self.private / ("accepted-" + identifier(event["event_id"]) + ".json"),
                event,
            )
        else:
            trigger = identifier(request["trigger_id"])
            if (
                not trigger.startswith("autoloop-")
                or not (
                    self.private / ("accepted-" + trigger[len("autoloop-") :] + ".json")
                ).exists()
            ):
                raise ValueError("continue has no registered accepted signal")
            with (self.private / "advance.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result = self.bridge._invoke("status")
                else:
                    result = (
                        self.bridge._continue(self.bridge._invoke("status"), trigger)
                        if self.config.get("allow_advance") is True
                        else self.bridge._invoke("reconcile", trigger_id=trigger)
                    )
        response = {"request_id": identity, "result": result}
        atomic_json(receipt, response)
        return response

    def serve(self, stop):
        """Process independent queries while one admitted CLI advances."""
        pending = set()

        def process(path):
            request = _read(path)
            try:
                result = self.handle(request)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                result = {"request_id": path.stem, "error": str(exc)}
            atomic_json(self.transport / "responses" / path.name, result)

        with ThreadPoolExecutor(max_workers=4) as pool:
            while not stop.is_set():
                for path in sorted((self.transport / "requests").glob("*.json")):
                    if (
                        path.name not in pending
                        and not (self.transport / "responses" / path.name).exists()
                    ):
                        pending.add(path.name)
                        pool.submit(process, path)
                stop.wait(0.02)
