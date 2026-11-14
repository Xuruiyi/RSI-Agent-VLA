"""Recoverable protocol10 coordination; business stores remain authoritative."""

import fcntl
import json
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

from .storage import atomic_json, identifier

EVENT_TYPES = frozenset(
    {
        "pause",
        "resume",
        "cancel",
        "terminate",
        "timer",
        "resource_changed",
        "receipt_available",
        "authorization_expired",
    }
)
TERMINAL = frozenset({"completed", "terminated", "cancelled"})


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("explicit ISO8601 time required")  # noqa: TRY004 - public input validation
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    if result.tzinfo is None:
        raise ValueError("time must include timezone")
    return result


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("resource values must be finite nonnegative numbers")  # noqa: TRY004 - public input validation
    if not isfinite(value) or value < 0:
        raise ValueError("resource values must be finite nonnegative numbers")
    return value


class Coordinator:
    """Persist stable intents and reconcile effects before dispatching once.

    Args:
        root: New scope's coordinator state directory, separate from business stores.
        binding: Immutable registration, protocol and version references.
        backend: Consumer with observe(progress, inputs), reconcile(operation, inputs)
            and execute(operation, inputs) methods. Reconciliation may close native
            evidence through idempotent consumers, but must never dispatch work.
    """

    def __init__(self, root: str | Path, binding: dict[str, Any], backend: Any):
        if not isinstance(binding, dict) or not binding:
            raise ValueError("nonempty immutable binding required")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "progress.json"
        self.binding = deepcopy(binding)
        self.backend = backend
        if self.path.exists():
            self._load()
        else:
            with self._lock():
                if self.path.exists():
                    self._load()
                    return
                self._save(
                    {
                        "schema_version": 1,
                        "binding": self.binding,
                        "status": "ready",
                        "current_stage": "baseline",
                        "wait": None,
                        "pending": None,
                        "pending_receipt": None,
                        "operations": {},
                        "receipts": {},
                        "events": {},
                        "paused": False,
                    }
                )

    @contextmanager
    def _lock(self):
        # This is deliberately independent of the host dynamic-job lock. Native
        # business consumers own that lock while they dispatch their work.
        with (self.root / "coordinator.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another protocol10 coordinator is active") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state["binding"] != self.binding:
            raise ValueError("conflicting coordinator binding")
        return self._signals(state)

    @contextmanager
    def _signal_lock(self):
        with (self.root / "signals.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _signals(self, state):
        with self._signal_lock():
            return self._apply_signals(state)

    def _apply_signals(self, state):
        path = self.root / "signals.json"
        signals = json.loads(path.read_text()) if path.exists() else {}
        for key, record in signals.items():
            event = record.get("event", record)
            if key in state["events"]:
                continue
            state["events"][key] = deepcopy(event)
            pending = state["pending"]
            applies = record.get(
                "applies",
                event["operation_id"] is None
                or (
                    pending is not None
                    and pending["operation_id"] == event["operation_id"]
                ),
            )
            if not applies or state["status"] in TERMINAL:
                continue
            kind = event["type"]
            if kind == "pause":
                state.update(
                    paused=True, status="paused", wait={"reason": "user_pause"}
                )
            elif kind == "resume":
                state.update(paused=False, status="ready", wait=None)
            elif kind in {"cancel", "terminate"}:
                state.update(
                    status="cancelled" if kind == "cancel" else "terminated",
                    wait={"reason": kind},
                )
            elif kind == "authorization_expired" and not state["paused"]:
                state.update(status="waiting", wait={"reason": kind})
        if state["paused"] and state["status"] not in TERMINAL:
            state.update(
                status="paused",
                wait={
                    "reason": "user_pause",
                    "continue_when": "explicit_resume_and_current_admission",
                },
            )
        return state

    def _save(self, state: dict[str, Any]) -> dict[str, Any]:
        self._signals(state)
        if state.get("wait"):
            reason = state["wait"]["reason"]
            conditions = {
                "user_pause": "explicit_resume_and_current_admission",
                "cancel": "closed_campaign_no_automatic_continuation",
                "terminate": "closed_campaign_no_automatic_continuation",
                "authorization_required": "new_applicable_authorization",
                "authorization_expired": "new_applicable_unexpired_authorization",
                "authorization_binding_mismatch": "authorization_matching_registered_binding",
                "authorization_time_required": "explicit_current_time_and_expiry",
                "operation_not_authorized": "explicit_authority_for_registered_operation",
                "budget_wait": "authoritative_remaining_envelope_fits",
                "phase_resource_evidence": "resolved_authoritative_phase_and_offline_usage",
                "resource_evidence_required": "known_costs_and_confirmed_resource_availability",
                "evaluation_repair": "repair_consumer_then_reparse_original_output",
                "technical_recovery_cause": "closed_fault_cost_cleanup_and_applicable_recovery_permit",
            }
            state["wait"].setdefault(
                "continue_when",
                conditions.get(
                    reason, "reconcile_same_operation_receipt_and_owner_cleanup"
                ),
            )
        with self._signal_lock():
            self._apply_signals(state)
            atomic_json(self.path, state)
        return deepcopy(state)

    def status(self) -> dict[str, Any]:
        """Read persisted coordination progress without querying business stores."""
        return self._load()

    def event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record a deduplicated control signal; signals never grant authority."""
        required = {"type", "event_id", "operation_id"}
        if not isinstance(event, dict) or not required <= event.keys():
            raise ValueError("event type, event_id and operation_id required")
        if event.keys() - required - {"reference"} or event["type"] not in EVENT_TYPES:
            raise ValueError("unsupported event type or fields")
        identifier(event["event_id"])
        if event["operation_id"] is not None:
            identifier(event["operation_id"])
        if "reference" in event and not isinstance(event["reference"], str):
            raise ValueError("event reference must be a string")
        with self._signal_lock():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self._apply_signals(state)
            previous = state["events"].get(event["event_id"])
            if previous is not None:
                if previous != event:
                    raise ValueError("conflicting event identity")
                return state
            path = self.root / "signals.json"
            signals = json.loads(path.read_text()) if path.exists() else {}
            pending = state["pending"]
            applies = event["operation_id"] is None or (
                pending is not None and pending["operation_id"] == event["operation_id"]
            )
            signals[event["event_id"]] = {"event": deepcopy(event), "applies": applies}
            atomic_json(path, signals)
            return deepcopy(self._apply_signals(state))

    def _operation(self, value: Any) -> dict[str, Any]:
        required = {"operation_id", "stage", "kind", "dynamic", "payload"}
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError("incomplete operation")
        identifier(value["operation_id"])
        if (
            not isinstance(value["stage"], str)
            or not value["stage"]
            or not isinstance(value["kind"], str)
            or not value["kind"]
            or type(value["dynamic"]) is not bool
            or not isinstance(value["payload"], dict)
        ):
            raise ValueError("invalid operation fields")
        return deepcopy(value)

    def _guard(self, operation: dict[str, Any], inputs: dict[str, Any]) -> str | None:
        if not operation["dynamic"]:
            return None
        authorization = inputs.get("authorization", {})
        if (
            not isinstance(authorization, dict)
            or authorization.get("dynamic_authorized") is not True
        ):
            return "authorization_required"
        if authorization.get("binding") != self.binding:
            return "authorization_binding_mismatch"
        if (
            not isinstance(authorization.get("operation_ids"), list)
            or operation["operation_id"] not in authorization["operation_ids"]
        ):
            return "operation_not_authorized"
        try:
            if _timestamp(inputs.get("now")) >= _timestamp(
                authorization.get("expires_at")
            ):
                return "authorization_expired"
        except (ValueError, TypeError):
            return "authorization_time_required"
        resources = inputs.get("resources", {})
        if (
            not isinstance(resources, dict)
            or resources.get("unresolved_costs")
            or resources.get("blockers")
        ):
            return "resource_evidence_required"
        try:
            available = resources["available"]
            supplied = resources["required"]
            reserve = resources["reserve"]
            required = operation.get("resource_requirements", {})
            if not all(
                isinstance(v, dict) for v in (available, supplied, reserve, required)
            ):
                return "resource_evidence_required"
            dimensions = set(supplied) | set(required) | set(reserve)
            if not dimensions or not any(
                _number(v) > 0
                for v in list(supplied.values()) + list(required.values())
            ):
                return "resource_evidence_required"
            for dimension in dimensions:
                need = max(
                    _number(supplied.get(dimension, 0)),
                    _number(required.get(dimension, 0)),
                )
                if _number(available[dimension]) < need + _number(
                    reserve.get(dimension, 0)
                ):
                    return "budget_wait"
        except (ValueError, KeyError, TypeError):
            return "resource_evidence_required"
        return None

    def _receipt(
        self, operation: dict[str, Any], method: str, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            receipt = getattr(self.backend, method)(
                deepcopy(operation), deepcopy(inputs)
            )
            if (
                not isinstance(receipt, dict)
                or receipt.get("operation_id") != operation["operation_id"]
                or receipt.get("status")
                not in {"completed", "not_submitted", "unknown", "waiting"}
                or (receipt["status"] == "completed" and not receipt.get("receipt_ref"))
            ):
                raise ValueError("invalid backend receipt")
            # Reject non-JSON evidence before it can prevent the unknown checkpoint.
            json.dumps(receipt, allow_nan=False)
            return deepcopy(receipt)
        except Exception as exc:  # noqa: BLE001 - preserve unknown submission after any backend failure
            evaluation = operation["kind"] in {
                "evaluation",
                "baseline_evaluation",
                "candidate_evaluation",
                "attempt_evaluation",
                "matrix_evaluation",
            }
            return {
                "operation_id": operation["operation_id"],
                "status": "waiting" if evaluation else "unknown",
                "reason": "evaluation_repair" if evaluation else "submission_unknown",
                "error": f"{type(exc).__name__}: {exc}",
                "receipt_ref": None,
            }

    def _record_receipt(self, state: dict[str, Any], receipt: dict[str, Any]) -> None:
        state["pending_receipt"] = receipt
        if receipt["status"] == "completed":
            key = receipt["operation_id"]
            previous = state["receipts"].get(key)
            if previous is not None and previous != receipt:
                raise ValueError("conflicting completed receipt")
            state["receipts"][key] = receipt
            state.update(pending=None, pending_receipt=None, status="ready", wait=None)
        else:
            reason = (
                receipt.get("reason")
                or {
                    "unknown": "submission_unknown",
                    "waiting": "backend_wait",
                    "not_submitted": "not_submitted",
                }[receipt["status"]]
            )
            state.update(status="waiting", wait={"reason": reason})

    def advance(
        self,
        inputs: dict[str, Any],
        reconcile_only: bool = False,
        trigger_id: str | None = None,
    ) -> dict[str, Any]:
        """Reconcile a stable intent, then execute at most one admitted operation.

        Dynamic execution requires explicit time, a matching unexpired authorization
        with operation_ids, and available/required/reserve resource vectors. An
        unknown receipt keeps its identity pending until native evidence resolves it.
        """
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be an object")  # noqa: TRY004 - public input validation
        with self._lock():
            state = self._load()
            if trigger_id is not None:
                identifier(trigger_id)
                triggers = state.setdefault("triggers", {})
                if trigger_id in triggers:
                    return state
                triggers[trigger_id] = {"reconcile_only": reconcile_only}
                self._save(state)
            stopped = state["status"] in TERMINAL or state["paused"]
            if stopped and (not reconcile_only or state["pending"] is None):
                return state
            saved_control = (state["status"], deepcopy(state["wait"]))
            operation = state["pending"]
            if operation is None:
                facts = self.backend.observe(deepcopy(state), deepcopy(inputs))
                if facts.get("terminal"):
                    state.update(
                        status="completed", wait=None, terminal=facts["terminal"]
                    )
                    return self._save(state)
                if facts.get("wait"):
                    state.update(status="waiting", wait=deepcopy(facts["wait"]))
                    return self._save(state)
                value = facts.get("next_operation")
                if value is None or reconcile_only:
                    return self._save(state)
                operation = self._operation(value)
                key = operation["operation_id"]
                previous = state["operations"].get(key)
                if previous is not None and previous != operation:
                    raise ValueError("conflicting operation identity")
                if key in state["receipts"]:
                    raise ValueError("backend selected an already completed operation")
                state["operations"][key] = operation
                state.update(
                    pending=operation,
                    pending_receipt=None,
                    status="ready",
                    wait=None,
                    current_stage=operation["stage"],
                )
                # Durable intent precedes even reconciliation and every side effect.
                self._save(state)
            receipt = self._receipt(operation, "reconcile", inputs)
            self._record_receipt(state, receipt)
            if stopped:
                state["status"], state["wait"] = saved_control
            self._save(state)
            if (
                state["paused"]
                or state["status"] in TERMINAL
                or receipt["status"] != "not_submitted"
                or reconcile_only
            ):
                return deepcopy(state)
            if receipt.get("submission_absent") is not True:
                state.update(status="waiting", wait={"reason": "submission_unknown"})
                return self._save(state)
            reason = self._guard(operation, inputs)
            if reason:
                state.update(status="waiting", wait={"reason": reason})
                return self._save(state)
            # Persist an uncertain submission before dispatch. If the process dies
            # after the native side effect, the next invocation reconciles this ID.
            state["pending_receipt"] = {
                "operation_id": operation["operation_id"],
                "status": "unknown",
                "reason": "dispatch_in_progress",
                "receipt_ref": None,
            }
            self._save(state)
            if state["paused"] or state["status"] in TERMINAL:
                return deepcopy(state)
            receipt = self._receipt(operation, "execute", inputs)
            self._record_receipt(state, receipt)
            return self._save(state)
