# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit run identity, action facts and admission budgets for RSI runs.

Ordinary RPent runs do not enable this opt-in journal. The launcher owns
authorization, profile/bundle validation and whole-process supervision.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class RuntimeBudgetExceeded(RuntimeError):
    """The next operation cannot fit its enforced budget."""


def runtime_context() -> dict[str, Any] | None:
    """Read the launcher's explicit identity and limits, if supplied."""
    path = os.environ.get("RSI_RUNTIME_CONTEXT")
    if not path:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported RSI_RUNTIME_CONTEXT schema")
    for key in (
        "run_id",
        "episode_id",
        "bundle_id",
        "protocol_id",
        "profile_ref",
        "output_dir",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"RSI_RUNTIME_CONTEXT missing {key}")
    return value


def planner_admission(backend: str, model: str | None, effort: str) -> None:
    """Reject a model request when this backend cannot enforce finite tokens.

    Codex/Claude expose a whole turn, while API currently lacks a verified
    pre-request multimodal token count. Post-response usage cannot constrain
    the next request. A deterministic Flash replay makes no planner request.
    """
    context = runtime_context()
    if context is None:
        return
    expected = context.get("planner", {})
    actual = {"backend": backend, "model": model, "reasoning_effort": effort}
    if any(actual[key] != expected[key] for key in actual if key in expected):
        raise ValueError("actual planner options differ from RSI runtime context")
    limits = context.get("limits", {})
    if backend == "codex" and context.get("token_policy") == "observed_only":
        if (
            not os.environ.get("RSI_REQUEST_GATE_URL")
            or type(limits.get("model_requests")) is not int
            or limits["model_requests"] < 1
        ):
            raise RuntimeBudgetExceeded(
                "Codex observed-token protocol requires the active HTTP request gate"
            )
        return
    if backend != "flash" and any(
        limits.get(key) is not None for key in ("input_tokens", "output_tokens")
    ):
        raise RuntimeBudgetExceeded(
            f"{backend}: cannot bound the next model request by remaining input/output tokens; "
            "missing verified pre-request token count and per-request provider cap; no request dispatched"
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_budget_stop(
    path: str | Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    """Read the first published budget fact and verify its run identity."""
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0" or any(
        value.get("identity", {}).get(key) != identity.get(key)
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    ):
        raise ValueError("budget stop identity/schema mismatch")
    return value


class RuntimeJournal:
    """Append L0 facts at their producer, reserving steps before RPC dispatch."""

    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.output_dir = Path(context["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "action_events.jsonl"
        self.costs_path = self.output_dir / "runtime_costs.json"
        # An existing journal belongs to an already-started run, never silently append.
        self.events_path.open("x", encoding="utf-8").close()
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.sequence = 0
        self.reserved_steps = 0
        self.confirmed_steps = 0
        self.tool_calls = 0
        self.active_tools = 0
        self.budget_stop_path = Path(
            context.get("budget_stop_ref", self.output_dir / "budget_stop.json")
        )
        self.planner_usage: dict[str, Any] = {}
        self.uncertain_steps = 0
        self.chunk_steps: int | None = None
        self.finish_requires_native_success = False
        self.planner_finish_reserve_calls = 0
        rules_ref = context.get("runtime_rules_ref")
        if rules_ref:
            rules = json.loads(Path(rules_ref).read_text(encoding="utf-8"))
            cap, baseline = rules.get("chunk_steps"), rules.get("baseline_chunk_steps")
            version = rules.get("schema")
            expected = {"schema", "chunk_steps", "baseline_chunk_steps"}
            if version in {"rsi-runtime-rules-v2", "rsi-runtime-rules-v3"}:
                expected.add("finish_requires_native_success")
            if version == "rsi-runtime-rules-v3":
                expected.add("planner_finish_reserve_calls")
            if (
                set(rules) != expected
                or version
                not in {
                    "rsi-runtime-rules-v1",
                    "rsi-runtime-rules-v2",
                    "rsi-runtime-rules-v3",
                }
                or type(cap) is not int
                or type(baseline) is not int
                or not 1 <= cap <= baseline
                or type(rules.get("finish_requires_native_success", False)) is not bool
            ):
                raise ValueError("invalid RSI runtime rule artifact")
            reserve = rules.get("planner_finish_reserve_calls", 0)
            if type(reserve) is not int or reserve not in {0, 1}:
                raise ValueError("invalid finish reserve hint")
            self.planner_finish_reserve_calls = reserve
            self.chunk_steps = cap
            self.finish_requires_native_success = rules.get(
                "finish_requires_native_success", False
            )
        self.emit(
            "run_start",
            runtime_rules_ref=rules_ref,
            effective_rules={
                "chunk_steps": self.chunk_steps,
                "finish_requires_native_success": self.finish_requires_native_success,
                "planner_finish_reserve_calls": self.planner_finish_reserve_calls,
            },
            missing=["service_initialization_actions", "initial_reset_settling_steps"],
        )

    @classmethod
    def from_environment(cls) -> RuntimeJournal | None:
        """Enable only for an explicitly bound launcher context."""
        context = runtime_context()
        return None if context is None else cls(context)

    def identity(self) -> dict[str, Any]:
        """Return identity shared by actions, native results and cost records."""
        return {
            key: self.context.get(key)
            for key in (
                "run_id",
                "episode_id",
                "bundle_id",
                "protocol_id",
                "profile_ref",
                "parent_decision_id",
            )
        }

    def emit(self, kind: str, **fields: Any) -> int:
        """Persist one fact before its dependent operation starts."""
        with self.lock:
            if kind == "tool_end" and fields.get("status") != "rejected":
                self.active_tools = max(0, self.active_tools - 1)
            sequence = self.sequence
            record = {
                **self.identity(),
                "schema_version": "1.0",
                "sequence": sequence,
                "kind": kind,
                "monotonic_ns": time.monotonic_ns(),
                "utc_ns": time.time_ns(),
                "clock_domain": "runtime_client",
                **fields,
            }
            line = json.dumps(record, default=_json_value, allow_nan=False) + "\n"
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self.sequence += 1
            return sequence

    def complete_native_success(self, *, step: int) -> None:
        """Close Planner admission after native success and state capture."""
        from rpent.evaluation import write_json_atomic

        path = self.output_dir / "native_completion.json"
        if not path.exists():
            sequence = self.emit(
                "native_completion", step=step, reason="native_success"
            )
            write_json_atomic(
                path,
                {
                    "identity": self.identity(),
                    "reason": "native_success",
                    "step": step,
                    "sequence": sequence,
                },
            )

    def native_completion(self) -> dict[str, Any] | None:
        """Read the matching run's native success closure."""
        path = self.output_dir / "native_completion.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if (
            value.get("identity") != self.identity()
            or value.get("reason") != "native_success"
        ):
            raise ValueError("native completion identity/reason mismatch")
        return value

    def budget_snapshot(self) -> dict[str, Any]:
        """Read current owner snapshots; never reserve or charge another call."""
        limits = self.context["limits"]
        missing = []

        def owner(name: str) -> dict[str, Any]:
            try:
                value = json.loads((self.output_dir / name).read_text())
                if any(
                    value.get("identity", {}).get(k) != self.context.get(k)
                    for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
                ):
                    raise ValueError("owner identity mismatch")
                return value
            except (OSError, ValueError):
                missing.append(name)
                return {}

        gate = owner("request_gate_live.json")
        service = owner("environment_service/runtime_costs.json")
        requests = gate.get("requests_forwarded")
        steps = service.get("environment_steps_reserved")
        elapsed = time.monotonic() - self.context.get(
            "job_started_monotonic", self.started
        )

        def remaining(key: str, used: Any) -> float | int | None:
            cap = limits.get(key)
            return max(0, cap - used) if cap is not None and used is not None else None

        return {
            "tool_calls_used": self.tool_calls,
            "tool_calls_remaining": remaining("tool_calls", self.tool_calls),
            "actual_model_requests": requests,
            "model_requests_remaining": remaining("model_requests", requests),
            "wall_seconds_remaining": remaining("wall_seconds", elapsed),
            "environment_steps_reserved": steps,
            "environment_steps_remaining": remaining("environment_steps", steps),
            "budget_stop": self.budget_stop(),
            "source": "owner snapshots at tool completion; admission supervisors remain authoritative",
            "missing": missing,
        }

    def planner_finish_hint(self) -> str | None:
        """Render the frozen optional advice from this attempt's actual cap."""
        if self.planner_finish_reserve_calls == 0:
            return None
        limit = self.context["limits"]["tool_calls"]
        if type(limit) is not int or limit < 1:
            raise ValueError("finish reserve hint requires a finite tool cap")
        return (
            f"This attempt permits at most {limit} admitted RPent tool calls in total, "
            "including observations and finish. Keep the last admitted call available "
            "for finish. Use it to report success, failure, or stuck according to the "
            "evidence already available. This advice grants no extra calls and does "
            "not change the native task goal."
        )

    def budget_stop(self) -> dict[str, Any] | None:
        """Return a client or service stop, without changing the first cause."""
        return read_budget_stop(self.budget_stop_path, self.identity())

    def mark_budget_stop(
        self, reason: str, *, limit: int | float, observed: int | float, **fields: Any
    ) -> dict[str, Any]:
        """Atomically publish the first stop while admitted operations drain."""
        with self.lock:
            previous = self.budget_stop()
            if previous is not None:
                return previous
            sequence = self.emit(
                "budget_stop",
                reason=reason,
                limit=limit,
                observed=observed,
                stop_after_inflight=True,
                **fields,
            )
            record = {
                "schema_version": "1.0",
                "identity": self.identity(),
                "reason": reason,
                "limit": limit,
                "observed": observed,
                "stop_after_inflight": True,
                "producer_events_ref": str(self.events_path),
                "sequence": sequence,
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            }
            # The link publishes a complete JSON inode without replacing a stop
            # concurrently published by the environment-service process.
            self.budget_stop_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.budget_stop_path.parent,
                prefix=".budget-stop-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                try:
                    os.link(temporary, self.budget_stop_path)
                except FileExistsError:
                    return self.budget_stop()
            finally:
                temporary.unlink()
            return record

    def check(self) -> None:
        """Check the client wall budget between operations."""
        limit = self.context.get("limits", {}).get("wall_seconds")
        if (
            limit is not None
            and time.monotonic()
            - self.context.get("job_started_monotonic", self.started)
            >= limit
        ):
            self.mark_budget_stop(
                "wall_seconds", limit=limit, observed=time.monotonic() - self.started
            )
            raise RuntimeBudgetExceeded("runtime wall_seconds exhausted")

    def admit_tool(self, name: str) -> None:
        """Charge all toolkit calls, including reads and rejected operations."""
        with self.lock:
            self.check()
            if self.native_completion() is not None:
                raise RuntimeBudgetExceeded("native task already completed")
            if (self.output_dir / "runtime_stop.json").exists():
                raise RuntimeBudgetExceeded("runtime stopped by supervisor")
            limit = self.context.get("limits", {}).get("tool_calls")
            if limit is not None and self.tool_calls >= limit:
                self.mark_budget_stop(
                    "tool_calls", limit=limit, observed=self.tool_calls
                )
                self.emit("tool_rejected", tool=name, reason="tool_calls exhausted")
                raise RuntimeBudgetExceeded("tool_calls exhausted")
            stop = self.budget_stop()
            if stop is not None:
                self.emit(
                    "tool_rejected",
                    tool=name,
                    reason="runtime budget stopped",
                    resource=stop["reason"],
                )
                raise RuntimeBudgetExceeded(f"runtime budget stopped: {stop['reason']}")
            self.tool_calls += 1
            self.active_tools += 1
            self.emit("tool_call", tool=name)
            if limit is not None and self.tool_calls == limit:
                self.mark_budget_stop(
                    "tool_calls", limit=limit, observed=self.tool_calls
                )

    def action(
        self,
        callback: Callable[..., Any],
        actions: Any,
        *,
        source: str,
        proposal_sequence: int | None = None,
        proposal_length: int | None = None,
        chunk: bool = False,
        callback_with_send_sequence: bool = False,
        already_admitted: bool = False,
    ) -> Any:
        """Record send intent and native RPC feedback; disconnects remain unknown."""
        with self.lock:
            self.check()
            stop = self.budget_stop()
            # The last admitted tool may drain its actions; no new tool can enter.
            if stop is not None and not (
                stop["reason"] in {"tool_calls", "model_requests"}
                and (self.active_tools or already_admitted)
            ):
                raise RuntimeBudgetExceeded(f"runtime budget stopped: {stop['reason']}")
            count = len(actions) if chunk else 1
            limit = self.context.get("limits", {}).get("environment_steps")
            if limit is not None and self.reserved_steps + count > limit:
                self.mark_budget_stop(
                    "environment_steps",
                    limit=limit,
                    observed=self.reserved_steps,
                    requested_steps=count,
                )
                self.emit(
                    "action_rejected",
                    source=source,
                    reason="environment_steps exhausted",
                    requested_steps=count,
                )
                raise RuntimeBudgetExceeded(
                    "environment_steps cannot fit next action/chunk"
                )
            self.reserved_steps += count
            if limit is not None and self.reserved_steps == limit:
                self.mark_budget_stop(
                    "environment_steps", limit=limit, observed=self.reserved_steps
                )
            send_sequence = self.emit(
                "action_send",
                source=source,
                sent=actions,
                proposal_sequence=proposal_sequence,
                proposal_indices=list(range(count)),
                execution_mask=None,
                applied=None,
                missing=[
                    "controller_applied_action",
                    "measured_motion",
                    "feedback_pending",
                ],
            )
        try:
            result = (
                callback(send_sequence) if callback_with_send_sequence else callback()
            )
        except BaseException as exc:
            with self.lock:
                self.uncertain_steps += count
                self.emit(
                    "action_feedback",
                    source=source,
                    send_sequence=send_sequence,
                    proposal_sequence=proposal_sequence,
                    execution_mask=None,
                    applied=None,
                    error=f"{type(exc).__name__}: {exc}",
                    missing=[
                        "native_rpc_feedback",
                        "execution_mapping",
                        "controller_applied_action",
                        "measured_motion",
                    ],
                )
            raise
        obs, reward, terminated, truncated, _info = result
        with self.lock:
            # A successful RPC confirms returned transition slots, not actuator application.
            returned = (
                len(terminated) if chunk and hasattr(terminated, "__len__") else count
            )
            if returned != count:
                self.uncertain_steps += count
                mask = None
            else:
                self.confirmed_steps += count
                mask = [True] * count + [False] * max(
                    0, (proposal_length or count) - count
                )
            state = (
                obs[-1].get("states") if isinstance(obs, list) else obs.get("states")
            )
            self.emit(
                "action_feedback",
                source=source,
                send_sequence=send_sequence,
                proposal_sequence=proposal_sequence,
                execution_mask=mask,
                applied=None,
                feedback={
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "observed_state": state,
                },
                feedback_source="env.step/chunk_step RPC return",
                missing=[
                    "controller_applied_action",
                    "measured_motion",
                    "server_capture_clock",
                ]
                + ([] if mask is not None else ["execution_mapping"]),
            )
        return result

    def record_planner_stats(self, stats: dict[str, Any]) -> None:
        """Keep provider usage and missing fields separate, including cached input."""
        self.planner_usage = dict(stats)
        self.emit("planner_usage", stats=self.planner_usage)

    def costs(self) -> dict[str, Any]:
        """Snapshot measured client costs; other producers remain explicitly missing."""
        mapping = {
            "input_tokens": "total_input_tokens",
            "cached_input_tokens": "total_cached_input_tokens",
            "output_tokens": "total_output_tokens",
        }
        values = {
            name: (
                self.planner_usage.get(key)
                if self.planner_usage.get("usage_observed") is not False
                else None
            )
            for name, key in mapping.items()
        }
        return {
            **self.identity(),
            "identity": self.identity(),
            "schema_version": "1.0",
            "runtime_client_wall_seconds": time.monotonic() - self.started,
            "environment_steps_reserved": self.reserved_steps,
            "environment_steps_confirmed": self.confirmed_steps,
            "environment_steps_unknown": self.uncertain_steps,
            "tool_calls": self.tool_calls,
            "active_tools": self.active_tools,
            "budget_stop_ref": str(self.budget_stop_path),
            "budget_stop": self.budget_stop(),
            **values,
            "cached_input_is_subset": True,
            "planner_requested": self.context.get("planner"),
            "planner_backend_confirmed": None,
            "training": None,
            "failed_candidates": None,
            "audit_evaluation": None,
            "reset_cost": None,
            "recovery_cost": None,
            "human_cost": None,
            "gpu_device_seconds": None,
            "whole_process_wall_seconds": None,
            "storage_bytes": None,
            "missing": [name for name, value in values.items() if value is None]
            + [
                "planner_backend_confirmation",
                "training",
                "failed_candidates",
                "audit_evaluation",
                "reset_cost",
                "recovery_cost",
                "human_cost",
                "gpu_device_seconds",
                "whole_process_wall_seconds:see_supervisor.json",
                "storage_bytes:see_supervisor.json",
            ],
        }

    def finalize(self) -> None:
        """Write a cost snapshot without hiding unavailable measurements."""
        with self.lock:
            self.costs_path.write_text(
                json.dumps(self.costs(), indent=2), encoding="utf-8"
            )
