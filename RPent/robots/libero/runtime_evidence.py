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

"""Owner-side LIBERO vector actions, including native reset settling.

The adapter wraps one environment instance. It delegates the installed RLinf
constructor/reset/step implementations and does not replace their control loop.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.session.runtime import RuntimeJournal


@dataclass
class _RpcEvidence:
    metadata: dict[str, Any]
    next_slot: int = 0
    current_link: dict[str, Any] | None = None
    confirmed: list[int] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)


_RPC_EVIDENCE: ContextVar[_RpcEvidence | None] = ContextVar(
    "libero_rpc_evidence", default=None
)


@contextmanager
def rpc_evidence(journal: EnvironmentJournal, metadata: dict[str, Any], slots: int):
    """Bind one mutating RPC while RpcFacade holds its existing write lock."""
    if not isinstance(metadata, dict):
        raise ValueError("RSI RPC evidence must be an object")
    identity = journal.identity()
    if any(
        metadata.get(key) != identity[key]
        for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
    ):
        raise ValueError("RSI RPC evidence identity mismatch")
    indices = metadata.get("proposal_indices")
    proposal_length = metadata.get("proposal_length")
    proposal_sequence = metadata.get("proposal_sequence")
    if (
        type(metadata.get("client_send_sequence")) is not int
        or metadata["client_send_sequence"] < 0
        or (
            proposal_sequence is not None
            and (type(proposal_sequence) is not int or proposal_sequence < 0)
        )
        or not isinstance(indices, list)
        or len(indices) != slots
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
        or type(proposal_length) is not int
        or proposal_length < 1
        or any(index >= proposal_length for index in indices)
    ):
        raise ValueError("invalid RSI RPC proposal mapping")
    metadata = {**metadata, "proposal_indices": list(indices)}
    state = _RpcEvidence(metadata)
    token = _RPC_EVIDENCE.set(state)
    try:
        journal.emit("rpc_dispatch_start", rpc_evidence=metadata)
    except BaseException:
        _RPC_EVIDENCE.reset(token)
        raise
    error = None
    try:
        yield
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        mask = [False] * proposal_length
        for index in state.confirmed:
            mask[index] = True
        try:
            journal.emit(
                "rpc_dispatch_end",
                rpc_evidence=metadata,
                execution_mask=None if state.unknown else mask,
                confirmed_proposal_indices=state.confirmed,
                unknown_proposal_indices=state.unknown,
                not_dispatched_proposal_indices=[
                    index
                    for index in indices
                    if index not in state.confirmed + state.unknown
                ],
                error=error,
            )
            journal.finalize()
        finally:
            _RPC_EVIDENCE.reset(token)


class EnvironmentJournal(RuntimeJournal):
    """Record environment-service facts in a separate, ordered stream."""

    def __init__(self, context: dict[str, Any]) -> None:
        self.construction_complete = False
        self.construction_wall_seconds = 0.0
        self.reset_calls = 0
        self.reset_completed = 0
        self.reset_wall_seconds = 0.0
        self.setup_steps_confirmed = 0
        self.policy_steps_confirmed = 0
        self.service_closed = False
        self.pending_action = False
        self.last_native_success: bool | None = None
        self.last_feedback_sequence: int | None = None
        output = Path(context["output_dir"]) / "environment_service"
        super().__init__({**context, "output_dir": str(output)})

    def emit(self, kind: str, **fields: Any) -> int:
        """Use the owning process clock and its precise coverage scope."""
        fields["clock_domain"] = "environment_service"
        rpc = _RPC_EVIDENCE.get()
        if rpc is not None and rpc.current_link is not None:
            fields["rpc_link"] = rpc.current_link
        if kind == "run_start":
            fields["missing"] = [
                "controller_applied_action",
                "simulator_internal_physics_ticks",
            ]
            fields["coverage_scope"] = (
                "installed LiberoEnv vector-env.step dispatch and native worker feedback"
            )
        return super().emit(kind, **fields)

    def costs(self) -> dict[str, Any]:
        """Report total native actions once, separating setup from policy."""
        return {
            "schema_version": "1.0",
            **self.identity(),
            "identity": self.identity(),
            "environment_service_wall_seconds": time.monotonic() - self.started,
            "environment_steps_reserved": self.reserved_steps,
            "environment_steps_confirmed": self.confirmed_steps,
            "environment_steps_unknown": self.uncertain_steps,
            "setup_steps_confirmed": self.setup_steps_confirmed,
            "policy_steps_confirmed": self.policy_steps_confirmed,
            "construction_complete": self.construction_complete,
            "construction_wall_seconds": self.construction_wall_seconds,
            "reset_calls": self.reset_calls,
            "reset_completed": self.reset_completed,
            "reset_wall_seconds": self.reset_wall_seconds,
            "service_closed": self.service_closed,
            "pending_action": self.pending_action,
            "last_native_success": self.last_native_success,
            "last_feedback_sequence": self.last_feedback_sequence,
            "native_success_source": "native vector worker step <- LIBERO BDDLBaseDomain.step: _check_success()",
            "coverage_scope": "vector-env.step; setup and policy counts partition total confirmed actions",
            "missing": [
                "controller_applied_action",
                "simulator_internal_physics_ticks",
            ],
        }


class _ObservedVectorEnv:
    """Delegate the vector API while observing the actual native step boundary."""

    def __init__(self, vector: Any, owner: Any, journal: EnvironmentJournal) -> None:
        self._vector = vector
        self._owner = owner
        self._journal = journal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vector, name)

    def step(self, actions: Any, *args: Any, **kwargs: Any) -> Any:
        journal = self._journal
        source = self._owner._rsi_phase
        rpc = _RPC_EVIDENCE.get()
        if rpc is not None:
            slot = rpc.next_slot
            indices = rpc.metadata["proposal_indices"]
            if slot >= len(indices):
                raise ValueError("native step exceeds declared RSI RPC slots")
            rpc.next_slot += 1
            rpc.current_link = {
                "client_send_sequence": rpc.metadata["client_send_sequence"],
                "client_proposal_sequence": rpc.metadata["proposal_sequence"],
                "client_proposal_index": indices[slot],
                "rpc_slot_index": slot,
                "rpc_expected_slots": len(indices),
            }
        proposal = journal.emit(
            "action_proposal",
            source=source,
            proposal=actions,
            reset_index=journal.reset_calls,
            producer="installed RLinf LiberoEnv.reset/step",
        )
        original = []
        dispatched = False

        def dispatch():
            nonlocal dispatched
            dispatched = True
            journal.pending_action = True
            journal.finalize()
            result = self._vector.step(actions, *args, **kwargs)
            original.append(result)
            raw_obs, reward, terminated, _info = result
            values = (
                terminated.tolist() if hasattr(terminated, "tolist") else terminated
            )
            if isinstance(values, (list, tuple)) and len(values) == 1:
                values = values[0]
            journal.last_native_success = values if type(values) is bool else None
            journal.pending_action = False
            # Keep numeric observed state, not default camera frames, in L0.
            state = [
                {
                    key: observation.get(key)
                    for key in (
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    )
                }
                for observation in raw_obs
            ]
            # Native vector feedback has no RLinf episode-horizon truncation.
            return {"states": state}, reward, terminated, None, {}

        before = journal.confirmed_steps
        try:
            journal.action(
                dispatch,
                actions,
                source=source,
                proposal_sequence=proposal,
                proposal_length=len(actions),
                chunk=True,
                already_admitted=rpc is not None,
            )
            journal.last_feedback_sequence = journal.sequence - 1
        finally:
            confirmed = journal.confirmed_steps - before
            if rpc is not None:
                index = rpc.current_link["client_proposal_index"]
                if confirmed:
                    rpc.confirmed.append(index)
                elif dispatched:
                    rpc.unknown.append(index)
                rpc.current_link = None
            if source == "rlinf.policy":
                journal.policy_steps_confirmed += confirmed
            else:
                journal.setup_steps_confirmed += confirmed
            journal.finalize()
        return original[0]

    def close(self) -> None:
        """Retain closing cost and completion without changing vector cleanup."""
        try:
            self._vector.close()
            self._journal.service_closed = True
            self._journal.emit("environment_service_closed")
        except BaseException as exc:
            self._journal.emit(
                "environment_service_close_error", error=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            self._journal.finalize()


def observed_libero_class(base: type, context: dict[str, Any]) -> type:
    """Wrap the installed class at its actual vector-env initialization hook."""

    class ObservedLiberoEnv(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._rsi_journal = EnvironmentJournal(context)
            self._rsi_phase = "rlinf.construction"
            started = time.monotonic()
            self._rsi_journal.emit("environment_construction_start")
            try:
                super().__init__(*args, **kwargs)
                self._rsi_journal.construction_complete = True
                self._rsi_journal.emit("environment_construction_complete")
            except BaseException as exc:
                self._rsi_journal.emit(
                    "environment_construction_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                self._rsi_journal.construction_wall_seconds = time.monotonic() - started
                self._rsi_phase = "rlinf.policy"
                self._rsi_journal.finalize()

        def _init_env(self) -> None:
            super()._init_env()
            self.env = _ObservedVectorEnv(self.env, self, self._rsi_journal)

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            journal = self._rsi_journal
            journal.reset_calls += 1
            journal.emit("reset_start", reset_index=journal.reset_calls)
            previous, self._rsi_phase = self._rsi_phase, "rlinf.reset_settling"
            started = time.monotonic()
            try:
                result = super().reset(*args, **kwargs)
                journal.reset_completed += 1
                journal.emit("reset_complete", reset_index=journal.reset_calls)
                return result
            except BaseException as exc:
                journal.emit(
                    "reset_error",
                    reset_index=journal.reset_calls,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                journal.reset_wall_seconds += time.monotonic() - started
                self._rsi_phase = previous
                journal.finalize()

    return ObservedLiberoEnv
