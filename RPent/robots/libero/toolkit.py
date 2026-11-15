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

"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.session.runtime import RuntimeBudgetExceeded, RuntimeJournal
from rpent.tools import ToolResult, iter_tools
from rpent.tools.base import tool
from rpent.tools.direct_action import direct_action_tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("libero_toolkit")


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO robot."""

    runtime_journal: RuntimeJournal | None = None
    _common_execution: dict[str, Any] | None = None
    _common_prediction_state: ToolResult | None = None

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        enable_direct_action: bool = False,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        policy_kind: str = "vla",
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported LIBERO toolkit mode: {mode!r}")
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._mode = mode
        self._policy_kind = policy_kind
        self._solved: bool = False
        self._attempt: int = 1
        # Bound the resettable attempts owned by this planner session.
        self._attempts_per_session: int = max(0, int(attempts_per_session))
        self._session_attempt: int = 1
        self.runtime_journal = RuntimeJournal.from_environment()
        try:
            self.init_primitives(runtime_kwargs=runtime_kwargs)
        except BaseException as exc:
            if self.runtime_journal is not None:
                self.runtime_journal.emit(
                    "toolkit_initialization_error", error=f"{type(exc).__name__}: {exc}"
                )
                self.runtime_journal.finalize()
            raise
        self._register_libero_tools()
        if policy_kind == "wam" or (
            self.runtime_journal is not None
            and getattr(self.runtime_journal, "finish_requires_native_success", False)
        ):
            finish = self._tools["finish"]
            self.add_tool(
                replace(
                    finish.with_handler(partial(self._wam_finish, finish)),
                    description="End the episode with its observed outcome. "
                    "Success requires native environment success.",
                ),
                replace=True,
            )
        if enable_direct_action:
            self.add_tool(
                direct_action_tool(
                    self._primitives.env.action_specs,
                    partial(
                        self._execute_primitive,
                        "execute_action",
                        self._primitives.execute_action,
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        from robots.libero.execution_arms import validate_common_execution

        journal = self.runtime_journal
        common = journal.context.get("common_execution") if journal else None
        self._common_execution = (
            validate_common_execution(
                common,
                action_prefix_steps=journal.chunk_steps or 5,
                environment_steps=journal.context["limits"]["environment_steps"],
                tool_calls=journal.context["limits"]["tool_calls"],
            )
            if common is not None
            else None
        )
        self._common_prediction_state = None
        for definition in iter_tools(libero_tools):
            if definition.name == "view_env_state" and self.runtime_journal is not None:
                detail = self.runtime_journal.context.get(
                    "planner_observation", {}
                ).get("image_detail", "full")
                original = definition

                def view_handler(*, _definition=original, _detail=detail, **kwargs):
                    if kwargs.get("image_detail") is None:
                        kwargs["image_detail"] = _detail
                    return _definition(state=self._state, **kwargs)

                self.add_tool(definition.with_handler(view_handler))
            else:
                self.add_tool(
                    definition.with_handler(partial(definition, state=self._state))
                )
        for definition in iter_tools(self._primitives):
            if self._common_execution and definition.name not in {
                "vla_execute",
                "segment",
            }:
                continue
            excluded = (
                {"pi0_pick", "pi0_doubled", "vla_execute"}
                if self._policy_kind == "wam"
                else {"wam_act"}
            )
            if definition.name in excluded:
                continue
            if definition.name == "segment":
                handler = partial(definition, state=self._state)
            else:
                handler = partial(self._execute_primitive, definition.name, definition)
            self.add_tool(definition.with_handler(handler))
        from robots.libero.skill_routing import load_control

        self._relation_control = load_control(self.runtime_journal, set(self._tools))
        if self._relation_control is not None:
            self.runtime_journal.emit(
                "relation_policy_loaded", relation="rhoB",
                rho_ref=self._relation_control["rho_ref"],
                version_signature=self._relation_control["version_signature"],
                consumer="LiberoToolkit._execute_primitive",
                route_ids=[row["route_id"] for row in self._relation_control["routes"]],
            )
        if self._mode == "exploration":
            self.add_tool(self._reset_episode)
            finish = self._tools["finish"]
            self.add_tool(
                finish.with_handler(partial(self._guarded_finish, finish)), replace=True
            )

    def _execute_primitive(
        self, name: str, handler: Callable[..., ToolResult], **kwargs: Any
    ) -> ToolResult:
        if self._common_execution and name != "vla_execute":
            raise ValueError("common execution permits only vla_execute motion")
        env = self._primitives.env
        if self._policy_kind == "wam" and (env.terminated or env.truncated):
            status = "success" if env.terminated else "failure"
            raise ValueError(
                "Episode already ended; no further motion is allowed. "
                f"Call finish with status={status!r}."
            )
        from robots.libero.skill_routing import route_for_call

        route = route_for_call(getattr(self, "_relation_control", None), name, kwargs)
        if route is None:
            self._primitives.begin_primitive(name)
            try:
                if self._common_execution and name == "vla_execute":
                    return self._execute_common_vla_segment(handler, **kwargs)
                return handler(**kwargs)
            finally:
                self._primitives.end_primitive()
        journal = self.runtime_journal
        selected = route["selected_worker"]
        binding = {"rho_ref": self._relation_control["rho_ref"],
                   "route_id": route["route_id"], "state": route["state"],
                   "requested_worker": route["requested_worker"], "trigger_worker": name,
                   "selected_worker": selected,
                   "criterion_ref": route["criterion"]["criterion_ref"]}
        decision = journal.emit("relation_route_decision", **binding)
        binding["decision_sequence"] = decision
        if selected in {"handoff", "defer"}:
            return ToolResult(data={
                "relation_route": binding, "action": selected, "executed_steps": 0,
                "next_action": "Inspect the current observations before choosing the next skill."
                if selected == "handoff" else "This registered skill route is deferred.",
            })
        drawer = route["criterion"].get("schema") == "rsi-skill-drawer-open-criterion-v1"
        allowance = kwargs.get("max_chunks", 24)
        if type(allowance) is not int or not 1 <= allowance <= 24:
            raise ValueError("routed skills require an unchanged 1..24 chunk allowance")
        if drawer and (allowance != route["max_chunks"] or set(kwargs) - {
                "instruction" if name == "vla_execute" else "prompt", "max_chunks"}):
            raise ValueError("registered drawer route requires its exact 24chunk argument contract")
        actual_kwargs = kwargs if selected == name and not drawer else {
            "prompt" if selected in {"pi0_pick", "pi0_doubled"} else "instruction": route["instruction"],
            "max_chunks": allowance,
        }
        actual_handler = handler if selected == name else getattr(self._primitives, selected)
        evidence = {**journal.identity(), "route_id": route["route_id"],
                    "worker": selected, "client_route_sequence": decision}
        before = env.rsi_skill_observation(evidence={**evidence, "phase": "start"})
        journal.emit("skill_route_start", **binding, client_route_sequence=decision,
                     actual_arguments=actual_kwargs, environment_observation=before)
        self._primitives.begin_primitive(selected)
        try:
            result = actual_handler(**actual_kwargs)
        finally:
            self._primitives.end_primitive()
        after = env.rsi_skill_observation(evidence={**evidence, "phase": "handoff"})
        journal.emit("skill_route_handoff", **binding, client_route_sequence=decision,
                     actual_arguments=actual_kwargs, chunks_used=result.data.get("chunks_used"),
                     execution_error=result.data.get("error"), environment_observation=after)
        result.data["relation_route"] = binding
        return result

    def _execute_common_vla_segment(
        self, handler: Callable[..., ToolResult], *, instruction: str, max_chunks: int
    ) -> ToolResult:
        """Persist one native prediction at a time for every execution arm."""
        journal = self.runtime_journal
        logical_call = journal.emit(
            "common_execution_call_start",
            instruction=instruction,
            max_chunks=max_chunks,
            common_execution=self._common_execution,
        )
        totals = dict.fromkeys(
            (
                "predictions_started",
                "predictions_completed",
                "executed_steps",
                "unknown_steps",
            ),
            0,
        )
        segments = []
        admission_error = None
        for index in range(max_chunks):
            if self._primitives.env.terminated or self._primitives.env.truncated:
                break
            if index:
                internal_call = journal.sequence
                try:
                    journal.admit_tool("vla_execute")
                except RuntimeBudgetExceeded as exc:
                    admission_error = str(exc)
                    break
                journal.emit(
                    "common_internal_tool_admission",
                    logical_call_sequence=logical_call,
                    tool_call_sequence=internal_call,
                )
            started = time.monotonic()
            segment = None
            try:
                segment = handler(instruction=instruction, max_chunks=1).data
                segments.append(segment)
                for key in totals:
                    totals[key] += segment.get(key, 0)
                if segment.get("predictions_started", 0):
                    self._common_prediction_state = self._capture_env_state(
                        command={
                            "action": "vla_execute",
                            "instruction": instruction,
                            "max_chunks": 1,
                        },
                        result=segment,
                        elapsed_s=time.monotonic() - started,
                    )
            finally:
                if index:
                    journal.emit(
                        "tool_end",
                        tool="vla_execute",
                        logical_call_sequence=logical_call,
                        tool_call_sequence=internal_call,
                        status="failed"
                        if segment is None or segment.get("error")
                        else "completed",
                        elapsed_seconds=time.monotonic() - started,
                    )
            journal.emit(
                "common_prediction_call",
                logical_call_sequence=logical_call,
                prediction_index=index,
                segment_sequence=segment.get("segment_sequence"),
                tool_allowance_source="outer_call"
                if index == 0
                else "internal_prediction",
                recording_cadence="prediction",
            )
            if segment.get("stop_reason") != "chunk_allowance":
                break
        if not segments:
            return handler(instruction=instruction, max_chunks=1)
        data = {
            **segments[-1],
            **totals,
            "max_chunks": max_chunks,
            "chunks_used": totals["predictions_completed"],
            "common_execution_call_sequence": logical_call,
            "common_execution_segments": segments,
        }
        if admission_error is not None:
            data.update(
                stop_reason="budget",
                segment_status="budget",
                diagnostic=admission_error,
            )
        journal.emit(
            "common_execution_call_end", logical_call_sequence=logical_call, **data
        )
        return ToolResult(data=data)

    def _wam_finish(self, inner: Any, **kwargs: Any) -> ToolResult:
        """Require native task success before accepting a success claim."""
        if (
            kwargs.get("status") == "success"
            and getattr(self._primitives.env, "terminated", None) is not True
        ):
            return ToolResult(
                data={
                    "error": "finish refused: success requires native terminated=true",
                    "terminated": self._primitives.env.terminated,
                    "truncated": self._primitives.env.truncated,
                    "reason": "Inspect the current state. Continue if the episode is "
                    "active; report failure if it was truncated.",
                }
            )
        return inner(**kwargs)

    def _guarded_finish(self, inner: Any, **kwargs: Any) -> ToolResult:
        """Refuse to end an unsolved session while attempts remain."""
        budget = self._attempts_per_session
        if budget and not self.solved() and self._session_attempt < budget:
            remaining = budget - self._session_attempt
            return ToolResult(
                data={
                    "error": "finish refused",
                    "reason": f"This session has {remaining} of its {budget} attempts left and the task is not solved. Archive this attempt, call `reset`, and try another approach.",
                }
            )
        return inner(**kwargs)

    @tool(name="reset")
    def _reset_episode(self, reason: str) -> ToolResult:
        """EXPLORE MODE ONLY. Abandon the current episode and restore the same initial scene. Archive the failed attempt first and state which strategy lever will change in the next attempt.

        Args:
            reason: Why this episode is unrecoverable and what will change.
        """
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return ToolResult(
                data={
                    "error": "reset refused",
                    "reason": f"This session's attempt budget is spent ({budget} attempts). Archive the attempt, update the handoff notes, and call `finish` so the next session can continue.",
                }
            )
        self._attempt += 1
        self._session_attempt += 1
        result = self._primitives.reset_episode(reason=reason)
        result["attempt"] = self._attempt
        result["notice"] = (
            f"Episode restarted; this is attempt {self._attempt}. The original "
            "layout was restored. Re-run perception before acting."
        )
        return ToolResult(data=result)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        if (
            command.get("action") == "vla_execute"
            and self._common_prediction_state is not None
        ):
            saved = self._common_prediction_state
            out = ToolResult(data=dict(saved.data), images=list(saved.images))
            out.data["execution_segment"] = result
            out.data["agent_elapsed_s"] = elapsed_s
            return out
        return self._capture_env_state(
            command=command, result=result, elapsed_s=elapsed_s
        )

    def _capture_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Save the common observation without adding another action boundary."""
        media_started = time.monotonic()
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        self._solved |= record.terminated
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        detail = (
            self.runtime_journal.context.get("planner_observation", {}).get(
                "image_detail", "full"
            )
            if self.runtime_journal
            else "full"
        )
        out = libero_tools.view_env_state(
            record.step_idx, state=self._state, image_detail=detail
        )
        if command.get("action") == "vla_execute":
            out.data["execution_segment"] = result
        if (
            record.terminated
            and not result.get("error")
            and self.runtime_journal is not None
            and self.runtime_journal.context.get("execution_version")
            == "bounded-vla-v1"
        ):
            self.runtime_journal.complete_native_success(step=record.step_idx)
        out.data["agent_elapsed_s"] = elapsed_s
        if self._policy_kind == "wam" and (record.terminated or record.truncated):
            status = "success" if record.terminated else "failure"
            out.data["next_action"] = (
                f"Episode ended. Call finish with status={status!r}; "
                "further motion is invalid."
            )
        if result.get("interrupted"):
            out.data.update(result)
        if self.runtime_journal is not None:
            self.runtime_journal.emit(
                "media_state_end", elapsed_seconds=time.monotonic() - media_started
            )
        return out

    @property
    def primitives(self) -> "libero_tools.LiberoPrimitives":
        """Return the action primitives this toolkit drives."""
        return self._primitives

    def init_primitives(
        self,
        *,
        runtime_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            policy_kind=self._policy_kind,
            runtime_journal=self.runtime_journal,
            **runtime_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Finalize collected data and save the episode video independently."""
        super().close()
        try:
            media_started = time.monotonic()
            episode = self._primitives.finalize_flywheel()
            if episode is not None:
                logger.info("flywheel episode finalized: %s", episode)
        except Exception as e:
            logger.warning("failed to finalize flywheel episode: %s", e)
            if self.runtime_journal is not None:
                self.runtime_journal.emit(
                    "collection_error", error=f"{type(e).__name__}: {e}"
                )

        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            logger.warning(f"failed to save episode video: {e}")
        if self.runtime_journal is not None:
            self.runtime_journal.emit(
                "media_cleanup_end", elapsed_seconds=time.monotonic() - media_started
            )
            self.runtime_journal.finalize()

    def solved(self) -> bool:
        """Return whether this run has completed the task."""
        return self._solved

    def evaluation_metrics(self, *, success: bool | None) -> dict[str, Any]:
        """Snapshot native episode flags and separately retain tool success claims."""
        env = self._primitives.env
        tool_successes = []
        execution_segments = []
        for record in self._state.records():
            result = record.result or {}
            action = (record.command or {}).get("action")
            if action == "vla_execute":
                execution_segments.append(
                    {
                        **result,
                        "step_idx": record.step_idx,
                        "evidence_ref": str(self._state_output_dir / "states.json"),
                    }
                )
            if "success" not in result:
                continue
            tool_successes.append(
                {
                    "step_idx": record.step_idx,
                    "action": action,
                    "success": result["success"],
                    "source": "states.json:steps[].result.success",
                    "meaning": (
                        "pick_heuristic_or_native_termination"
                        if action == "pi0_pick"
                        else "tool_reported"
                    ),
                }
            )
        return {
            "terminated": getattr(env, "terminated", None),
            "truncated": getattr(env, "truncated", None),
            "episode_execution": getattr(self, "_episode_execution", None),
            "execution_arm": (
                self.runtime_journal.context.get("execution_arm")
                if self.runtime_journal
                else None
            ),
            "mode": self._mode,
            "attempt": self._attempt,
            "tool_successes": tool_successes,
            "execution_segments": execution_segments,
            "runtime": (
                {
                    "identity": self.runtime_journal.identity(),
                    "action_events_ref": str(self.runtime_journal.events_path),
                    "costs_ref": str(self.runtime_journal.costs_path),
                    "costs": self.runtime_journal.costs(),
                    "environment_service": {
                        "action_events_ref": str(
                            self.runtime_journal.output_dir
                            / "environment_service"
                            / "action_events.jsonl"
                        ),
                        "costs_ref": str(
                            self.runtime_journal.output_dir
                            / "environment_service"
                            / "runtime_costs.json"
                        ),
                    },
                }
                if getattr(self, "runtime_journal", None) is not None
                else None
            ),
        }

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(
            self._state, recipe_tag, output_dir=get_output_dir()
        )

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        """Admit tools against the active run's total call budget."""
        journal = self.runtime_journal
        if journal is None:
            return super().execute_tool(name, arguments)
        started = time.monotonic()
        status, admitted = "failed", False
        try:
            journal.admit_tool(name)
            admitted = True
            from robots.libero.skill_routing import private_evidence_access

            if private_evidence_access(journal, name, arguments):
                return ToolResult(data={
                    "error": "Environment evaluation evidence is private to the independent evaluator.",
                    "runtime_budget": journal.budget_snapshot(),
                })
            result = super().execute_tool(name, arguments)
            status = (
                "interrupted"
                if result.data.get("interrupted")
                else "failed"
                if result.data.get("error")
                else "completed"
            )
            result.data["runtime_budget"] = journal.budget_snapshot()
            return result
        finally:
            journal.emit(
                "tool_end",
                tool=name,
                status=status if admitted else "rejected",
                elapsed_seconds=time.monotonic() - started,
            )
