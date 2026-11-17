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

"""Independent LIBERO result artifacts from native episode signals."""

from __future__ import annotations

import json
from pathlib import Path

from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.session.runtime import read_budget_stop


def execution_health(runtime: dict) -> dict:
    """Read bound execution faults; normal Planner exit is no recovery evidence."""
    reference = runtime.get("action_events_ref")
    identity = runtime.get("identity") or {}
    issues, stops, missing = [], [], []
    try:
        events = [json.loads(line) for line in Path(reference).read_text().splitlines()]
        for index, event in enumerate(events):
            if event.get("sequence") != index or any(
                event.get(key) != identity.get(key)
                for key in ("run_id", "episode_id", "bundle_id", "protocol_id")
            ):
                raise ValueError("execution event identity/sequence mismatch")
            kind = event.get("kind")
            if (
                kind == "inference_end"
                and event.get("completed") is False
                or kind == "action_feedback"
                and (event.get("error") or event.get("execution_mask") is None)
                or kind == "vla_segment_end"
                and event.get("stop_reason") == "technical_error"
                or kind == "skill_execution_result"
                and (
                    event.get("tool_status") == "failed"
                    or event.get("action_feedback_unknown") is True
                )
            ):
                issues.append(event)
            if (
                kind == "vla_segment_end"
                and event.get("stop_reason") in {"cancelled", "budget"}
                or kind == "skill_execution_result"
                and event.get("tool_status") == "interrupted"
            ):
                stops.append(event)
    except (OSError, ValueError, TypeError) as exc:
        missing.append(f"execution_events_unconfirmed:{type(exc).__name__}")
    return {
        "status": "unresolved" if issues else "unconfirmed" if missing else "clear",
        "issues": issues,
        "stops": stops,
        "events_ref": reference,
        "missing": missing,
        "recovery_criterion": "not_defined_for_execution_faults" if issues else None,
    }


def finalize_run(context: RunFinalizationContext) -> Path:
    """Persist native task results separately from planner and tool claims.

    LIBERO's BDDLBaseDomain.step replaces done with _check_success(); RLinf
    forwards it as terminated when ignore_terminations=False. RLinf owns the
    separate max_episode_steps truncation. This mapping is LIBERO-specific.
    An active episode or missing native flags has no completed benchmark result.
    """
    metrics = context.environment_metrics
    terminated = metrics.get("terminated")
    truncated = metrics.get("truncated")
    benchmark_success = None
    benchmark_status = "unknown"
    if terminated is True:
        benchmark_success, benchmark_status = True, "success"
    elif terminated is False and truncated is True:
        benchmark_success, benchmark_status = False, "truncated"
    elif terminated is False and truncated is False:
        benchmark_status = "active"

    # The runner drains tools and closes owned services before this finalizer.
    # Its normal return closes the attempt, while native success still comes
    # solely from the matching environment worker's confirmed last step.
    runtime = metrics.get("runtime") or {}
    episode = metrics.get("episode_execution") or {}
    health = execution_health(runtime) if runtime else None
    segments = metrics.get("execution_segments", [])
    if segments:
        health = health or {
            "status": "clear",
            "issues": [],
            "stops": [],
            "events_ref": None,
            "missing": [],
            "recovery_criterion": None,
        }
        for segment in segments:
            if (
                segment.get("stop_reason") == "technical_error"
                or segment.get("tool_status") == "failed"
                or segment.get("action_feedback_unknown") is True
            ):
                health["issues"].append(segment)
                health["status"] = "unresolved"
                health["recovery_criterion"] = "not_defined_for_execution_faults"
            elif segment.get("stop_reason") in {"cancelled", "budget"}:
                health["stops"].append(segment)
    if episode.get("status") == "technical_error":
        health = health or {
            "status": "unresolved",
            "issues": [],
            "stops": [],
            "events_ref": runtime.get("action_events_ref"),
            "missing": [],
        }
        health["status"] = "unresolved"
        health["issues"].append(episode)
        health["recovery_criterion"] = "not_defined_for_execution_faults"
    technical = bool(
        context.agent_error
        or health
        and health["status"] != "clear"
        or episode.get("status") == "technical_error"
    )
    segment_cancelled = bool(
        episode.get("status") == "cancelled"
        or health
        and any(e.get("stop_reason") == "cancelled" for e in health["stops"])
    )
    identity = runtime.get("identity")
    service_ref = runtime.get("environment_service", {}).get("costs_ref")
    service = None
    service_missing = []
    if service_ref and identity:
        try:
            service = json.loads(Path(service_ref).read_text(encoding="utf-8"))
            if any(
                service.get("identity", {}).get(key) != value
                for key, value in identity.items()
            ):
                service_missing.append("environment_service_identity_mismatch")
                service = None
        except (OSError, ValueError) as exc:
            service_missing.append(
                f"environment_service_unavailable:{type(exc).__name__}"
            )
    finish = context.finish_result or {}
    service_confirmed = bool(
        service
        and service.get("service_closed") is True
        and service.get("pending_action") is False
        and service.get("environment_steps_unknown") == 0
        and type(service.get("last_native_success")) is bool
        and service.get("last_feedback_sequence") is not None
        and service.get("native_success_source")
    )
    planner_stop = context.stats.get("planner_budget_stop")
    budget_verified = False
    if (
        isinstance(planner_stop, dict)
        and identity
        and planner_stop.get("budget_stop_ref")
    ):
        try:
            marker = read_budget_stop(planner_stop["budget_stop_ref"], identity)
            if marker is not None:
                reason, limit, observed = (
                    marker.get("reason"),
                    marker.get("limit"),
                    marker.get("observed"),
                )
                numeric = type(limit) in (int, float) and type(observed) in (int, float)
                exhausted = numeric and observed >= limit
                if reason == "environment_steps" and numeric:
                    requested = marker.get("requested_steps", 0)
                    exhausted = exhausted or (
                        type(requested) is int and observed + requested > limit
                    )
                budget_verified = bool(
                    marker.get("stop_after_inflight") is True
                    and reason
                    in {
                        "tool_calls",
                        "environment_steps",
                        "max_turns",
                        "model_requests",
                    }
                    and exhausted
                    and all(
                        planner_stop.get(key) == marker.get(key)
                        for key in (
                            "identity",
                            "reason",
                            "limit",
                            "observed",
                            "producer_events_ref",
                            "sequence",
                        )
                    )
                    and type(planner_stop.get("planner_event_sequence")) is int
                    and planner_stop.get("planner_events_ref")
                    == runtime.get("action_events_ref")
                )
        except (OSError, ValueError, TypeError):
            service_missing.append("unverified_budget_stop")
    # A local admission rejection may be reported as a failed SDK turn. Keep
    # that raw status; only its bound, persisted receipt permits budget closure.
    local_budget_failure = False
    if budget_verified and context.stats.get("planner_turn_status") == "failed":
        local = planner_stop.get("local_model_stop") or {}
        errors = context.stats.get("sdk_errors") or []
        try:
            saved = json.loads(Path(planner_stop["local_model_stop_ref"]).read_text())
            local_budget_failure = bool(
                saved == local
                and local.get("identity") == identity
                and local.get("reason") == "budget_exhausted"
                and local.get("forwarded") is False
                and local.get("marker_ref") == planner_stop["budget_stop_ref"]
                and local.get("marker_sequence") == planner_stop["sequence"]
                and errors
                and all(e.get("local_rejection_candidate") is True for e in errors)
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    budget_closed = bool(
        budget_verified
        and service_confirmed
        and service["last_native_success"] is False
        and not technical
        and not segment_cancelled
        and not finish.get("operator_aborted")
        and not finish.get("interrupted")
        and context.stats.get("attempt_closed") is True
        and (
            context.stats.get("planner_turn_status") in {"interrupted", "completed"}
            or local_budget_failure
        )
    )
    # Deterministic consumers have no Planner receipt. Their persisted budget
    # marker is still produced by the same journal before the rejected action.
    if episode.get("status") == "budget_exhausted" and identity:
        try:
            marker = read_budget_stop(episode["budget_stop_ref"], identity)
            limit, observed = (
                (marker or {}).get("limit"),
                (marker or {}).get("observed"),
            )
            numeric = type(limit) in (int, float) and type(observed) in (int, float)
            requested = (marker or {}).get("requested_steps", 0)
            exhausted = numeric and (
                observed >= limit
                or marker.get("reason") == "environment_steps"
                and type(requested) is int
                and observed + requested > limit
            )
            producer = (marker or {}).get("producer_events_ref")
            client_producer = bool(
                producer and producer == runtime.get("action_events_ref")
            )
            # The native owner counts reset settling and policy actions in the
            # same physical allowance. Its first-cause marker may precede the
            # last admitted action's feedback, which service closure drains.
            environment_producer = bool(
                producer
                and marker.get("reason") == "environment_steps"
                and producer
                == runtime.get("environment_service", {}).get("action_events_ref")
                and service
                and service.get("environment_steps_reserved")
                == service.get("environment_steps_confirmed")
                == observed
            )
            budget_closed = bool(
                marker
                and marker.get("stop_after_inflight") is True
                and exhausted
                and (client_producer or environment_producer)
                and service_confirmed
                and service["last_native_success"] is False
                and not technical
                and not segment_cancelled
                and not finish.get("operator_aborted")
                and not finish.get("interrupted")
            )
        except (OSError, ValueError, TypeError, KeyError):
            service_missing.append("unverified_execution_arm_budget_stop")
    normal_closed = bool(
        not technical
        and not segment_cancelled
        and not finish.get("operator_aborted")
        and not finish.get("interrupted")
        and service_confirmed
        and episode.get("status")
        not in {"unknown", "technical_error", "cancelled", "budget_exhausted"}
        and context.stats.get("planner_turn_status")
        not in {"interrupted", "cancelled", "failed"}
        and (
            finish.get("_finish") is True
            or context.stats.get("attempt_closed") is True
            or context.stats.get("planner_turn_status") == "completed"
        )
    )
    attempt_closed = benchmark_success is not None or normal_closed or budget_closed
    termination_reason = (
        "native_success"
        if benchmark_success is True
        else "native_episode_horizon"
        if benchmark_success is False
        else "verified_budget_exhaustion"
        if budget_closed
        else "normal_runner_completion"
        if normal_closed
        else "unconfirmed_or_interrupted"
    )
    if benchmark_success is None and (normal_closed or budget_closed):
        benchmark_success = service["last_native_success"]
        benchmark_status = "success" if benchmark_success else "failure"

    record = {
        "schema_version": "1.0",
        "robot": context.robot_name,
        "task": dict(context.task_desc),
        "mode": metrics.get("mode"),
        "attempt": metrics.get("attempt"),
        "planner_finish": dict(context.finish_result)
        if context.finish_result is not None
        else None,
        "tool_successes": metrics.get("tool_successes", []),
        "terminated": terminated,
        "truncated": truncated,
        "native_result_source": {
            "terminated": (
                "LiberoEnvClient.terminated <- RLinf LiberoEnv.step/chunk_step <- "
                "LIBERO BDDLBaseDomain.step: _check_success()"
                if terminated is not None
                else None
            ),
            "truncated": (
                "LiberoEnvClient.truncated <- RLinf LiberoEnv.step/chunk_step: "
                "elapsed_steps >= max_episode_steps"
                if truncated is not None
                else None
            ),
        },
        "benchmark_success": benchmark_success,
        "benchmark_status": benchmark_status,
        "attempt_closed": attempt_closed,
        "termination_reason": termination_reason,
        "verified_budget_exhaustion": budget_closed,
        "attempt_outcome_source": {
            "native_success": service.get("native_success_source") if service else None,
            "evidence_ref": service_ref,
            "last_feedback_sequence": service.get("last_feedback_sequence")
            if service
            else None,
            "runner_completion": "RunFinalizationContext after toolkit drain and service close"
            if normal_closed
            else None,
            "budget_completion": "explicit local budget stop after toolkit drain and service close"
            if budget_closed
            else None,
            "planner_budget_stop": planner_stop,
        },
        "environment_service": service,
        "missing": service_missing,
        "process_labels": {
            "target_object_correct": None,
            "destination_correct": None,
            "released": None,
            "settled": None,
        },
        "error": context.agent_error,
        "planner_diagnostics": {
            key: context.stats[key]
            for key in (
                "sdk_errors",
                "planner_turn_status",
                "planner_native_stop",
                "planner_budget_stop",
            )
            if key in context.stats
        },
        "execution_health": health,
        "execution_arm": metrics.get("execution_arm"),
        "episode_execution": metrics.get("episode_execution"),
        "elapsed_s": round(context.elapsed_s, 1),
        "planner": {
            "backend": context.planner,
            "model": context.model,
            "reasoning_effort": context.reasoning_effort,
            "max_turns": context.max_turns,
            "timeout_s": context.planner_timeout_s,
        },
        "runtime": metrics.get("runtime"),
        "interrupted": bool(
            context.agent_error
            or segment_cancelled
            or context.stats.get("planner_turn_status")
            in {"interrupted", "cancelled", "failed"}
        )
        and benchmark_success is None,
        "validity": "technical_invalid" if technical else "valid",
        "costs": (metrics.get("runtime") or {}).get("costs"),
    }
    if metrics.get("runtime"):
        record.update(metrics["runtime"]["identity"])
    else:
        record["missing"] = [
            "runtime_identity",
            "action_level_evidence",
            "complete_cost_ledger",
        ]
    return write_json_atomic(context.output_dir / "result.json", record)
