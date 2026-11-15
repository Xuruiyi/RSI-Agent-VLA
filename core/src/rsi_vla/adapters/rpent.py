"""Read existing RPent JSON only. No RPent import, services, model or device."""
import json
from pathlib import Path
from ..contracts import ExecutionEvent


def action_origin(action):
    """Separate Planner invocation from the generator of robot actions."""
    if action in {"vla_execute", "pi0_pick", "pi0_doubled"}:
        return "vla"
    if action in {"move_to", "rotate_wrist", "rotate_pitch", "move_pose", "release", "set_gripper"}:
        return "scripted"
    return "unknown"


def execution_health(events, identity, *, source_ref=None):
    """Audit recorded execution faults without treating arbitrary tool errors as fatal.

    No current producer defines evidence that resolves inference/RPC faults.
    Later tool completion, native success and Planner narration cannot clear them.
    """
    issues, stops, missing = [], [], []
    for index, event in enumerate(events):
        if event.get('sequence') != index or any(event.get(key) != identity[key]
                for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')):
            missing.append('execution_event_identity_or_sequence_mismatch')
            continue
        kind = event.get('kind')
        if (kind == 'inference_end' and event.get('completed') is False
                or kind == 'action_feedback' and (event.get('error') or event.get('execution_mask') is None)
                or kind == 'vla_segment_end' and event.get('stop_reason') == 'technical_error'):
            issues.append(event)
        if kind == 'vla_segment_end' and event.get('stop_reason') in {'budget', 'cancelled'}:
            stops.append(event)
    return dict(status='unresolved' if issues else 'unconfirmed' if missing else 'clear',
                issues=issues, stops=stops, events_ref=source_ref, missing=missing,
                recovery_criterion='not_defined_for_execution_faults' if issues else None)


def _frozen_budget_receipt(result):
    """Read the identity-bound journal marker for a non-Planner budget stop."""
    episode = result.get('episode_execution') or {}
    if episode.get('arm') not in {'D', 'S', 'R'} or episode.get('status') != 'budget_exhausted':
        return False
    try:
        marker = json.loads(Path(episode['budget_stop_ref']).read_text())
        identity = {key: result[key] for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')}
        limit, observed = marker.get('limit'), marker.get('observed')
        numeric = type(limit) in (int, float) and type(observed) in (int, float)
        requested = marker.get('requested_steps', 0)
        exhausted = numeric and (observed >= limit or marker.get('reason') == 'environment_steps'
            and type(requested) is int and observed + requested > limit)
        runtime = result.get('runtime') or {}
        service = result.get('environment_service') or {}
        producer = marker.get('producer_events_ref')
        client_producer = bool(producer and producer == runtime.get('action_events_ref'))
        environment_producer = bool(producer and marker.get('reason') == 'environment_steps'
            and producer == runtime.get('environment_service', {}).get('action_events_ref')
            and service.get('environment_steps_reserved')
            == service.get('environment_steps_confirmed') == observed)
        return bool(all(marker.get('identity', {}).get(k) == v for k, v in identity.items()) and exhausted
            and marker.get('reason') in {'tool_calls', 'environment_steps', 'max_turns', 'model_requests'}
            and marker.get('stop_after_inflight') is True
            and (client_producer or environment_producer))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def closed_native_failure(result):
    """Verify the service evidence behind a normally closed negative attempt."""
    service = result.get('environment_service') or {}
    source = result.get('attempt_outcome_source') or {}
    return (result.get('attempt_closed') is True and not result.get('error')
            and result.get('validity') != 'technical_invalid'
            and (result.get('execution_health') or {}).get('status', 'clear') == 'clear'
            and service.get('service_closed') is True and service.get('pending_action') is False
            and service.get('environment_steps_unknown') == 0 and service.get('last_native_success') is False
            and source.get('native_success') and (source.get('runner_completion') or
                 result.get('verified_budget_exhaustion') is True and source.get('budget_completion')
                 and (source.get('planner_budget_stop') or _frozen_budget_receipt(result)))
            and source.get('last_feedback_sequence') == service.get('last_feedback_sequence')
            and all(service.get('identity', {}).get(key) == result.get(key)
                    for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id')))


def read_trajectory(states_path, transcript_path, *, bundle_id, run_id, episode_id,
                    independent_result_path=None, action_events_path=None, protocol_id=None):
    states_path, transcript_path = Path(states_path), Path(transcript_path)
    states = json.loads(states_path.read_text(encoding="utf-8"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    events = []
    expected_identity = {"run_id": run_id, "bundle_id": bundle_id, "episode_id": episode_id}
    if protocol_id is not None:
        expected_identity["protocol_id"] = protocol_id
    runtime_identity = (transcript.get("runtime") or {}).get("identity")
    if runtime_identity is not None and any(runtime_identity.get(key) != value
                                             for key, value in expected_identity.items()):
        raise ValueError("transcript runtime identity mismatch")
    arm = None
    context_path = states_path.parent / "runtime_context.json"
    if context_path.is_file():
        context = json.loads(context_path.read_text())
        if any(context.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("execution arm context identity mismatch")
        arm = context.get("execution_arm", {}).get("arm")
    selector = "frozen_program" if arm in {"D", "S", "R"} else "planner"
    retention = None
    if (states_path.parent / "artifact_retention.json").exists():
        from ..artifact_retention import policy_unavailable_media
        retention = policy_unavailable_media(states_path.parent)
    steps = states.get("steps", []) if isinstance(states, dict) else states
    for sequence, step in enumerate(steps):
        command = step.get("command")
        result = step.get("result", {})
        action = (command or {}).get("action", "initial_observation")
        media_paths = [states_path.parent / str(item) / (f"{step.get('step_idx', sequence):02d}" + Path(item).suffix)
                       for item in step.get("artifacts", [])]
        events.append(ExecutionEvent(
            event_id=f"{run_id}.{episode_id}.step-{sequence}", run_id=run_id,
            episode_id=episode_id, sequence=sequence, active_bundle=bundle_id,
            kind="tool_transition" if command else "observation", source="rpent.states",
            timestamps={"elapsed_s": step.get("elapsed_s"), "clock_domain": "tool_duration"},
            # High-level command is not a raw proposed/sent/applied action array.
            feedback={"state": step.get("state"), "tool_result": result,
                      "command": command, "source_step_idx": step.get("step_idx"),
                      "declared_artifacts": step.get("artifacts", [])},
            tool_status="failed" if result.get("error") else "completed" if "result" in step else "unknown",
            flags={"terminated": step.get("terminated"), "truncated": step.get("truncated"),
                   "action_source": action_origin(action),
                   "call_initiator": selector if command else "unknown",
                   "instruction_and_allowance_selector": selector if action == "vla_execute" else "unknown",
                   "task_selector": ("original_task" if arm == "D" else selector) if action == "vla_execute" else "unknown",
                   "execution_arm": arm,
                   **({"artifact_retention": {**retention, "paths": [str(path) for path in media_paths
                         if str(path) in retention["paths"]]}} if retention else {}),
                   "heuristic_pick_success": result.get("success") if action == "pi0_pick" else None},
            missing=["raw_action_proposal", "sent_action", "controller_applied_action",
                     "execution_mask", "capture_send_feedback_clocks", "independent_task_outcome"]
                    + (["source_step_sequence_gap"] if step.get("step_idx", sequence) != sequence else [])
                    + ["media_missing:" + str(path) for path in media_paths if not path.exists()],
            media=[str(path) for path in media_paths if path.exists()],
        ))
    if action_events_path is not None:
        raw_events = []
        with Path(action_events_path).open(encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                if raw.get("schema_version") != "1.0" or any(
                    raw.get(key) != value for key, value in expected_identity.items()
                ):
                    raise ValueError("action journal schema/run/version identity mismatch")
                if raw.get("sequence") != len(raw_events):
                    raise ValueError("action journal sequence gap or duplicate")
                raw_events.append(raw)
        for raw in raw_events:
            events.append(ExecutionEvent(
                event_id=f"{run_id}.{episode_id}.action-{raw['sequence']}", run_id=run_id,
                episode_id=episode_id, sequence=len(events), active_bundle=bundle_id,
                kind=raw["kind"], source="rpent.runtime_journal",
                timestamps={key: raw.get(key) for key in ("monotonic_ns", "utc_ns", "clock_domain")},
                parent_decision=raw.get("parent_decision_id"), proposal=raw.get("proposal"),
                sent=raw.get("sent"), applied=raw.get("applied"), feedback=raw.get("feedback"),
                execution_mask=raw.get("execution_mask"), missing=raw.get("missing", []),
                flags={"source": raw.get("source"), "source_sequence": raw["sequence"],
                       "proposal_sequence": raw.get("proposal_sequence"),
                       "send_sequence": raw.get("send_sequence"),
                       "protocol_id": raw.get("protocol_id"),
                       "error": raw.get("error"), "raw_event": raw},
            ))
    # Explicit independent environment boolean outranks Planner finish and termination.
    # Legacy environment_success has no source/closed-attempt guarantee.
    native = None
    source = None
    independent = None
    result_missing = []
    if independent_result_path is not None:
        independent = json.loads(Path(independent_result_path).read_text(encoding="utf-8"))
        if independent.get("schema_version") != "1.0" or independent.get("robot") != "libero":
            raise ValueError("unsupported independent result schema/backend")
        task = independent.get("task", {})
        for key in ("suite", "task", "seed"):
            if key in transcript and task.get(key) != transcript[key]:
                raise ValueError("independent result task identity mismatch")
        for key, expected in expected_identity.items():
            if key in independent and independent[key] != expected:
                raise ValueError("independent result run/version identity mismatch")
            if key not in independent:
                result_missing.append(key + "_not_recorded_by_runtime")
        native = independent.get("benchmark_success")
        native_sources = independent.get("native_result_source", {})
        if type(native) is bool and not any(native_sources.values()):
            native = None
            result_missing.append("independent_native_source")
        terminated, truncated = independent.get("terminated"), independent.get("truncated")
        expected_native = True if terminated is True else False if terminated is False and (truncated is True or closed_native_failure(independent)) else None
        if type(native) is bool and native != expected_native:
            native = None
            result_missing.append("inconsistent_independent_native_result")
        if steps and any(type(steps[-1].get(key)) is bool and type(independent.get(key)) is bool
                         and steps[-1][key] != independent[key] for key in ("terminated", "truncated")):
            native = None
            result_missing.append("conflicting_states_independent_result")
        source = str(independent_result_path) + ":benchmark_success" if type(native) is bool else None
    technical_invalid = bool(independent and (independent.get('validity') == 'technical_invalid'
        or (independent.get('execution_health') or {}).get('status', 'clear') != 'clear'))
    outcome = 'unknown' if technical_invalid else "success" if native is True else "failure" if native is False else "unknown"
    if technical_invalid:
        source = None
    events.append(ExecutionEvent(
        event_id=f"{run_id}.{episode_id}.result", run_id=run_id, episode_id=episode_id,
        sequence=len(events), active_bundle=bundle_id, kind="task_result", source="rpent.transcript",
        outcome=outcome, outcome_source=source,
        flags={"planner_finish": transcript.get("finish"), "technical_invalid": bool(transcript.get("error") or independent and (independent.get("validity") == "technical_invalid" or independent.get("error"))),
               "error": transcript.get("error"), "elapsed_s": transcript.get("elapsed_s"),
               "statistics": transcript.get("stats", {}),
               "legacy_environment_success": transcript.get("environment_success"),
               "native_terminated": steps[-1].get("terminated") if steps else None,
               "independent_result": independent,
               "costs": independent.get("costs") if independent else None,
               "import_binding": {"run_id": run_id, "bundle_id": bundle_id,
                                  "meaning": "matched runtime identity" if runtime_identity is not None
                                  else "caller supplied, not runtime-verified"}},
        missing=result_missing + ([] if type(native) is bool else ["independent_task_outcome"]),
    ))
    return events
