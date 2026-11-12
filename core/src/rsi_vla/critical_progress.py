"""Independent progression evidence for the concrete bounded VLA consumer."""

import json
from pathlib import Path


def inspect_progress(run_root, identity):
    """Verify used prerequisites; unknown paths or missing bindings stay unresolved."""
    root = Path(run_root)
    refs = [
        str(root / "action_events.jsonl"),
        str(root / "states.json"),
        str(root / "code_snapshot/RPent/robots/libero/tools.py"),
    ]
    report = {
        "value": None,
        "checks": {},
        "missing": [],
        "evidence_refs": refs,
        "consumer": "LiberoPrimitives.vla_execute -> _vlm_chunk",
        "not_applicable": {},
    }
    try:
        events = [
            json.loads(line)
            for line in (root / "action_events.jsonl").read_text().splitlines()
        ]
        states = json.loads((root / "states.json").read_text())
    except (OSError, ValueError):
        report["missing"].append(
            "action_events_or_states; retain these artifacts in future runs"
        )
        return report
    if not events or any(
        e.get("sequence") != i or any(e.get(k) != v for k, v in identity.items())
        for i, e in enumerate(events)
    ):
        report["missing"].append("complete_matching_event_sequence")
        return report
    calls = [e["tool"] for e in events if e["kind"] == "tool_call"]
    if (
        not calls
        or "vla_execute" not in calls
        or set(calls) - {"view_env_state", "vla_execute"}
    ):
        report["missing"].append(
            "independent prerequisite/coordinate/commit audit for actually used non-bounded tools"
        )
        return report
    commands = [
        step["command"] for step in states.get("steps", []) if step.get("command")
    ]
    if any(c.get("action") != "vla_execute" for c in commands):
        report["missing"].append("states/journal tool path disagreement")
        return report
    starts = [e for e in events if e["kind"] == "vla_segment_start"]
    if len(starts) != calls.count("vla_execute") or len(commands) != len(starts):
        report["missing"].append("all segment starts and high-level commands")
        return report
    # These subchecks are inapplicable for this actual call path, not universally true.
    report["not_applicable"] = {
        "historical_coordinates": "only view/vla_execute; no geometric coordinate argument consumer",
        "holding_or_settled": "vla_execute only consumes instruction, allowance and latest observation",
        "completion_submissions": "bounded path does not call TaskState.update or a completion-commit API",
    }
    report["checks"]["segment_arguments"] = all(
        isinstance(e.get("instruction"), str)
        and bool(e["instruction"].strip())
        and type(e.get("max_chunks")) is int
        and 1 <= e["max_chunks"] <= 24
        and all(c.get(k) == e.get(k) for k in ("instruction", "max_chunks"))
        for e, c in zip(starts, commands)
    )
    ends = [e["tool"] for e in events if e["kind"] == "tool_end"]
    report["checks"]["closed_call_journal"] = True if ends == calls else None
    if ends != calls:
        report["missing"].append("all admitted calls need matching terminal events")
    source_path = root / "code_snapshot/RPent/robots/libero/tools.py"
    report["checks"]["consumer_source_available"] = (
        True if source_path.is_file() else None
    )
    if not source_path.is_file():
        report["missing"].append(
            "frozen concrete consumer source required for independent audit"
        )
    bindings = []
    for e in events:
        if e["kind"] != "inference_start":
            continue
        seq = e.get("observation_sequence")
        updates = [
            u for u in events[: e["sequence"]] if u["kind"] == "observation_update"
        ]
        if seq is None or not updates:
            bindings.append(None)
            continue
        segment = next(
            (s for s in starts if s["sequence"] == e.get("segment_sequence")), None
        )
        bindings.append(
            seq == updates[-1]["sequence"]
            and e.get("observation_mode") == "latest"
            and e.get("source") == "vla"
            and segment is not None
            and e.get("instruction") == segment["instruction"]
        )
    report["checks"]["latest_observation_binding"] = (
        False
        if False in bindings
        else None
        if not bindings or None in bindings
        else True
    )
    if report["checks"]["latest_observation_binding"] is None:
        report["missing"].append(
            "per-inference observation_sequence/mode; future set_obs emits observation_update"
        )
    values = list(report["checks"].values())
    report["value"] = (
        False if False in values else True if all(v is True for v in values) else None
    )
    return report
