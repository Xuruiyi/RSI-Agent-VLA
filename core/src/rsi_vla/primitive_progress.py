"""Audit recorded high-resolution projection, literal-gripper move_to and release.

This reads saved arrays and replays float32 arithmetic against live feedback.
It makes no claim that a model's supplied target was derived from a projected
pixel or that the model identified the right object. Native task facts remain
the independent outcome authority.
"""

import json
from pathlib import Path


def _data(item):
    content = item["result"]["content"]
    texts = [part["text"] for part in content if part.get("type") == "text"]
    if len(texts) != 1:
        raise ValueError("one SDK text result required")
    value = json.loads(texts[0])
    if "content" in value:
        texts = [
            part["text"] for part in value["content"] if part.get("type") == "text"
        ]
        if len(texts) != 1:
            raise ValueError("one nested SDK text result required")
        value = json.loads(texts[0])
    return value


def audit_primitives(root, identity, events, states):
    """Check actual high projection, no-yaw move_to with gripper ±1, and release."""
    import numpy as np

    from .adapters.runtime_acceptance import audit_sdk_tools

    root = Path(root)
    report = {
        "value": None,
        "checks": {},
        "missing": [],
        "evidence_refs": [],
        "scope": "fresh saved projection and literal-target/live-EEF servo; no model-internal coordinate causality or target-identification claim",
    }
    checks = report["checks"]
    try:
        if not events or any(
            event.get("sequence") != index
            or any(event.get(key) != value for key, value in identity.items())
            for index, event in enumerate(events)
        ):
            raise ValueError("primitive event sequence/identity unavailable")
        for name in (
            "robots/libero/tools.py",
            "robots/libero/toolkit.py",
            "rpent/tools/toolkit.py",
            "rpent/session/base.py",
        ):
            source = root / "code_snapshot/RPent" / name
            report["evidence_refs"].append(str(source))
            reviewed = Path(__file__).resolve().parents[3] / "RPent" / name
            if source.read_bytes() != reviewed.read_bytes():
                raise ValueError(
                    "primitive consumer differs from reviewed source: " + name
                )
        checks["reviewed_readonly_projection_and_live_eef_consumer"] = True
        audit = audit_sdk_tools(root, identity, events)
        if audit["issues"]:
            raise ValueError("SDK/L0 association: " + "; ".join(audit["issues"]))
        paths = list(root.glob("codex_*.stream.jsonl"))
        if len(paths) != 1:
            raise ValueError("one SDK stream required")
        report["evidence_refs"].append(str(paths[0]))
        records = [json.loads(line) for line in paths[0].read_text().splitlines()]
        pairs = {}
        for event in records:
            item = (event.get("payload") or {}).get("item") or {}
            if item.get("type") == "mcpToolCall" and event.get("method") in {
                "item/started",
                "item/completed",
            }:
                pairs.setdefault(item["id"], {}).setdefault(event["method"], []).append(
                    item
                )
        steps = states["steps"]
        if [step["step_idx"] for step in steps] != list(range(len(steps))):
            raise ValueError("state trace step sequence unavailable")
        latest = 0
        primitive_sends = set()
        projected = moved = released = 0
        for call in sorted(audit["tools"], key=lambda item: item["l0_start_sequence"]):
            pair = pairs[call["id"]]
            if (
                len(pair.get("item/started", [])) != 1
                or len(pair.get("item/completed", [])) != 1
            ):
                raise ValueError("unique SDK arguments/result missing")
            start, end = pair["item/started"][0], pair["item/completed"][0]
            if start["arguments"] != end["arguments"]:
                raise ValueError("SDK tool arguments changed")
            args, returned = end["arguments"], _data(end)
            name = call["tool"]
            span = events[call["l0_start_sequence"] + 1 : call["l0_terminal_sequence"]]
            if name == "back_project_batch":
                projected += 1
                if (
                    args.get("step", -1) not in (-1, latest)
                    or returned.get("step") != latest
                ):
                    checks["projection_uses_latest_step"] = False
                    continue
                if args.get("resolution", "high") != "high" or args.get(
                    "camera", "agentview"
                ) not in {"agentview", "wrist"}:
                    raise ValueError(
                        "projection path needs a separate resolution/camera audit"
                    )
                checks["projection_is_readonly"] = checks.get(
                    "projection_is_readonly", True
                ) and not any(
                    event["kind"]
                    in {
                        "action_proposal",
                        "action_send",
                        "action_feedback",
                        "inference_start",
                        "observation_update",
                    }
                    for event in span
                )
                camera = args.get("camera", "agentview")
                artifact = camera + "_world_high.npz"
                if artifact not in steps[latest]["artifacts"]:
                    raise ValueError("projection artifact absent from selected state")
                path = root / artifact / f"{latest:02d}.npz"
                report["evidence_refs"].append(str(path))
                with np.load(path) as archive:
                    world = archive["array"]
                expected = [
                    {
                        "index": index,
                        "pixel": point,
                        "camera": camera,
                        "resolution": "high",
                        "world_xyz": [
                            round(float(value), 4)
                            for value in world[point[0], point[1], :3]
                        ],
                        "step": latest,
                        "image_size": list(world.shape[:2]),
                        "source_artifact": artifact,
                    }
                    for index, point in enumerate(args["points"])
                ]
                checks.setdefault("projection_uses_latest_step", True)
                checks["projection_matches_saved_pixels"] = (
                    checks.get("projection_matches_saved_pixels", True)
                    and returned["results"] == expected
                )
            elif name in {
                "vla_execute",
                "pi0_pick",
                "pi0_doubled",
                "move_to",
                "release",
            }:
                current = latest
                latest += 1
                row = steps[latest]
                command = {"action": name, **args}
                if (
                    returned.get("step") != latest
                    or returned.get("state") != row["state"]
                    or returned.get("log", {}).get("command") != command
                    or row.get("command") != command
                    or returned.get("log", {}).get("result") != row.get("result")
                ):
                    raise ValueError(
                        "SDK result/state trace command binding incomplete"
                    )
                if name in {"vla_execute", "pi0_pick", "pi0_doubled"}:
                    # Model-driven skills have their own control/argument audit.
                    continue
                if name == "move_to":
                    moved += 1
                    if (
                        set(args) - {"xyz", "gripper", "max_steps", "step_clip"}
                        or type(args.get("gripper", -1)) not in (int, float)
                        or args.get("gripper", -1) not in {-1, 1}
                    ):
                        raise ValueError(
                            "move_to path needs a separate orientation/gripper audit"
                        )
                    target = np.asarray(
                        [float(value) for value in args["xyz"]], dtype=np.float32
                    )
                else:
                    released += 1
                    if set(args) - {"max_steps"}:
                        raise ValueError("release arguments outside reviewed signature")
                limit = args.get("max_steps", 80 if name == "move_to" else 20)
                feedbacks = [
                    event
                    for event in events[: call["l0_start_sequence"]]
                    if event["kind"] == "action_feedback"
                ]
                if not feedbacks:
                    raise ValueError(
                        "live observation feedback preceding primitive missing"
                    )
                cur = np.asarray(
                    feedbacks[-1]["feedback"]["observed_state"][:3], dtype=np.float32
                )
                if (
                    cur.tolist()
                    != np.asarray(
                        steps[current]["state"]["robot0_eef_pos"], dtype=np.float32
                    ).tolist()
                ):
                    raise ValueError("state/live EEF binding disagrees")
                if name == "release":
                    gp = np.asarray(
                        feedbacks[-1]["feedback"]["observed_state"][6:8],
                        dtype=np.float32,
                    )
                    start_grip = peak_grip = float(abs(gp[0]) + abs(gp[1]))
                sends = [event for event in span if event["kind"] == "action_send"]
                proposals = {
                    event["sequence"]: event
                    for event in span
                    if event["kind"] == "action_proposal"
                }
                feedback = {
                    event["send_sequence"]: event
                    for event in span
                    if event["kind"] == "action_feedback"
                }
                if (
                    not sends
                    or len(sends) != len(proposals)
                    or {event["sequence"] for event in sends} != set(feedback)
                    or len(sends) > limit
                ):
                    raise ValueError(
                        "primitive proposal/send/feedback closure incomplete"
                    )
                for send in sends:
                    action = np.zeros(7, dtype=np.float32)
                    if name == "move_to":
                        action[:3] = (
                            np.clip(
                                target - cur,
                                -args.get("step_clip", 0.025),
                                args.get("step_clip", 0.025),
                            )
                            / 0.05
                        )
                        action[:3] = np.clip(action[:3], -1.0, 1.0)
                    action[6] = args.get("gripper", -1) if name == "move_to" else -1
                    proposal, end = (
                        proposals[send["proposal_sequence"]],
                        feedback[send["sequence"]],
                    )
                    key = name + "_exact_live_actions"
                    checks[key] = checks.get(key, True) and (
                        (
                            name != "move_to"
                            or float(np.linalg.norm(target - cur)) >= 0.012
                        )
                        and send.get("source")
                        == proposal.get("source")
                        == end.get("source")
                        == name
                        and proposal["sequence"] < send["sequence"] < end["sequence"]
                        and send["sent"] == proposal["proposal"] == action.tolist()
                        and end["execution_mask"] == [True]
                        and send.get("proposal_indices") == [0]
                        and end.get("proposal_sequence") == proposal["sequence"]
                    )
                    primitive_sends.add(send["sequence"])
                    observed = end["feedback"]["observed_state"]
                    cur = np.asarray(observed[:3], dtype=np.float32)
                    if name == "release":
                        gp = np.asarray(observed[6:8], dtype=np.float32)
                        grip = float(abs(gp[0]) + abs(gp[1]))
                        peak_grip = max(peak_grip, grip)
                terminal = feedback[sends[-1]["sequence"]]["feedback"]
                key = name + "_final_eef_binding"
                checks[key] = (
                    checks.get(key, True)
                    and cur.tolist()
                    == np.asarray(
                        row["state"]["robot0_eef_pos"], dtype=np.float32
                    ).tolist()
                )
                native_stop = (
                    terminal.get("terminated") is True
                    or terminal.get("truncated") is True
                )
                expected_steps = len(sends)
                if name == "move_to" and len(sends) < limit and not native_stop:
                    # The original loop appends its final converged observation
                    # before testing the existing 0.012 position tolerance.
                    checks["move_to_existing_convergence"] = (
                        checks.get("move_to_existing_convergence", True)
                        and float(np.linalg.norm(target - cur)) < 0.012
                    )
                    expected_steps += 1
                key = name + "_step_result_closure"
                checks[key] = (
                    checks.get(key, True)
                    and row["result"]["steps_used"] == expected_steps
                )
                if name == "release":
                    checks["release_actual_opening_observations"] = checks.get(
                        "release_actual_opening_observations", True
                    ) and all(
                        row["result"].get(key) == round(value, 4)
                        for key, value in (
                            ("start_gripper_opening", start_grip),
                            ("peak_gripper_opening", peak_grip),
                            ("final_gripper_opening", grip),
                        )
                    )
                    checks["release_loop_limit_or_native_stop"] = checks.get(
                        "release_loop_limit_or_native_stop", True
                    ) and (len(sends) == limit or native_stop)
        direct_sends = {
            event["sequence"]
            for event in events
            if event["kind"] == "action_send" and event.get("source") != "vla"
        }
        checks["all_primitive_sends_accounted"] = direct_sends == primitive_sends
        native_path = root / "environment_service/action_events.jsonl"
        native = [json.loads(line) for line in native_path.read_text().splitlines()]
        report["evidence_refs"].append(str(native_path))
        if any(
            event.get("sequence") != index
            or any(event.get(key) != value for key, value in identity.items())
            for index, event in enumerate(native)
        ):
            raise ValueError("native primitive event identity/sequence unavailable")
        native_ends = [
            event
            for event in native
            if event["kind"] == "rpc_dispatch_end"
            and event.get("rpc_evidence", {}).get("client_send_sequence")
            in primitive_sends
        ]
        native_starts = [
            event
            for event in native
            if event["kind"] == "rpc_dispatch_start"
            and event.get("rpc_evidence", {}).get("client_send_sequence")
            in primitive_sends
        ]
        starts_by_send = {
            event["rpc_evidence"]["client_send_sequence"]: event
            for event in native_starts
        }
        sends_by_sequence = {
            event["sequence"]: event
            for event in events
            if event["kind"] == "action_send"
        }
        checks["primitive_native_rpc_closure"] = (
            len(native_ends) == len(primitive_sends)
            and {event["rpc_evidence"]["client_send_sequence"] for event in native_ends}
            == primitive_sends
            and all(
                event.get("execution_mask") == [True]
                and event.get("unknown_proposal_indices") == []
                and not event.get("error")
                for event in native_ends
            )
        )
        checks["primitive_native_rpc_closure"] &= (
            len(native_starts) == len(primitive_sends)
            and set(starts_by_send) == primitive_sends
            and all(
                starts_by_send[event["rpc_evidence"]["client_send_sequence"]][
                    "sequence"
                ]
                < event["sequence"]
                and starts_by_send[event["rpc_evidence"]["client_send_sequence"]][
                    "rpc_evidence"
                ]
                == event["rpc_evidence"]
                and event["rpc_evidence"].get("proposal_sequence")
                == sends_by_sequence[event["rpc_evidence"]["client_send_sequence"]][
                    "proposal_sequence"
                ]
                and event["rpc_evidence"].get("proposal_indices") == [0]
                and event["rpc_evidence"].get("proposal_length") == 1
                for event in native_ends
            )
        )
        checks["complete_state_trace"] = latest == len(steps) - 1
        checks["actual_non_vla_paths_audited"] = (
            projected > 0 or moved > 0 or released > 0
        )
        report["not_applicable"] = {
            "holding_or_goal_commit": "reviewed move_to consumes a literal target, latest EEF and literal ±1 gripper command; release sends zero pose/open-gripper vectors and records opening observations. Neither consumes a holding/object-identity/goal-commit predicate; no physical holding or successful release is inferred",
            "projection_target_causality": "no equality/tolerance or model-internal derivation is inferred between projected points and the literal move_to target",
        }
        report["value"] = (
            False
            if False in checks.values()
            else True
            if checks and all(value is True for value in checks.values())
            else None
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report["missing"].append(str(exc))
    return report


def audit_direct_picks(root, identity, events, states):
    """Audit the observed direct pick loop, retaining the v8 result contract."""
    report = _audit_direct_model_skill(root, identity, events, states, tool="pi0_pick")
    report["pick_spans"] = report.pop("skill_spans")
    return report


def audit_direct_contacts(root, identity, events, states):
    """Audit pi0_doubled native-termination loops without a pick heuristic."""
    report = _audit_direct_model_skill(
        root, identity, events, states, tool="pi0_doubled"
    )
    report["contact_spans"] = report.pop("skill_spans")
    return report


def _audit_direct_model_skill(root, identity, events, states, *, tool):
    """Shared SDK/state/chunk/native closure for the two observed direct skills."""
    import math
    import numpy as np
    from .adapters.runtime_acceptance import audit_sdk_tools

    root = Path(root)
    picking = tool == "pi0_pick"
    prefix = "direct_pick" if picking else "direct_contact"
    report = {
        "value": None,
        "checks": {},
        "missing": [],
        "evidence_refs": [],
        "skill_spans": [],
        "not_applicable": {
            "route_and_B_feedback": f"direct Planner {tool}; no registered route or private object handoff evidence is inferred"
        },
    }
    checks = report["checks"]
    try:
        for name in (
            "robots/libero/tools.py",
            "robots/libero/toolkit.py",
            "rpent/tools/toolkit.py",
            "rpent/session/base.py",
            "robots/libero/env_client.py",
            "robots/libero/env_server.py",
            "robots/libero/action_spec.py",
        ):
            path = root / "code_snapshot/RPent" / name
            report["evidence_refs"].append(str(path))
            if (
                path.read_bytes()
                != (Path(__file__).resolve().parents[3] / "RPent" / name).read_bytes()
            ):
                raise ValueError(
                    "direct skill consumer differs from reviewed source: " + name
                )
        if any(
            event.get("sequence") != index
            or any(event.get(key) != value for key, value in identity.items())
            for index, event in enumerate(events)
        ):
            raise ValueError("direct skill event identity/sequence unavailable")
        audited = audit_sdk_tools(root, identity, events)
        if audited["issues"]:
            raise ValueError(
                "direct skill SDK/L0 association: " + "; ".join(audited["issues"])
            )
        streams = list(root.glob("codex_*.stream.jsonl"))
        if len(streams) != 1:
            raise ValueError("one SDK stream required for direct skill")
        report["evidence_refs"].append(str(streams[0]))
        pairs = {}
        for record in map(json.loads, streams[0].read_text().splitlines()):
            item = (record.get("payload") or {}).get("item") or {}
            if item.get("type") == "mcpToolCall" and record.get("method") in {
                "item/started",
                "item/completed",
            }:
                pairs.setdefault(item["id"], {}).setdefault(
                    record["method"], []
                ).append(item)
        steps = states["steps"]
        if [step["step_idx"] for step in steps] != list(range(len(steps))):
            raise ValueError("direct skill state sequence unavailable")
        count = 0
        for call in audited["tools"]:
            if call["tool"] != tool:
                continue
            low, high = call["l0_start_sequence"], call["l0_terminal_sequence"]
            span = events[low + 1 : high]
            if any(event["kind"] == "relation_route_decision" for event in span):
                continue  # Registered routes retain their separate audit.
            pair = pairs[call["id"]]
            if (
                len(pair.get("item/started", [])) != 1
                or len(pair.get("item/completed", [])) != 1
            ):
                raise ValueError("direct skill unique SDK start/end missing")
            start, end = pair["item/started"][0], pair["item/completed"][0]
            if (
                start["arguments"] != end["arguments"]
                or not call["sdk_completed"]
                or call.get("l0_status") != "completed"
            ):
                raise ValueError("direct skill SDK arguments or terminal unavailable")
            args, returned = end["arguments"], _data(end)
            allowed = {"prompt", "max_chunks"}
            if picking:
                allowed.update(
                    {
                        "lift_thresh",
                        "gripper_closed_thresh",
                        "gripper_open_thresh",
                        "descent_thresh",
                    }
                )
            allowance = args.get("max_chunks", 24 if picking else 20)
            if (
                set(args) - allowed
                or not isinstance(args.get("prompt"), str)
                or not args["prompt"].strip()
                or type(allowance) is not int
                or not 1 <= allowance <= 24
            ):
                raise ValueError("direct skill arguments need a separate audit")
            limits = {
                key: args.get(key, default)
                for key, default in {
                    "lift_thresh": 0.05,
                    "gripper_closed_thresh": 0.06,
                    "gripper_open_thresh": 0.0,
                    "descent_thresh": 0.1,
                }.items()
            }
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in limits.values()
            ):
                raise ValueError("direct skill heuristic arguments unavailable")
            step = returned.get("step")
            if type(step) is not int or not 0 < step < len(steps):
                raise ValueError("direct skill state result index unavailable")
            before, row = steps[step - 1], steps[step]
            command = {"action": tool, **args}
            result = row.get("result", {})
            if (
                row.get("command") != command
                or result.get("relation_route") is not None
                or returned.get("state") != row["state"]
                or returned.get("log", {}).get("command") != command
                or returned.get("log", {}).get("result") != result
            ):
                raise ValueError("direct skill SDK/state command binding incomplete")
            initial = None
            if picking:
                feedback_before = [
                    event
                    for event in events[:low]
                    if event["kind"] == "action_feedback"
                ]
                if not feedback_before:
                    raise ValueError("direct skill preceding live observation missing")
                initial = np.asarray(
                    feedback_before[-1]["feedback"]["observed_state"], dtype=np.float32
                )
                if (
                    initial[:3].tolist()
                    != np.asarray(
                        before["state"]["robot0_eef_pos"], dtype=np.float32
                    ).tolist()
                    or initial[6:8].tolist()
                    != np.asarray(
                        before["state"]["robot0_gripper_qpos"], dtype=np.float32
                    ).tolist()
                ):
                    raise ValueError(
                        "direct skill initial EEF/gripper binding disagrees"
                    )
            for key in ("terminated", "truncated"):
                if (
                    type(before.get(key)) is not bool
                    or type(row.get(key)) is not bool
                    or type(returned.get(key)) is not bool
                    or returned[key] != row[key]
                ):
                    raise ValueError("direct skill state native flags unavailable")
            native_terminated, native_truncated = (
                before["terminated"],
                before["truncated"],
            )
            checks[prefix + "_initial_env_active"] = checks.get(
                prefix + "_initial_env_active", True
            ) and not (native_terminated or native_truncated)
            if picking:
                initial_z = peak_z = min_z = post_min_peak = float(initial[2])
                min_grip = final_grip = float(abs(initial[6]) + abs(initial[7]))
            descent = success = stopped = False
            inferences = [event for event in span if event["kind"] == "inference_start"]
            if not inferences or type(result.get("chunks_used")) is not int:
                raise ValueError("direct skill inference count unavailable")
            checks[prefix + "_count_and_arguments"] = checks.get(
                prefix + "_count_and_arguments", True
            ) and (
                len(inferences) == result["chunks_used"] <= allowance
                and all(
                    event.get("source") == "vla"
                    and event.get("segment_sequence") is None
                    and event.get("instruction") == args["prompt"]
                    for event in inferences
                )
            )
            observed = initial
            terminal = None
            for inference in inferences:
                ends = [
                    event
                    for event in span
                    if event["kind"] == "inference_end"
                    and event.get("inference_sequence") == inference["sequence"]
                ]
                chunks = [
                    event
                    for event in span
                    if event["kind"] == "chunk_execution_end"
                    and event.get("inference_sequence") == inference["sequence"]
                ]
                proposals = [
                    event
                    for event in span
                    if event["kind"] == "action_proposal"
                    and event.get("inference_sequence") == inference["sequence"]
                ]
                if (
                    len(ends) != 1
                    or ends[0].get("completed") is not True
                    or len(chunks) != 1
                    or len(proposals) != 1
                    or chunks[0].get("predictions_started") != 1
                    or chunks[0].get("predictions_completed") != 1
                    or chunks[0].get("unknown_steps") != 0
                    or type(chunks[0].get("executed_steps")) is not int
                    or chunks[0]["executed_steps"] <= 0
                ):
                    raise ValueError("direct skill inference/chunk closure incomplete")
                sends = [
                    event
                    for event in span
                    if event["kind"] == "action_send"
                    and event.get("proposal_sequence") == proposals[0]["sequence"]
                ]
                feedback = [
                    event
                    for event in span
                    if event["kind"] == "action_feedback"
                    and sends
                    and event.get("send_sequence") == sends[0]["sequence"]
                ]
                if (
                    len(sends) != 1
                    or len(feedback) != 1
                    or not inference["sequence"]
                    < ends[0]["sequence"]
                    < proposals[0]["sequence"]
                    < sends[0]["sequence"]
                    < feedback[0]["sequence"]
                    < chunks[0]["sequence"]
                ):
                    raise ValueError("direct skill action closure incomplete")
                if stopped:
                    checks[prefix + "_existing_stop_rule"] = False
                terminal = feedback[0]["feedback"]
                mask = feedback[0].get("execution_mask")
                size = chunks[0]["executed_steps"]
                if (
                    not isinstance(mask, list)
                    or len(mask) != size
                    or not mask
                    or any(value is not True for value in mask)
                    or len(sends[0]["sent"]) != size
                ):
                    raise ValueError(
                        "direct skill executed chunk/mask length unavailable"
                    )
                flags = {}
                for key in ("terminated", "truncated"):
                    values = terminal.get(key)
                    if (
                        not isinstance(values, list)
                        or len(values) != size
                        or any(type(value) is not bool for value in values)
                    ):
                        raise ValueError(
                            "direct skill native flag list missing, invalid or wrong length"
                        )
                    flags[key] = any(values)
                # LiberoEnvClient.check_done ORs any flag in each executed chunk
                # into the existing scalar state; it does not use list truthiness.
                native_terminated = native_terminated or flags["terminated"]
                native_truncated = native_truncated or flags["truncated"]
                if not picking:
                    stopped = native_terminated or native_truncated
                    success = native_terminated if stopped else False
                    continue
                observed = np.asarray(terminal["observed_state"], dtype=np.float32)
                z = float(observed[2])
                final_grip = float(abs(observed[6]) + abs(observed[7]))
                peak_z = max(peak_z, z)
                if z < min_z:
                    min_z = post_min_peak = z
                else:
                    post_min_peak = max(post_min_peak, z)
                descent = descent or initial_z - min_z >= limits["descent_thresh"]
                min_grip = min(min_grip, final_grip)
                if (
                    descent
                    and post_min_peak - min_z >= limits["lift_thresh"]
                    and limits["gripper_open_thresh"]
                    <= final_grip
                    < limits["gripper_closed_thresh"]
                ):
                    success = stopped = True
                elif native_terminated or native_truncated:
                    success, stopped = native_terminated, True
            checks[prefix + "_existing_stop_rule"] = checks.get(
                prefix + "_existing_stop_rule", True
            ) and (stopped or len(inferences) == allowance)
            if picking:
                if (
                    observed[:3].tolist()
                    != np.asarray(
                        row["state"]["robot0_eef_pos"], dtype=np.float32
                    ).tolist()
                    or observed[6:8].tolist()
                    != np.asarray(
                        row["state"]["robot0_gripper_qpos"], dtype=np.float32
                    ).tolist()
                ):
                    raise ValueError("direct skill final EEF/gripper binding disagrees")
                diagnostics = {
                    "start_eef_z": round(initial_z, 4),
                    "peak_eef_z": round(peak_z, 4),
                    "min_eef_z": round(min_z, 4),
                    "post_min_peak_z": round(post_min_peak, 4),
                    "descent_m": round(initial_z - min_z, 4),
                    "post_min_ascent_m": round(post_min_peak - min_z, 4),
                    "descent_done": descent,
                    **limits,
                }
                expected = dict(
                    name="pick",
                    instruction=args["prompt"],
                    success=success,
                    chunks_used=len(inferences),
                    max_chunks=allowance,
                    peak_lift_m=post_min_peak - min_z,
                    min_gripper_opening=min_grip,
                    final_gripper_opening=final_grip,
                    terminated=native_terminated,
                    truncated=native_truncated,
                    diagnostics=diagnostics,
                )
            else:
                expected = dict(
                    name="pi0_doubled",
                    instruction=args["prompt"],
                    success=success,
                    task_success=success,
                    contact_skill_executed=len(inferences) > 0,
                    chunks_used=len(inferences),
                    max_chunks=allowance,
                    terminated=native_terminated,
                    truncated=native_truncated,
                    diagnostics={
                        "mode": "contact_skill_success_by_termination",
                        "success_meaning": "`success` mirrors official LIBERO task termination only; for intermediate contact skills, inspect image/state evidence.",
                    },
                )
                report["not_applicable"]["contact_effect_or_holding"] = (
                    "contact_skill_executed records execution only; native failure/success is not an independently measured contact effect or holding claim"
                )
            checks[prefix + "_scalar_native_state"] = checks.get(
                prefix + "_scalar_native_state", True
            ) and (
                row["terminated"] == native_terminated
                and row["truncated"] == native_truncated
            )
            checks[
                (
                    "direct_pick_recorded_heuristic_result"
                    if picking
                    else "direct_contact_recorded_result"
                )
            ] = checks.get(
                (
                    "direct_pick_recorded_heuristic_result"
                    if picking
                    else "direct_contact_recorded_result"
                ),
                True,
            ) and all(result.get(key) == value for key, value in expected.items())
            report["skill_spans"].append(
                dict(start=low, end=high, instruction=args["prompt"], step_idx=step)
            )
            count += 1
        checks[prefix + "_actual_calls"] = count > 0
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        report["missing"].append(str(exc))
    report["value"] = (
        False
        if False in checks.values()
        else True
        if checks
        and not report["missing"]
        and all(value is True for value in checks.values())
        else None
    )
    return report
