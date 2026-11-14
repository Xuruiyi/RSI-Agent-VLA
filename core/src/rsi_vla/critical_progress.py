"""Independent progression evidence for the concrete bounded VLA consumer."""

import ast
import json
from pathlib import Path


AUDIT_VERSION = "critical-progress-v10-routed-drawer-contact"


def inspect_progress(run_root, identity, *, read_source_ref=None):
    """Verify used prerequisites; unknown paths or missing bindings stay unresolved."""
    root = Path(run_root)
    refs = [
        str(root / "action_events.jsonl"),
        str(root / "states.json"),
        str(root / "code_snapshot/RPent/robots/libero/tools.py"),
    ]
    report = {
        "value": None,
        "audit_version": AUDIT_VERSION,
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
    call_events = [e for e in events if e["kind"] == "tool_call"]
    end_events = [e for e in events if e["kind"] == "tool_end"]
    calls = [e["tool"] for e in call_events]
    pick_spans = []
    direct = None
    direct_calls = []
    for event in call_events:
        if event["tool"] != "pi0_pick":
            continue
        terminal = next(
            (
                end
                for end in end_events
                if end["sequence"] > event["sequence"] and end["tool"] == "pi0_pick"
            ),
            None,
        )
        span = events[event["sequence"] + 1 : terminal["sequence"]] if terminal else []
        if not any(item["kind"] == "relation_route_decision" for item in span):
            direct_calls.append(event)
    if direct_calls:
        from .primitive_progress import audit_direct_picks

        direct = audit_direct_picks(root, identity, events, states)
        report["direct_pick_progress"] = {
            key: value for key, value in direct.items() if key != "pick_spans"
        }
        report["checks"]["actual_direct_pick_audit"] = direct["value"]
        report["missing"].extend(direct["missing"])
        report["evidence_refs"].extend(direct["evidence_refs"])
        if direct["value"] is not True:
            report["value"] = direct["value"]
            return report
        pick_spans.extend(direct["pick_spans"])
    contact = None
    direct_contacts = []
    for call in call_events:
        if call["tool"] != "pi0_doubled":
            continue
        terminal = next((end for end in end_events
                         if end["sequence"] > call["sequence"] and end["tool"] == "pi0_doubled"), None)
        span = events[call["sequence"] + 1:terminal["sequence"]] if terminal else []
        if not any(event["kind"] == "relation_route_decision" for event in span):
            direct_contacts.append(call)
    if direct_contacts:
        from .primitive_progress import audit_direct_contacts

        contact = audit_direct_contacts(root, identity, events, states)
        report["direct_contact_progress"] = {
            key: value for key, value in contact.items() if key != "contact_spans"
        }
        report["checks"]["actual_direct_contact_audit"] = contact["value"]
        report["missing"].extend(contact["missing"])
        report["evidence_refs"].extend(contact["evidence_refs"])
        if contact["value"] is not True:
            report["value"] = contact["value"]
            return report
        pick_spans.extend(contact["contact_spans"])
    routed = None
    if any(event["kind"] == "relation_route_decision" for event in events):
        routed = audit_skill_routes(
            root, identity, events, states, direct_pick_spans=pick_spans
        )
        report["skill_routes"] = {
            key: value
            for key, value in routed.items()
            if key not in {"vla_commands", "pick_spans"}
        }
        report["checks"]["actual_skill_route_audit"] = routed["value"]
        report["missing"].extend(routed["missing"])
        report["evidence_refs"].extend(routed["evidence_refs"])
        if routed["value"] is not True:
            return report
        pick_spans.extend(routed["pick_spans"])
    primitive_tools = {"back_project_batch", "move_to", "release"}
    mixed = bool(set(calls) & primitive_tools)
    supported = {"view_env_state", "vla_execute", "read_text_file", "finish"}
    if routed is not None or direct is not None:
        supported.add("pi0_pick")
    if contact is not None or routed is not None:
        supported.add("pi0_doubled")
    unsupported = set(calls) - supported - primitive_tools
    if mixed:
        from .primitive_progress import audit_primitives

        primitive = audit_primitives(root, identity, events, states)
        report["primitive_progress"] = primitive
        report["checks"]["actual_primitive_audit"] = primitive["value"]
        report["missing"].extend(primitive["missing"])
        report["evidence_refs"].extend(primitive["evidence_refs"])
    if (
        not calls
        or (
            "vla_execute" not in calls
            and routed is None
            and direct is None
            and contact is None
        )
        or unsupported
    ):
        report["checks"]["supported_actual_tools"] = None
        report["missing"].append(
            "independent prerequisite/coordinate/commit audit for actually used non-bounded tools"
        )
    if "finish" in calls:
        finishing = audit_finish(root, identity, events)
        report["finish"] = finishing
        report["checks"]["actual_finish_audit"] = finishing["value"]
        report["missing"].extend(finishing["missing"])
        report["evidence_refs"].extend(finishing["evidence_refs"])
    if "read_text_file" in calls:
        from .read_text_audit import audit_read

        reading = audit_read(
            root, identity, call_events, end_events, source_ref=read_source_ref
        )
        report["read_text_file"] = reading
        report["checks"]["actual_read_audit"] = reading["value"]
        report["missing"].extend(reading["missing"])
        report["evidence_refs"].extend(reading["evidence_refs"])
    commands = [
        step["command"] for step in states.get("steps", []) if step.get("command")
    ]
    allowed_commands = {"vla_execute", "move_to", "release"}
    if routed is not None or direct is not None:
        allowed_commands.add("pi0_pick")
    if contact is not None or routed is not None:
        allowed_commands.add("pi0_doubled")
    if any(c.get("action") not in allowed_commands for c in commands):
        report["missing"].append("states/journal tool path disagreement")
        return report
    commands = (
        routed["vla_commands"]
        if routed is not None
        else [command for command in commands if command["action"] == "vla_execute"]
    )
    starts = [e for e in events if e["kind"] == "vla_segment_start"]
    expected_segments = (
        len(commands) if routed is not None else calls.count("vla_execute")
    )
    if len(starts) != expected_segments or len(commands) != len(starts):
        report["missing"].append("all segment starts and high-level commands")
        return report
    # These subchecks are inapplicable for this actual call path, not universally true.
    report["not_applicable"] = {
        "historical_coordinates": "view/vla_execute plus audited text read; no geometric coordinate argument consumer",
        "holding_or_settled": "vla_execute only consumes instruction, allowance and latest observation",
        "completion_submissions": "audited finish returns only a Planner termination claim; the native worker owns the benchmark outcome; no TaskState.update or geometric completion-commit API"
        if "finish" in calls
        else "bounded path does not call TaskState.update or a completion-commit API",
    }
    if pick_spans:
        report["consumer"] = (
            "audited LiberoPrimitives.pi0_pick/pi0_doubled/vla_execute -> _vlm_chunk"
        )
        report["not_applicable"].pop("holding_or_settled", None)
        if routed is not None and routed["pick_spans"]:
            report["skill_handoff_scope"] = (
                "live EEF/gripper pick heuristics are not an independent holding claim; private object handoff geometry is evaluated separately"
            )
        if direct is not None:
            report["direct_pick_scope"] = (
                "saved EEF/gripper heuristic and closed control only; route, rhoB feedback and object holding are not applicable to the direct call"
            )
    if contact is not None:
        report["direct_contact_scope"] = (
            "closed native-termination contact loop only; no EEF/gripper pick heuristic, holding, route or rhoB feedback is inferred"
        )
    if mixed:
        report["not_applicable"] = primitive.get("not_applicable", {})
    if unsupported:
        report["not_applicable"] = {}
    report["checks"]["segment_arguments"] = all(
        isinstance(e.get("instruction"), str)
        and bool(e["instruction"].strip())
        and type(e.get("max_chunks")) is int
        and 1 <= e["max_chunks"] <= 24
        and all(c.get(k) == e.get(k) for k in ("instruction", "max_chunks"))
        for e, c in zip(starts, commands)
    )
    ends = [e["tool"] for e in events if e["kind"] == "tool_end"]
    report["checks"]["closed_call_journal"] = (
        True
        if ends == calls
        and all(e.get("status", "completed") == "completed" for e in end_events)
        else None
    )
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
        pick = next(
            (
                span
                for span in pick_spans
                if span["start"] < e["sequence"] < span["end"]
            ),
            None,
        )
        bindings.append(
            seq == updates[-1]["sequence"]
            and e.get("observation_mode") == "latest"
            and e.get("source") == "vla"
            and (
                (segment is not None and e.get("instruction") == segment["instruction"])
                or (
                    pick is not None
                    and e.get("segment_sequence") is None
                    and e.get("instruction") == pick["instruction"]
                )
            )
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
    if (root / "runtime_context.json").exists():
        _control_checks(root, identity, events, starts, report, pick_spans=pick_spans)
    values = list(report["checks"].values())
    report["value"] = (
        False
        if False in values
        else True
        if not report["missing"] and all(v is True for v in values)
        else None
    )
    return report


def audit_finish(root, identity, events):
    """Audit the saved CommonTools.finish path without trusting its success claim."""
    from .adapters.runtime_acceptance import audit_sdk_tools

    root = Path(root)
    report = {"value": None, "checks": {}, "missing": [], "evidence_refs": []}
    names = (
        "rpent/tools/common.py",
        "rpent/tools/toolkit.py",
        "robots/libero/toolkit.py",
        "robots/libero/robot_spec.py",
        "rpent/planner/codex.py",
        "robots/libero/result.py",
    )
    try:
        snapshot = root / "code_snapshot/RPent"
        reviewed = Path(__file__).resolve().parents[3] / "RPent"
        for name in names:
            path = snapshot / name
            report["evidence_refs"].append(str(path))
            if path.read_bytes() != (reviewed / name).read_bytes():
                raise OSError("finish consumer differs from reviewed source: " + name)
        tree = ast.parse((snapshot / "rpent/tools/common.py").read_text())
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CommonTools"
        )
        method = next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "finish"
        )
        body = [
            node
            for node in method.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        expected = ast.parse(
            'return ToolResult(data={"_finish": True, "status": status, "summary": summary})'
        ).body
        if [ast.dump(node) for node in body] != [ast.dump(node) for node in expected]:
            raise OSError("finish is not the reviewed non-actuating return")
        context = json.loads((root / "runtime_context.json").read_text())
        raw = json.loads((root / "result.json").read_text())
        service = json.loads(
            (root / "environment_service/runtime_costs.json").read_text()
        )
        if any(
            any(
                doc.get("identity", doc).get(k) != value
                for k, value in identity.items()
            )
            for doc in (context, raw, service)
        ):
            raise OSError("finish owner identity mismatch")
        rules = next(
            event["effective_rules"] for event in events if event["kind"] == "run_start"
        )
        launch = json.loads((root / "launch_plan.json").read_text())
        command = launch["command"]
        guarded = rules.get("finish_requires_native_success")
        if (
            type(guarded) is not bool
            or command[command.index("--am-backend") + 1] != "pi05"
        ):
            raise OSError("finish override path needs its own audit")
        registry = (snapshot / "rpent/tools/toolkit.py").read_text()
        toolkit = (snapshot / "robots/libero/toolkit.py").read_text()
        report["checks"]["plain_common_dispatch"] = all(
            text in registry
            for text in (
                "common = CommonTools(memory=self._memory)",
                "for declaration in iter_tools(common):",
            )
        ) and all(
            text in toolkit
            for text in (
                'policy_kind: str = "vla"',
                'if policy_kind == "wam" or (',
                'getattr(self.runtime_journal, "finish_requires_native_success", False)',
            )
        )
        if guarded:
            owner = next(
                node
                for node in ast.parse(toolkit).body
                if isinstance(node, ast.ClassDef) and node.name == "LiberoToolkit"
            )
            method = next(
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == "_wam_finish"
            )
            body = [
                node
                for node in method.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            expected = ast.parse("""
if kwargs.get("status") == "success" and getattr(self._primitives.env, "terminated", None) is not True:
    return ToolResult(data={
        "error": "finish refused: success requires native terminated=true",
        "terminated": self._primitives.env.terminated,
        "truncated": self._primitives.env.truncated,
        "reason": "Inspect the current state. Continue if the episode is active; report failure if it was truncated.",
    })
return inner(**kwargs)
""").body
            if [ast.dump(node) for node in body] != [
                ast.dump(node) for node in expected
            ]:
                raise OSError(
                    "finish guard differs from the reviewed native predicate and return"
                )
            report["checks"]["guarded_common_dispatch"] = True
        tool_audit = audit_sdk_tools(root, identity, events)
        if tool_audit["issues"]:
            raise OSError(
                "finish SDK/L0 association incomplete: "
                + "; ".join(tool_audit["issues"])
            )
        calls = [item for item in tool_audit["tools"] if item.get("tool") == "finish"]
        paths = list(root.glob("codex_*.stream.jsonl"))
        if len(calls) != 1 or len(paths) != 1 or not calls[0]["sdk_completed"]:
            raise OSError("one complete finish call required")
        report["evidence_refs"].append(str(paths[0]))
        sdk = [json.loads(line) for line in paths[0].read_text().splitlines()]
        pair = [
            (event["method"], (event.get("payload") or {}).get("item", {}))
            for event in sdk
            if (event.get("payload") or {}).get("item", {}).get("id") == calls[0]["id"]
        ]
        starts = [item for method, item in pair if method == "item/started"]
        ends = [item for method, item in pair if method == "item/completed"]
        if (
            len(starts) != 1
            or len(ends) != 1
            or starts[0].get("arguments") != ends[0].get("arguments")
        ):
            raise OSError("finish arguments or terminal missing")
        args = ends[0]["arguments"]
        if set(args) != {"status", "summary"} or any(
            not isinstance(value, str) for value in args.values()
        ):
            raise OSError("finish arguments outside the reviewed signature")
        content = ends[0]["result"]["content"]
        if len(content) != 1 or content[0].get("type") != "text":
            raise OSError("finish return unavailable")
        returned = json.loads(content[0]["text"])
        claim = {"_finish": True, **args}
        report["checks"]["returned_claim_and_finalizer_match"] = all(
            returned.get(k) == value
            and (raw.get("planner_finish") or {}).get(k) == value
            for k, value in claim.items()
        )
        last = calls[0]["l0_terminal_sequence"]
        report["checks"]["no_actions_after_finish"] = not any(
            event["sequence"] > last
            and event["kind"] in {"tool_call", "action_send", "inference_start"}
            for event in events
        )
        if (
            service.get("service_closed") is not True
            or service.get("pending_action") is not False
            or service.get("environment_steps_unknown") != 0
            or type(service.get("last_native_success")) is not bool
        ):
            raise OSError("native outcome owner not closed")
        report["checks"]["native_outcome_independent_of_claim"] = (
            raw.get("benchmark_success") is service["last_native_success"]
            and raw.get("attempt_closed") is True
            and bool((raw.get("attempt_outcome_source") or {}).get("native_success"))
        )
        if guarded:
            report["checks"]["guarded_success_matches_native"] = (
                args["status"] != "success" or service["last_native_success"] is True
            )
        report["checks"]["source_pure_return"] = True
        report["value"] = all(value is True for value in report["checks"].values())
        report["non_applicability"] = (
            "finish returns a termination claim, makes no geometry/action/state or completion-commit call; outcome comes from the closed native worker"
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError) as exc:
        report["missing"].append(str(exc))
    return report


def _control_checks(root, identity, events, starts, report, *, pick_spans=()):
    """Bind the observed native dispatch, not an unmeasured controller trajectory."""
    from .adapters.runtime_acceptance import inspect_attempt

    checks = report["checks"]
    try:
        audit = inspect_attempt(root)
        for name in (
            "ordered_l0",
            "execution_health",
            "action_mapping",
            "setup_coverage",
            "no_mid_attempt_reset",
            "vla_actual_loading_and_prediction",
        ):
            checks["control_" + name] = audit["checks"].get(name)
        proposals = {
            e["sequence"]: e
            for e in events
            if e["kind"] == "action_proposal" and e.get("source") == "vla"
        }
        inferences = {
            e["sequence"]: e for e in events if e["kind"] == "inference_start"
        }
        sends = [
            e for e in events if e["kind"] == "action_send" and e.get("source") == "vla"
        ]
        checks["proposal_to_native_send"] = (
            bool(sends)
            and len(sends) == len(proposals) == len(inferences)
            and all(
                e.get("source") == "vla"
                and e["sent"] == proposals[e["proposal_sequence"]]["proposal"][:5]
                and e["proposal_indices"] == list(range(len(e["sent"])))
                and 0 < len(e["sent"]) <= 5
                and proposals[e["proposal_sequence"]]["inference_sequence"]
                in inferences
                and proposals[e["proposal_sequence"]]["instruction"]
                == inferences[proposals[e["proposal_sequence"]]["inference_sequence"]][
                    "instruction"
                ]
                and proposals[e["proposal_sequence"]]["segment_sequence"]
                == inferences[proposals[e["proposal_sequence"]]["inference_sequence"]][
                    "segment_sequence"
                ]
                and proposals[e["proposal_sequence"]]["inference_sequence"]
                < e["proposal_sequence"]
                < e["sequence"]
                and (
                    proposals[e["proposal_sequence"]]["segment_sequence"]
                    in {s["sequence"] for s in starts}
                    or (
                        proposals[e["proposal_sequence"]]["segment_sequence"] is None
                        and any(
                            span["start"]
                            < proposals[e["proposal_sequence"]]["inference_sequence"]
                            < e["sequence"]
                            < span["end"]
                            for span in pick_spans
                        )
                    )
                )
                for e in sends
            )
        )
        ends = [e for e in events if e["kind"] == "vla_segment_end"]
        checks["segment_execution_closure"] = len(ends) == len(starts) and all(
            e["segment_sequence"] == start["sequence"]
            and all(e.get(k) == start.get(k) for k in ("instruction", "max_chunks"))
            and e["predictions_started"]
            == e["predictions_completed"]
            == e["chunks_used"]
            == sum(
                i.get("segment_sequence") == start["sequence"]
                for i in inferences.values()
            )
            and 0 < e["chunks_used"] <= e["max_chunks"]
            and e["unknown_steps"] == 0
            for start, e in zip(starts, ends)
        )
        source = (root / "code_snapshot/RPent/robots/libero/tools.py").read_text()
        checks["native_control_consumer_source"] = all(
            s in source
            for s in (
                "actions = actions[: journal.chunk_steps]",
                "self.env.rsi_chunk_step(",
                '"proposal_indices": list(range(len(actions)))',
            )
        ) and all(
            (root / ("code_snapshot/RPent/robots/libero/" + name)).is_file()
            for name in ("env_server.py", "env_client.py", "action_spec.py")
        )
        report["control_evidence_scope"] = (
            "exact proposals -> first five native OSC_POSE slots -> service confirmed execution masks; controller-applied telemetry and physics ticks remain unmeasured"
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        checks["native_control_chain"] = None
        report["missing"].append("native control chain: " + str(exc))


def audit_skill_routes(root, identity, events, states, *, direct_pick_spans=()):
    """Bind the actual same-goal consumer without rewriting Planner requests."""
    root = Path(root)
    report = {
        "value": None,
        "checks": {},
        "missing": [],
        "evidence_refs": [],
        "vla_commands": [],
        "pick_spans": [],
    }
    try:
        for name in (
            "tools.py",
            "toolkit.py",
            "skill_routing.py",
            "env_client.py",
            "env_server.py",
        ):
            snapshot = root / "code_snapshot/RPent/robots/libero" / name
            reviewed = (
                Path(__file__).resolve().parents[3] / "RPent/robots/libero" / name
            )
            report["evidence_refs"].append(str(snapshot))
            if snapshot.read_bytes() != reviewed.read_bytes():
                raise ValueError(
                    "routed skill consumer differs from reviewed source: " + name
                )
        context = json.loads((root / "runtime_context.json").read_text())
        control = context["relation_control"]
        if any(context.get(key) != value for key, value in identity.items()):
            raise ValueError("route context identity mismatch")
        if control["version_signature"] != {
            key: context[key]
            for key in ("theta_ref", "processor_ref", "phi_ref", "code_ref")
        }:
            raise ValueError("route context executable version mismatch")
        routes = {row["route_id"]: row for row in control["routes"]}
        if any(row["criterion"].get("schema") == "rsi-skill-drawer-open-criterion-v1" for row in routes.values()):
            name = "drawer_observation.py"
            snapshot = root / "code_snapshot/RPent/robots/libero" / name
            reviewed = Path(__file__).resolve().parents[3] / "RPent/robots/libero" / name
            report["evidence_refs"].append(str(snapshot))
            if snapshot.read_bytes() != reviewed.read_bytes():
                raise ValueError("drawer worker observation differs from reviewed source")
        loaded = [
            event
            for event in events
            if event["kind"] == "relation_policy_loaded"
            and event.get("relation") == "rhoB"
        ]
        if len(loaded) != 1 or loaded[0].get("rho_ref") != control["rho_ref"]:
            raise ValueError("bound rhoB was not loaded by the actual Toolkit")
        decisions = [
            event for event in events if event["kind"] == "relation_route_decision"
        ]
        steps = [
            step
            for step in states["steps"]
            if (step.get("command") or {}).get("action") in {"pi0_pick", "vla_execute", "pi0_doubled"}
        ]
        seen = set()
        private_path = root / "environment_service/action_events.jsonl"
        report["evidence_refs"].append(str(private_path))
        private = [json.loads(line) for line in private_path.read_text().splitlines()]
        if any(
            event.get("sequence") != i
            or any(event.get(key) != value for key, value in identity.items())
            for i, event in enumerate(private)
        ):
            raise ValueError(
                "private skill observation stream identity or sequence mismatch"
            )
        for step in steps:
            command, result = step["command"], step.get("result", {})
            binding = result.get("relation_route")
            if binding is None:
                if command["action"] in {"pi0_pick", "pi0_doubled"} and any(
                    span["step_idx"] == step.get("step_idx")
                    for span in direct_pick_spans
                ):
                    continue
                if command["action"] != "vla_execute":
                    raise ValueError(
                        "unregistered non-bounded skill call needs its own audit"
                    )
                report["vla_commands"].append(command)
                continue
            decision = events[binding["decision_sequence"]]
            if decision not in decisions or decision["sequence"] in seen:
                raise ValueError("actual skill result has no unique route decision")
            seen.add(decision["sequence"])
            route = routes[decision["route_id"]]
            from .cooperation import skill_route_workers, DRAWER_CRITERION
            workers = skill_route_workers(route["criterion"])
            if (decision["selected_worker"] not in workers | {"handoff", "defer"}
                    or not set(route["trigger_workers"]) <= workers):
                raise ValueError("actual route exceeds the registered goal compatibility group")
            if (
                any(
                    binding.get(key) != decision.get(key)
                    for key in (
                        "route_id",
                        "state",
                        "rho_ref",
                        "requested_worker",
                        "trigger_worker",
                        "selected_worker",
                        "criterion_ref",
                    )
                )
                or decision["rho_ref"] != control["rho_ref"]
                or decision["state"] != route["state"]
                or decision["selected_worker"] != route["selected_worker"]
                or command["action"] != decision["trigger_worker"]
                or command["action"] not in route["trigger_workers"]
                or command.get(
                    "prompt" if command["action"] in {"pi0_pick", "pi0_doubled"} else "instruction"
                )
                != route["instruction"]
            ):
                raise ValueError(
                    "route result, actual request and registered goal disagree"
                )
            before = [
                event
                for event in events
                if event["kind"] == "tool_call"
                and event["sequence"] < decision["sequence"]
            ]
            after = [
                event
                for event in events
                if event["kind"] == "tool_end"
                and event["sequence"] > decision["sequence"]
            ]
            if (
                not before
                or not after
                or before[-1]["tool"] != command["action"]
                or after[0]["tool"] != command["action"]
            ):
                raise ValueError("route decision is outside its actual tool call")
            low, high = before[-1]["sequence"], after[0]["sequence"]
            span = events[low + 1 : high]
            selected = decision["selected_worker"]
            if selected in {"handoff", "defer"}:
                if any(
                    event["kind"] in {"inference_start", "action_send"}
                    for event in span
                ):
                    raise ValueError(
                        "a deferred or handed-off route dispatched actions"
                    )
                continue
            starts = [event for event in span if event["kind"] == "skill_route_start"]
            ends = [event for event in span if event["kind"] == "skill_route_handoff"]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError("actual skill start/handoff unavailable")
            start, end = starts[0], ends[0]
            expected = {
                "prompt" if selected in {"pi0_pick", "pi0_doubled"} else "instruction": route[
                    "instruction"
                ],
                "max_chunks": command.get("max_chunks", 24),
            }
            if route["criterion"]["schema"] == DRAWER_CRITERION and (
                    route.get("max_chunks") != 24 or expected["max_chunks"] != 24):
                raise ValueError("actual drawer route changed its registered 24chunk allowance")
            if selected == command["action"] == "pi0_pick":
                expected.update(
                    {
                        key: command.get(key, value)
                        for key, value in {
                            "lift_thresh": 0.05,
                            "gripper_closed_thresh": 0.06,
                            "gripper_open_thresh": 0.0,
                            "descent_thresh": 0.1,
                        }.items()
                    }
                )
            if (
                start.get("actual_arguments") != expected
                or end.get("actual_arguments") != expected
                or end.get("execution_error")
                or not start["sequence"] < end["sequence"]
            ):
                raise ValueError(
                    "actual skill arguments or execution closure unavailable"
                )
            for phase, client in (("start", start), ("handoff", end)):
                env = private[client["environment_observation"]["source_sequence"]]
                if (
                    env["kind"] != "skill_route_observation"
                    or env["phase"] != phase
                    or env["route_id"] != route["route_id"]
                    or env["worker"] != selected
                    or env["client_route_sequence"] != decision["sequence"]
                    or env["criterion_ref"] != route["criterion"]["criterion_ref"]
                    or env["version_signature"] != control["version_signature"]
                ):
                    raise ValueError(
                        "actual skill handoff does not bind its environment observation"
                    )
            inferences = [event for event in span if event["kind"] == "inference_start"]
            if (
                type(end.get("chunks_used")) is not int
                or not 0 < end["chunks_used"] <= expected["max_chunks"]
                or len(inferences) != end["chunks_used"]
                or any(
                    event.get("instruction") != route["instruction"]
                    or not start["sequence"] < event["sequence"] < end["sequence"]
                    for event in inferences
                )
            ):
                raise ValueError(
                    "actual prediction count, allowance or goal binding unavailable"
                )
            for inference in inferences:
                completed = [
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
                if (
                    len(completed) != 1
                    or completed[0].get("completed") is not True
                    or len(chunks) != 1
                    or chunks[0].get("predictions_started") != 1
                    or chunks[0].get("predictions_completed") != 1
                    or chunks[0].get("unknown_steps") != 0
                    or type(chunks[0].get("executed_steps")) is not int
                    or chunks[0]["executed_steps"] <= 0
                ):
                    raise ValueError(
                        "actual routed inference or chunk execution is not closed"
                    )
            if selected == "vla_execute":
                report["vla_commands"].append({"action": "vla_execute", **expected})
            else:
                if (any(event.get("segment_sequence") is not None for event in inferences)
                        or any(event["kind"] in {"vla_segment_start", "vla_segment_end"} for event in span)):
                    raise ValueError("direct routed consumer has an unexpected segment binding")
                report["pick_spans"].append(
                    {
                        "start": start["sequence"],
                        "end": end["sequence"],
                        "instruction": route["instruction"],
                    }
                )
        if seen != {event["sequence"] for event in decisions}:
            raise ValueError(
                "unassociated route decision or missing recorded tool result"
            )
        report["checks"]["actual_goal_arguments_and_private_observation_binding"] = True
        report["value"] = True
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        report["missing"].append("skill route control chain: " + str(exc))
    return report
