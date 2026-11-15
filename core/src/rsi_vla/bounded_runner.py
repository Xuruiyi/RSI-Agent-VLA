"""One explicitly permitted Research transport. No automatic retries or experiments."""

import json
import os
import sys
import time
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

from .phase_receipts import read_phase, write_phase
from .research_transport import sdk_input, unpack
from .storage import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def permitted_input(permit):
    """Verify compact referenced input against the host's immutable content grant."""
    if 'input' in permit:
        return permit['input']
    path = Path(permit['input_ref']).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != permit.get('input_sha256'):
        raise PermissionError('referenced Research input differs from authorized content')
    return json.loads(payload)


def validate_permit(tools, transport, permit, *, root, output):
    """Match full input and current source bytes, not an old authorization boolean."""
    from .task_policy import task_policy

    request = unpack(transport)
    if request.get("task_policy", {}) != task_policy(tools.protocol):
        raise PermissionError("prepared task policy changed; prepare a new request")
    if (
        request.get("base_bundle") != asdict(tools.base)
        or request.get("job_id") != tools.job.state["job_id"]
        or request.get("bundle_id") != tools.base.bundle_id
        or request.get("model") != "gpt-6.1-sol"
        or request.get("reasoning_effort") != "high"
        or tools.store.get("bounded_transports", request["request_id"]) != transport
    ):
        raise PermissionError(
            "prepared input belongs to a different job/bundle/settings"
        )
    expected = {
        "schema": "rsi-bounded-research-permit-v1",
        "authorized": True,
        "job_id": tools.job.state["job_id"],
        "request_id": request["request_id"],
        "bundle_id": tools.base.bundle_id,
        "protocol_id": tools.protocol["protocol_id"],
        "output_dir": str(Path(output).resolve()),
        "model": "gpt-6.1-sol",
        "reasoning_effort": "high",
        "job_limits": tools.job.state["limits"],
    }
    if any(permit.get(k) != v for k, v in expected.items()):
        raise PermissionError(
            "permit does not match new job/bundle/protocol/input/output/budget"
        )
    if permitted_input(permit) != transport:
        raise PermissionError('permit does not bind this complete Research transport')
    from .task_policy import number

    proposal = task_policy(tools.protocol).get(
        "research_per_request", {"wall_seconds": 300, "storage_bytes": 10485760}
    )
    wall_cap = number(proposal["wall_seconds"], "Research wall cap")
    storage_cap = number(proposal["storage_bytes"], "Research storage cap")
    limits = permit["limits"]
    if (
        set(limits) != {"model_requests", "wall_seconds", "storage_bytes"}
        or limits["model_requests"] != 1
        or not 0 < number(limits["wall_seconds"], "Research wall limit") <= wall_cap
        or not 0
        < number(limits["storage_bytes"], "Research storage limit")
        <= storage_cap
    ):
        raise PermissionError("permit exceeds the one-request Research proposal")
    for k, jk in (
        ("model_requests", "model_requests"),
        ("wall_seconds", "whole_job_wall_seconds"),
        ("storage_bytes", "artifact_bytes"),
    ):
        if limits[k] > tools.job.remaining().get(jk, 0):
            raise PermissionError("original job budget insufficient")
    phase = read_phase(permit["phase_snapshot_ref"])
    if phase.get("unresolved_costs"):
        raise PermissionError(
            "phase actual costs unresolved; reconcile before Research"
        )
    if any(
        limits[k] > phase["remaining"][pk]
        for k, pk in (
            ("model_requests", "model_requests"),
            ("wall_seconds", "wall_seconds"),
            ("storage_bytes", "storage_bytes"),
        )
    ):
        raise PermissionError("remaining original phase budget insufficient")
    manifest = read(permit["source_manifest_ref"])
    if manifest.get("version") not in {
        "batch4-preparation",
        "5a-assets-offline",
    } or not manifest.get("files"):
        raise PermissionError("new source manifest required")
    for name in manifest["files"]:
        if (Path(root) / name).read_bytes() != (
            Path(permit["source_manifest_ref"]).parent / "source" / name
        ).read_bytes():
            raise PermissionError("prepared source changed: " + name)
    return request


def _research_codex_config(output):
    """Keep SDK plugin discovery out of the bounded Research child."""
    from rpent.planner.codex import build_codex_config

    config = build_codex_config(
        mcp_url=None, base_url=None, api_key=None, cwd=str(output)
    )
    return replace(
        config,
        config_overrides=(
            *config.config_overrides,
            "features.plugins=false",
            "features.remote_plugin=false",
        ),
    )


def child(output):
    """Actual SDK consumer of the compact full-fact transport; never used by offline tests."""
    import openai_codex
    from rpent.planner.codex import ReasoningEffort

    output = Path(output)
    context = read(output / "runtime_context.json")
    permit = read(context["authorization_ref"])
    transport = read(output / "request.json")
    if unpack(transport)["reasoning_effort"] != "high":
        raise PermissionError(
            "new Research dispatch requires high; historical responses may only be reconciled"
        )
    if (
        not permit.get("authorized")
        or permitted_input(permit) != transport
        or not (output / "dispatch.json").exists()
    ):
        raise PermissionError("child lacks matching new one-request dispatch")
    if (output / "sdk-result.json").exists() or (
        output / "child-started.json"
    ).exists():
        raise FileExistsError("request child already started; reconcile, never retry")
    atomic_json(
        output / "child-started.json", {"request_id": unpack(transport)["request_id"]}
    )
    actual_input = sdk_input(transport, read(output / "response-schema.json"))
    atomic_json(output / "sdk-input.json", actual_input)
    config = _research_codex_config(output)
    with openai_codex.Codex(config=config) as client:
        thread = client.thread_start(
            model="gpt-6.1-sol",
            approval_mode=openai_codex.ApprovalMode.deny_all,
            sandbox=openai_codex.Sandbox.read_only,
            developer_instructions=actual_input["developer_instructions"],
        )
        result = thread.turn(
            actual_input["prompt"],
            model="gpt-6.1-sol",
            effort=ReasoningEffort.high,
            output_schema=actual_input["output_schema"],
        ).run()
    atomic_json(output / "sdk-input-consumption.json", {
        "request_id": actual_input["request_id"], "sdk_input_ref": str(output / "sdk-input.json"),
        "sdk_turn_returned": True, "media_mode": "text_only", "image_bytes_sent": False,
        "http_evidence_ref": str(output / "model_requests.jsonl"), "model_attention": "unknown"})
    atomic_json(
        output / "sdk-result.json",
        {
            "status": getattr(result.status, "value", str(result.status)),
            "final_response": result.final_response,
        },
    )


def archive_sdk_input(tools, output, transport):
    """Preserve the actual validated SDK bytes under its existing canonical name."""
    output = Path(output)
    actual = read(output/'sdk-input.json')
    if actual != sdk_input(transport, read(output/'response-schema.json')):
        raise ValueError('actual SDK input differs from prepared request')
    target = tools.store.path('research_sdk_inputs', unpack(transport)['request_id'])
    if target.exists():
        if read(target) != actual:
            raise ValueError('stored SDK input differs from actual consumed input')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(output/'sdk-input.json', target)


def _reconcile(tools, adapter, output, *, cost_limit):
    """Reparse existing response and usage; idempotent charge even after schema errors."""
    output = Path(output)
    transport = read(output / "request.json")
    request = unpack(transport)
    if tools.store.get("bounded_transports", request["request_id"]) != transport:
        raise ValueError("saved request differs from this job's prepared input")
    context = read(output / "runtime_context.json")
    permit = read(context['authorization_ref'])
    if (
        any(context.get(k) != request[k] for k in ("job_id", "bundle_id"))
        or context["protocol_id"] != tools.protocol["protocol_id"]
    ):
        raise ValueError("response context belongs to another job")
    supervision = read(output / "supervisor.json")
    costs = read(output / "execution-costs.json")["costs"]
    events = [
        json.loads(s)
        for s in (output / "model_requests.jsonl").read_text().splitlines()
    ]
    admitted = [e for e in events if e["kind"] == "request_admitted"]
    terminal = [e for e in events if e["kind"] == "response_terminal"]
    if len(admitted) > 1 or costs["model_requests"] != len(admitted):
        raise ValueError("unexpected Research request accounting")
    facts = (
        {
            "effective_model": admitted[0]["model"],
            "effective_reasoning_effort": admitted[0]["reasoning_effort"],
            "evidence_ref": str(output / "model_requests.jsonl"),
        }
        if admitted
        else None
    )
    if (output / "sdk-input.json").exists():
        actual_input = read(output / "sdk-input.json")
        if actual_input != sdk_input(transport, read(output / "response-schema.json")):
            raise ValueError("actual SDK input differs from the prepared business request")
        if permit.get('input_ref'):
            archive_sdk_input(tools, output, transport)
        else:
            tools.store.put("research_sdk_inputs", request["request_id"], actual_input)
        receipt_path = output / "sdk-input-consumption.json"
        if receipt_path.exists():
            tools.store.put("research_sdk_consumption", request["request_id"], read(receipt_path))
    if facts:
        adapter.record_execution(tools, request["request_id"], costs, facts)
    else:
        key = request["request_id"] + "-startup"
        if not tools.store.path("research_startup_costs", key).exists():
            reason = "failed LLM startup " + request["request_id"]
            hard = {k: v for k, v in costs.items() if k in tools.job.state["limits"]}
            prepaid = [c for c in tools.job.state["charges"] if c["reason"] == reason]
            if prepaid and prepaid != [{"reason": reason, "costs": hard}]:
                raise ValueError("startup charge mismatch")
            if not prepaid:
                tools.job.consume(hard, reason)
            tools.store.put("research_startup_costs", key, {"costs": costs})
    # Charge the existing phase exactly once, including failed requests/startup.
    permit = read(context["authorization_ref"])
    phase_path = Path(permit["phase_snapshot_ref"])
    phase = read_phase(phase_path)
    phase_costs = {
        "wall_seconds": costs["whole_job_wall_seconds"],
        "model_requests": len(admitted),
        "tool_calls": 0,
        "environment_steps": 0,
        "gpu_seconds": 0,
        "simulation_episodes": 0,
        # Match the job's raw measurement; finish() books subsequent output growth.
        "storage_bytes": costs["artifact_bytes"],
    }
    existing = [
        c
        for c in phase.get("research_charges", [])
        if c["request_id"] == request["request_id"]
    ]
    if existing:
        if any(
            existing[0]["costs"][k] != v
            for k, v in phase_costs.items()
            if k != "storage_bytes"
        ):
            raise ValueError("phase Research charge mismatch")
    else:
        phase.setdefault("research_charges", []).append(
            {
                "request_id": request["request_id"],
                "costs": phase_costs,
                "evidence_ref": str(output / "execution-costs.json"),
            }
        )
        for k, v in phase_costs.items():
            phase["used"][k] += v
            phase["remaining"][k] = max(0, phase["limits"][k] - phase["used"][k])
        write_phase(phase_path, phase)
    if (
        costs["whole_job_wall_seconds"] > context["limits"]["wall_seconds"]
        or costs.get("artifact_bytes", 0) > context["limits"]["storage_bytes"]
    ):
        raise RuntimeError(
            "Research execution exceeded its permit; costs retained, no response application"
        )
    if (
        supervision.get("status") != "completed"
        or supervision.get("exit_code") != 0
        or not supervision.get("cleanup_confirmed")
        or len(admitted) != 1
        or len(terminal) != 1
        or terminal[0].get("backend_model") != "gpt-6.1-sol"
    ):
        raise RuntimeError(
            "transport incomplete; retain evidence and reconcile offline, never retry"
        )
    sdk = read(output / "sdk-result.json")
    if sdk["status"].rsplit(".", 1)[-1] != "completed":
        raise ValueError("Research SDK incomplete")
    response = json.loads(sdk["final_response"])
    # apply_response rechecks trusted llm_executions and all finite policy constraints.
    result = adapter.apply_response(
        tools, response, evidence_level=tools.protocol["scope"], cost_limit=cost_limit
    )
    atomic_json(output / "validated-response.json", response)
    return result


def reconcile(tools, adapter, output, *, cost_limit):
    from .artifact_accounting import begin, finish

    output = Path(output)
    request = unpack(read(output / "request.json"))
    if (
        request.get("job_id") != tools.job.state["job_id"]
        or request.get("bundle_id") != tools.base.bundle_id
        or tools.store.get("bounded_transports", request["request_id"])
        != read(output / "request.json")
    ):
        raise ValueError("reconciliation input belongs to another job")
    key = request["request_id"] + "-derived"
    context = read(output / "runtime_context.json")
    permit = read(context["authorization_ref"])
    if permit.get('input_ref') and (output/'sdk-input.json').exists():
        # This is an alias of an already supervised inode, not new derived bytes.
        archive_sdk_input(tools, output, read(output/'request.json'))
    begin(tools, key)
    try:
        return _reconcile(tools, adapter, output, cost_limit=cost_limit)
    finally:
        measurement = read(output / "execution-costs.json")
        costs = measurement["costs"]
        accounting = finish(
            tools,
            key,
            phase_path=permit["phase_snapshot_ref"],
            output=output,
            billed_output_bytes=measurement.get(
                "raw_output_bytes_at_measurement", costs.get("artifact_bytes")
            ),
        )
        if (
            costs.get("artifact_bytes", 0) + accounting["booked_bytes"]
            > context["limits"]["storage_bytes"]
        ):
            raise RuntimeError(
                "Research output and derived evidence exceed storage bound; retained, no new dispatch"
            )


def execute(
    tools,
    adapter,
    transport,
    *,
    root,
    output,
    permit_path,
    preflight,
    cost_limit,
    gate_factory=None,
    supervisor=None,
    lock_factory=None,
):
    """Production execution uses existing gate/supervisor/lock; tests inject in-process doubles."""
    request = validate_permit(
        tools, transport, read(permit_path), root=root, output=output
    )
    diagnostic = unpack(transport).get("research_mode") == "technical_diagnostic"
    goal_only = request.get("opportunity_only") is True
    if goal_only and (request.get("task_policy", {}).get("opportunity_discovery") is not True
                      or tools.domain_policy.get("opportunity_only") is not True
                      or request.get("allowed_operators")):
        raise PermissionError("goal-only dispatch requires registered discovery and no trial operators")
    if diagnostic:
        request = unpack(transport)
        if not tools.domain_policy.get("diagnostic_scope_approved") or not any(
            r["invalid_count"] and r["data_refs"] for r in request["evidence"]
        ):
            raise PermissionError(
                "diagnosis needs approved scope and legal technical evidence"
            )
    if not diagnostic and not goal_only and not any(row["currently_trialable"] for row in preflight):
        raise PermissionError(
            "no executable evidenced operator; do not spend Research allowance"
        )
    if not diagnostic and not goal_only:
        trialable = {row["operator"] for row in preflight if row["currently_trialable"]}
        if not set(unpack(transport)["allowed_operators"]) <= trialable:
            raise PermissionError(
                "actual request operators exceed approved feasible preflight"
            )
    if gate_factory is None:
        from .adapters.request_gate import RequestGate

        gate_factory = RequestGate
    if supervisor is None:
        from .adapters.runtime_supervisor import supervise

        supervisor = supervise
    if lock_factory is None:
        from .runtime import dynamic_job_lock

        lock_factory = dynamic_job_lock
    permit = read(permit_path)
    output = Path(output).resolve()
    context = {
        "run_id": output.name,
        "episode_id": "research-only",
        "job_id": tools.job.state["job_id"],
        "bundle_id": tools.base.bundle_id,
        "protocol_id": tools.protocol["protocol_id"],
        "output_dir": str(output),
        "gpu_count": 0,
        "dynamic_authorized": True,
        "authorization_ref": str(Path(permit_path).resolve()),
        "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "high"},
        "limits": permit["limits"],
        "storage_write_roots": permit.get("storage_write_roots", [str(output)]),
    }
    with lock_factory():
        from .process_accounting import StorageMeter

        if permit.get('input_ref'):
            context['storage_write_roots'] = [*context['storage_write_roots'], permit['input_ref']]
        storage_meter = StorageMeter(context["storage_write_roots"])
        context["storage_baseline"] = storage_meter.baseline
        output.mkdir(parents=True, exist_ok=False)
        if permit.get('input_ref'):
            os.link(permit['input_ref'],output/'request.json')
        else:
            atomic_json(output / "request.json", transport)
        atomic_json(
            output / "response-schema.json",
            adapter.response_schema(request["request_id"], tools=tools),
        )
        atomic_json(
            output / "dispatch.json",
            {"request_id": request["request_id"], "automatic_retries": 0},
        )
        started = time.monotonic()
        context["job_started_monotonic"] = started
        atomic_json(output / "runtime_context.json", context)
        previous = os.environ.get("RSI_REQUEST_GATE_URL")
        try:
            with gate_factory(context) as gate:
                os.environ["RSI_REQUEST_GATE_URL"] = gate.url
                report = supervisor(
                    [
                        sys.executable,
                        str(Path(root) / "scripts/run_bounded_research.py"),
                        "--child",
                        str(output),
                    ],
                    context_path=output / "runtime_context.json",
                )
                atomic_json(output / "supervisor.json", report)
        except BaseException as exc:  # noqa: BLE001 - preserve supervisor interruption and all measured costs
            atomic_json(
                output / "supervisor.json",
                {
                    "status": "failed",
                    "exit_code": None,
                    "cleanup_confirmed": False,
                    "error": str(exc),
                },
            )
        finally:
            if previous is None:
                os.environ.pop("RSI_REQUEST_GATE_URL", None)
            else:
                os.environ["RSI_REQUEST_GATE_URL"] = previous
            events = (
                [
                    json.loads(s)
                    for s in (output / "model_requests.jsonl").read_text().splitlines()
                ]
                if (output / "model_requests.jsonl").exists()
                else []
            )
            admitted = [e for e in events if e["kind"] == "request_admitted"]
            terminals = [e for e in events if e["kind"] == "response_terminal"]
            costs = {
                "whole_job_wall_seconds": time.monotonic() - started,
                "model_requests": len(admitted),
                "tool_calls": 0,
            }
            complete = (
                bool(terminals)
                and len(terminals) == len(admitted)
                and all(
                    type((e.get("usage") or {}).get(k)) is int
                    for e in terminals
                    for k in ("input_tokens", "output_tokens")
                )
            )
            if complete:
                for name in ("input_tokens", "output_tokens"):
                    costs[name] = sum(e["usage"][name] for e in terminals)
                cached = [
                    (e["usage"].get("input_tokens_details") or {}).get("cached_tokens")
                    for e in terminals
                ]
                if all(type(v) is int for v in cached):
                    costs["cached_input_tokens"] = sum(cached)
            from .process_accounting import accounted_storage_bytes, snapshot

            if permit.get('input_ref'):
                # The request hard link aliases the prepare-owned canonical
                # transport; the native meter has that inode in its baseline.
                costs['artifact_bytes'] = max(
                    read(output/'supervisor.json').get('storage_bytes', 0),
                    storage_meter.sample()['cumulative_new_bytes_lower_bound'],
                )
            else:
                costs["artifact_bytes"] = max(
                    accounted_storage_bytes(output),
                    storage_meter.sample()["cumulative_new_bytes_lower_bound"],
                )
            atomic_json(
                output / "execution-costs.json",
                {
                    "costs": costs,
                    "raw_output_bytes_at_measurement": sum(snapshot([output]).values()),
                    "usage_complete": complete,
                    "cached_is_input_subset": True,
                    "missing_usage": "retain unknown; do not fill zero"
                    if not complete
                    else None,
                },
            )
        return reconcile(tools, adapter, output, cost_limit=cost_limit)
