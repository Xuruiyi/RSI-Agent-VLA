"""One explicitly permitted Research transport. No automatic retries or experiments."""

import json
import os
import sys
import time
from pathlib import Path
from .storage import atomic_json
from .research_transport import model_prompt, unpack


def read(path):
    return json.loads(Path(path).read_text())


def validate_permit(tools, transport, permit, *, root, output):
    """Match full input and current source bytes, not an old authorization boolean."""
    request = unpack(transport)
    if (
        request.get("job_id") != tools.job.state["job_id"]
        or request.get("bundle_id") != tools.base.bundle_id
        or request.get("model") != "gpt-6.1-sol"
        or request.get("reasoning_effort") != "medium"
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
        "reasoning_effort": "medium",
        "input": transport,
        "job_limits": tools.job.state["limits"],
    }
    if any(permit.get(k) != v for k, v in expected.items()):
        raise PermissionError(
            "permit does not match new job/bundle/protocol/input/output/budget"
        )
    limits = permit["limits"]
    if (
        set(limits) != {"model_requests", "wall_seconds", "storage_bytes"}
        or limits["model_requests"] != 1
        or not 0 < limits["wall_seconds"] <= 300
        or not 0 < limits["storage_bytes"] <= 10485760
    ):
        raise PermissionError("permit exceeds the one-request Research proposal")
    for k, jk in (
        ("model_requests", "model_requests"),
        ("wall_seconds", "whole_job_wall_seconds"),
        ("storage_bytes", "artifact_bytes"),
    ):
        if limits[k] > tools.job.remaining().get(jk, 0):
            raise PermissionError("original job budget insufficient")
    phase = read(permit["phase_snapshot_ref"])
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
    if manifest.get("version") != "batch4-preparation" or not manifest.get("files"):
        raise PermissionError("new source manifest required")
    for name in manifest["files"]:
        if (Path(root) / name).read_bytes() != (
            Path(permit["source_manifest_ref"]).parent / "source" / name
        ).read_bytes():
            raise PermissionError("prepared source changed: " + name)
    return request


def child(output):
    """Actual SDK consumer of the compact full-fact transport; never used by offline tests."""
    import openai_codex
    from rpent.planner.codex import ReasoningEffort, build_codex_config

    output = Path(output)
    context = read(output / "runtime_context.json")
    permit = read(context["authorization_ref"])
    transport = read(output / "request.json")
    if (
        not permit.get("authorized")
        or permit.get("input") != transport
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
    config = build_codex_config(
        mcp_url=None, base_url=None, api_key=None, cwd=str(output)
    )
    with openai_codex.Codex(config=config) as client:
        thread = client.thread_start(
            model="gpt-6.1-sol",
            approval_mode=openai_codex.ApprovalMode.deny_all,
            sandbox=openai_codex.Sandbox.read_only,
            developer_instructions="Use only supplied evidence. Do not call tools, read files, access networks, or launch experiments.",
        )
        result = thread.turn(
            model_prompt(transport),
            model="gpt-6.1-sol",
            effort=ReasoningEffort.medium,
            output_schema=read(output / "response-schema.json"),
        ).run()
    atomic_json(
        output / "sdk-result.json",
        {
            "status": getattr(result.status, "value", str(result.status)),
            "final_response": result.final_response,
        },
    )


def reconcile(tools, adapter, output, *, cost_limit):
    """Reparse existing response and usage; idempotent charge even after schema errors."""
    output = Path(output)
    transport = read(output / "request.json")
    request = unpack(transport)
    if tools.store.get("bounded_transports", request["request_id"]) != transport:
        raise ValueError("saved request differs from this job's prepared input")
    context = read(output / "runtime_context.json")
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
    phase = read(phase_path)
    phase_costs = {
        "wall_seconds": costs["whole_job_wall_seconds"],
        "model_requests": len(admitted),
        "tool_calls": 0,
        "environment_steps": 0,
        "gpu_seconds": 0,
        "simulation_episodes": 0,
        "storage_bytes": sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        ),
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
        atomic_json(phase_path, phase)
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
    if not any(row["currently_trialable"] for row in preflight):
        raise PermissionError(
            "no executable evidenced operator; do not spend Research allowance"
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
        "planner": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
        "limits": permit["limits"],
    }
    with lock_factory():
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "request.json", transport)
        atomic_json(
            output / "response-schema.json",
            adapter.response_schema(request["request_id"]),
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
        except BaseException as exc:
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
            costs["artifact_bytes"] = sum(
                p.stat().st_size for p in output.rglob("*") if p.is_file()
            )
            atomic_json(
                output / "execution-costs.json",
                {
                    "costs": costs,
                    "usage_complete": complete,
                    "cached_is_input_subset": True,
                    "missing_usage": "retain unknown; do not fill zero"
                    if not complete
                    else None,
                },
            )
        return reconcile(tools, adapter, output, cost_limit=cost_limit)
