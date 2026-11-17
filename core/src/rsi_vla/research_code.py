# Copyright 2026 RSI-Agent-VLA Authors.
"""Single-file Research rule candidates through the existing no-device sandbox."""

import ast
import difflib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import Candidate
from .process_accounting import StorageMeter
from .storage import atomic_json, identifier

ACTIONS = ("inspect_evidence", "bounded_experiment", "defer")


def validate_research_rule(source):
    """Check syntax and the explicit select_action(context) plugin interface."""
    tree = ast.parse(source, filename="research_rule.py")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(tree.body) != 1:
        raise ValueError("candidate may change only one research rule function")
    function = functions[0]
    args = function.args
    if (
        function.name != "select_action"
        or function.decorator_list
        or len(args.args) != 1
        or args.args[0].arg != "context"
        or args.posonlyargs
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
    ):
        raise ValueError("research rule interface is select_action(context)")
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.If,
        ast.IfExp,
        ast.While,
        ast.Assign,
        ast.Expr,
        ast.Pass,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Subscript,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.In,
        ast.NotIn,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(
            "research rules support bounded logic, not imports, calls or attributes"
        )
    if function.returns is not None or args.args[0].annotation is not None:
        raise ValueError("research rule interface has no executable annotations")
    compile(tree, "research_rule.py", "exec")


def _domain(tools, identity):
    return tools.store.root / "code_rule_artifacts" / identifier(identity)


def prepare_code_candidate(
    tools,
    candidate_id,
    *,
    base_source_ref,
    source,
    declared_paths,
    inputs,
    config,
    timeout_seconds,
):
    """Make an independent plugin copy; governance and scoring paths are excluded."""
    identity = identifier(candidate_id)
    if declared_paths != ["research_rule.py"]:
        raise ValueError("candidate scope is exactly research_rule.py")
    if not isinstance(source, str) or len(source.encode()) > 64 * 1024:
        raise ValueError("research rule source exceeds registered size")
    validate_research_rule(source)
    base_source_ref = str(Path(base_source_ref).resolve(strict=True))
    before = Path(base_source_ref).read_text()
    if source == before:
        raise ValueError("candidate must change the research rule")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"evidence_available", "budget_available"}
        or any(type(value) is not bool for value in inputs.values())
    ):
        raise ValueError("research rule receives only bounded evidence/budget facts")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 10:
        raise ValueError("research rule timeout must be within ten seconds")
    request = {
        "candidate_id": identity,
        "base_source_ref": base_source_ref,
        "source": source,
        "inputs": inputs,
        "timeout_seconds": timeout_seconds,
        "task_root": str(Path(config["isolation_task_root"]).resolve()),
    }
    if tools.store.path("research_code_proposals", identity).exists():
        saved = tools._owned("research_code_proposals", identity)
        if saved["request"] != request:
            raise ValueError("conflicting research code candidate")
        return saved
    domain = _domain(tools, identity)
    domain.mkdir(parents=True, exist_ok=True)
    (domain / "base.py").write_text(before)
    (domain / "research_rule.py").write_text(source)
    (domain / "candidate.diff").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="base/research_rule.py",
                tofile="candidate/research_rule.py",
            )
        )
    )
    return tools._save(
        "research_code_proposals",
        identity,
        {
            "candidate_id": identity,
            "request": request,
            "source_ref": str((domain / "research_rule.py").resolve()),
            "base_copy_ref": str((domain / "base.py").resolve()),
            "diff_ref": str((domain / "candidate.diff").resolve()),
            "interface": "research-rule-select-action-v1",
            "syntax_valid": True,
            "execution_authorized": False,
        },
    )


_RUNNER = """import argparse, contextlib, json, resource, runpy
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--input', action='append', default=[])
a=p.parse_args()
resource.setrlimit(resource.RLIMIT_AS, (536870912,536870912))
resource.setrlimit(resource.RLIMIT_CPU, (2,2))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576,1048576))
resource.setrlimit(resource.RLIMIT_NPROC, (8,8))
out=Path(a.output)
with (out/'plugin.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
 plugin=runpy.run_path(a.input[0])
 context=json.loads(Path(a.input[1]).read_text())
 action=plugin['select_action'](context)
 if action not in ('inspect_evidence','bounded_experiment','defer'):
  raise ValueError('unsupported research rule action')
 if action == 'bounded_experiment' and not (context['evidence_available'] and context['budget_available']):
  raise ValueError('rule cannot experiment without evidence and budget')
 (out/'selection.json').write_text(json.dumps({'action':action}))
"""


def _run(tools, identity, *, source_ref, inputs, config, timeout_seconds):
    from .protocol10_roles import run_role

    identity = identifier(identity)
    domain = _domain(tools, identity)
    domain.mkdir(parents=True, exist_ok=True)
    runner = domain / "runner.py"
    context = domain / "context.json"
    request = {
        "source_ref": source_ref,
        "inputs": inputs,
        "config": config,
        "timeout_seconds": timeout_seconds,
    }
    marker = tools.store.path("research_code_launches", identity)
    if (
        marker.exists()
        and tools._owned("research_code_launches", identity)["request"] != request
    ):
        raise ValueError("conflicting research rule execution")
    if not marker.exists():
        runner.write_text(_RUNNER)
        atomic_json(context, inputs)
        tools._save("research_code_launches", identity, {"request": request})
    registered = dict(config)
    registered["code_candidate_executions"] = [
        {
            "execution_id": identity,
            "source_ref": str(runner.resolve()),
            "input_refs": [source_ref, str(context.resolve())],
            "read_grants": [],
            "timeout_seconds": timeout_seconds,
        }
    ]
    if os.environ.get("RSI_ISOLATED_ROLE") == "research":
        from .protocol10_roles import broker_request

        socket_path = os.environ.get("RSI_RESEARCH_RULE_BROKER")
        if not socket_path:
            raise PermissionError("isolated Research requires the trusted rule broker")
        receipt = broker_request(
            {"action": "inherited_research_rule", "execution_id": identity},
            socket_path=socket_path,
        )
    else:
        receipt = run_role(
            registered, {"action": "code_candidate_execute", "execution_id": identity}
        )
    if receipt.get("wall_seconds") is None or not receipt.get("storage"):
        raise RuntimeError(
            "original candidate costs unknown; reconcile its receipt before new work"
        )
    cost_path = tools.store.path("research_code_costs", identity)
    if not cost_path.exists():
        persistent = StorageMeter(
            [domain, tools.store.path("research_code_proposals", identity)], baseline={}
        )
        measured = persistent.sample()
        costs = {
            "whole_job_wall_seconds": receipt["wall_seconds"],
            "artifact_bytes": measured["cumulative_new_bytes_lower_bound"]
            + receipt["storage"]["cumulative_new_bytes_lower_bound"],
        }
        tools._save(
            "research_code_costs",
            identity,
            {
                "costs": costs,
                "persistent": measured,
                "execution_storage": receipt["storage"],
                "persistent_current_bytes": measured["current_new_bytes"]
                + receipt["persistent_current_bytes"],
                "temporary_current_bytes": receipt["temporary_current_bytes"],
            },
        )
    observation = tools._owned("research_code_costs", identity)
    costs = observation["costs"]
    reason = "research code " + identity
    charged = [row for row in tools.job.state["charges"] if row["reason"] == reason]
    if not charged:
        tools.job.consume(costs, reason)
    selection = None
    output_ref = Path(receipt["output_ref"]) / "selection.json"
    if (
        not receipt.get("error")
        and output_ref.is_file()
        and output_ref.stat().st_size < 1024
    ):
        selection = json.loads(output_ref.read_text())
        if set(selection) != {"action"} or selection["action"] not in ACTIONS:
            raise ValueError("invalid sandbox research rule result")
    if os.environ.get("RSI_ISOLATED_ROLE") == "research":
        # The trusted host snapshots the result into its external return receipt.
        selection = receipt.get("selection_snapshot") if not receipt.get("error") else None
    return {
        "receipt": receipt,
        "costs": costs,
        "storage_observation": observation,
        "selection": selection,
        "technical_valid": selection is not None,
        "capability_gain": None,
    }


def _book_metadata(tools, identity):
    roots = [
        tools.store.path(kind, key)
        for kind, key in (
            ("research_code_feedback", identity),
            ("research_code_launches", identity),
            ("research_rule_consumptions", identity),
            ("candidates", identity),
            ("candidate_context", identity),
            ("psi", identity + "-psi"),
            ("bundles", identity + "-bundle"),
        )
    ]
    record = StorageMeter(roots, baseline={}).sample()
    reason = "research code metadata " + identity
    if not any(c["reason"] == reason for c in tools.job.state["charges"]):
        tools.job.record_storage(record["cumulative_new_bytes_lower_bound"], reason)
    tools.store.put("research_code_metadata_costs", identity, record)
    return record


def execute_code_candidate(tools, candidate_id, *, config):
    """Execute a bounded real subprocess and retain candidate diff and measured costs."""
    identity = identifier(candidate_id)
    proposal = tools._owned("research_code_proposals", identity)
    request = proposal["request"]
    if str(Path(config["isolation_task_root"]).resolve()) != request["task_root"]:
        raise ValueError("candidate task domain differs from preparation")
    launch_path = tools.store.path("research_code_launches", identity)
    if launch_path.exists():
        original = tools._owned("research_code_launches", identity)["request"]
        if original["config"] != config:
            raise ValueError("candidate execution configuration changed")
    if Path(proposal["source_ref"]).read_text() != request["source"]:
        raise ValueError("prepared research rule source changed")
    if tools.store.path("research_code_feedback", identity).exists():
        _book_metadata(tools, identity)
        return tools._owned("research_code_feedback", identity)
    if {"whole_job_wall_seconds", "artifact_bytes"} - tools.job.state["limits"].keys():
        raise PermissionError(
            "research rule requires registered wall and storage budgets"
        )
    if (
        tools.job.state["status"] != "running"
        or tools.job.remaining().get("whole_job_wall_seconds", 0)
        < request["timeout_seconds"]
        or tools.job.remaining()["artifact_bytes"] < 128 * 1024
    ):
        raise PermissionError("candidate lacks registered remaining wall budget")
    result = _run(
        tools,
        identity,
        source_ref=proposal["source_ref"],
        inputs=request["inputs"],
        config=config,
        timeout_seconds=request["timeout_seconds"],
    )
    candidate = None
    if result["technical_valid"]:
        psi_ref = identity + "-psi"
        psi = tools.store.get("psi", tools.base.psi_ref)
        tools.store.put(
            "psi",
            psi_ref,
            {
                **psi,
                "research_rule_plugin": {
                    "schema": "research-rule-select-action-v1",
                    "source_ref": proposal["source_ref"],
                    "diff_ref": proposal["diff_ref"],
                    "candidate_id": identity,
                    "source": request["source"],
                },
            },
        )
        target = replace(
            tools.base,
            bundle_id=identity + "-bundle",
            psi_ref=psi_ref,
            parent_bundle_id=tools.base.bundle_id,
            diff_ref=proposal["diff_ref"],
        )
        tools.store.save_bundle(target)
        candidate = Candidate(
            identity,
            tools.base.bundle_id,
            target.bundle_id,
            tools.protocol["protocol_id"],
            tools.protocol["benchmark_id"],
            {"psi_ref": psi_ref, "diff_ref": proposal["diff_ref"]},
            [proposal["source_ref"], proposal["diff_ref"]],
            [],
            tools._candidate_costs(),
            {
                "interface": proposal["interface"],
                "technical_valid": True,
                "task_improvement": None,
                "requires_independent_evaluation": True,
            },
            novelty_type="research_rule_program",
        )
        tools.store.save_candidate(candidate)
        from .candidate_archive import register_context

        register_context(
            tools.store,
            identity,
            selection_reason="interface-valid isolated research rule candidate; task gain remains unmeasured",
            dependencies={
                "versions": {
                    "theta_ref": tools.base.theta_ref,
                    "rho_ref": tools.base.rho_ref,
                    "psi_ref": psi_ref,
                    "code_ref": tools.base.code_ref,
                }
            },
            origin={"kind": "program", "ref": proposal["source_ref"]},
            contribution_ids=[identity],
            job_ids=[tools.job.state["job_id"]],
        )
    feedback = tools._save(
        "research_code_feedback",
        identity,
        {
            "candidate_id": identity,
            **result,
            "candidate": asdict(candidate) if candidate else None,
            "next_action": "research_followup",
            "hypotheses_resolved": False,
        },
    )

    _book_metadata(tools, identity)
    return feedback


def run_inherited_research_rule(
    tools, execution_id, *, inputs, config, timeout_seconds=3
):
    """Consume the starting bundle's actual plugin in a subsequent Research job."""
    plugin = tools.store.get("psi", tools.base.psi_ref).get("research_rule_plugin")
    if not plugin or plugin["schema"] != "research-rule-select-action-v1":
        raise ValueError("starting research bundle has no supported rule plugin")
    if Path(plugin["source_ref"]).read_text() != plugin["source"]:
        raise ValueError("inherited research rule source changed")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 10:
        raise ValueError("inherited research rule requires a bounded timeout")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"evidence_available", "budget_available"}
        or any(type(value) is not bool for value in inputs.values())
    ):
        raise ValueError("inherited research rule receives bounded boolean facts")
    identity = identifier(execution_id)
    if tools.store.path("research_rule_consumptions", identity).exists():
        original = tools._owned("research_code_launches", identity)["request"]
        if (
            original["inputs"] != inputs
            or original["config"] != config
            or original["timeout_seconds"] != timeout_seconds
        ):
            raise ValueError("conflicting inherited research rule execution")
        _book_metadata(tools, identity)
        return tools._owned("research_rule_consumptions", identity)
    if {"whole_job_wall_seconds", "artifact_bytes"} - tools.job.state["limits"].keys():
        raise PermissionError(
            "research rule requires registered wall and storage budgets"
        )
    if (
        tools.job.state["status"] != "running"
        or tools.job.remaining().get("whole_job_wall_seconds", 0) < timeout_seconds
        or tools.job.remaining()["artifact_bytes"] < 128 * 1024
    ):
        raise PermissionError("inherited research rule lacks remaining budget")
    result = _run(
        tools,
        identity,
        source_ref=plugin["source_ref"],
        inputs=inputs,
        config=config,
        timeout_seconds=timeout_seconds,
    )
    feedback = tools._save(
        "research_rule_consumptions",
        identity,
        {
            "execution_id": identity,
            "psi_ref": tools.base.psi_ref,
            "source_ref": plugin["source_ref"],
            **result,
        },
    )

    _book_metadata(tools, identity)
    return feedback
