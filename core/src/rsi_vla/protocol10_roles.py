"""Host broker for fixed production consumers in separate mount write domains."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

from .phase_receipts import commit
from .storage import atomic_json, identifier

ROLES = {
    "runtime": "runtime",
    "research": "state/research",
    "evaluation": "state/evaluation",
    "operator": "operator",
    "release": "release",
}
OP_ROLES = {
    "runtime": "runtime",
    "research": "research",
    "development_registration": "research",
    "candidate_submit": "research",
    "attempt_evaluation": "evaluation",
    "matrix_evaluation": "evaluation",
    "gate": "evaluation",
    "candidate_deployment_prepare": "research",
    "candidate_evidence_import": "evaluation",
    "candidate_upgrade_qualify": "release",
    "candidate_upgrade_activate": "release",
}


def _sandbox_command(
    task_root,
    role,
    argv,
    gpu=False,
    scratch_root=None,
    release_root=None,
    protected_paths=(),
):
    path = Path(__file__).resolve().parents[3] / "scripts/run_isolated_role.py"
    spec = importlib.util.spec_from_file_location("_rsi_isolation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {"gpu": gpu, "scratch_root": scratch_root}
    if release_root is not None:
        kwargs["release_root"] = release_root
    if protected_paths:
        kwargs["protected_paths"] = protected_paths
    return module.command(task_root, role, argv, **kwargs)


def broker_request(request):
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(os.environ["RSI_ROLE_BROKER"])
        client.sendall(json.dumps(request).encode() + b"\n")
        data = client.makefile("rb").readline()
    result = json.loads(data)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["result"]


def operator_command(configuration_ref, argv, *, boundary_factory=None):
    """Return None inside operator; otherwise run the CLI through a host broker."""
    if os.environ.get("RSI_ISOLATED_ROLE") == "operator":
        return None
    config = json.loads(Path(configuration_ref).read_text())
    return run_operator(config, argv, boundary_factory=boundary_factory)


def run_operator(configuration, argv, *, boundary_factory=None):
    """Keep host launch authority outside every role's read-only parent mounts."""
    root = Path(configuration["isolation_task_root"]).resolve()
    for role, relative in ROLES.items():
        if role == "release":
            continue
        (root / relative).mkdir(parents=True, exist_ok=True)
        (root / "scratch" / role).mkdir(parents=True, exist_ok=True)
    lock = Path("/tmp/rsi-agent-vla-dynamic.lock")
    lock.touch(exist_ok=True)
    socket_path = root / "scratch/operator" / ("broker-" + uuid4().hex + ".sock")
    server = socket.socket(socket.AF_UNIX)
    directory_fd = os.open(socket_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # A short descriptor path keeps deeply nested registered roots usable.
        server.bind(f"/proc/self/fd/{directory_fd}/{socket_path.name}")
    finally:
        os.close(directory_fd)
    server.listen()
    server.settimeout(0.2)
    stopped = threading.Event()

    operator_pid = []
    operator_ready = threading.Event()
    delivered_proposals = []
    delivered_audits = {}

    def serve():
        while not stopped.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                try:
                    operator_ready.wait(timeout=5)
                    import struct

                    peer_pid = struct.unpack(
                        "3i",
                        connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, 12
                        ),
                    )[0]
                    ancestor = peer_pid
                    while operator_pid and ancestor != operator_pid[0] and ancestor > 1:
                        ancestor = int(
                            Path(f"/proc/{ancestor}/stat")
                            .read_text()
                            .rsplit(")", 1)[1]
                            .split()[1]
                        )
                    if not operator_pid or ancestor != operator_pid[0]:
                        raise PermissionError(
                            "only the active operator process can request host actions"
                        )
                    request = json.loads(connection.makefile("rb").readline())
                    if request["action"] == "phase_commit":
                        proposal = request["proposal"]
                        if proposal not in delivered_proposals:
                            audit_task, audit = delivered_audits.get(
                                proposal.get("operation_id"), (None, None)
                            )
                            if not audit_task:
                                raise ValueError(
                                    "phase update lacks a delivered role cost receipt"
                                )
                            op = audit_task["operation"]
                            key = "protocol10_trial_charges"
                            old = proposal["before"].get(key, [])
                            new = proposal["after"].get(key, [])
                            changes = [entry for entry in new if entry not in old]
                            if (
                                len(changes) != 1
                                or changes[0]["run_id"] != op["payload"]["run_id"]
                            ):
                                raise ValueError(
                                    "phase update does not match audited operation"
                                )
                            allowed = {
                                "used",
                                "remaining",
                                key,
                                "unresolved_costs",
                                "trial_cost_reconciliations",
                            }
                            if any(
                                proposal["before"].get(k) != proposal["after"].get(k)
                                for k in proposal["before"].keys()
                                | proposal["after"].keys()
                                if k not in allowed
                            ):
                                raise ValueError(
                                    "operator settlement changes unrelated role costs"
                                )
                            from .protocol10_backend import COST_MAP

                            measured = audit.get("costs", {})
                            for dimension, field in COST_MAP.items():
                                if (
                                    field
                                    and field in measured
                                    and measured[field] is not None
                                ):
                                    actual = changes[0]["costs"].get(dimension)
                                    if (
                                        dimension == "storage_bytes"
                                        and actual < measured[field]
                                    ) or (
                                        dimension != "storage_bytes"
                                        and actual != measured[field]
                                    ):
                                        raise ValueError(
                                            "phase charge differs from role cost receipt"
                                        )
                        commit(proposal, configuration["phase_ref"])
                        result = {"committed": True}
                    elif request["action"] == "task_admission":
                        if set(request) != {"action", "operation_id"}:
                            raise ValueError("resource probe accepts only an operation identity")
                        from .task_continuation import probe_admission
                        result = probe_admission(configuration, request["operation_id"])
                    elif request["action"] == "role":
                        task = request["task"]
                        result = run_role(
                            configuration, task, boundary_factory=boundary_factory
                        )
                        delivered_proposals.extend(result.get("phase_proposals", []))
                        if task["action"] == "audit" and "value" in result:
                            delivered_audits[task["operation"]["operation_id"]] = (
                                task,
                                result["value"],
                            )
                    else:
                        raise ValueError("unsupported broker action")
                    response = {"result": result}
                except Exception as exc:  # noqa: BLE001 - preserve broker failures as explicit receipts
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                connection.sendall(json.dumps(response).encode() + b"\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    env = dict(os.environ, RSI_ROLE_BROKER="/tmp/" + socket_path.name)
    try:
        process = subprocess.Popen(_sandbox_command(root, "operator", argv), env=env)
        operator_pid.append(process.pid)
        operator_ready.set()
        return process.wait()
    finally:
        stopped.set()
        server.close()
        thread.join(timeout=1)
        socket_path.unlink(missing_ok=True)


def dispatch(config, backend_root, action, *, operation=None, inputs=None, **kwargs):
    task = {
        "backend_root": str(backend_root),
        "action": action,
        "operation": operation,
        "inputs": inputs or {},
        **kwargs,
    }
    result = broker_request({"action": "role", "task": task})
    for proposal in result.get("phase_proposals", []):
        broker_request({"action": "phase_commit", "proposal": proposal})
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["value"]


def deployment_runtime_protected_paths(config, task):
    """Keep completed runtime facts read-only during a new deployment job."""
    operation = task.get("operation", {})
    if operation.get("kind") != "runtime":
        return ()
    slots = config.get("runtime_slots", [])
    current = next(
        (slot for slot in slots
         if slot["run_id"] == operation["payload"]["run_id"]
         and slot.get("split") == "deployment"),
        None,
    )
    if current is None:
        return ()
    output = Path(current["output_ref"]).resolve()
    paths = {
        Path(slot["output_ref"]).resolve()
        for slot in slots
        if slot["run_id"] != current["run_id"]
        and Path(slot["output_ref"]).exists()
        and Path(slot["output_ref"]).resolve() != output
    }
    protocol = json.loads(Path(config["protocol_ref"]).read_text())
    derivation = protocol.get("deployment_derivation", {})
    if derivation.get("kind") == "candidate_upgrade":
        from .upgrade_bridge import isolation_paths

        paths.update(isolation_paths(config, {
            "action": "runtime",
            "registration_ref": derivation["authorization_ref"],
        }))
    return tuple(sorted(paths))


def run_role(config, task, *, boundary_factory=None):
    root = Path(config["isolation_task_root"]).resolve()
    action = task["action"]
    if action in {"execute", "reconcile"}:
        if config.get("schema") == "rsi-task-completion-v1":
            from .task_completion import ACTION_ROLES

            role = ACTION_ROLES[task["operation"]["kind"]]
        else:
            role = OP_ROLES[task["operation"]["kind"]]
    else:
        role = {
            "audit": "evaluation",
            "development_collect": "evaluation",
            "development_dispatch": "research",
            "development_reconcile": "research",
            "runtime_plan": "runtime",
            "candidate_deployment_prepare": "research",
            "candidate_evidence_import": "evaluation",
            "candidate_upgrade_qualify": "release",
            "candidate_upgrade_activate": "release",
        }[action]
    identity = identifier(
        task["operation"]["operation_id"] if task.get("operation") else task["identity"]
    )
    token = identity + "-" + action
    role_root = (
        Path(config["release_root"]).resolve()
        if role == "release"
        else root / ROLES[role]
    )
    receipt_directory = role_root / "role_receipts"
    latest = sorted(receipt_directory.glob(token + "-revision-*.json"))
    if latest:
        saved = json.loads(latest[-1].read_text())

        def comparable(value):
            return {k: v for k, v in value.items() if k != "inputs"}

        if saved["request"]["configuration"] != config or comparable(
            saved["request"]["task"]
        ) != comparable(task):
            raise ValueError("conflicting role task identity")
        if not saved.get("error") and (
            not isinstance(saved.get("value"), dict)
            or saved["value"].get("status", "completed") == "completed"
        ):
            return saved
        if action == "execute" and task["operation"]["dynamic"]:
            return saved
    revision = len(latest) + 1
    token += "-revision-" + str(revision).zfill(5)
    request_path = root / "operator/role_requests" / (token + ".json")
    receipt_path = receipt_directory / (token + ".json")
    request = {
        "configuration": config,
        "task": task,
        "role": role,
        "receipt_ref": str(receipt_path),
    }
    if boundary_factory:
        if config.get("offline_fixture") is not True:
            raise ValueError("CPU boundary injection requires offline fixture scope")
        request["boundary_factory"] = boundary_factory
    if request_path.exists():
        if (
            not receipt_path.exists()
            and action in {"execute", "runtime_plan"}
            and task["operation"]["dynamic"]
        ):
            raise RuntimeError(
                "role launch marker has no return receipt; reconcile native output, never resubmit"
            )
        if json.loads(request_path.read_text()) != request:
            raise ValueError("conflicting role request identity")
    else:
        atomic_json(request_path, request)
    if receipt_path.exists():
        result = json.loads(receipt_path.read_text())
    else:
        argv = [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "scripts/run_protocol10_role.py"),
            "--request",
            str(request_path),
        ]
        scratch_root = None
        if (
            role == "runtime"
            or role == "research"
            and task.get("operation", {}).get("kind") == "research"
        ) and task.get("operation", {}).get("dynamic"):
            cfg = (
                next(request for request in config["research_requests"]
                     if request["request_id"] == task["operation"]["payload"]["request_id"])
                if config.get("schema") == "rsi-task-completion-v1" and task["operation"]["kind"] == "research"
                else next(slot for slot in config["runtime_slots"]
                          if slot["run_id"] == task["operation"]["payload"]["run_id"])
                if config.get("schema") == "rsi-task-completion-v1"
                else config["research"]
                if task["operation"]["kind"] == "research"
                else task["operation"]["payload"]
            )
            scratch_root = cfg.get("scratch_ref")
        protected_paths = ()
        if action.startswith("candidate_"):
            from .upgrade_bridge import isolation_paths

            protected_paths = isolation_paths(config, task)
        elif role == "runtime":
            protected_paths = deployment_runtime_protected_paths(config, task)
        returned = subprocess.run(
            _sandbox_command(
                root,
                role,
                argv,
                gpu=role == "runtime" and not config.get("offline_fixture"),
                scratch_root=scratch_root,
                release_root=config.get("release_root") if role == "release" else None,
                protected_paths=protected_paths,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if not receipt_path.exists():
            raise RuntimeError(
                "role failed without receipt: " + returned.stderr[-2000:]
            )
        result = json.loads(receipt_path.read_text())
    if result.get("role") != role or result.get("request") != request:
        raise ValueError("role receipt identity mismatch")
    return result


def role_main(request_ref):
    """Execute a fixed consumer; model output cannot choose code or write domains."""
    from .contracts import ExperimentResult
    from .phase_receipts import proposals
    from .protocol10_backend import Protocol10Backend
    from .research_cycle import _collect

    request = json.loads(Path(request_ref).read_text())
    config, task, role = request["configuration"], request["task"], request["role"]
    root = Path(config["isolation_task_root"]).resolve()
    if os.environ.get("RSI_ISOLATED_ROLE") != role or os.environ.get(
        "RSI_ISOLATION_TASK_ROOT"
    ) != str(root):
        raise PermissionError("role consumer requires its registered OS isolation")
    receipt_path = Path(request["receipt_ref"])
    role_root = (
        Path(config["release_root"]).resolve()
        if role == "release"
        else root / ROLES[role]
    )
    if not receipt_path.is_relative_to(role_root):
        raise ValueError("role receipt escapes its write domain")
    overrides = {}
    if request.get("boundary_factory"):
        if config.get("offline_fixture") is not True:
            raise PermissionError("CPU boundary unavailable in production")
        import importlib

        module, name = request["boundary_factory"].split(":")
        overrides = getattr(importlib.import_module(module), name)(config)
    upgrade_actions = {
        "candidate_deployment_prepare",
        "candidate_evidence_import",
        "candidate_upgrade_qualify",
        "candidate_upgrade_activate",
    }
    if task["action"] in upgrade_actions:
        backend = None
    elif config.get("schema") == "rsi-task-completion-v1":
        from .task_completion import TaskCompletionBackend

        backend = TaskCompletionBackend(config)
    else:
        backend = Protocol10Backend(task["backend_root"], config, **overrides)
    result = {
        "request": deepcopy(request),
        "role": role,
        "pid": os.getpid(),
    }
    try:
        action, op = task["action"], task.get("operation")
        if action in upgrade_actions:
            from .upgrade_bridge import role_action

            value = role_action(config, task)
        elif action in {"execute", "reconcile"}:
            value = getattr(backend, action)(op, task["inputs"])
        elif action == "audit":
            value = backend.audit(task["output"], require_vla=False)
        elif action == "runtime_plan":
            if not config.get("offline_fixture"):
                from .protocol10_admission import validate_dispatch_admission

                validate_dispatch_admission(
                    config,
                    backend.protocol,
                    op,
                    authorization=task["inputs"].get("authorization"),
                )
            backend.runtime_execute(task["plan"])
            value = {"returned": True}
        elif action == "development_dispatch":
            tools = backend._tools()
            value = tools._save(
                "trial_dispatches",
                task["experiment_id"] + "-dispatch",
                {
                    "experiment_id": task["experiment_id"],
                    "output": task["output"],
                    "status": "dispatched",
                },
            )
        elif action == "development_collect":
            from dataclasses import asdict

            derived = root / ROLES[role] / "development" / task["experiment_id"]
            from copy import copy

            from .storage import ResearchStore

            class EvaluationImports(ResearchStore):
                def get(self, kind, key):
                    if self.path(kind, key).exists():
                        return super().get(kind, key)
                    return backend.store.get(kind, key)

            tools = copy(backend._tools())
            tools.store = EvaluationImports(derived / "imports")
            collector = backend.trial_collector
            result_value, checks = (
                collector(tools, task["experiment_id"], task["output"])
                if collector
                else _collect(
                    tools, task["experiment_id"], task["output"], derived_root=derived
                )
            )
            value = {"result": asdict(result_value), "checks": checks}
        elif action == "development_reconcile":
            collection = task["collection"]
            value = backend.trial_reconcile(
                backend._tools(),
                task["experiment_id"],
                output=task["output"],
                permit_path=task["permit_ref"],
                collector=lambda *a: (
                    ExperimentResult(**collection["result"]),
                    collection["checks"],
                ),
                auto_submit=False,
                settle_before_collect=True,
                cost_observer=backend.audit,
            )
        else:
            raise ValueError("unsupported role action")
        result["value"] = value
    except Exception as exc:  # noqa: BLE001 - retain failed role evidence for reconciliation
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["phase_proposals"] = proposals()
    atomic_json(receipt_path, result)
    return 1 if result.get("error") else 0
