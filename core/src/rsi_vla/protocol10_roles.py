"""Host broker for fixed production consumers in separate mount write domains."""

import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

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
    hidden_paths=(),
    read_grants=(),
    request_ref=None,
    write_root=None,
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
    for name, value in {
        "hidden_paths": hidden_paths, "read_grants": read_grants,
        "request_ref": request_ref, "write_root": write_root,
    }.items():
        if value:
            kwargs[name] = value
    return module.command(task_root, role, argv, **kwargs)


def _public_task_subset(public, source):
    """Verify task dictionaries and ordered subsets of complete list members."""
    if isinstance(public, dict) and isinstance(source, dict):
        return public.keys() <= source.keys() and all(
            _public_task_subset(value, source[key]) for key, value in public.items()
        )
    if isinstance(public, list) and isinstance(source, list):
        remaining = iter(source)
        return all(any(json.dumps(item, sort_keys=True) == json.dumps(value, sort_keys=True)
                       for item in remaining) for value in public)
    return type(public) is type(source) and public == source


def role_configuration(configuration, role):
    """Validate reviewed public task inputs before passing them to a role.

    New-task configurations may register ``public_protocol_ref`` and
    ``public_frozen_protocol_ref``. Public ``tasks`` and ``task_sets`` may omit
    dictionary fields and complete list members while retaining their order.
    Other fields, including identity, policy and budget, remain equal to the
    private source. An optional
    ``public_configuration_ref`` supplies an independently reviewed configuration
    without private top-level fields and requires ``private_configuration_ref``
    so the original file can be hidden. This helper never copies private data
    into a generated projection. Without a separate public configuration file,
    ``configuration_is_public=True`` explicitly declares all configuration values
    public; only protocol references are replaced.
    """
    if role not in {"runtime", "research", "candidate"}:
        return configuration
    if not configuration.get("public_protocol_ref"):
        if configuration.get("public_configuration_ref"):
            raise ValueError("public configuration requires registered public protocol references")
        return configuration
    if configuration.get("schema") != "rsi-task-completion-v1":
        raise ValueError("public role protocols require the new-task consumer")
    from .evaluation import load_protocol

    source = load_protocol(configuration["protocol_ref"])
    public_ref = Path(configuration["public_protocol_ref"]).resolve(strict=True)
    public = load_protocol(public_ref)
    frozen = json.loads(Path(configuration["frozen_protocol_ref"]).read_text())
    public_frozen_ref = Path(configuration["public_frozen_protocol_ref"]).resolve(strict=True)
    public_frozen = json.loads(public_frozen_ref.read_text())
    private_refs = {Path(configuration[key]).resolve() for key in ("protocol_ref", "frozen_protocol_ref")}
    if public_ref in private_refs or public_frozen_ref in private_refs:
        raise ValueError("public and private protocol paths must be distinct")
    if source != frozen or public != public_frozen:
        raise ValueError("public role protocol differs from its frozen source or snapshot")
    required = {"protocol_id", "scope_id", "scope", "status", "benchmark_id", "anchor_bundle_id"}
    required |= {key for key in ("task_policy", "budget") if key in source}
    if not required <= public.keys() or any(
        key not in source or (
            not _public_task_subset(public[key], source[key])
            if key in {"tasks", "task_sets"}
            else json.dumps(public[key], sort_keys=True) != json.dumps(source[key], sort_keys=True)
        )
        for key in public
    ):
        raise ValueError("public role protocol changes source identity, policy or fields")
    scope = json.loads(Path(configuration["scope_authorization_ref"]).read_text())
    admin = json.loads(Path(configuration["administrative_authorization_ref"]).read_text())
    if (scope.get("authorized") is not True or admin.get("authorized") is not True
            or scope.get("scope_id") != source["scope_id"]
            or admin.get("scope") != source["scope_id"]
            or admin.get("protocol_id") != source["protocol_id"]):
        raise ValueError("public role protocol lacks matching scope and administrative authority")
    if configuration.get("public_configuration_ref"):
        if not configuration.get("private_configuration_ref"):
            raise ValueError("public configuration requires its hidden private source reference")
        value = json.loads(Path(configuration["public_configuration_ref"]).read_text())
        controls = {
            "schema", "isolation_task_root", "scope_authorization_ref",
            "administrative_authorization_ref", "phase_ref", "source_manifest_ref",
            "operations", "runtime_slots", "research_requests", "offline_fixture",
            "research_store_ref", "evaluation_store_ref", "repository_root",
        }
        for key in controls & configuration.keys():
            if value.get(key) != configuration[key]:
                raise ValueError("public configuration changes a registered role control: " + key)
        for key in value:
            if key not in {"protocol_ref", "frozen_protocol_ref"} and value[key] != configuration.get(key):
                raise ValueError("public configuration changes a registered input: " + key)
        if (Path(value["protocol_ref"]).resolve() != public_ref
                or Path(value["frozen_protocol_ref"]).resolve() != public_frozen_ref):
            raise ValueError("public configuration references a different public protocol")
    else:
        if configuration.get("configuration_is_public") is not True:
            raise ValueError("role projection requires reviewed public configuration inputs")
        value = deepcopy(configuration)
    value["protocol_ref"] = str(public_ref)
    value["frozen_protocol_ref"] = str(public_frozen_ref)
    for key in ("public_protocol_ref", "public_frozen_protocol_ref",
                "public_configuration_ref", "private_configuration_ref"):
        value.pop(key, None)
    return value


def role_isolation(configuration, role):
    """Return host-owned hidden paths and explicit per-role read grants.

    ``role_isolation.hidden_paths`` are hidden from operator, research, runtime
    and candidate. Evaluation and release retain trusted reads. Grants expose
    public projections inside a hidden directory and must contain public data.
    """
    root = Path(configuration["isolation_task_root"]).resolve()
    policy = configuration.get("role_isolation", {})
    hidden = []
    if role not in {"evaluation", "release"}:
        hidden.extend(policy.get("hidden_paths", []))
        if role in {"runtime", "research", "candidate"} and configuration.get("public_protocol_ref"):
            hidden.extend(configuration[key] for key in (
                "protocol_ref", "frozen_protocol_ref", "private_configuration_ref"
            ) if configuration.get(key))
            # Old returns may embed full configurations. Mask their individual
            # inodes while leaving this role's new receipt directory writable.
            hidden.extend(str(path) for path in root.glob("**/role_receipts/*.json"))
            hidden.extend(str(path) for path in (root / "operator/code_candidate_receipts").glob("*.json"))
        for path in (root / "state/evaluation/role_receipts", root / "operator/role_requests"):
            if path.exists():
                hidden.append(str(path))
    if role in {"research", "runtime", "candidate", "operator"}:
        evaluations = Path(configuration.get("evaluation_store_ref", root / "state/evaluation"))
        release = Path(configuration.get("release_root", root / "release"))
        for path in [evaluations / kind for kind in
                     ("meta_exam_scores", "meta_submissions", "meta_attempt_costs")] + [
                         release / "research_policy_qualifications"]:
            if path.exists():
                hidden.append(str(path))
    if role in {"research", "runtime", "candidate"}:
        host = Path(configuration.get("host_root", root / "operator/midterm-host"))
        continuation = Path(configuration.get("continuation_root", root / "operator/continuation"))
        # Raw coordinator returns and Host input provenance may contain private
        # evaluator payloads. Expose only fixed CPU data and cost receipts.
        for path in (host, continuation):
            if path.exists():
                hidden.append(str(path))
        grants = list(policy.get("read_grants", {}).get(role, []))
        if role != "candidate":
            for pattern in ("jobs/*/public-inputs.json", "jobs/*/completion.json", "jobs/*/supervisor.json"):
                grants.extend(str(path) for path in host.glob(pattern))
        for operation in configuration.get("operations", []):
            if operation.get("kind") == "goal_materialization":
                payload = operation["payload"]
                domain = Path(payload["output_root"]) / payload["new_protocol_id"]
                for name in ("protocol.json", "evaluation-materialization.json"):
                    if (domain / name).exists():
                        hidden.append(str(domain / name))
    else:
        grants = list(policy.get("read_grants", {}).get(role, []))
    for directory in (root / "scratch").glob("*"):
        if directory.name != role:
            hidden.append(str(directory))
    return {
        "hidden_paths": tuple(hidden),
        "read_grants": tuple(grants),
    }


def broker_request(request, *, socket_path=None):
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(socket_path or os.environ["RSI_ROLE_BROKER"])
        client.sendall(json.dumps(request).encode() + b"\n")
        data = client.makefile("rb").readline()
    result = json.loads(data)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["result"]



class _ResearchRuleBroker:
    """Serve the bound ψ selection outside the Research mount namespace.

    The child submits only the registered execution identity. Source, bounded
    inputs, grants and timeout are reconstructed from its frozen Job and launch
    record; the socket confers no generic role or command dispatch authority.
    """

    def __init__(self, configuration, task, scratch):
        self.configuration = configuration
        self.task = task
        self.path = Path(scratch) / ("rule-broker-" + uuid4().hex + ".sock")
        self.server = socket.socket(socket.AF_UNIX)
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.server.bind(f"/proc/self/fd/{descriptor}/{self.path.name}")
        finally:
            os.close(descriptor)
        self.server.listen(1)
        self.server.settimeout(0.2)
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.pid = None
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        import struct

        while not self.stopped.is_set():
            try:
                connection, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                try:
                    connection.settimeout(5)
                    if not self.ready.wait(5):
                        raise PermissionError("Research role did not start")
                    peer = struct.unpack("3i", connection.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[0]
                    while peer != self.pid and peer > 1:
                        peer = int(Path(f"/proc/{peer}/stat").read_text()
                                   .rsplit(")", 1)[1].split()[1])
                    if peer != self.pid:
                        raise PermissionError("rule broker requires the active Research child")
                    raw = connection.makefile("rb").readline(8193)
                    if len(raw) > 8192 or not raw.endswith(b"\n"):
                        raise ValueError("rule broker request exceeds its bound")
                    request = json.loads(raw)
                    result = self.execute(request)
                    response = {"result": result}
                except Exception as exc:  # noqa: BLE001 - preserve explicit broker failure
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                try:
                    connection.sendall(json.dumps(response).encode() + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    # A lost receiver does not erase the native candidate receipt.
                    pass

    def execute(self, request):
        from .research_code import _RUNNER, validate_research_rule
        from .task_completion import TaskCompletionBackend

        operation = self.task.get("operation", {})
        if operation.get("kind") not in {"research_prepare", "meta_exam_research"}:
            raise PermissionError("rule broker only serves registered Research preparation")
        request_id = operation["payload"]["request_id"]
        identity = identifier(request_id + "-psi-selection")
        if request != {"action": "inherited_research_rule", "execution_id": identity}:
            raise ValueError("rule broker accepts only the bound selection identity")
        tools, _ = TaskCompletionBackend(self.configuration).tools(request_id)
        if tools.job.state["status"] != "running":
            raise PermissionError("frozen Research Job cannot launch a rule")
        plugin = tools.store.get("psi", tools.base.psi_ref).get("research_rule_plugin")
        if not plugin or plugin.get("schema") != "research-rule-select-action-v1":
            raise ValueError("bound Job has no supported ψ program")
        validate_research_rule(plugin["source"])
        source = Path(plugin["source_ref"]).resolve(strict=True)
        if source.read_text() != plugin["source"]:
            raise ValueError("bound ψ source changed")
        launch = tools._owned("research_code_launches", identity)["request"]
        timeout = launch["timeout_seconds"]
        if (launch["config"] != self.configuration
                or launch["source_ref"] != str(source)
                or type(timeout) not in (int, float) or not 0 < timeout <= 3
                or tools.job.remaining().get("whole_job_wall_seconds", 0) < timeout
                or tools.job.remaining().get("artifact_bytes", 0) < 128 * 1024):
            raise PermissionError("rule launch differs from its bounded Job authority")
        inputs = launch["inputs"]
        if (not isinstance(inputs, dict)
                or set(inputs) != {"evidence_available", "budget_available"}
                or any(type(value) is not bool for value in inputs.values())):
            raise ValueError("rule launch requires bounded boolean facts")
        domain = tools.store.root / "code_rule_artifacts" / identity
        runner = domain / "runner.py"
        context = domain / "context.json"
        if runner.read_text() != _RUNNER or json.loads(context.read_text()) != inputs:
            raise ValueError("rule runner or context changed after registration")
        registered = {**self.configuration, "code_candidate_executions": [{
            "execution_id": identity, "source_ref": str(runner.resolve()),
            "input_refs": [str(source), str(context.resolve())],
            "read_grants": [], "timeout_seconds": timeout,
            "result_contract": "research-rule-select-action-v1",
        }]}
        return run_code_candidate(registered, {
            "action": "code_candidate_execute", "execution_id": identity})

    def close(self):
        self.stopped.set()
        self.ready.set()
        self.server.close()
        self.thread.join(timeout=1)
        self.path.unlink(missing_ok=True)


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
        command = _sandbox_command(root, "operator", argv, **role_isolation(configuration, "operator"))
        marker = command.index("--")
        command[marker:marker] = ["--setenv", "RSI_ROLE_BROKER", env["RSI_ROLE_BROKER"]]
        process = subprocess.Popen(command, env=env)
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


def run_role(config, task, *, boundary_factory=None, resolved_operation=None):
    root = Path(config["isolation_task_root"]).resolve()
    action = task["action"]
    if action == "code_candidate_execute":
        return run_code_candidate(config, task)
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
    public_config = role_configuration(config, role)
    projected_protocols = public_config is not config
    if resolved_operation is not None:
        from .host_inputs import resolved_role_configuration

        public_config = resolved_role_configuration(
            public_config, task["operation"], resolved_operation
        )
        task = {**task, "operation": deepcopy(resolved_operation)}
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

        if saved["request"]["configuration"] != public_config or comparable(
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
        "configuration": public_config,
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
            if config.get("release_root") and task.get("operation", {}).get("kind") != "release_loaded":
                from .release import ReleaseManager
                from .release_safety import registered_publication_paths
                from .evaluation import EvaluationStore
                from .storage import ResearchStore
                manager = ReleaseManager(config["release_root"],
                    ResearchStore(config["research_store_ref"]),
                    EvaluationStore(config["evaluation_store_ref"]))
                protected_paths = (*protected_paths, *registered_publication_paths(manager))
        if task.get("operation", {}).get("kind", "").startswith("release_"):
            from .release_flow import protected_paths as release_protected_paths
            protected_paths = (*protected_paths, *release_protected_paths(config, task))
        if task.get("operation", {}).get("kind", "").startswith("meta_"):
            from .meta_exam import protected_paths as meta_protected_paths
            protected_paths = (*protected_paths, *meta_protected_paths(config, task))
        if projected_protocols:
            protected_paths = (*protected_paths, public_config["protocol_ref"],
                               public_config["frozen_protocol_ref"])
        command = _sandbox_command(
            root, role, argv,
            gpu=role == "runtime" and not config.get("offline_fixture"),
            scratch_root=scratch_root,
            release_root=config.get("release_root") if role == "release" else None,
            protected_paths=protected_paths, request_ref=request_path,
            **role_isolation(config, role),
        )
        broker = None
        if (role == "research" and config.get("schema") == "rsi-task-completion-v1"
                and task.get("operation", {}).get("kind") in {"research_prepare", "meta_exam_research"}):
            scratch = Path(scratch_root) if scratch_root else root / "scratch/research"
            broker = _ResearchRuleBroker(public_config, task, scratch)
            marker = command.index("--")
            command[marker:marker] = ["--setenv", "RSI_RESEARCH_RULE_BROKER",
                                     "/tmp/" + broker.path.name]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            if broker:
                broker.pid = process.pid
                broker.ready.set()
            stdout, stderr = process.communicate()
            returned = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            if broker:
                broker.close()
        if not receipt_path.exists():
            raise RuntimeError(
                "role failed without receipt: " + returned.stderr[-2000:]
            )
        result = json.loads(receipt_path.read_text())
    if result.get("role") != role or result.get("request") != request:
        raise ValueError("role receipt identity mismatch")
    return result


def run_code_candidate(config, task):
    """Execute one host-registered Python candidate without host command authority.

    Entries in ``code_candidate_executions`` register ``execution_id``, a single
    ``source_ref`` file, ``input_refs``, optional dependency ``read_grants``, and
    ``timeout_seconds``. Output is fixed to the task's candidate execution domain.
    The host writes the return receipt outside the candidate's mounted output.
    """
    if set(task) != {"action", "execution_id"}:
        raise ValueError("code candidate accepts only a registered execution identity")
    identity = identifier(task["execution_id"])
    matches = [entry for entry in config.get("code_candidate_executions", [])
               if entry["execution_id"] == identity]
    if len(matches) != 1:
        raise ValueError("code candidate execution is not uniquely registered")
    entry = matches[0]
    timeout = entry["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("code candidate requires a positive registered timeout")
    root = Path(config["isolation_task_root"]).resolve()
    domain = root / "state/research/code_candidates" / identity
    output = domain / "output"
    if output.resolve() != output:
        raise ValueError("candidate output escapes its registered path through a symlink")
    output.mkdir(parents=True, exist_ok=True)
    receipt_ref = root / "operator/code_candidate_receipts" / (identity + ".json")
    request = {"configuration": config, "task": task}
    if receipt_ref.exists():
        saved = json.loads(receipt_ref.read_text())
        if saved["request"] != request:
            raise ValueError("conflicting code candidate execution identity")
        return saved
    source = Path(entry["source_ref"]).resolve(strict=True)
    if not source.is_file() or source.suffix != ".py":
        raise ValueError("code candidate requires a registered Python source file")
    inputs = [Path(value).resolve(strict=True) for value in entry.get("input_refs", [])]
    policy = role_isolation(config, "candidate")
    policy["read_grants"] = (*policy["read_grants"], source, *inputs,
                             *entry.get("read_grants", []))
    # A virtualenv Python resolves outside /usr. Its interpreter and prefix are
    # explicit dependencies; site packages still require a registered read grant.
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_relative_to(Path("/usr")):
        policy["read_grants"] = (*policy["read_grants"], executable, sys.base_prefix)
    argv = [str(executable), "-I", str(source), "--output", str(output)]
    for path in inputs:
        argv.extend(["--input", str(path)])
    marker = receipt_ref.with_suffix(".started.json")
    if marker.exists():
        raise RuntimeError("candidate launch lacks a return receipt; reconcile before resubmission")
    command = _sandbox_command(root, "candidate", argv, write_root=output,
                               scratch_root=root / "scratch/candidate" / identity,
                               release_root=config.get("release_root"), **policy)
    from time import monotonic

    from .process_accounting import StorageMeter, snapshot

    scratch = root / "scratch/candidate" / identity
    meter = StorageMeter([output, scratch])
    began = monotonic()
    atomic_json(marker, request)
    result = {"request": request, "role": "candidate", "output_ref": str(output)}
    try:
        returned = subprocess.run(command, capture_output=True, text=True,
                                  check=False, timeout=timeout)
        result["value"] = {"returncode": returned.returncode,
                           "stdout": returned.stdout, "stderr": returned.stderr}
        if returned.returncode:
            result["error"] = "code candidate exited unsuccessfully"
    except subprocess.TimeoutExpired:
        result["error"] = "code candidate exceeded its registered timeout"
    if entry.get("result_contract") == "research-rule-select-action-v1" and not result.get("error"):
        selection = output / "selection.json"
        if selection.is_file() and selection.stat().st_size < 1024:
            value = json.loads(selection.read_text())
            if (set(value) == {"action"} and value["action"] in
                    {"inspect_evidence", "bounded_experiment", "defer"}):
                result["selection_snapshot"] = value
            else:
                result["error"] = "candidate returned an invalid research action"
    result["wall_seconds"] = monotonic() - began
    result["storage"] = meter.sample()
    result["persistent_current_bytes"] = sum(snapshot([output]).values())
    result["temporary_current_bytes"] = sum(snapshot([scratch]).values())
    atomic_json(receipt_ref, result)
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
