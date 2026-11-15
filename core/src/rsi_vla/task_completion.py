# ruff: noqa: UP017, FURB162 - Python 3.10 support
"""New-task role consumers over the existing Runtime, Research and release APIs.

The operator owns the registered sequence. This module does not issue permits,
select model output, reopen protocol10, or change the phase authorization.
"""

import json
import math
from copy import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from rsi_vla.contracts import Candidate, ExperimentResult
from rsi_vla.evaluation import EvaluationStore, Gate, load_protocol
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import BoundedResearchAdapter, ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json, identifier

SCHEMA = "rsi-task-completion-v1"
ACTION_ROLES = {
    "runtime": "runtime",
    "research_prepare": "research",
    "research_feedback": "research",
    "evidence_followup_prepare": "research",
    "research": "research",
    "development_register": "research",
    "development_collect": "evaluation",
    "development_reconcile": "research",
    "development_base_collect": "evaluation",
    "development_base_import": "research",
    "development_reparse": "evaluation",
    "candidate_submit": "research",
    "audit": "evaluation",
    "matrix_evaluation": "evaluation",
    "matrix_reparse": "evaluation",
    "gate": "evaluation",
    "gate_reconsider": "evaluation",
    "goal_value_review": "evaluation",
}


class TaskResearchAdapter(BoundedResearchAdapter):
    """Keep the bounded schema and register one or two permitted development slots."""

    def __init__(self, request):
        super().__init__()
        self.request = request

    def response_schema(self, request_id, *, tools=None):
        schema = super().response_schema(request_id, tools=tools)
        if tools is not None:
            request = tools.store.get("bounded_requests", request_id)
            allowed = set(request["allowed_operators"])
            schema["properties"]["intervention"]["anyOf"] = [
                item
                for item in schema["properties"]["intervention"]["anyOf"]
                if item.get("type") == "null"
                or set(item.get("properties", {})) <= allowed
            ]
            if request.get("research_mode") == "exploratory":
                schema["properties"]["source"] = {
                    "type": "string",
                    "const": "exploration",
                }
        schema["properties"]["development_slots"] = {
            "type": "array",
            "minItems": 0,
            "maxItems": 2,
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task_key": {"type": "string", "const": row["task_key"]},
                            "seed": {"type": "integer", "const": row["seed"]},
                        },
                        "required": ["task_key", "seed"],
                    }
                    for row in self.request["development_slots"]
                ]
            },
        }
        if not self.request["development_slots"]:
            schema["properties"]["development_slots"].update(
                maxItems=0, items={"type": "object", "additionalProperties": False,
                                  "properties": {}, "required": []})
        schema["required"].append("development_slots")
        return schema

    def apply_response(
        self, tools, response, *, evidence_level, cost_limit, synthetic=False
    ):
        slots = response.get("development_slots")
        if not isinstance(slots, list) or any(
            row not in self.request["development_slots"] for row in slots
        ):
            raise ValueError("Research selected an unregistered development slot")
        identities = {(row["task_key"], row["seed"]) for row in slots}
        if len(identities) != len(slots) or len(slots) > 2:
            raise ValueError(
                "Research development slots must be unique and at most two"
            )
        if response.get("decision") == "prepare_rule_trial" and not slots:
            raise ValueError("Research trial requires a registered development slot")
        if response.get("decision") in {"defer", "propose_goal"} and slots:
            raise ValueError("deferred Research cannot dispatch development slots")
        proposal = {k: v for k, v in response.items() if k != "development_slots"}
        trial = super().apply_response(
            tools,
            proposal,
            evidence_level=evidence_level,
            cost_limit=cost_limit,
            synthetic=synthetic,
            reserved_bundle_id=self.request["reserved_bundle_id"],
            development_states=[row["task_key"] for row in slots],
        )
        tools._save(
            "task_completion_selections",
            response["request_id"],
            {
                "request_id": response["request_id"],
                "development_slots": [] if trial.get("action") == "defer" else slots,
                "reserved_bundle_id": self.request["reserved_bundle_id"],
            },
        )
        return trial


def read(ref):
    return json.loads(Path(ref).read_text())


def recovery_choice(attempts, *, max_recoveries=4):
    """Return the next actual technical fault in original attempt order.

    Only original faults are eligible, once each. A valid benchmark failure is
    retained. Unresolved cost, cleanup or cause blocks its own recovery rather
    than authorizing a new experiment. Recoveries are created after this choice.
    """
    originals = [a for a in attempts if not a.get("recovery_of_run_id")]
    recovered = {
        a["recovery_of_run_id"] for a in attempts if a.get("recovery_of_run_id")
    }
    if len(recovered) >= max_recoveries:
        return None
    for attempt in originals:
        if attempt["run_id"] in recovered or attempt.get("validity") == "valid":
            continue
        if attempt.get("failure_class") != "technical":
            continue
        if not all(
            attempt.get(k) is True
            for k in ("cost_closed", "cleanup_complete", "recovery_cause_resolved")
        ):
            return {"state": "waiting_fault_closure", "run_id": attempt["run_id"]}
        return {"state": "recovery_eligible", "run_id": attempt["run_id"]}
    return None


def selected_manifest(slots, attempts):
    """Keep the fixed denominator and select first valid outcomes, including failure."""
    selected = []
    for slot in slots:
        choices = [a for a in attempts if a.get("slot_id") == slot["slot_id"]]
        valid = [a for a in choices if a.get("validity") == "valid"]
        if valid:
            chosen = valid[0]
            context = read(Path(chosen["output_ref"]) / "runtime_context.json")
            selected.append(
                {
                    "task_key": slot["task_key"],
                    "seed": slot["seed"],
                    "run_id": chosen["run_id"],
                    "episode_id": context["episode_id"],
                    "result_path": str(
                        Path(chosen["output_ref"]) / "independent_result.json"
                    ),
                    "action_events_path": str(
                        Path(chosen["output_ref"]) / "action_events.jsonl"
                    ),
                    "costs_path": str(
                        Path(chosen["output_ref"]) / "runtime_costs.json"
                    ),
                    "supervisor_path": str(
                        Path(chosen["output_ref"]) / "supervisor.json"
                    ),
                }
            )
    return selected


class TaskCompletionBackend:
    """Execute a registered operation in its existing isolated role domain.

    Config contains protocol_ref, profile_ref, repository_root, artifacts_root,
    research_store_ref, evaluation_store_ref, research_requests, candidates,
    runtime_slots and operations. Entries are immutable operator preparation;
    each operation has operation_id/kind/payload and the registered references.
    New Research requests share an existing job, with distinct request/permit/
    output identities. The host still gates budgets, expiry, pause and resources.
    """

    def __init__(self, config):
        from rsi_vla.task_policy import task_policy

        if config.get("schema") != SCHEMA:
            raise ValueError("new-task consumer requires its own configuration")
        self.config = config
        self.protocol = load_protocol(config["protocol_ref"])
        if not task_policy(self.protocol):
            raise ValueError("new-task consumer requires explicit new task policy")
        self.store = ResearchStore(config["research_store_ref"], llm_archive_format=config.get("llm_archive_format"), archive_record_kinds=config.get("research_archive_record_kinds", ("llm_requests",)))
        self.evaluations = EvaluationStore(config["evaluation_store_ref"])

    def registered(self, op):
        matches = [
            item
            for item in self.config["operations"]
            if item["operation_id"] == op["operation_id"]
        ]
        if matches != [op]:
            raise ValueError("operation differs from registered fixed consumer input")
        return op["kind"], op["payload"]

    def tools(self, request_id):
        cfg = next(
            r for r in self.config["research_requests"] if r["request_id"] == request_id
        )
        job = ResearchJob(
            self.store,
            cfg["job_id"],
            cfg.get("base_bundle_id", self.config.get("base_bundle_id")),
            cfg.get("job_limits"),
        )
        tools = ResearchTools(
            self.store,
            job,
            self.protocol,
            domain_policy=read(cfg["domain_policy_ref"]),
        )
        return tools, cfg

    def preserve_development_views(self, results):
        """Materialize exact independently imported views for ordinary consumers.

        Older development findings reference evaluation-owned views rather than
        copies in ResearchStore. Preserve their identities and complete facts;
        never reinterpret a formal view as development input.
        """
        allowed = (self.evaluations.root / "development").resolve()
        for result in results:
            view_id = result.get("intervention", {}).get("evidence_view_id")
            if not view_id or self.store.path("evidence_views", view_id).exists():
                continue
            if result.get("role") != "development" or result["intervention"].get("information_scope") != "development":
                raise ValueError("only independent development views may be materialized")
            refs = {Path(ref).resolve() for ref in result["data_refs"]
                    if Path(ref).name == view_id + ".json"}
            if len(refs) != 1 or not next(iter(refs)).is_relative_to(allowed):
                raise ValueError("development view reference is outside its independent import domain")
            view = read(next(iter(refs)))
            identity = result["intervention"].get("identity", {})
            if view.get("view_id") != view_id or view.get("bundle_id") != result["bundle_id"] or any(
                event["active_bundle"] != result["bundle_id"]
                or any(event.get(key) != identity[key] for key in ("run_id", "episode_id") if key in identity)
                for event in view["l0"]):
                raise ValueError("development view identity differs from the original result")
            self.store.put("evidence_views", view_id, view)

    def authority(self):
        now = datetime.now(timezone.utc)
        scope = read(self.config["scope_authorization_ref"])
        admin = read(self.config["administrative_authorization_ref"])
        for value in (scope, admin):
            expires = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
            if (
                expires.tzinfo is None
                or now >= expires
                or value.get("authorized") is not True
            ):
                raise PermissionError("new-task authorization missing or expired")
        if (
            scope["scope_id"] != self.protocol["scope_id"]
            or admin["scope"] != scope["scope_id"]
            or admin["protocol_id"] != self.protocol["protocol_id"]
            or Path(admin["phase_snapshot_ref"]).resolve()
            != Path(self.config["phase_ref"]).resolve()
            or Path(admin["approval_reference"]).resolve()
            != Path(self.config["scope_authorization_ref"]).resolve()
            or read(self.config["frozen_protocol_ref"]) != self.protocol
        ):
            raise PermissionError(
                "new-task authorization/protocol/phase binding mismatch"
            )
        return scope, admin

    def validate_dynamic(self, kind, payload):
        """Recheck real authority, permit, phase and source in the consuming role."""
        scope, admin = self.authority()
        if (
            scope.get("dynamic_authorized") is not True
            or admin.get("dynamic_authorized") is not True
        ):
            raise PermissionError("new-task dynamic authorization missing")
        cfg = (
            next(
                r
                for r in self.config["research_requests"]
                if r["request_id"] == payload["request_id"]
            )
            if kind == "research"
            else next(
                s
                for s in self.config["runtime_slots"]
                if s["run_id"] == payload["run_id"]
            )
        )
        permit = read(cfg["permit_ref"])
        expires = datetime.fromisoformat(permit["expires_at"].replace("Z", "+00:00"))
        if expires.tzinfo is None or datetime.now(timezone.utc) >= expires:
            raise PermissionError("single-operation permit expired")
        expected = {
            "authorized": True,
            "scope": scope["scope_id"],
            "protocol_id": self.protocol["protocol_id"],
            "phase_snapshot_ref": self.config["phase_ref"],
            "administrative_authorization_ref": self.config[
                "administrative_authorization_ref"
            ],
            "source_manifest_ref": self.config["source_manifest_ref"],
            "output_dir": cfg["output_ref"],
            "bundle_id": cfg.get("base_bundle_id", self.config["base_bundle_id"])
            if kind == "research"
            else cfg["bundle_id"],
        }
        if kind == "research":
            expected.update(
                job_id=cfg["job_id"],
                request_id=cfg["request_id"],
                **scope["models"]["research"],
            )
            cap = self.protocol["task_policy"]["research_per_request"]
            cap = {**cap, "model_requests": 1}
        else:
            expected.update(
                run_id=cfg["run_id"],
                task_key=cfg["task_key"],
                seed=cfg["seed"],
                simulation_episodes=1,
                gpu_devices=[cfg["cuda_device"]],
            )
            cap = {
                key: value
                for key, value in self.protocol["budget"]["runtime_per_start"].items()
                if key != "simulation_episodes"
            }
            cap["planner_seconds"] = self.protocol["budget"]["per_attempt_proposal"][
                "planner_timeout_seconds"
            ]
        if any(permit.get(key) != value for key, value in expected.items()):
            raise PermissionError(
                "single-operation permit identity or source binding mismatch"
            )
        output, scratch = (
            Path(cfg[key]).resolve() for key in ("output_ref", "scratch_ref")
        )
        if (
            output == scratch
            or output.is_relative_to(scratch)
            or scratch.is_relative_to(output)
            or permit.get("storage_write_roots") != [str(output), str(scratch)]
        ):
            raise PermissionError(
                "single-operation write domains differ from registration"
            )
        limits = permit.get("limits", {})
        required = (
            {"model_requests", "wall_seconds", "storage_bytes"}
            if kind == "research"
            else set(cap)
        )
        if not required <= limits.keys() or any(
            key not in cap
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            or value > cap[key]
            for key, value in limits.items()
        ):
            raise PermissionError("single-operation permit exceeds registered limits")
        phase = read(self.config["phase_ref"])
        if phase.get("unresolved_costs") or any(
            value > phase["remaining"].get(key, -1)
            for key, value in limits.items()
            if key in phase["limits"]
        ):
            raise PermissionError("scope remaining resources cannot admit operation")
        if kind == "runtime" and phase["remaining"].get("simulation_episodes", 0) < 1:
            raise PermissionError("scope Runtime budget exhausted")
        started = sum(
            (Path(slot["output_ref"]) / "runtime_context.json").exists()
            for slot in self.config["runtime_slots"]
        )
        if (
            kind == "runtime"
            and started >= scope["dynamic_limits"]["simulation_episodes"]
        ):
            raise PermissionError("scope Runtime start cap exhausted")
        recovery_of = cfg.get("recovery_of_run_id")
        if kind == "runtime" and recovery_of:
            recoveries = [
                slot
                for slot in self.config["runtime_slots"]
                if slot.get("recovery_of_run_id")
                and (Path(slot["output_ref"]) / "runtime_context.json").exists()
            ]
            if len(recoveries) >= scope["runtime_technical_recovery_max"] or any(
                slot["recovery_of_run_id"] == recovery_of for slot in recoveries
            ):
                raise PermissionError("scope technical recovery cap exhausted")
            basis = read(cfg["recovery_basis_ref"])
            if (
                basis.get("run_id") != recovery_of
                or basis.get("validity") == "valid"
                or basis.get("failure_class") != "technical"
                or not all(
                    basis.get(key) is True
                    for key in (
                        "cost_closed",
                        "cleanup_complete",
                        "recovery_cause_resolved",
                    )
                )
            ):
                raise PermissionError(
                    "recovery requires the closed original technical fault"
                )
        requests = sum(
            entry["costs"].get("model_requests", 0)
            for entry in phase.get("research_charges", [])
        )
        if kind == "research" and requests >= scope["research_requests_max"]:
            raise PermissionError("scope Research request cap exhausted")
        manifest_ref = Path(self.config["source_manifest_ref"])
        manifest = read(manifest_ref)
        if not manifest.get("files"):
            raise PermissionError("reviewed source manifest required")
        for name in manifest["files"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("source manifest path escapes repository")
            if (Path(self.config["repository_root"]) / relative).read_bytes() != (
                manifest_ref.parent / "source" / relative
            ).read_bytes():
                raise PermissionError("reviewed source changed: " + name)
        return permit

    def execute(self, op, inputs):
        kind, payload = self.registered(op)
        if kind == "development_reparse":
            self.authority()
            from rsi_vla.development_reparse import reparse_development_attempt
            return reparse_development_attempt(self.config, payload)
        if kind in {"runtime", "research"}:
            self.validate_dynamic(kind, payload)
        if kind == "goal_value_review":
            import os

            from rsi_vla.task_policy import review_goal_value

            # The Research role proposes a target; only the independent
            # evaluation role consumes registered policy and measurements.
            if os.environ.get("RSI_ISOLATED_ROLE") != "evaluation":
                raise PermissionError("goal value requires the independent evaluation role")
            self.authority()
            root = Path(self.config["isolation_task_root"]).resolve(strict=True)
            policy_ref = Path(payload["policy_ref"]).resolve(strict=True)
            evidence_ref = Path(payload["evidence_ref"]).resolve()
            if not policy_ref.is_relative_to(root / "operator"):
                raise PermissionError("goal value policy must be operator-owned")
            if not evidence_ref.is_relative_to(self.evaluations.root.resolve()):
                raise PermissionError("goal value evidence must be independently evaluated")
            policy = read(policy_ref)
            if policy.get("policy_ref") != str(policy_ref):
                raise ValueError("goal value policy reference mismatch")
            opportunity = self.store.get("opportunities", payload["opportunity_id"])
            if payload.get("generate_measurement") is True:
                from rsi_vla.goal_value import evaluate_goal_value
                review_id = identifier(payload["review_id"])
                expected = self.evaluations.path("goal_value_measurements", review_id).resolve()
                if evidence_ref != expected:
                    raise PermissionError("goal value generation must use its registered evaluation identity")
                measurement = evaluate_goal_value(self.protocol, opportunity, policy=policy)
                self.evaluations.put("goal_value_measurements", review_id, measurement)
            evidence = read(evidence_ref)
            for ref in evidence.get("evidence_refs", []):
                if not Path(ref).is_file():
                    raise ValueError("goal value measurement source is missing")
            value = review_goal_value(self.protocol, opportunity, evidence, policy=policy)
            value.update(policy_source_ref=str(policy_ref), measurement_ref=str(evidence_ref))
            review_id = identifier(payload["review_id"])
            self.evaluations.put("goal_value_reviews", review_id, value)
            return {"review": value, "review_ref": str(
                self.evaluations.path("goal_value_reviews", review_id).resolve())}
        if kind == "runtime":
            from rsi_vla.runtime import execute_runtime, prepare_runtime

            slot = next(
                s
                for s in self.config["runtime_slots"]
                if s["run_id"] == payload["run_id"]
            )
            if slot["split"] == "deployment":
                from rsi_vla.runtime import main as runtime_main

                argv = [
                    "--root",
                    self.config["repository_root"],
                    "--artifacts",
                    self.config["artifacts_root"],
                    "--protocol",
                    self.config["protocol_ref"],
                    "--deployment-release",
                    self.config["release_root"],
                    "--run",
                    slot["run_id"],
                    "--task-key",
                    slot["task_key"],
                    "--seed",
                    str(slot["seed"]),
                    "--split",
                    "deployment",
                    "--output",
                    slot["output_ref"],
                    "--authorization",
                    slot["permit_ref"],
                    "--cuda-device",
                    str(slot["cuda_device"]),
                    "--storage-write-root",
                    slot["scratch_ref"],
                    "--execute",
                ]
                code = runtime_main(argv)
                context = read(Path(slot["output_ref"]) / "runtime_context.json")
                return {
                    "returned": True,
                    "run_id": slot["run_id"],
                    "runtime_exit_code": code,
                    "selection_argv": argv,
                    "selection": context["selection"],
                    "release_binding": context["release_binding"],
                }
            plan = prepare_runtime(
                self.config["repository_root"],
                self.config["artifacts_root"],
                protocol_path=self.config["protocol_ref"],
                run_id=slot["run_id"],
                bundle_id=slot["bundle_id"],
                task_key=slot["task_key"],
                seed=slot["seed"],
                split=slot["split"],
                output_dir=slot["output_ref"],
                authorization_path=slot["permit_ref"],
                cuda_device=slot["cuda_device"],
                storage_write_roots=[slot["scratch_ref"]],
            )
            request_id = slot.get("research_request_id")
            if request_id:
                from rsi_vla.contracts import RunBundle
                policy_tools, request = self.tools(request_id)
                policy_tools = copy(policy_tools)
                policy_tools.base = RunBundle(**self.store.get("bundles", slot["bundle_id"]))
                admission = policy_tools.research_relation_admission(
                    request["result_ids"], states=[slot["task_key"]],
                    legal_repairs=["relation_feedback_update"])
                if admission["action"] != "prepare":
                    return {"returned": False, "action": "defer", "run_id": slot["run_id"],
                            "relation_admission": admission}
                plan["context"]["relation_dispatch_admission"] = {
                    **admission, "actual_dispatch_gate": True,
                    "source_result_ids": request["result_ids"],
                    "job_state_before_dispatch": policy_tools.job.state}
            execute_runtime(plan)
            return {"returned": True, "run_id": slot["run_id"]}
        if kind == "research_feedback":
            tools, cfg = self.tools(payload["request_id"])
            execution = self.store.get("llm_executions", cfg["request_id"])
            facts = execution["transport_facts"]
            output = Path(cfg["output_ref"])
            supervisor = read(output / "supervisor.json")
            sdk = read(output / "sdk-result.json")
            technical = (
                facts.get("effective_model") == "gpt-6.1-sol"
                and facts.get("effective_reasoning_effort") == "high"
                and supervisor.get("status") == "completed"
                and supervisor.get("exit_code") == 0
                and supervisor.get("cleanup_confirmed") is True
                and sdk["status"].rsplit(".", 1)[-1] == "completed"
            )
            result_id = payload.get("result_id")
            feedback = tools.record_contextual_research_feedback(
                payload["feedback_id"], selection_id=cfg["request_id"],
                result_id=result_id,
                technical_valid=None if result_id else technical,
                costs=None if result_id else execution["costs"],
                source_refs=[str(output / name) for name in (
                    "sdk-result.json", "supervisor.json", "model_requests.jsonl")])
            trial_ref = self.store.path("rule_trials", cfg["request_id"] + "-experiment")
            supplement_ref = self.store.path("research_evidence_supplements", cfg["request_id"])
            if result_id:
                next_action = "experiment_completed"
            elif trial_ref.exists():
                next_action = "development"
            elif supplement_ref.exists() and read(supplement_ref)["reads"]:
                next_action = "prepare_evidence_followup"
            else:
                next_action = "defer"
            route = {"request_id": cfg["request_id"], "feedback_id": payload["feedback_id"],
                     "next_action": next_action, "result_id": result_id}
            self.store.put("research_feedback_routes", payload["feedback_id"], route)
            return {**feedback, **route}
        if kind == "evidence_followup_prepare":
            tools, cfg = self.tools(payload["request_id"])
            parent = payload["parent_request_id"]
            transport = TaskResearchAdapter(cfg).prepare_evidence_followup(
                tools, parent_request_id=parent, request_id=cfg["request_id"],
                max_input_tokens=cfg["max_input_tokens"],
                max_output_tokens=cfg["max_output_tokens"])
            if self.config.get('research_receipt_mode') == 'reference_v1':
                return {'request_id': cfg['request_id'], 'transport_ref': str(tools.store.path('bounded_transports', cfg['request_id']).resolve())}
            return transport
        if kind in {"research_prepare", "research"}:
            tools, cfg = self.tools(payload["request_id"])
            adapter = TaskResearchAdapter(cfg)
            if kind == "research_prepare":
                from rsi_vla.research_preflight import operator_preflight

                self.preserve_development_views(tools.query_development(cfg["result_ids"]))
                preflight = operator_preflight(
                    tools,
                    self.config["repository_root"],
                    protocol_path=self.config["protocol_ref"],
                    result_ids=cfg["result_ids"],
                )
                goal_only = tools.domain_policy.get("opportunity_only") is True
                if not goal_only and not any(r["currently_trialable"] for r in preflight):
                    return {
                        "action": "no_legal_operator",
                        "request_count": 0,
                        "preflight": preflight,
                    }
                feasible = {
                    row["operator"] for row in preflight if row["currently_trialable"]
                }
                tools.domain_policy = {
                    **tools.domain_policy,
                    "rule_operators": [
                        operator
                        for operator in tools.domain_policy["rule_operators"]
                        if operator in feasible
                    ],
                }
                if cfg.get("relation_operator"):
                    operator = cfg["relation_operator"]
                    if operator not in tools.domain_policy["rule_operators"]:
                        raise PermissionError("contextual selection requires a feasible registered operator")
                    gates = tools.domain_policy.get("relation_admission", {})
                    selection = tools.research_relation_admission(
                        cfg["result_ids"], legal_repairs=[operator],
                        selection_id=cfg["request_id"],
                        technical_admitted=gates.get("technical_admitted") is True,
                        data_admitted=gates.get("data_admitted") is True)
                    if selection["action"] != "prepare":
                        return {"action": "defer", "request_count": 0, "relation_admission": selection}
                transport = adapter.prepare_request(
                    tools,
                    request_id=cfg["request_id"],
                    result_ids=cfg["result_ids"],
                    max_input_tokens=cfg["max_input_tokens"],
                    max_output_tokens=cfg["max_output_tokens"],
                )
                if self.config.get('research_receipt_mode') == 'reference_v1':
                    tools.store.put('research_preflights', cfg['request_id'], preflight)
                    return {'request_id': cfg['request_id'], 'transport_ref': str(tools.store.path('bounded_transports', cfg['request_id']).resolve()),
                            'preflight_ref': str(tools.store.path('research_preflights', cfg['request_id']).resolve())}
                return {"transport": transport, "preflight": preflight}
            from rsi_vla.bounded_runner import execute

            return execute(
                tools,
                adapter,
                self.store.get("bounded_transports", cfg["request_id"]),
                root=self.config["repository_root"],
                output=cfg["output_ref"],
                permit_path=cfg["permit_ref"],
                preflight=read(cfg["preflight_ref"]),
                cost_limit=cfg["cost_limit"],
            )
        if kind == "audit":
            from rsi_vla.adapters.runtime_acceptance import inspect_attempt
            from rsi_vla.critical_progress import inspect_progress
            from rsi_vla.runtime_cleanup import inspect_cleanup

            audit = inspect_attempt(payload["output_ref"], require_vla=False)
            return {
                "audit": audit,
                "progress": inspect_progress(payload["output_ref"], audit["identity"]),
                "cleanup": inspect_cleanup(payload["output_ref"], audit["identity"]),
            }
        if kind == "development_register":
            tools, _ = self.tools(payload["request_id"])
            original = payload["source_experiment_id"]
            experiment_id = payload["experiment_id"]
            if experiment_id != original:
                for table in ("research_experiments", "rule_trials"):
                    record = tools._owned(table, original)
                    tools._save(
                        table, experiment_id, {**record, "experiment_id": experiment_id}
                    )
            tools._save(
                "trial_dispatches",
                experiment_id + "-dispatch",
                {
                    "experiment_id": experiment_id,
                    "output": payload["output_ref"],
                    "status": "dispatched",
                },
            )
            return {"registered": True}
        if kind == "development_base_collect":
            from rsi_vla.adapters.runtime_acceptance import inspect_attempt

            audit = inspect_attempt(payload["output_ref"], require_vla=True,
                                    evaluation_ref=payload.get("evaluation_revision_ref"),
                                    evaluation_store_ref=self.evaluations.root if payload.get("evaluation_revision_ref") else None)
            atomic_json(payload["audit_ref"], audit)
            if payload.get("relation_evidence_ref"):
                from rsi_vla.relation_evidence import review_skill_feedback
                target = Path(payload["relation_evidence_ref"]).resolve()
                if not target.is_relative_to(self.evaluations.root.resolve()):
                    raise PermissionError("skill feedback must be independently evaluated")
                atomic_json(target, review_skill_feedback(payload["output_ref"]))
            return audit
        if kind == "development_base_import":
            from rsi_vla.development import import_development

            result = import_development(
                self.store,
                self.protocol,
                payload["output_ref"],
                result_id=payload["result_id"],
                audit_path=payload["audit_ref"],
                evaluation_ref=payload.get("evaluation_revision_ref"),
                evaluation_store_ref=self.evaluations.root if payload.get("evaluation_revision_ref") else None,
            )
            if payload.get("relation_evidence_ref"):
                from rsi_vla.relation_evidence import attach_skill_feedback
                source_ref = Path(payload["relation_evidence_ref"]).resolve(strict=True)
                if not source_ref.is_relative_to(self.evaluations.root.resolve()):
                    raise PermissionError("skill feedback must come from the evaluation domain")
                result = attach_skill_feedback(self.store, result, read(source_ref), str(source_ref))
            return asdict(result)
        if kind == "development_collect":
            from rsi_vla.research_cycle import _collect

            tools, _ = self.tools(payload["request_id"])
            derived = (
                Path(self.config["evaluation_store_ref"])
                / "development"
                / payload["experiment_id"]
            )
            source = tools.store
            trial = tools._owned("rule_trials", payload["experiment_id"])
            if trial.get("rho_ref"):
                from rsi_vla.relation_consumption import evaluate_relation_consumption
                consumption = evaluate_relation_consumption(source, payload["output_ref"],
                    research_request_ids=payload.get("relation_research_request_ids", []))
                atomic_json(derived / "relation-consumption.json", consumption)

            class EvaluationImports(ResearchStore):
                def get(self, table, key):
                    return (
                        super().get(table, key)
                        if self.path(table, key).exists()
                        else source.get(table, key)
                    )

            tools = copy(tools)
            tools.store = EvaluationImports(derived / "imports")
            result, checks = _collect(
                tools,
                payload["experiment_id"],
                payload["output_ref"],
                derived_root=derived,
            )
            value = {"result": asdict(result), "checks": checks}
            atomic_json(payload["collection_ref"], value)
            return value
        if kind == "development_reconcile":
            from rsi_vla.research_cycle import reconcile_trial

            tools, _ = self.tools(payload["request_id"])
            if payload.get("costs_only") is True:
                from rsi_vla.research_cycle import _settle_trial_output
                context = read(Path(payload["output_ref"]) / "runtime_context.json")
                costs = _settle_trial_output(tools, payload["experiment_id"],
                    payload["output_ref"], payload["permit_ref"], context)
                reason = "experiment " + payload["experiment_id"]
                plan = tools._owned("research_experiments", payload["experiment_id"])
                hard = tools._hard_costs(plan, costs)
                prepaid = [row for row in tools.job.state["charges"] if row["reason"] == reason]
                if prepaid:
                    if len(prepaid) != 1 or prepaid[0]["costs"] != hard:
                        raise ValueError("development prepaid cost differs from the actual receipt")
                else:
                    tools.job.consume(hard, reason)
                return {"costs_settled": True, "costs": costs, "result_collected": False}
            collection = read(payload["collection_ref"])
            reconciled = reconcile_trial(
                tools,
                payload["experiment_id"],
                output=payload["output_ref"],
                permit_path=payload["permit_ref"],
                auto_submit=False,
                settle_before_collect=True,
                collector=lambda *a: (
                    ExperimentResult(**collection["result"]),
                    collection["checks"],
                ),
            )
            self.preserve_development_views([collection["result"]])
            return reconciled
        if kind == "candidate_submit":
            scope, _ = self.authority()
            existing = {item["candidate_id"] for item in self.store.all("candidates")
                        if item["protocol_id"] in scope.get("candidate_protocol_ids", [self.protocol["protocol_id"]])}
            if (
                payload["candidate_id"] not in existing
                and len(existing) >= scope["candidate_max"]
            ):
                raise PermissionError("scope Candidate cap exhausted")
            tools, _ = self.tools(payload["request_id"])
            return tools.submit_candidate(
                payload["candidate_id"],
                experiment_id=payload["experiment_ids"][0],
                additional_experiment_ids=payload["experiment_ids"][1:],
                cost_snapshot_ref=payload["cost_snapshot_ref"],
            )
        if kind in {"matrix_evaluation", "matrix_reparse"}:
            from rsi_vla.evaluation import apply_comparison_attempt_evidence
            from rsi_vla.evaluator import evaluate_matrix

            if kind == "matrix_reparse":
                self.authority()
                originals = [
                    item["payload"]
                    for item in self.config["operations"]
                    if item["kind"] == "matrix_evaluation"
                    and item["payload"]["result_id"] == payload["result_id"]
                ]
                keys = (
                    "result_id",
                    "bundle_id",
                    "candidate_id",
                    "manifest_ref",
                    "attempt_evidence_ref",
                    "side",
                )
                if len(originals) != 1 or any(
                    payload.get(key) != originals[0].get(key) for key in keys
                ):
                    raise ValueError(
                        "matrix reparse must retain the original matrix and all attempts"
                    )
                source_ref = Path(payload["source_ref"]).resolve(strict=True)
                manifest = read(source_ref)
                required = {
                    "core/src/rsi_vla/evaluator.py",
                    "core/src/rsi_vla/critical_progress.py",
                    "core/src/rsi_vla/primitive_progress.py",
                    "core/src/rsi_vla/evaluation.py",
                    "core/src/rsi_vla/task_completion.py",
                }
                if not required <= set(manifest.get("files", [])):
                    raise ValueError(
                        "matrix reparse requires the complete independent evaluation sources"
                    )
                for name in manifest["files"]:
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("matrix reparse source escapes repository")
                    if (
                        Path(self.config["repository_root"]) / relative
                    ).read_bytes() != (
                        source_ref.parent / "source" / relative
                    ).read_bytes():
                        raise ValueError(
                            "matrix reparse consumer source differs: " + name
                        )
            result = evaluate_matrix(
                self.protocol,
                read(self.config["profile_ref"]),
                read(payload["manifest_ref"]),
                result_id=payload["result_id"],
                bundle_id=payload["bundle_id"],
                candidate_id=payload.get("candidate_id"),
                audit_output_root=self.evaluations.root / "result_audits" / identifier(
                    payload["revision_id"] if kind == "matrix_reparse" else payload["result_id"]
                ),
            )
            apply_comparison_attempt_evidence(
                read(payload["attempt_evidence_ref"]), payload["side"], result
            )
            if kind == "matrix_reparse":
                previous = read(payload["expected_previous_ref"])
                refs = list(previous["data_refs"])
                repairs = []
                for index, ref in enumerate(refs):
                    missing = Path(ref)
                    if missing.exists():
                        continue
                    if missing.name != "independent-audit.json" or any(
                        row.get("evidence_ref") == ref for row in previous["assignments"]
                    ):
                        raise ValueError("matrix reparse cannot replace missing raw/native evidence")
                    context_ref = missing.parent / "runtime_context.json"
                    context = read(context_ref)
                    replacement = (self.evaluations.root / "result_audits" / identifier(payload["revision_id"])
                                   / identifier(context["run_id"]) / "independent-audit.json").resolve(strict=True)
                    if str(replacement) not in result.data_refs:
                        raise ValueError("missing derived audit has no freshly computed replacement")
                    identity = {key: context[key] for key in ("run_id", "episode_id", "bundle_id", "protocol_id")}
                    repairs.append(dict(missing_ref=ref, replacement_ref=str(replacement),
                                        runtime_context_ref=str(context_ref.resolve()), identity=identity))
                    refs[index] = str(replacement)
                refs.extend(
                    ref
                    for ref in [*result.data_refs, str(source_ref)]
                    if ref not in refs
                )
                result = replace(result, data_refs=refs)
                receipt = self.evaluations.reparse_result(
                    result,
                    revision_id=payload["revision_id"],
                    expected_previous_ref=payload["expected_previous_ref"],
                    reason=payload["reason"],
                    source_ref=source_ref,
                    protocol=self.protocol,
                    authorization_ref=self.config["administrative_authorization_ref"],
                    scope_authorization_ref=self.config["scope_authorization_ref"],
                    derived_ref_repairs=repairs,
                )
                return {
                    "result": asdict(result),
                    "revision_receipt": receipt,
                    "result_ref": str(
                        self.evaluations.path("results", result.result_id).resolve()
                    ),
                }
            self.evaluations.save_result(result)
            return asdict(result)
        if kind == "gate_reconsider":
            import os
            if os.environ.get("RSI_ISOLATED_ROLE") != "evaluation":
                raise PermissionError("Gate reconsideration requires the independent evaluation role")
            self.authority()
            candidate = Candidate(**self.store.get("candidates", payload["candidate_id"]))
            receipt = self.evaluations.reconsider_gate(
                decision_id=payload["decision_id"], candidate=candidate,
                current_bundle_id=payload["current_bundle_id"], result_ids=payload["result_ids"],
                previous_result_refs=payload["previous_result_refs"], revision_id=payload["revision_id"],
                expected_previous_ref=payload["expected_previous_ref"], reason=payload["reason"],
                source_ref=payload["source_ref"], protocol=self.protocol,
                frozen_protocol_ref=self.config["frozen_protocol_ref"],
                authorization_ref=self.config["administrative_authorization_ref"],
                scope_authorization_ref=self.config["scope_authorization_ref"])
            return {"decision": self.evaluations.get("decisions", payload["decision_id"]),
                    "revision_receipt": receipt,
                    "decision_ref": str(self.evaluations.path("decisions", payload["decision_id"]).resolve())}
        if kind == "gate":
            candidate = Candidate(
                **self.store.get("candidates", payload["candidate_id"])
            )
            results = {
                side: ExperimentResult(**self.evaluations.get("results", ref))
                for side, ref in payload["result_ids"].items()
            }
            decision = Gate(self.protocol).evaluate(
                payload["decision_id"],
                candidate,
                current_bundle_id=payload["current_bundle_id"],
                candidate_result=results["candidate"],
                baseline_result=results["baseline"],
                anchor_result=results["anchor"],
            )
            self.evaluations.put("decisions", decision.decision_id, decision)
            return asdict(decision)
        raise ValueError("unsupported new-task operation")

    def reconcile(self, op, inputs):
        kind, payload = self.registered(op)
        if kind == "runtime":
            raise RuntimeError(
                "Runtime reconcile must audit the original output; never execute again"
            )
        if kind == "research":
            from rsi_vla.bounded_runner import reconcile

            tools, cfg = self.tools(payload["request_id"])
            return reconcile(
                tools,
                TaskResearchAdapter(cfg),
                cfg["output_ref"],
                cost_limit=cfg["cost_limit"],
            )
        return self.execute(op, inputs)
