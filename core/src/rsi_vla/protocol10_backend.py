# Copyright 2026 RSI-Agent-VLA Authors.
"""Protocol10 business adapters over Runtime, ResearchTools and independent evaluation.

The coordinator owns durable intents. This backend owns the registered business
sequence; inputs never contain commands, replacement slots or model responses.
"""

import json
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import Candidate, ExperimentResult
from .evaluation import EvaluationStore, Gate, load_protocol
from .evaluator import evaluate_matrix
from .improvement_protocol import (
    BASELINE,
    TASKS,
    admit_resources,
    baseline_readiness,
    load_frozen_protocol,
)
from .phase_receipts import read_phase
from .research import ResearchJob
from .research_tools import ResearchTools
from .storage import Records, ResearchStore, atomic_json

CANDIDATE = "candidate-protocol10-01"
COST_MAP = {
    "simulation_episodes": None,
    "model_requests": "model_requests",
    "tool_calls": "tool_calls",
    "environment_steps": "environment_steps",
    "wall_seconds": "whole_job_wall_seconds",
    "gpu_seconds": "gpu_device_seconds_envelope",
    "storage_bytes": "artifact_bytes",
}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Protocol10Backend:
    """Run the fixed comparison using explicit artifacts, permits and role stores.

    Configuration supplies repository_root, protocol_ref, frozen_protocol_ref,
    runtime_protocol_ref, profile_ref, artifacts_root, phase_ref, runtime_slots
    and research. Runtime slots register both matrices, development task choices
    and optional technical recovery identities before dispatch. Research alone
    chooses the ordered development subset. Optional callables replace process
    boundaries for CPU fixtures; they are unavailable through the CLI.
    """

    def __init__(
        self,
        root: str | Path,
        configuration: dict[str, Any],
        *,
        runtime_prepare=None,
        runtime_execute=None,
        audit=None,
        progress_inspector=None,
        research_execute=None,
        research_reconcile=None,
        trial_prepare=None,
        trial_execute=None,
        trial_reconcile=None,
        trial_collector=None,
    ):
        self.root, self.config = Path(root).resolve(), deepcopy(configuration)
        fixture = self.config.get("offline_fixture") is True
        self.protocol = (
            load_protocol(self.config["protocol_ref"])
            if fixture
            else load_frozen_protocol(
                self.config["protocol_ref"], self.config["frozen_protocol_ref"]
            )
        )
        if fixture and self.protocol["scope"] != "offline":
            raise ValueError("offline fixtures must use an offline protocol")
        from .adapters.runtime_acceptance import inspect_attempt
        from .bounded_runner import execute, reconcile
        from .critical_progress import inspect_progress
        from .research_cycle import execute_trial, prepare_trial, reconcile_trial
        from .runtime import execute_runtime, prepare_runtime

        self.runtime_prepare = runtime_prepare or prepare_runtime
        self.runtime_execute = runtime_execute or execute_runtime
        self.audit = audit or inspect_attempt
        self.progress_inspector = progress_inspector or inspect_progress
        self.research_execute = research_execute or execute
        self.research_reconcile = research_reconcile or reconcile
        self.trial_prepare = trial_prepare or prepare_trial
        self.trial_execute = trial_execute or execute_trial
        self.trial_reconcile = trial_reconcile or reconcile_trial
        self.trial_collector = trial_collector
        self.records = Records(self.root / "backend")
        self.evaluations = EvaluationStore(
            Path(self.config["isolation_task_root"]) / "state/evaluation"
            if self.config.get("isolation_task_root")
            else self.root / "evaluation"
        )
        self.store = ResearchStore(self.config["research"]["store_ref"])
        self.slots = deepcopy(self.config["runtime_slots"])
        self._validate_registration()
        if not fixture:
            from .improvement_protocol import runtime_projection

            projected = load_protocol(self.config["runtime_protocol_ref"])
            expected = runtime_projection(
                self.protocol,
                frozen_ref=self.config["frozen_protocol_ref"],
                authorization_ref=projected["runtime_projection"]["authorization_ref"],
            )
            if projected != expected:
                raise ValueError(
                    "Runtime protocol differs from the authorized frozen projection"
                )
            if (
                self.config["research"].get("tools_protocol_ref")
                and load_protocol(self.config["research"]["tools_protocol_ref"])
                != expected
            ):
                raise ValueError(
                    "Research tools protocol differs from the registered Runtime projection"
                )
            if (
                Path(self.config["artifacts_root"]).resolve() / "research"
                != self.store.root.resolve()
            ):
                raise ValueError(
                    "Runtime and Research must consume the same registered artifact store"
                )
        refs = [
            self.config[name]
            for name in ("protocol_ref", "runtime_protocol_ref", "profile_ref")
        ]
        refs += [self.config["research"]["domain_policy_ref"]]
        if self.config["research"].get("preflight_ref"):
            refs.append(self.config["research"]["preflight_ref"])
        if self.config.get("source_manifest_ref"):
            refs.append(self.config["source_manifest_ref"])
        if self.config["research"].get("tools_protocol_ref"):
            refs.append(self.config["research"]["tools_protocol_ref"])
        self.registration = {
            "configuration": self.config,
            "inputs": {ref: read(ref) for ref in refs},
        }
        self.records.put("registration", "protocol10", self.registration)
        if not fixture and not self.config.get("isolation_task_root"):
            raise ValueError("production protocol10 requires isolated role domains")
        if self.config.get("isolation_task_root"):
            task_root = Path(self.config["isolation_task_root"]).resolve()
            if self.root != task_root / "operator/protocol10":
                raise ValueError("coordinator backend must belong to operator domain")
            if self.store.root.resolve() != task_root / "state/research":
                raise ValueError("Research store must belong to Research domain")
            for slot in self.slots:
                if (
                    not Path(slot["output_ref"])
                    .resolve()
                    .is_relative_to(task_root / "runtime")
                ):
                    raise ValueError("Runtime output escapes Runtime domain")
            if (
                not Path(self.config["research"]["output_ref"])
                .resolve()
                .is_relative_to(task_root / "state/research")
            ):
                raise ValueError("Research output escapes Research domain")

    def _validate_registration(self):
        expected = {
            (side, task, seed)
            for side in ("baseline", "candidate")
            for task in TASKS
            for seed in (0, 1)
        }
        primary = [
            s
            for s in self.slots
            if s["side"] in {"baseline", "candidate"} and not s.get("recovery_of")
        ]
        if {(s["side"], s["task_key"], s["seed"]) for s in primary} != expected or len(
            primary
        ) != 16:
            raise ValueError("register both complete fixed eight-slot matrices")
        runs, outputs = set(), set()
        for slot in self.slots:
            if slot["run_id"] in runs or slot["output_ref"] in outputs:
                raise ValueError("run/output identity reused")
            runs.add(slot["run_id"])
            outputs.add(slot["output_ref"])
            if slot["task_key"] not in TASKS or slot["side"] not in {
                "baseline",
                "candidate",
                "development",
            }:
                raise ValueError("unregistered task or side")
            if slot["seed"] not in ([2] if slot["side"] == "development" else [0, 1]):
                raise ValueError("slot outside fixed split")
            if slot["bundle_id"] != (
                BASELINE if slot["side"] == "baseline" else CANDIDATE
            ):
                raise ValueError("reserved bundle identity mismatch")
        recoveries = [s for s in self.slots if s.get("recovery_of")]
        if len(recoveries) > 2:
            raise ValueError("at most two technical recoveries may be registered")
        for slot in recoveries:
            original = self._slot(slot["recovery_of"])
            if original.get("recovery_of") or any(
                slot[k] != original[k]
                for k in ("side", "task_key", "seed", "bundle_id")
            ):
                raise ValueError("recovery must bind an original matching slot")
        if len({s["recovery_of"] for s in recoveries}) != len(recoveries):
            raise ValueError("at most one technical recovery per slot")

    def _slot(self, run_id):
        return next(s for s in self.slots if s["run_id"] == run_id)

    def _tools(self):
        cfg = self.config["research"]
        job = ResearchJob(self.store, cfg["job_id"])
        protocol = load_protocol(
            cfg.get("tools_protocol_ref", self.config["runtime_protocol_ref"])
        )
        return ResearchTools(
            self.store, job, protocol, domain_policy=read(cfg["domain_policy_ref"])
        )

    def _adapter(self):
        from .research_tools import Protocol10ResearchAdapter

        return Protocol10ResearchAdapter()

    def _op(self, kind, stage, payload=None, dynamic=False, key=None):
        identity = key or kind
        payload = deepcopy(payload or {})
        if dynamic:
            permit_ref = (
                self.config["research"]["permit_ref"]
                if kind == "research"
                else payload["permit_ref"]
            )
            payload["registered_permit"] = read(permit_ref)
        requirements = (
            (
                self.protocol["budget"]["research_proposal"]
                if kind == "research"
                else self.protocol["budget"]["runtime_per_start"]
            )
            if dynamic
            else {}
        )
        return {
            "operation_id": identity,
            "kind": kind,
            "stage": stage,
            "dynamic": dynamic,
            "payload": payload or {},
            "resource_requirements": requirements,
        }

    def _complete(self, op, **fields):
        receipt = {
            "operation_id": op["operation_id"],
            "status": "completed",
            "receipt_ref": str(self.records.path("receipts", op["operation_id"])),
            **fields,
        }
        if os.environ.get("RSI_ISOLATED_ROLE") not in {
            "runtime",
            "research",
            "evaluation",
        }:
            self.records.put("receipts", op["operation_id"], receipt)
        return receipt

    def _selection(self):
        return self.store.get(
            "protocol10_selections", self.config["research"]["request_id"]
        )

    def _development(self):
        selection = self._selection()
        slots = []
        for index, choice in enumerate(selection["development_slots"]):
            matches = [
                s
                for s in self.slots
                if s["side"] == "development"
                and not s.get("recovery_of")
                and (s["task_key"], s["seed"]) == (choice["task_key"], choice["seed"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Research-selected slot lacks one prebound permit/run/output"
                )
            slot = deepcopy(matches[0])
            slot["experiment_id"] = (
                self.config["research"]["request_id"]
                + "-experiment"
                + ("" if index == 0 else f"-slot{index + 1:02d}")
            )
            slots.append(slot)
        return slots

    def observe(self, progress: dict[str, Any], inputs: dict[str, Any]):
        """Derive one next operation from committed receipts, never model prose."""
        done = progress["receipts"]
        for side in ("baseline", "candidate"):
            if side == "candidate":
                if "baseline-evaluation" not in done:
                    break
                ready = done["baseline-evaluation"]["readiness"]
                if not ready["candidate_generation_allowed"]:
                    return {
                        "next_operation": None,
                        "terminal": ready["action"],
                        "wait": None,
                    }
                if "research" not in done:
                    return {
                        "next_operation": self._op(
                            "research",
                            "research",
                            dynamic=any(
                                row.get("currently_trialable") is True
                                for row in read(
                                    self.config["research"]["preflight_ref"]
                                )
                            ),
                        ),
                        "terminal": None,
                        "wait": None,
                    }
                if done["research"].get("stop"):
                    return {
                        "next_operation": None,
                        "terminal": "research_stop",
                        "wait": None,
                    }
                if "development-registration" not in done:
                    return {
                        "next_operation": self._op(
                            "development_registration",
                            "development",
                            key="development-registration",
                        ),
                        "terminal": None,
                        "wait": None,
                    }
                for original in self._development():
                    slot = original
                    while True:
                        key = "development-" + slot["run_id"]
                        if key not in done:
                            return {
                                "next_operation": self._op(
                                    "development", "development", slot, True, key
                                ),
                                "terminal": None,
                                "wait": None,
                            }
                        receipt = done[key]
                        if not receipt.get("stop"):
                            break
                        cycle = receipt["cycle"]
                        technical = (
                            cycle["screening_checks"].get("technical_valid") is False
                        )
                        audit = (
                            self._role_dispatch(
                                "audit",
                                operation=self._op(
                                    "settlement",
                                    "development",
                                    slot,
                                    key=slot["run_id"] + "-recovery-audit",
                                ),
                                output=slot["output_ref"],
                            )
                            if self._isolated_operator()
                            else self.audit(slot["output_ref"], require_vla=False)
                        )
                        if "owner_cleanup" not in audit.get("checks", {}):
                            from .runtime_cleanup import inspect_cleanup

                            audit.setdefault("checks", {})["owner_cleanup"] = (
                                inspect_cleanup(slot["output_ref"], audit["identity"])
                            )
                        costs_closed = all(
                            type(cycle["costs"].get(v)) in (int, float)
                            for v in COST_MAP.values()
                            if v
                        )
                        recoveries = [
                            r
                            for r in self.slots
                            if r.get("recovery_of") == slot["run_id"]
                        ]
                        if (
                            not technical
                            or not recoveries
                            or not costs_closed
                            or audit["checks"].get("owner_cleanup") is not True
                            or audit.get("confirmed_violations")
                        ):
                            return {
                                "next_operation": None,
                                "terminal": "development_stop",
                                "wait": None,
                            }
                        if (
                            inputs.get("recovery_cause_resolved", {}).get(
                                slot["run_id"]
                            )
                            is not True
                        ):
                            return {
                                "next_operation": None,
                                "terminal": None,
                                "wait": {
                                    "reason": "technical_recovery_cause",
                                    "operation_id": recoveries[0]["run_id"],
                                },
                            }
                        slot = {
                            **recoveries[0],
                            "experiment_id": original["experiment_id"] + "-recovery",
                        }
                if "candidate-submit" not in done:
                    return {
                        "next_operation": self._op(
                            "candidate_submit", "candidate", key="candidate-submit"
                        ),
                        "terminal": None,
                        "wait": None,
                    }
            for original in [
                s for s in self.slots if s["side"] == side and not s.get("recovery_of")
            ]:
                slot = original
                while True:
                    for kind in ("runtime", "settlement", "attempt_evaluation"):
                        key = slot["run_id"] + "-" + kind
                        if key not in done:
                            return {
                                "next_operation": self._op(
                                    kind, side, slot, kind == "runtime", key
                                ),
                                "terminal": None,
                                "wait": None,
                            }
                    result = done[slot["run_id"] + "-attempt_evaluation"]
                    if result["validity"] == "valid":
                        break
                    recoveries = [
                        s for s in self.slots if s.get("recovery_of") == slot["run_id"]
                    ]
                    if (
                        not recoveries
                        or result.get("failure_class") != "technical"
                        or result.get("cost_closed") is not True
                        or result.get("cleanup_complete") is not True
                    ):
                        return {
                            "next_operation": None,
                            "terminal": side + "_technical_stop",
                            "wait": None,
                        }
                    recovery = recoveries[0]
                    if (
                        inputs.get("recovery_cause_resolved", {}).get(slot["run_id"])
                        is not True
                    ):
                        return {
                            "next_operation": None,
                            "terminal": None,
                            "wait": {
                                "reason": "technical_recovery_cause",
                                "operation_id": recovery["run_id"] + "-runtime",
                            },
                        }
                    slot = recovery
            key = side + "-evaluation"
            if key not in done:
                return {
                    "next_operation": self._op(
                        "matrix_evaluation", side, {"side": side}, key=key
                    ),
                    "terminal": None,
                    "wait": None,
                }
        if "gate" not in done:
            return {
                "next_operation": self._op("gate", "gate"),
                "terminal": None,
                "wait": None,
            }
        if "qualification-handoff" not in done:
            return {
                "next_operation": self._op(
                    "qualification_handoff",
                    "qualification",
                    key="qualification-handoff",
                ),
                "terminal": None,
                "wait": None,
            }
        return {"next_operation": None, "terminal": "comparison_complete", "wait": None}

    def _role_dispatch(self, action, *, operation=None, inputs=None, **kwargs):
        from .protocol10_roles import dispatch

        return dispatch(
            self.config, self.root, action, operation=operation, inputs=inputs, **kwargs
        )

    def _isolated_operator(self):
        return (
            bool(self.config.get("isolation_task_root"))
            and os.environ.get("RSI_ISOLATED_ROLE") == "operator"
        )

    def _accept_role_receipt(self, op, receipt):
        if receipt.get("operation_id") != op["operation_id"]:
            raise ValueError("role operation receipt mismatch")
        if receipt.get("status") == "completed":
            self.records.put("receipts", op["operation_id"], receipt)
            if op["kind"] == "runtime":
                self.records.put(
                    "runtime_returns",
                    op["payload"]["run_id"],
                    {
                        k: receipt[k]
                        for k in (
                            "run_id",
                            "output_ref",
                            "wall_seconds",
                            "launcher_error",
                        )
                    },
                )
        return receipt

    def reconcile(self, op: dict[str, Any], inputs: dict[str, Any]):
        """Inspect native markers before allowing a first or resumed dispatch."""
        path = self.records.path("receipts", op["operation_id"])
        if path.exists():
            return read(path)
        kind, payload = op["kind"], op["payload"]
        if self._isolated_operator() and kind == "research":
            return self._accept_role_receipt(
                op, self._role_dispatch("reconcile", operation=op, inputs=inputs)
            )
        if (
            self._isolated_operator()
            and kind == "development"
            and Path(payload["output_ref"]).exists()
        ):
            return self._isolated_development(op, inputs, reconcile_only=True)
        if kind == "runtime":
            returned = self.records.path("runtime_returns", payload["run_id"])
            if returned.exists():
                return self._complete(op, **read(returned))
            output = Path(payload["output_ref"])
            if all(
                (output / name).exists()
                for name in (
                    "runtime_context.json",
                    "supervisor.json",
                    "launcher_costs.json",
                )
            ):
                context, supervisor, launcher = (
                    read(output / name)
                    for name in (
                        "runtime_context.json",
                        "supervisor.json",
                        "launcher_costs.json",
                    )
                )
                expected = {
                    "run_id": payload["run_id"],
                    "bundle_id": payload["bundle_id"],
                    "protocol_id": self.protocol["protocol_id"],
                    "episode_id": f"{payload['task_key']}-s{payload['seed']}",
                }
                if (
                    all(context.get(k) == v for k, v in expected.items())
                    and launcher.get("identity") == expected
                    and supervisor.get("cleanup_confirmed") is True
                    and supervisor.get("status")
                    in {"completed", "failed", "stopped", "budget_exhausted"}
                    and type(launcher.get("whole_job_wall_seconds")) in (int, float)
                ):
                    native = {
                        "run_id": payload["run_id"],
                        "output_ref": payload["output_ref"],
                        "wall_seconds": launcher["whole_job_wall_seconds"],
                        "launcher_error": None
                        if not (output / "launcher_error.json").exists()
                        else read(output / "launcher_error.json"),
                    }
                    self.records.put("runtime_returns", payload["run_id"], native)
                    return self._complete(op, **native)
            if (
                Path(payload["output_ref"]).exists()
                or self.records.path("dispatches", payload["run_id"]).exists()
            ):
                return {
                    "operation_id": op["operation_id"],
                    "status": "unknown",
                    "reason": "runtime submission unresolved",
                    "receipt_ref": str(returned),
                }
        elif kind == "research":
            output = Path(self.config["research"]["output_ref"])
            if output.exists():
                try:
                    result = self.research_reconcile(
                        self._tools(),
                        self._adapter(),
                        output,
                        cost_limit=self.config["research"]["cost_limit"],
                    )
                    return self._research_receipt(op, result)
                except (OSError, ValueError, RuntimeError) as exc:
                    terminal = self._closed_research_stop(op, exc)
                    if terminal is not None:
                        return terminal
                    return {
                        "operation_id": op["operation_id"],
                        "status": "waiting",
                        "reason": "research_reconciliation",
                        "detail": str(exc),
                        "receipt_ref": str(output),
                    }
        elif kind == "development":
            tools = self._tools()
            marker = tools.store.path(
                "trial_dispatches", payload["experiment_id"] + "-dispatch"
            )
            if marker.exists() or Path(payload["output_ref"]).exists():
                try:
                    result = self.trial_reconcile(
                        tools,
                        payload["experiment_id"],
                        output=payload["output_ref"],
                        permit_path=payload["permit_ref"],
                        collector=self.trial_collector,
                        auto_submit=False,
                        settle_before_collect=True,
                        cost_observer=self.audit,
                    )
                    return self._complete(
                        op, cycle=result, stop=result["action"] == "stop"
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    return {
                        "operation_id": op["operation_id"],
                        "status": "waiting",
                        "reason": "evaluation_repair",
                        "detail": str(exc),
                        "receipt_ref": str(marker),
                    }
        elif kind in {
            "candidate_submit",
            "gate",
            "qualification_handoff",
            "development_registration",
        }:
            # Receipt repair may observe a prior submission, but pause/reconcile
            # must never create a new Candidate or Gate decision.
            if (
                kind == "candidate_submit"
                and self.store.path("candidates", CANDIDATE).exists()
            ):
                return self._complete(
                    op, candidate=self.store.get("candidates", CANDIDATE)
                )
            if (
                kind == "gate"
                and self.evaluations.path(
                    "decisions", "protocol10-independent-gate"
                ).exists()
            ):
                return self._complete(
                    op,
                    decision=self.evaluations.get(
                        "decisions", "protocol10-independent-gate"
                    ),
                )
        else:
            # Evidence parsing and fee settlement have native idempotent consumers.
            try:
                return self.execute(op, inputs)
            except (OSError, ValueError, RuntimeError) as exc:
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "evaluation_repair",
                    "detail": str(exc),
                    "receipt_ref": str(path),
                }
        return {
            "operation_id": op["operation_id"],
            "status": "not_submitted",
            "submission_absent": True,
            "receipt_ref": str(path),
        }

    def _prepare_runtime(self, slot):
        return self.runtime_prepare(
            self.config["repository_root"],
            self.config["artifacts_root"],
            protocol_path=self.config["runtime_protocol_ref"],
            run_id=slot["run_id"],
            bundle_id=slot["bundle_id"],
            task_key=slot["task_key"],
            seed=slot["seed"],
            split="release",
            output_dir=slot["output_ref"],
            authorization_path=slot["permit_ref"],
            cuda_device=slot.get("cuda_device", 0),
            storage_write_roots=[slot["scratch_ref"]],
        )

    def _isolated_development(self, op, inputs, *, reconcile_only=False):
        payload = op["payload"]
        if not reconcile_only and not Path(payload["output_ref"]).exists():
            if not self.config.get("offline_fixture"):
                from .protocol10_admission import validate_dispatch_admission

                validate_dispatch_admission(
                    self.config,
                    self.protocol,
                    op,
                    authorization=inputs.get("authorization"),
                )
            plan = self.trial_prepare(
                self._tools(),
                payload["experiment_id"],
                root=self.config["repository_root"],
                artifacts=self.config["artifacts_root"],
                protocol_path=self.config["runtime_protocol_ref"],
                output=payload["output_ref"],
                run_id=payload["run_id"],
                permit_path=payload["permit_ref"],
                task_key=payload["task_key"],
                seed=payload["seed"],
                cuda_device=payload.get("cuda_device", 0),
                storage_write_roots=[payload["scratch_ref"]],
            )
            if plan["blockers"]:
                raise ValueError("development Runtime preparation blocked")
            marker = self.store.path(
                "trial_dispatches", payload["experiment_id"] + "-dispatch"
            )
            if marker.exists():
                raise RuntimeError(
                    "development dispatch unresolved; never repeat Runtime"
                )
            self._role_dispatch(
                "development_dispatch",
                operation=op,
                experiment_id=payload["experiment_id"],
                output=payload["output_ref"],
            )
            self._role_dispatch("runtime_plan", operation=op, inputs=inputs, plan=plan)
        collection = self._role_dispatch(
            "development_collect",
            operation=op,
            experiment_id=payload["experiment_id"],
            output=payload["output_ref"],
        )
        result = self._role_dispatch(
            "development_reconcile",
            operation=op,
            experiment_id=payload["experiment_id"],
            output=payload["output_ref"],
            permit_ref=payload["permit_ref"],
            collection=collection,
        )
        return self._complete(op, cycle=result, stop=result["action"] == "stop")

    def _closed_research_stop(self, op, error):
        """End a failed proposal only after native request, cost and owner closure."""
        from .research_tools import Protocol10ResponseRejected

        cfg = self.config["research"]
        output = Path(cfg["output_ref"])
        try:
            context = read(output / "runtime_context.json")
            if (
                context.get("job_id") != cfg["job_id"]
                or context.get("bundle_id") != BASELINE
                or context.get("protocol_id") != self.protocol["protocol_id"]
                or context.get("authorization_ref")
                != str(Path(cfg["permit_ref"]).resolve())
            ):
                return None
            supervision = read(output / "supervisor.json")
            if supervision.get("cleanup_confirmed") is not True or supervision.get(
                "status"
            ) not in {"completed", "failed", "interrupted"}:
                return None
            events = [
                json.loads(line)
                for line in (output / "model_requests.jsonl").read_text().splitlines()
                if line
            ]
            admitted = [
                event for event in events if event["kind"] == "request_admitted"
            ]
            terminals = [
                event for event in events if event["kind"] == "response_terminal"
            ]
            costs = read(output / "execution-costs.json")["costs"]
            if (
                len(admitted) > 1
                or len(admitted) != len(terminals)
                or costs.get("model_requests") != len(admitted)
            ):
                return None
            phase = read_phase(self.config["phase_ref"])
            charges = [
                entry
                for entry in phase.get("research_charges", [])
                if entry["request_id"] == cfg["request_id"]
            ]
            if len(charges) != 1 or phase.get("unresolved_costs"):
                return None
            mapped = {
                "model_requests": "model_requests",
                "wall_seconds": "whole_job_wall_seconds",
                "storage_bytes": "artifact_bytes",
            }
            if any(
                type(costs.get(raw)) not in (int, float)
                or charges[0]["costs"].get(key) != costs[raw]
                for key, raw in mapped.items()
            ):
                return None
            no_response = not admitted
            sdk_path = output / "sdk-result.json"
            if not sdk_path.exists():
                no_response = True
            else:
                sdk = read(sdk_path)
                no_response |= sdk.get("status", "").rsplit(".", 1)[-1] != "completed"
                final = sdk.get("final_response")
                if not isinstance(final, str) or not final.strip():
                    no_response = True
                else:
                    try:
                        json.loads(final)
                    except json.JSONDecodeError:
                        no_response = True
            if no_response or isinstance(error, Protocol10ResponseRejected):
                return self._complete(
                    op,
                    stop=True,
                    reason="no_legal_research_response",
                    detail=str(error),
                    cost_closed=True,
                    cleanup_complete=True,
                    request_count=len(admitted),
                )
        except (OSError, KeyError, TypeError, ValueError):
            return None
        return None

    def _research_receipt(self, op, result):
        selection = self._selection()
        return self._complete(
            op,
            result=result,
            selection=selection,
            stop=selection["decision"] == "defer",
        )

    def execute(self, op: dict[str, Any], inputs: dict[str, Any]):
        """Execute the registered consumer after Coordinator admission."""
        from .phase_receipts import set_operation

        set_operation(op["operation_id"])
        kind, payload = op["kind"], op["payload"]
        if any(
            read(ref) != content for ref, content in self.registration["inputs"].items()
        ):
            raise ValueError("registered protocol/profile/domain inputs changed")
        if op["dynamic"]:
            permit_ref = (
                self.config["research"]["permit_ref"]
                if kind == "research"
                else payload["permit_ref"]
            )
            if read(permit_ref) != payload["registered_permit"]:
                raise ValueError("registered single-operation permit changed")
            phase = read_phase(self.config["phase_ref"])
            run_ids = {slot["run_id"] for slot in self.slots}
            attempts = {
                entry["run_id"]
                for key in ("protocol10_trial_charges", "development_trial_charges")
                for entry in phase.get(key, [])
                if entry["run_id"] in run_ids
            }
            research_done = any(
                entry["request_id"] == self.config["research"]["request_id"]
                for entry in phase.get("research_charges", [])
            )
            try:
                admission = admit_resources(
                    self.protocol,
                    phase,
                    offline_used_seconds=inputs["offline_used_seconds"],
                    offline_used_bytes=inputs["offline_used_bytes"],
                    runtime_starts_used=len(attempts),
                    research_completed=research_done,
                )
            except (ValueError, KeyError) as exc:
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "phase_resource_evidence",
                    "detail": str(exc),
                    "receipt_ref": self.config["phase_ref"],
                }
            if not admission["resource_fit"]:
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "budget_wait",
                    "admission": admission,
                    "receipt_ref": self.config["phase_ref"],
                }
        if self._isolated_operator():
            from .protocol10_roles import OP_ROLES

            if kind in OP_ROLES:
                return self._accept_role_receipt(
                    op, self._role_dispatch("execute", operation=op, inputs=inputs)
                )
            if kind == "development":
                return self._isolated_development(op, inputs)
        if op["dynamic"] and not self.config.get("offline_fixture"):
            from .protocol10_admission import validate_dispatch_admission

            validate_dispatch_admission(
                self.config,
                self.protocol,
                op,
                authorization=inputs.get("authorization"),
            )
        if kind == "runtime":
            plan = self._prepare_runtime(payload)
            if plan["blockers"]:
                raise ValueError(
                    "Runtime preparation blocked: " + str(plan["blockers"])
                )
            if os.environ.get("RSI_ISOLATED_ROLE") != "runtime":
                self.records.put(
                    "dispatches", payload["run_id"], {"operation": op, "plan": plan}
                )
            started, error = time.monotonic(), None
            try:
                if not self.config.get("offline_fixture"):
                    from .protocol10_admission import validate_dispatch_admission

                    validate_dispatch_admission(
                        self.config,
                        self.protocol,
                        op,
                        authorization=inputs.get("authorization"),
                    )
                self.runtime_execute(plan)
            except Exception as exc:  # noqa: BLE001 - persist the returned Runtime even on boundary failure
                error = f"{type(exc).__name__}: {exc}"
            returned = {
                "run_id": payload["run_id"],
                "output_ref": payload["output_ref"],
                "wall_seconds": time.monotonic() - started,
                "launcher_error": error,
            }
            if os.environ.get("RSI_ISOLATED_ROLE") != "runtime":
                self.records.put("runtime_returns", payload["run_id"], returned)
            return self._complete(op, **returned)
        if kind == "settlement":
            from .process_accounting import accounted_storage_bytes
            from .research_cycle import _phase_charge

            audited = (
                self._role_dispatch("audit", operation=op, output=payload["output_ref"])
                if self._isolated_operator()
                else self.audit(payload["output_ref"], require_vla=False)
            )
            if "owner_cleanup" not in audited.get("checks", {}):
                from .runtime_cleanup import inspect_cleanup

                audited.setdefault("checks", {})["owner_cleanup"] = inspect_cleanup(
                    payload["output_ref"], audited["identity"]
                )
            returned = self.records.get("runtime_returns", payload["run_id"])
            costs = {k: v for k, v in audited["costs"].items() if v is not None}
            booked = [
                entry
                for entry in read(self.config["phase_ref"]).get(
                    "protocol10_trial_charges", []
                )
                if entry["run_id"] == payload["run_id"]
            ]
            if booked:
                if (
                    len(booked) != 1
                    or booked[0]["permit_ref"] != payload["permit_ref"]
                    or booked[0]["output_ref"] != payload["output_ref"]
                ):
                    raise ValueError("conflicting native attempt charge")
                costs.update(
                    {
                        field: booked[0]["costs"][dimension]
                        for dimension, field in COST_MAP.items()
                        if field and dimension in booked[0]["costs"]
                    }
                )
            if not booked:
                costs["artifact_bytes"] = max(
                    costs.get("artifact_bytes", 0),
                    accounted_storage_bytes(payload["output_ref"]),
                )
            costs.setdefault("whole_job_wall_seconds", returned["wall_seconds"])
            costs.setdefault("gpu_device_seconds_envelope", returned["wall_seconds"])
            permit = read(payload["permit_ref"])
            if str(Path(permit["phase_snapshot_ref"]).resolve()) != str(
                Path(self.config["phase_ref"]).resolve()
            ):
                raise ValueError("attempt permit uses another phase ledger")
            _phase_charge(
                {**permit, "permit_ref": payload["permit_ref"]},
                payload["run_id"],
                costs,
                payload["output_ref"],
                charge_kind="protocol10_trial_charges",
            )
            closed = all(
                type(costs.get(v)) in (int, float) for v in COST_MAP.values() if v
            )
            if not closed or audited.get("checks", {}).get("owner_cleanup") is not True:
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "cost_or_cleanup_reconciliation",
                    "receipt_ref": str(
                        self.records.path("runtime_returns", payload["run_id"])
                    ),
                }
            return self._complete(
                op, costs=costs, cost_closed=True, cleanup_complete=True, audit=audited
            )
        if kind == "attempt_evaluation":
            settlement = self.records.get("receipts", payload["run_id"] + "-settlement")
            if (
                settlement["cost_closed"] is not True
                or settlement["cleanup_complete"] is not True
            ):
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "cost_or_cleanup_reconciliation",
                    "receipt_ref": settlement["receipt_ref"],
                }
            audit = settlement["audit"]
            progress = self.progress_inspector(payload["output_ref"], audit["identity"])
            if progress["value"] is None and audit["passed"]:
                raise ValueError("critical progress evaluation incomplete")
            valid = (
                audit["passed"]
                and progress["value"] is True
                and audit["outcome"] in {"success", "failure"}
            )
            return self._complete(
                op,
                validity="valid" if valid else "invalid",
                outcome=audit["outcome"],
                failure_class=None
                if valid
                else "governance"
                if progress["value"] is False or audit.get("confirmed_violations")
                else "technical",
                critical_progress=progress["value"],
                cost_closed=settlement["cost_closed"],
                cleanup_complete=settlement["cleanup_complete"],
            )
        if kind == "matrix_evaluation":
            side = payload["side"]
            manifest = []
            for slot in self.slots:
                if (
                    slot["side"] != side
                    or not self.records.path(
                        "receipts", slot["run_id"] + "-attempt_evaluation"
                    ).exists()
                ):
                    continue
                report = self.records.get(
                    "receipts", slot["run_id"] + "-attempt_evaluation"
                )
                if report["validity"] != "valid":
                    continue
                output = Path(slot["output_ref"])
                context = read(output / "runtime_context.json")
                manifest.append(
                    {
                        "task_key": slot["task_key"],
                        "seed": slot["seed"],
                        "run_id": slot["run_id"],
                        "episode_id": context["episode_id"],
                        "result_path": str(output / "independent_result.json"),
                        "action_events_path": str(output / "action_events.jsonl"),
                        "costs_path": str(output / "runtime_costs.json"),
                        "supervisor_path": str(output / "supervisor.json"),
                    }
                )
            protocol = deepcopy(self.protocol)
            protocol["task_sets"]["release"].setdefault(
                "benchmark_id", protocol["benchmark_id"]
            )
            result = evaluate_matrix(
                protocol,
                read(self.config["profile_ref"]),
                manifest,
                result_id=side + "-protocol10",
                bundle_id=BASELINE if side == "baseline" else CANDIDATE,
                candidate_id=None if side == "baseline" else CANDIDATE,
            )
            # Include every attempted run, including unselected technical failures.
            from .evaluation import apply_comparison_attempt_evidence

            attempt_slots, violations = [], []
            for slot in self.slots:
                receipt_path = self.records.path(
                    "receipts", slot["run_id"] + "-settlement"
                )
                if slot["side"] != side or not receipt_path.exists():
                    continue
                settlement = read(receipt_path)
                audit = settlement["audit"]
                attempt_slots.append(
                    {"side": side, "attempts": [{"costs": settlement["costs"]}]}
                )
                if audit.get("checks", {}).get("identity") is True:
                    checks = set(audit.get("confirmed_violations", []))
                    report = self.records.get(
                        "receipts", slot["run_id"] + "-attempt_evaluation"
                    )
                    if report["critical_progress"] is False:
                        checks.add("critical_progress")
                    violations.extend(
                        {"side": side, "constraint": check} for check in checks
                    )
            apply_comparison_attempt_evidence(
                {"slots": attempt_slots, "final_attempt_violations": violations},
                side,
                result,
            )
            self.evaluations.save_result(result)
            fields = {
                "result_ref": str(self.evaluations.path("results", result.result_id))
            }
            if side == "baseline":
                fields["readiness"] = baseline_readiness(self.protocol, result)
            return self._complete(op, **fields)
        if kind == "research":
            cfg, tools, adapter = (
                self.config["research"],
                self._tools(),
                self._adapter(),
            )
            feasible_rows = read(cfg["preflight_ref"])
            if not any(row.get("currently_trialable") is True for row in feasible_rows):
                return self._complete(
                    op,
                    stop=True,
                    reason="no executable evidenced operator in the approved domain",
                    request_count=0,
                )
            if not tools.store.path("bounded_transports", cfg["request_id"]).exists():
                return self._complete(
                    op,
                    stop=True,
                    reason="no legal prepared Research input",
                    request_count=0,
                )
            transport = tools.store.get("bounded_transports", cfg["request_id"])
            allowed = tools.store.get("bounded_requests", cfg["request_id"])[
                "allowed_operators"
            ]
            if (
                self.protocol["research_selection"]["rejected_hint_not_revived"]
                and "planner_segment_handoff_hint" in allowed
            ):
                return self._complete(
                    op,
                    stop=True,
                    reason="no fresh legal intervention; rejected hint is excluded",
                )
            feasible = {
                row["operator"]
                for row in read(cfg["preflight_ref"])
                if row.get("currently_trialable") is True
            }
            if not allowed or not set(allowed) <= feasible:
                return self._complete(
                    op,
                    stop=True,
                    reason="no executable evidenced operator in the approved domain",
                )
            try:
                if not self.config.get("offline_fixture"):
                    from .protocol10_admission import validate_dispatch_admission

                    validate_dispatch_admission(
                        self.config,
                        self.protocol,
                        op,
                        authorization=inputs.get("authorization"),
                    )
                result = self.research_execute(
                    tools,
                    adapter,
                    transport,
                    root=self.config["repository_root"],
                    output=cfg["output_ref"],
                    permit_path=cfg["permit_ref"],
                    preflight=read(cfg["preflight_ref"]),
                    cost_limit=cfg["cost_limit"],
                )
            except (OSError, ValueError, RuntimeError) as exc:
                terminal = self._closed_research_stop(op, exc)
                if terminal is not None:
                    return terminal
                return {
                    "operation_id": op["operation_id"],
                    "status": "waiting",
                    "reason": "research_reconciliation",
                    "detail": str(exc),
                    "receipt_ref": cfg["output_ref"],
                }
            return self._research_receipt(op, result)
        if kind == "development_registration":
            tools = self._tools()
            slots = self._development()
            if not slots or len(slots) > 2:
                raise ValueError("candidate must register one or two development slots")
            original_id = self.config["research"]["request_id"] + "-experiment"
            experiment = tools._owned("research_experiments", original_id)
            trial = tools._owned("rule_trials", original_id)
            if trial["bundle_id"] != CANDIDATE:
                raise ValueError("Research did not use reserved candidate identity")
            registrations = slots[1:]
            for original in slots:
                registrations += [
                    {
                        **recovery,
                        "experiment_id": original["experiment_id"] + "-recovery",
                    }
                    for recovery in self.slots
                    if recovery.get("recovery_of") == original["run_id"]
                ]
            for slot in registrations:
                for kind_name, record in (
                    ("research_experiments", experiment),
                    ("rule_trials", trial),
                ):
                    tools._save(
                        kind_name,
                        slot["experiment_id"],
                        {**record, "experiment_id": slot["experiment_id"]},
                    )
            return self._complete(
                op,
                slots=slots,
                selection_ref=str(
                    self.store.path(
                        "protocol10_selections", self.config["research"]["request_id"]
                    )
                ),
            )
        if kind == "development":
            tools = self._tools()
            plan = self.trial_prepare(
                tools,
                payload["experiment_id"],
                root=self.config["repository_root"],
                artifacts=self.config["artifacts_root"],
                protocol_path=self.config["runtime_protocol_ref"],
                output=payload["output_ref"],
                run_id=payload["run_id"],
                permit_path=payload["permit_ref"],
                task_key=payload["task_key"],
                seed=payload["seed"],
                cuda_device=payload.get("cuda_device", 0),
                storage_write_roots=[payload["scratch_ref"]],
            )
            result = self.trial_execute(
                tools,
                plan,
                permit_path=payload["permit_ref"],
                executor=self.runtime_execute,
                collector=self.trial_collector,
                auto_submit=False,
                settle_before_collect=True,
                cost_observer=self.audit,
            )
            return self._complete(op, cycle=result, stop=result["action"] == "stop")
        if kind == "candidate_submit":
            tools, slots = self._tools(), self._development()
            selected_slots = []
            for slot in slots:
                cycle = self.records.get("receipts", "development-" + slot["run_id"])
                if cycle.get("stop"):
                    recovery = next(
                        r for r in self.slots if r.get("recovery_of") == slot["run_id"]
                    )
                    slot = {
                        **recovery,
                        "experiment_id": slot["experiment_id"] + "-recovery",
                    }
                selected_slots.append(slot)
            slots = selected_slots
            phase = read_phase(self.config["phase_ref"])
            snapshot = self.store.path("cost_snapshots", CANDIDATE)
            if not snapshot.exists():
                atomic_json(snapshot, {"phase": phase})
            candidate = tools.submit_candidate(
                CANDIDATE,
                experiment_id=slots[0]["experiment_id"],
                additional_experiment_ids=[slot["experiment_id"] for slot in slots[1:]],
                cost_snapshot_ref=str(snapshot),
            )
            return self._complete(op, candidate=candidate)
        if kind == "gate":
            candidate = Candidate(**self.store.get("candidates", CANDIDATE))
            baseline = ExperimentResult(
                **self.evaluations.get("results", "baseline-protocol10")
            )
            current = ExperimentResult(
                **self.evaluations.get("results", "candidate-protocol10")
            )
            protocol = deepcopy(self.protocol)
            protocol["dynamic_authorized"] = load_protocol(
                self.config["runtime_protocol_ref"]
            ).get("dynamic_authorized", False)
            decision = Gate(protocol).evaluate(
                "protocol10-independent-gate",
                candidate,
                current_bundle_id=BASELINE,
                candidate_result=current,
                baseline_result=baseline,
                anchor_result=baseline,
            )
            self.evaluations.put("decisions", decision.decision_id, decision)
            return self._complete(op, decision=asdict(decision))
        if kind == "qualification_handoff":
            decision = self.records.get("receipts", "gate")["decision"]
            return self._complete(
                op,
                verdict=decision["verdict"],
                release_eligible=decision["release_eligible"],
                action="comparison_only_no_release_qualification",
                active_bundle_id=self.protocol["active_bundle_id"],
            )
        raise ValueError("unsupported protocol10 operation")
