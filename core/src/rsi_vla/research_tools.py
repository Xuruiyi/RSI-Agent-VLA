"""Callable research tools over the existing job, results, bundles and candidates.

These tools prepare trials and consume development evidence. They have no Gate,
release or runtime activation API. Online execution remains a backend responsibility.
"""
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from .contracts import Candidate, ExperimentResult, RunBundle
from .rule_updates import write_rule_trial


def _costs(costs):
    if not isinstance(costs, dict) or any(type(v) not in (int, float) or not isfinite(v) or v < 0
                                          for v in costs.values()):
        raise ValueError("costs must be nonnegative finite resource values")
    return dict(costs)


def _token_dimension(name):
    return name == "tokens" or "_tokens" in name


def _token_policy(protocol):
    value = protocol.get("budget", {}).get("token_policy", "registered_known_cost")
    return value.split(";", 1)[0].strip() if isinstance(value, str) else value


def _minimum_hypotheses(protocol, *, diagnostic=False):
    """Keep frozen comparisons; new tasks and diagnosis need one testable explanation."""
    from .task_policy import task_policy
    policy = task_policy(protocol)
    frozen = (protocol.get("scope_id") == "continuation10-20261113" or
              protocol.get("protocol_id") == "rsi-libero-pro-improvement-protocol10-public-seeds01")
    return 1 if policy or (diagnostic and not frozen) else 2


def _validate_hypotheses(hypotheses, minimum):
    required = {"explanation", "prediction", "distinguishing_test"}
    if (not isinstance(hypotheses, list) or len(hypotheses) < minimum or
            any(not isinstance(h, dict) or not required <= h.keys() or
                any(not isinstance(h[k], str) or not h[k].strip() for k in required)
                for h in hypotheses) or
            len({h["explanation"] for h in hypotheses}) != len(hypotheses)):
        raise ValueError(f"retain at least {minimum} distinct testable explanation(s)")


class ResearchTools:
    def __init__(self, store, job, protocol, *, domain_policy):
        if job.store is not store:
            raise ValueError("tools and job must share the same research store")
        self.store, self.job, self.protocol = store, job, protocol
        self.base = RunBundle(**store.get("bundles", job.state["bundle_id"]))
        if self.base.protocol_id != protocol["protocol_id"]:
            raise ValueError("job/protocol mismatch")
        self.domain_policy = dict(domain_policy)

    def _save(self, kind, key, record):
        value = {"job_id": self.job.state["job_id"], "base_bundle_id": self.base.bundle_id,
                 "protocol_id": self.protocol["protocol_id"], **record}
        self.store.put(kind, key, value)
        return value

    def _owned(self, kind, key):
        value = self.store.get(kind, key)
        if (value["job_id"] != self.job.state["job_id"] or
                value["base_bundle_id"] != self.base.bundle_id or
                value["protocol_id"] != self.protocol["protocol_id"]):
            raise ValueError("record belongs to another research job")
        return value

    def _observed_only(self, plan):
        return (plan["evidence_level"] != "offline" and
                _token_policy(self.protocol) == "observed_only")

    def _hard_costs(self, plan, costs):
        if not self._observed_only(plan):
            return dict(costs)
        dimensions = set(plan["cost_limit"])
        if any(_token_dimension(name) or name not in self.job.state["limits"] for name in dimensions):
            raise ValueError("experiment limits must use registered hard job dimensions")
        missing = dimensions - costs.keys()
        if missing:
            raise ValueError("experiment result missing required hard cost dimensions: " + ", ".join(sorted(missing)))
        return {name: costs[name] for name in sorted(dimensions)}

    def _candidate_costs(self):
        if _token_policy(self.protocol) != "observed_only":
            return dict(self.job.state["used"])
        total, accounted = {}, set()
        for finding in self.store.all("research_findings"):
            if (finding["job_id"] != self.job.state["job_id"] or
                    finding["base_bundle_id"] != self.base.bundle_id):
                continue
            accounted.add("experiment " + finding["experiment_id"])
            for name, value in _costs(finding["costs"]).items():
                total[name] = total.get(name, 0) + value
        for execution in self.store.all("llm_executions"):
            if (execution["job_id"] != self.job.state["job_id"] or
                    execution["base_bundle_id"] != self.base.bundle_id):
                continue
            accounted.add("llm " + execution["request_id"])
            for name, value in _costs(execution["costs"]).items():
                total[name] = total.get(name, 0) + value
        for charge in self.job.state["charges"]:
            if charge["reason"] not in accounted:
                for name, value in _costs(charge["costs"]).items():
                    total[name] = total.get(name, 0) + value
        incomplete = any(finding.get("cost_observation", {}).get("token_status") == "incomplete_observed_lower_bound"
                         for finding in self.store.all("research_findings")
                         if finding["job_id"] == self.job.state["job_id"] and finding.get("cost_observation"))
        if incomplete:
            for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
                total["observed_" + name] = total.get("observed_" + name, 0) + total.pop(name, 0)
        # Input and cached input remain separate vector entries. Cached input
        # is a subset of input; there is deliberately no input+cached total.
        return total

    def _validate_result(self, result, bundles):
        p = self.protocol
        if result.bundle_id in self.domain_policy.get('historical_development_bundle_ids', []):
            reference=self.domain_policy.get('historical_protocol_refs',{}).get(result.protocol_id)
            if reference:
                from .evaluation import load_protocol
                p=load_protocol(reference)
        split = p.get("task_sets", {}).get("development")
        benchmark = split.get("benchmark_id", p["benchmark_id"]) if split else p["benchmark_id"]
        if (result.role != "development" or result.bundle_id not in bundles or
                result.protocol_id != p["protocol_id"] or result.benchmark_id != benchmark):
            raise ValueError("only matching development evidence is legal")
        if result.intervention.get("information_scope") != "development":
            raise ValueError("explicit development information scope is required")
        if split is not None:
            allowed = {(task, seed) for task in split["task_keys"] for seed in split["seeds"]}
            slots = [(r["task_key"], r["seed"]) for r in result.assignments]
            if (not slots or len(set(slots)) != len(slots) or not set(slots) <= allowed or
                    result.assigned_count != len(slots)):
                raise ValueError("evidence includes nondevelopment/missing/duplicate assignments")
        elif p["scope"] != "offline":
            raise ValueError("runtime evidence needs a frozen development split")
        if result.evidence_level not in {"offline", p["scope"]}:
            raise ValueError("evidence level does not match the protocol")
        if result.outcome not in {"success", "failure", "unknown"}:
            raise ValueError("invalid development outcome")
        if result.assignments:
            for row in result.assignments:
                if row.get("outcome") not in {"success", "failure", "unknown"} or row.get(
                        "validity") not in {"valid", "invalid"}:
                    raise ValueError("invalid development assignment labels")
                if result.evidence_level != "offline" and row["outcome"] != "unknown":
                    if (not row.get("evidence_ref") or row.get("outcome_source") not in {
                            "native_benchmark", "verified_budget_exhaustion"}):
                        raise ValueError("runtime outcome needs native/terminal development evidence")
                    if row["outcome_source"] == "verified_budget_exhaustion" and row["outcome"] != "failure":
                        raise ValueError("budget exhaustion establishes failure only")
        _costs(result.costs)
        return asdict(result)

    def query_development(self, result_ids, *, bundle_ids=None):
        """Read only the development namespace, never sealed/release results."""
        historical=set(self.domain_policy.get('historical_development_bundle_ids', []))
        bundles = set(bundle_ids or [self.base.bundle_id, *historical])
        allowed = {self.base.bundle_id} | historical
        allowed.update(t["bundle_id"] for t in self.store.all("rule_trials")
                       if t["job_id"] == self.job.state["job_id"])
        if not bundles <= allowed:
            raise ValueError("evidence bundle outside this job")
        return [self._validate_result(ExperimentResult(**self.store.get("development_results", key)), bundles)
                for key in result_ids]

    def read_development_events(self, result_id, event_ids=None):
        """Read exact L0 events only from an admitted development result's own view."""
        result = self.query_development([result_id])[0]
        view_id = result["intervention"].get("evidence_view_id")
        if not view_id:
            raise ValueError("development result has no L0 view")
        view = self.store.get("evidence_views", view_id)
        if view["bundle_id"] != result["bundle_id"]:
            raise ValueError("development result/view bundle mismatch")
        if event_ids is None:
            return {"result_id": result_id, "view_id": view_id,
                    "event_index": [{k: e.get(k) for k in ("event_id", "sequence", "kind")} for e in view["l0"]],
                    "available_event_count": len(view["l0"])}
        wanted = set(event_ids)
        events = [e for e in view["l0"] if e["event_id"] in wanted]
        if wanted != {e["event_id"] for e in events}:
            raise ValueError("unknown development event ID")
        return {"result_id": result_id, "view_id": view_id, "events": events,
                "available_event_count": len(view["l0"])}

    def relation_evidence_views(self, result_ids):
        """Apply the bound rhoA to admitted views; original L0 remains complete."""
        from .relation_learning import select_media
        rho = self.store.get("rho", self.base.rho_ref)
        views = []
        for result in self.query_development(result_ids):
            view_id = result["intervention"].get("evidence_view_id")
            if not view_id:
                continue
            view = self.store.get("evidence_views", view_id)
            if view["bundle_id"] != result["bundle_id"]:
                raise ValueError("development result/view bundle mismatch")
            if rho["rhoA"].get("type") != "fixed":
                view = {**view, "selected_media": select_media(rho["rhoA"], view["l0"]),
                        "policy_ref": self.base.rho_ref}
            views.append(view)
        return views

    def prepare_relation_step(self, result_ids, *, requested_worker, allowed_workers,
                              state=None, inspected=False, legal_repairs=(),
                              experiment_permitted=False):
        """Prepare permission-bounded relation decisions without dispatch or charging."""
        from .cooperation import fixed_route, fixed_research_step
        from .relation_learning import version_signature
        evidence = self.query_development(result_ids)
        rho = self.store.get("rho", self.base.rho_ref)
        return {"rho_ref": self.base.rho_ref,
                "evidence_views": self.relation_evidence_views(result_ids),
                "worker": fixed_route(rho, requested_worker, allowed_workers,
                    state=state, version_signature=version_signature(self.base)),
                "research_step": fixed_research_step(rho, self.job,
                    evidence_available=bool(evidence), legal_repairs=legal_repairs,
                    inspected=inspected, experiment_permitted=experiment_permitted, state=state),
                "execution": "prepared_only"}

    def research_relation_admission(self, result_ids, states=None, legal_repairs=()):
        """Consume rhoC for preparation only; Runtime admission remains host-owned."""
        from .cooperation import fixed_research_step
        evidence = self.query_development(result_ids)
        contexts = list(states) if states is not None else sorted({
            row["task_key"] for result in evidence for row in result["assignments"]})
        rho = self.store.get("rho", self.base.rho_ref)
        decisions = [{"state": state, "action": fixed_research_step(rho, self.job,
            evidence_available=bool(evidence), inspected=True, legal_repairs=legal_repairs,
            experiment_permitted=bool(legal_repairs), state=state)} for state in contexts]
        return {"rho_ref": self.base.rho_ref, "decisions": decisions,
                "action": "prepare" if decisions and all(row["action"] == "bounded_experiment"
                                                         for row in decisions) else "defer",
                "execution": "prepared_only", "dynamic_execution_authorized": False}

    def plan_relation_experiment(self, experiment_id, *, opportunity_id, bundle_id,
                                 relations, criterion_ref, distinguishing_prediction,
                                 cost_limit, evidence_level):
        """Plan one program-recomputed rho update within explicit new-task scope."""
        from .relation_learning import preview_relation_feedback
        opportunity = self._owned("opportunities", opportunity_id)
        if opportunity.get("technical_diagnostic_only") or opportunity["source"] in {"new_goal", "exploration"}:
            raise ValueError("relation trial needs observed development opportunity")
        support = preview_relation_feedback(self, opportunity["result_ids"])
        if (not isinstance(relations, list) or not relations or any(not isinstance(name, str) for name in relations)
                or len(set(relations)) != len(relations)
                or not set(relations) <= set(support["supported_relations"])):
            raise ValueError("relation intervention requires supported finite development feedback")
        if not criterion_ref or not distinguishing_prediction or evidence_level not in {"offline", self.protocol["scope"]}:
            raise ValueError("relation experiment needs matching criterion, prediction and evidence level")
        limit = _costs(cost_limit)
        if (not limit or any(v > self.job.remaining().get(k, -1) for k, v in limit.items())
                or (evidence_level != "offline" and _token_policy(self.protocol) == "observed_only"
                    and any(_token_dimension(name) for name in limit))):
            raise ValueError("relation experiment exceeds original hard job resources")
        return self._save("research_experiments", experiment_id, {
            "experiment_id": experiment_id, "opportunity_id": opportunity_id, "bundle_id": bundle_id,
            "intervention": {"relation_feedback_update": list(relations)},
            "relation_result_ids": list(opportunity["result_ids"]), "relation_feedback_support": support,
            "criterion_ref": criterion_ref, "distinguishing_prediction": distinguishing_prediction,
            "cost_limit": limit, "evidence_level": evidence_level,
            "runtime_rule_schema": None, "requires_new_timing_baseline": False,
            "evidence_basis": "observed_development", "exploration": None,
            "frozen_theta_ref": self.base.theta_ref, "execution_status": "prepared",
            "dynamic_execution_authorized": False})

    def query_experience(self, result_ids=None, **filters):
        from .experience import query_experience
        return query_experience(self, result_ids, **filters)

    def discover_opportunity(self, opportunity_id, *, source, result_ids, external_value,
                             competing_explanations, expected_cost, target=None):
        if source not in {"failure", "normal", "technical_diagnostic", "capability_boundary", "new_goal", "exploration"}:
            raise ValueError("unsupported opportunity source")
        evidence = self.query_development(result_ids)
        from .task_policy import exploration_policy
        exploration = exploration_policy(self) if source == "exploration" else None
        if source == "exploration" and exploration is None:
            raise ValueError("exploration requires a registered new-protocol scope")
        if not external_value or (exploration is None and (not evidence or not any(r["data_refs"] for r in evidence))):
            raise ValueError("opportunity needs external value and legal evidence")
        outcomes = {row["outcome"] for r in evidence for row in r["assignments"]
                    if row["validity"] == "valid"}
        outcomes.update(r["outcome"] for r in evidence if not r["assignments"])
        if source == "normal" and "success" not in outcomes:
            raise ValueError("normal opportunity needs successful execution evidence")
        if source == "failure" and not outcomes.intersection({"failure", "unknown"}):
            raise ValueError("failure opportunity needs failed or unknown execution evidence")
        diagnostic = [r for r in evidence if r["data_refs"] and
            any(row["validity"] == "invalid" and row["outcome"] == "unknown" and row.get("evidence_ref") in r["data_refs"]
                for row in r["assignments"])]
        if source == "technical_diagnostic" and not diagnostic:
            raise ValueError("technical diagnosis requires referenced invalid/unknown execution evidence")
        _validate_hypotheses(competing_explanations, _minimum_hypotheses(
            self.protocol, diagnostic=source == "technical_diagnostic"))
        goal_review = None
        if source == "new_goal":
            policy = self.domain_policy
            if (not target or target.get("domain") != policy["domain"] or
                    target.get("kind") not in policy["allowed_goal_kinds"] or
                    not target.get("value_criterion_ref") or not target.get("independent_value_test_ref")):
                raise ValueError("new goal lacks domain eligibility or independent value test")
            goal_review = {"policy_ref": policy["policy_ref"], "eligible_for_proposal": True,
                           "value_status": "unverified", "capability_gain": None}
        elif target is not None:
            raise ValueError("target belongs to the new_goal entry")
        opportunity = {
            "opportunity_id": opportunity_id, "source": source, "result_ids": list(result_ids),
            "external_value": external_value, "hypotheses": competing_explanations,
            "expected_cost": _costs(expected_cost), "target": target, "goal_review": goal_review,
            "evidence_basis": "registered_hypothesis" if exploration else "observed_development",
            "exploration": exploration,
            "experiment_authorized": False, "technical_diagnostic_only": source == "technical_diagnostic"}
        from .task_policy import opportunity_admission, task_policy
        if task_policy(self.protocol).get("opportunity_discovery") is True:
            opportunity["admission"] = opportunity_admission(self.protocol, opportunity, self.job.remaining())
        return self._save("opportunities", opportunity_id, opportunity)

    def inspect_native_finish_opportunity(self, result_ids):
        """Find observed premature success intents in legal medium development records.

        This reports a mismatch, preserving explanations of its cause and the
        guard's benefit for a later comparison. planner_finish records an intent;
        a budget rejection may prevent its tool from accepting that intent.
        This method never infers acceptance or constructs a candidate.
        """
        observations = []
        for result in self.query_development(result_ids):
            if result["evidence_level"] != "simulation":
                continue
            for row in result["assignments"]:
                ref = row.get("evidence_ref")
                if row["validity"] != "valid" or not ref or ref not in result["data_refs"]:
                    continue
                raw = json.loads(Path(ref).read_text(encoding="utf-8"))
                planner = raw.get("planner", {})
                if (raw.get("bundle_id") != self.base.bundle_id or
                        raw.get("protocol_id") != self.protocol["protocol_id"] or
                        planner.get("model") != "gpt-6.1-sol" or planner.get("reasoning_effort") != "medium"):
                    continue
                native = raw.get("native_result_source", {}) or {}
                attempt = raw.get("attempt_outcome_source", {}) or {}
                if ((raw.get("planner_finish") or {}).get("status") == "success" and
                        raw.get("benchmark_success") is False and raw.get("attempt_closed") is True and
                        (native.get("terminated") or attempt.get("native_success"))):
                    observations.append({"development_result_id": result["result_id"],
                        "task_key": row["task_key"], "seed": row["seed"], "evidence_ref": ref,
                        "planner_finish_status": "success", "benchmark_success": False,
                        "attempt_closed": True, "model": "gpt-6.1-sol", "reasoning_effort": "medium",
                        "finish_intent_accepted": None, "finish_acceptance_evidence": "not_established_by_planner_finish",
                        "cause": "unresolved", "guard_benefit": "unverified"})
        return observations

    def inspect_finish_budget_opportunity(self, result_ids):
        """Inspect same-run counted dispatch facts, without inferring repair benefit."""
        observations = []
        for result in self.query_development(result_ids):
            if result["evidence_level"] != "simulation":
                continue
            for row in result["assignments"]:
                ref = row.get("evidence_ref")
                if row["validity"] != "valid" or not ref or ref not in result["data_refs"]:
                    continue
                raw = json.loads(Path(ref).read_text(encoding="utf-8"))
                planner = raw.get("planner", {})
                if (raw.get("bundle_id") != self.base.bundle_id or
                        raw.get("protocol_id") != self.protocol["protocol_id"] or
                        planner.get("model") != "gpt-6.1-sol" or planner.get("reasoning_effort") != "medium"):
                    continue
                root = Path(ref).parent
                context_ref, events_ref = root / "runtime_context.json", root / "action_events.jsonl"
                if not context_ref.exists() or not events_ref.exists():
                    continue
                context = json.loads(context_ref.read_text(encoding="utf-8"))
                identity = {name: raw.get(name) for name in ("run_id", "episode_id", "bundle_id", "protocol_id")}
                if (any(not value or context.get(name) != value for name, value in identity.items()) or
                        context.get("limits", {}).get("tool_calls") != 40):
                    continue
                events = [json.loads(line) for line in events_ref.read_text(encoding="utf-8").splitlines() if line]
                if any(event.get("sequence") != index or any(event.get(name) != value
                       for name, value in identity.items()) for index, event in enumerate(events)):
                    continue
                counted = sum(event.get("kind") == "tool_call" for event in events)
                if counted != 40:
                    continue
                for index, event in enumerate(events[:-1]):
                    end = events[index + 1]
                    if (event.get("kind") == "tool_rejected" and event.get("tool") == "finish" and
                            event.get("reason") == "tool_calls exhausted" and end.get("kind") == "tool_end" and
                            end.get("tool") == "finish" and end.get("status") == "rejected"):
                        observations.append({"development_result_id": result["result_id"],
                            "task_key": row["task_key"], "seed": row["seed"], "evidence_ref": ref,
                            "context_ref": str(context_ref), "action_events_ref": str(events_ref),
                            "tool_limit": 40, "counted_dispatches": counted,
                            "finish_rejection_sequence": event["sequence"], "finish_end_sequence": end["sequence"],
                            "finish_intent_accepted": False, "cause": "unresolved", "hint_benefit": "unverified"})
        return observations

    def plan_rule_experiment(self, experiment_id, *, opportunity_id, bundle_id, chunk_steps=None,
                             finish_requires_native_success=None, planner_finish_reserve_calls=None,
                             planner_segment_handoff_hint=None, planner_task_hint=None, relation_feedback_update=None,
                             criterion_ref, distinguishing_prediction, cost_limit,
                             evidence_level="offline"):
        if relation_feedback_update is not None:
            if any(value is not None for value in (chunk_steps, finish_requires_native_success,
                    planner_finish_reserve_calls, planner_segment_handoff_hint, planner_task_hint)):
                raise ValueError("relation update cannot be mixed with a phi operator")
            return self.plan_relation_experiment(experiment_id, opportunity_id=opportunity_id,
                bundle_id=bundle_id, relations=relation_feedback_update, criterion_ref=criterion_ref,
                distinguishing_prediction=distinguishing_prediction, cost_limit=cost_limit,
                evidence_level=evidence_level)
        opportunity = self._owned("opportunities", opportunity_id)
        if opportunity.get("technical_diagnostic_only"):
            raise ValueError("technical diagnosis does not establish a task behavior intervention prerequisite")
        from .rule_updates import read_phi_chunk, rule_update
        before = read_phi_chunk(self.base.phi_ref)
        after, changes = rule_update(before, chunk_steps=chunk_steps,
                                     finish_requires_native_success=finish_requires_native_success,
                                     planner_finish_reserve_calls=planner_finish_reserve_calls,
                                     planner_segment_handoff_hint=planner_segment_handoff_hint,
                                     planner_task_hint=planner_task_hint)
        intervention = {name: change["after"] for name, change in changes.items()}
        if not criterion_ref or not distinguishing_prediction:
            raise ValueError("experiment needs a criterion and distinguishing prediction")
        if opportunity["source"] == "new_goal":
            raise ValueError("a finite rule trial cannot establish a new capability's independent value")
        if evidence_level not in {"offline", self.protocol["scope"]}:
            raise ValueError("unsupported experiment evidence level")
        from .task_policy import exploration_policy
        exploration = None
        if opportunity["source"] == "exploration":
            if len(intervention) != 1:
                raise ValueError("exploration requires one registered operator")
            exploration = exploration_policy(self, next(iter(intervention)))
            if exploration is None:
                raise ValueError("operator outside registered exploration scope")
            if criterion_ref != exploration["criterion_ref"]:
                raise ValueError("exploration criterion mismatch")
            if any(v > exploration["cost_limit"].get(k, -1) for k, v in _costs(cost_limit).items()):
                raise ValueError("experiment exceeds exploration cost limit")
        trigger_evidence = []
        if "planner_task_hint" in intervention:
            from .task_policy import task_policy
            if not task_policy(self.protocol) or "planner_task_hint" not in self.domain_policy.get("rule_operators", []):
                raise ValueError("task hint outside approved new-task operator scope")
            trigger_evidence = self.query_development(opportunity["result_ids"])
            if not trigger_evidence and not exploration:
                raise ValueError("task hint needs legal development evidence or registered exploration")
        if "finish_requires_native_success" in intervention and evidence_level != "offline":
            trigger_evidence = self.inspect_native_finish_opportunity(opportunity["result_ids"])
            if not trigger_evidence and not exploration:
                raise ValueError("native finish trial needs observed medium development finish/native mismatch")
        if "planner_finish_reserve_calls" in intervention and evidence_level != "offline":
            trigger_evidence = self.inspect_finish_budget_opportunity(opportunity["result_ids"])
            if not trigger_evidence and not exploration:
                raise ValueError("finish reserve trial needs observed counted-tool budget rejection of finish")
        if "planner_segment_handoff_hint" in intervention:
            if "planner_segment_handoff_hint" not in self.domain_policy.get("rule_operators", []):
                raise ValueError("handoff hint outside approved human scope")
            if evidence_level != "offline" and self.domain_policy.get('handoff_scope_approved') is not True:
                raise ValueError("new handoff research scope has not been approved")
            from .experience import handoff_evidence
            trigger_evidence = handoff_evidence(self, opportunity["result_ids"])
            if evidence_level != "offline" and not trigger_evidence and not exploration:
                raise ValueError("handoff hint requires recorded segment handoff evidence")
        for previous in self.store.all("research_experiments"):
            if previous["experiment_id"] == experiment_id or not self.store.path("research_findings", previous["experiment_id"]).exists():
                continue
            old = self.store.get("opportunities", previous["opportunity_id"])
            from .task_policy import task_policy
            same_scope = not task_policy(self.protocol) or (
                previous["protocol_id"] == self.protocol["protocol_id"]
                and previous["base_bundle_id"] == self.base.bundle_id
                and previous.get("exploration") == exploration
            )
            if (same_scope and previous["intervention"] == intervention
                    and old["result_ids"] == opportunity["result_ids"]):
                raise ValueError("same processed evidence/operator requires new evidence")
        limit = _costs(cost_limit)
        if (evidence_level != "offline" and _token_policy(self.protocol) == "observed_only"
                and any(_token_dimension(name) for name in limit)):
            raise ValueError("observed-only tokens cannot be declared hard experiment limits")
        if not limit or any(v > self.job.remaining().get(k, -1) for k, v in limit.items()):
            raise ValueError("experiment exceeds the job's original budget")
        return self._save("research_experiments", experiment_id, {
            "experiment_id": experiment_id, "opportunity_id": opportunity_id,
            "bundle_id": bundle_id, "intervention": intervention,
            "runtime_rule_schema": after["schema"], "requires_new_timing_baseline": "chunk_steps" in changes,
            "criterion_ref": criterion_ref, "distinguishing_prediction": distinguishing_prediction,
            "cost_limit": limit, "evidence_level": evidence_level,
            "trigger_evidence": trigger_evidence,
            "exploration": exploration,
            "evidence_basis": "registered_hypothesis" if exploration else "observed_development",
            "frozen_theta_ref": self.base.theta_ref,
            "execution_status": "prepared", "dynamic_execution_authorized": False})

    def prepare_rule_trial(self, experiment_id):
        plan = self._owned("research_experiments", experiment_id)
        if "relation_feedback_update" in plan["intervention"]:
            from .relation_learning import write_relation_trial, SCHEMA
            from .runtime import resolve_reference
            base_phi_schema = resolve_reference(self.base.phi_ref).get("schema")
            target, changes = write_relation_trial(self, experiment_id=experiment_id,
                bundle_id=plan["bundle_id"], result_ids=plan["relation_result_ids"],
                relations=plan["intervention"]["relation_feedback_update"])
            return self._save("rule_trials", experiment_id, {
                "experiment_id": experiment_id, "bundle_id": target.bundle_id,
                "runtime_rules_ref": target.phi_ref, "diff_ref": target.diff_ref,
                "rho_ref": target.rho_ref, "rho_artifact_ref": str(self.store.path("rho", target.rho_ref).resolve()),
                "changes": changes, "relation_policy_schema": SCHEMA,
                "runtime_rule_schema": base_phi_schema, "runtime_loaded": False, "requires_new_timing_baseline": False})
        target, diff = write_rule_trial(self.store, self.base, bundle_id=plan["bundle_id"],
                                       artifact_id=experiment_id,
                                       **plan["intervention"])
        return self._save("rule_trials", experiment_id, {
            "experiment_id": experiment_id, "bundle_id": target.bundle_id,
            "runtime_rules_ref": target.phi_ref, "diff_ref": target.diff_ref,
            "changes": diff["changes"], "runtime_loaded": False,
            "runtime_rule_schema": plan.get("runtime_rule_schema", "rsi-runtime-rules-v1"),
            "requires_new_timing_baseline": diff["requires_new_timing_baseline"]})

    def record_experiment_result(self, experiment_id, result):
        plan = self._owned("research_experiments", experiment_id)
        self._owned("rule_trials", experiment_id)
        if isinstance(result, dict):
            result = ExperimentResult(**result)
        self._validate_result(result, {plan["bundle_id"]})
        if (result.criteria_ref != plan["criterion_ref"] or result.evidence_level != plan["evidence_level"] or
                any(type(result.intervention.get(name)) is not type(value) or
                    result.intervention.get(name) != value for name, value in plan["intervention"].items())):
            raise ValueError("result does not match the prepared intervention/criterion/level")
        hard_costs = self._hard_costs(plan, result.costs)
        if any(v > plan["cost_limit"].get(k, self.job.state["limits"].get(k, -1)
                                       if self._observed_only(plan) else -1)
               for k, v in hard_costs.items()):
            raise ValueError("experiment result exceeds its registered resource limit")
        previous = self.store.path("research_findings", experiment_id)
        if previous.exists():
            finding = self._owned("research_findings", experiment_id)
            if finding["result_id"] != result.result_id or self.store.get(
                    "development_results", result.result_id) != asdict(result):
                raise ValueError("experiment already has a different finding")
            return finding
        reason = "experiment " + experiment_id
        prepaid = [c for c in self.job.state["charges"] if c["reason"] == reason]
        if prepaid:
            if len(prepaid) != 1 or prepaid[0]["costs"] != hard_costs:
                raise ValueError("result costs do not match the experiment charge")
        else:
            if any(v > self.job.remaining().get(k, -1) for k, v in hard_costs.items()):
                raise ValueError("result exceeds the original remaining job budget")
            self.job.consume(hard_costs, reason)
        self.store.save_development_result(result)
        return self._save("research_findings", experiment_id, {
            "experiment_id": experiment_id, "result_id": result.result_id,
            "outcome": result.outcome, "evidence_level": result.evidence_level,
            "costs": result.costs, "hard_costs": hard_costs,
            "token_policy": "observed_only" if self._observed_only(plan) else "registered_known_cost",
            "cost_observation": result.intervention.get("cost_observation"),
            "hypotheses_resolved": False})

    def run_offline_experiment(self, experiment_id, evaluator, *, cost):
        """Call a known-cost bounded offline evaluator; no dynamic backend here."""
        plan = self._owned("research_experiments", experiment_id)
        trial = self._owned("rule_trials", experiment_id)
        cost = _costs(cost)
        if plan["evidence_level"] != "offline":
            raise ValueError("dynamic experiments require the authorized backend")
        if any(v > plan["cost_limit"].get(k, -1) for k, v in cost.items()):
            raise ValueError("offline action exceeds the experiment limit")
        if self.store.path("research_findings", experiment_id).exists():
            raise ValueError("experiment already has a finding; use a new experiment identity")
        if any(c["reason"] == "experiment " + experiment_id for c in self.job.state["charges"]):
            raise ValueError("experiment was already dispatched; reconcile its result before retrying")
        result = self.job.run(lambda: evaluator(plan, trial), cost=cost,
                              reason="experiment " + experiment_id)
        actual = result.costs if isinstance(result, ExperimentResult) else result["costs"]
        if actual != cost:
            raise ValueError("offline evaluator cost differs from its prepaid known cost")
        return self.record_experiment_result(experiment_id, result)

    def submit_candidate(self, candidate_id, *, experiment_id, finding_revision_id=None, cost_snapshot_ref=None,
                         additional_experiment_ids=()):
        plan = self._owned("research_experiments", experiment_id)
        trial = self._owned("rule_trials", experiment_id)
        finding = self._owned("research_finding_revisions", finding_revision_id) if finding_revision_id else self._owned("research_findings", experiment_id)
        if finding_revision_id and (finding["experiment_id"] != experiment_id or not finding.get("screening_checks")
                                    or not all(v is True for v in finding["screening_checks"].values())
                                    or cost_snapshot_ref is None):
            raise ValueError("revised candidate requires complete program screening and cost lineage")
        candidate_costs = self._candidate_costs()
        cost_scope = ("job_findings_plus_other_job_charges" if
                      _token_policy(self.protocol) == "observed_only" else "job_used_at_submission")
        cost_mapping = {}
        if cost_snapshot_ref:
            snapshot = json.loads(Path(cost_snapshot_ref).read_text())
            phase = snapshot["phase"]
            if phase.get("unresolved_costs"):
                raise ValueError("candidate phase has unresolved actual costs")
            if (finding_revision_id and snapshot.get("revision_key") != finding_revision_id):
                raise ValueError("candidate cost snapshot revision mismatch")
            # Match the Runtime audit/evaluator vocabulary. Keep all other
            # measured phase dimensions, including episodes and original Research.
            cost_mapping = {"wall_seconds": "whole_job_wall_seconds",
                            "gpu_seconds": "gpu_device_seconds_envelope",
                            "storage_bytes": "artifact_bytes"}
            candidate_costs = {cost_mapping.get(k, k): v for k, v in _costs(phase["used"]).items()}
            cost_scope = "authoritative_phase_history_at_submission"
        self.query_development([finding["result_id"]], bundle_ids=[trial["bundle_id"]])
        result = self.store.get("development_results", finding["result_id"])
        development_results = [finding["result_id"]]
        for extra_id in additional_experiment_ids:
            extra_plan = self._owned("research_experiments", extra_id)
            extra_trial = self._owned("rule_trials", extra_id)
            extra_finding = self._owned("research_findings", extra_id)
            if (extra_id == experiment_id or extra_plan["bundle_id"] != trial["bundle_id"]
                    or extra_trial["runtime_rules_ref"] != trial["runtime_rules_ref"]
                    or extra_plan["intervention"] != plan["intervention"]):
                raise ValueError("additional development finding must test the same candidate artifact")
            self.query_development([extra_finding["result_id"]], bundle_ids=[trial["bundle_id"]])
            if extra_finding["result_id"] in development_results:
                raise ValueError("duplicate candidate development finding")
            development_results.append(extra_finding["result_id"])
        relation_update = "relation_feedback_update" in plan["intervention"]
        if relation_update:
            selected = plan["intervention"]["relation_feedback_update"]
            consumption = result["intervention"].get("relation_consumption", {})
            required = {"native_success", "technical_valid", "runtime_intervention_loaded",
                        "critical_progress", "collection_complete"}
            for checked_id in development_results:
                checked = self.store.get("development_results", checked_id)
                proof = checked["intervention"].get("relation_consumption", {})
                screening = checked["intervention"].get("screening_checks", {})
                if (checked["intervention"].get("runtime_loaded") is not True or not required <= screening.keys()
                        or any(value is not True for value in screening.values())
                        or proof.get("schema") != "rsi-relation-consumption-v1"
                        or proof.get("rho_ref") != trial["rho_ref"]
                        or proof.get("identity") != checked["intervention"].get("identity")
                        or any(proof.get("relations", {}).get(name, {}).get("value") is not True
                               or not proof.get("relations", {}).get(name, {}).get("source_refs") for name in selected)):
                    raise ValueError("relation Candidate requires new development screening and actual consumer receipts")
        candidate_changes = ({"rho_ref": trial["rho_ref"], "diff_ref": trial["diff_ref"]} if relation_update
                             else {"phi_ref": trial["runtime_rules_ref"], "diff_ref": trial["diff_ref"]})
        artifact_refs = ([trial["rho_artifact_ref"], trial["diff_ref"]] if relation_update
                         else [trial["runtime_rules_ref"], trial["diff_ref"]])
        candidate = Candidate(candidate_id, self.base.bundle_id, trial["bundle_id"],
            self.protocol["protocol_id"], self.protocol["benchmark_id"],
            candidate_changes, artifact_refs, development_results,
            candidate_costs, {"theta_frozen": True,
                **({"relation_policy_schema": trial["relation_policy_schema"],
                    "relation_consumption": consumption} if relation_update else {}),
                **({"finding_revision_id": finding_revision_id, "cost_snapshot_ref": str(cost_snapshot_ref),
                    "cost_lineage_scope": "authoritative_phase_history_including_original_Research_and_failed_attempts"} if finding_revision_id else {}),
                "runtime_rule_schema": trial.get("runtime_rule_schema", "rsi-runtime-rules-v1"),
                "evidence_level": plan["evidence_level"],
                "runtime_loaded": result["intervention"].get("runtime_loaded", False),
                "cost_observation": finding.get("cost_observation"),
                "requires_new_timing_baseline": trial.get("requires_new_timing_baseline", True),
                "cost_scope": cost_scope,
                **({"cost_dimension_mapping": cost_mapping,
                    "phase_token_observations_ref": str(cost_snapshot_ref),
                    "phase_token_policy": "observed_only; source observations retained; unknown totals omitted"}
                   if cost_snapshot_ref else {}),
                "token_policy": _token_policy(self.protocol)},
            approval_scope=(self.protocol["scope"] if plan["evidence_level"] != "offline" and
                            self.protocol.get("dynamic_authorized") is True else "offline"),
            novelty_type="configuration")
        self.store.save_candidate(candidate)
        return asdict(candidate)

    def defer(self, decision_id, *, opportunity_id, reason, missing_evidence):
        self._owned("opportunities", opportunity_id)
        if not reason:
            raise ValueError("deferral requires a decision reason")
        if self.store.path('research_decisions', decision_id).exists():
            previous = self._owned('research_decisions', decision_id)
            if (previous['opportunity_id'] != opportunity_id or previous['reason'] != reason
                    or previous['missing_evidence'] != list(missing_evidence) or previous['action'] != 'defer'):
                raise ValueError('conflicting deferral decision')
            return previous
        return self._save("research_decisions", decision_id, {
            "decision_id": decision_id, "opportunity_id": opportunity_id, "action": "defer",
            "reason": reason, "missing_evidence": list(missing_evidence),
            "remaining_budget": self.job.remaining()})


class ResearchToolkit:
    """Explicit mutating tools; RPent readonly scheduling is not an ACL."""
    names = ("query_development", "read_development_events", "query_experience", "discover_opportunity", "inspect_native_finish_opportunity",
             "inspect_finish_budget_opportunity", "plan_rule_experiment",
             "prepare_rule_trial", "submit_candidate", "defer")

    def __init__(self, tools):
        self.tools = tools

    def call(self, name, arguments):
        if name not in self.names:
            raise ValueError("tool outside the finite research interface")
        return getattr(self.tools, name)(**arguments)


class ResearchLLMAdapter:
    """Offline request factory/response validation; never sends model requests."""
    def __init__(self, *, model="gpt-6.1-sol", reasoning_effort="high"):
        if model != "gpt-6.1-sol" or reasoning_effort not in {"high", "medium"}:
            raise ValueError("ResearchLLM requires gpt-6.1-sol/high; medium is historical parsing only")
        self.model, self.reasoning_effort = model, reasoning_effort

    def prepare_request(self, tools, *, request_id, result_ids, max_input_tokens, max_output_tokens):
        if self.reasoning_effort != "high":
            raise ValueError("new Research requests require gpt-6.1-sol/high; medium is historical parsing only")
        job = tools.job
        evidence = tools.query_development(result_ids)
        remaining = job.remaining()
        observed_only = _token_policy(tools.protocol) == "observed_only"
        for key, value in (("input_tokens", max_input_tokens), ("output_tokens", max_output_tokens)):
            if type(value) is not int or value <= 0 or (not observed_only and value > remaining.get(key, 0)):
                raise ValueError("request exceeds the original token budget")
        if observed_only and (remaining.get("model_requests", 0) < 1 or
                              remaining.get("whole_job_wall_seconds", 0) <= 0 or
                              job.state["status"] != "running"):
            raise ValueError("request needs remaining model_requests and whole-job wall budget")
        request = {"request_id": request_id, "model": self.model, "reasoning_effort": self.reasoning_effort,
                "job_id": job.state["job_id"], "bundle_id": job.state["bundle_id"],
                "evidence": evidence, "tools": list(ResearchToolkit.names),
                "native_finish_observations": tools.inspect_native_finish_opportunity(result_ids),
                "finish_budget_observations": tools.inspect_finish_budget_opportunity(result_ids),
                "max_input_tokens": max_input_tokens, "max_output_tokens": max_output_tokens,
                "token_policy": "observed_only" if observed_only else "registered_known_cost",
                "token_parameters_role": "reference_only" if observed_only else "budget_bound",
                "execution": {"status": "prepared_only", "requested_model": self.model,
                              "requested_reasoning_effort": self.reasoning_effort,
                              "effective_model": None, "effective_reasoning_effort": None}}
        views = tools.relation_evidence_views(result_ids)
        if views:
            request["evidence_views"] = views
        tools.store.put("llm_requests", request_id, request)
        return request

    def response_schema(self, request_id):
        text = {"type": "string", "minLength": 1}
        hypothesis = {"type": "object", "additionalProperties": False,
            "properties": {name: text for name in ("explanation", "prediction", "distinguishing_test")},
            "required": ["explanation", "prediction", "distinguishing_test"]}
        rule = {"type": "object", "additionalProperties": False,
                "properties": {"planner_finish_reserve_calls": {"type": "integer", "const": 1}},
                "required": ["planner_finish_reserve_calls"]}
        experiment = {"type": "object", "additionalProperties": False,
            "properties": {"criterion_ref": text, "distinguishing_prediction": text},
            "required": ["criterion_ref", "distinguishing_prediction"]}
        properties = {"schema": {"type": "string", "const": "rsi-research-response-v1"},
            "request_id": {"type": "string", "const": request_id},
            "decision": {"type": "string", "enum": ["propose_finish_reserve_hint", "defer"]}, "reason": text,
            "evidence_refs": {"type": "array", "items": text, "minItems": 1},
            "competing_explanations": {"type": "array", "items": hypothesis, "minItems": 2},
            "proposed_rule": {"anyOf": [rule, {"type": "null"}]},
            "experiment": {"anyOf": [experiment, {"type": "null"}]},
            "tool_calls": {"type": "array", "maxItems": 0, "items": {"type": "object", "properties": {},
                             "additionalProperties": False, "required": []}}}
        return {"type": "object", "additionalProperties": False,
                "properties": properties, "required": list(properties)}

    def _validate_proposal(self, tools, request, response):
        schema = self.response_schema(request["request_id"])
        if set(response) != set(schema["required"]) or response.get("schema") != "rsi-research-response-v1":
            raise ValueError("research response fields do not match the structured proposal schema")
        hypotheses = response.get("competing_explanations")
        required = {"explanation", "prediction", "distinguishing_test"}
        if (not isinstance(hypotheses, list) or len(hypotheses) < 2 or
                any(not isinstance(item, dict) or set(item) != required or
                    any(not isinstance(item[name], str) or not item[name].strip() for name in required)
                    for item in hypotheses) or len({item["explanation"] for item in hypotheses}) != len(hypotheses)):
            raise ValueError("research proposal requires distinct testable competing explanations")
        legal_refs = {ref for result in request["evidence"] for ref in result["data_refs"]}
        legal_refs.update(result["result_id"] for result in request["evidence"])
        for observation in request.get("finish_budget_observations", []):
            legal_refs.update(observation[name] for name in ("evidence_ref", "context_ref", "action_events_ref"))
        refs = response.get("evidence_refs")
        if (not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in legal_refs for ref in refs)
                or not isinstance(response.get("reason"), str) or not response["reason"].strip() or response["tool_calls"] != []):
            raise ValueError("proposal needs legal evidence references, a reason and no dispatched tools")
        if response["decision"] == "defer":
            if response["proposed_rule"] is not None or response["experiment"] is not None:
                raise ValueError("defer must not smuggle a candidate or experiment")
        elif response["decision"] == "propose_finish_reserve_hint":
            rule, experiment = response["proposed_rule"], response["experiment"]
            if (not request.get("finish_budget_observations") or rule != {"planner_finish_reserve_calls": 1} or
                    type(rule.get("planner_finish_reserve_calls")) is not int or not isinstance(experiment, dict) or
                    set(experiment) != {"criterion_ref", "distinguishing_prediction"} or
                    experiment["criterion_ref"] != tools.protocol["protocol_id"] or
                    not isinstance(experiment["distinguishing_prediction"], str) or not experiment["distinguishing_prediction"].strip()):
                raise ValueError("finish reserve proposal lacks observed eligibility or a frozen distinguishing experiment")
        else:
            raise ValueError("unsupported research proposal decision")

    def record_execution(self, tools, request_id, actual_costs, transport_facts):
        """Record the trusted caller's transport facts; this does not dispatch a request.

        RequestGate and the external process supervisor own actual admission and
        stopping. Model-produced text is never a source of effective settings.
        """
        request = tools.store.get("llm_requests", request_id)
        if (request["job_id"] != tools.job.state["job_id"] or request["bundle_id"] != tools.base.bundle_id):
            raise ValueError("execution belongs to another research job")
        if request.get("token_policy") != "observed_only":
            raise ValueError("transport cost recording requires the observed-only request interface")
        if (transport_facts.get("effective_model") != self.model or
                transport_facts.get("effective_reasoning_effort") != self.reasoning_effort or
                not transport_facts.get("evidence_ref")):
            raise ValueError("execution needs matching model/effort from transport evidence")
        costs = _costs(actual_costs)
        if not {"whole_job_wall_seconds", "model_requests"} <= costs.keys() or costs["model_requests"] < 1:
            raise ValueError("LLM execution is missing its wall/request hard costs")
        hard = {name: value for name, value in costs.items()
                if name in tools.job.state["limits"] and not _token_dimension(name)}
        record = {"request_id": request_id, "job_id": tools.job.state["job_id"],
                  "base_bundle_id": tools.base.bundle_id, "costs": costs, "hard_costs": hard,
                  "token_policy": "observed_only", "transport_facts": dict(transport_facts)}
        if tools.store.path("llm_executions", request_id).exists():
            previous = tools.store.get("llm_executions", request_id)
            if previous != record:
                raise ValueError("LLM request already has a different execution record")
            return previous
        reason = "llm " + request_id
        prepaid = [c for c in tools.job.state["charges"] if c["reason"] == reason]
        if prepaid:
            if len(prepaid) != 1 or prepaid[0]["costs"] != hard:
                raise ValueError("LLM costs do not match the existing hard charge")
        else:
            if any(value > tools.job.remaining()[name] for name, value in hard.items()):
                raise ValueError("LLM execution exceeds the original hard budget")
            tools.job.consume(hard, reason)
        tools.store.put("llm_executions", request_id, record)
        return record

    def validate_response(self, tools, response):
        if not response.get("request_id"):
            raise ValueError("response lacks matching observed model/effort/request identity")
        request = tools.store.get("llm_requests", response["request_id"])
        if (request["job_id"] != tools.job.state["job_id"] or
                request["bundle_id"] != tools.base.bundle_id):
            raise ValueError("response belongs to another research job")
        facts = (tools.store.get("llm_executions", response["request_id"])["transport_facts"]
                 if request.get("token_policy") == "observed_only" else response)
        if (facts.get("effective_model") != self.model or
                facts.get("effective_reasoning_effort") != self.reasoning_effort):
            raise ValueError("response lacks matching transport-observed model/effort")
        if "schema" in response:
            self._validate_proposal(tools, request, response)
        calls = response.get("tool_calls")
        if not isinstance(calls, list) or any(c.get("name") not in ResearchToolkit.names or
                not isinstance(c.get("arguments"), dict) for c in calls):
            raise ValueError("invalid research tool calls")
        stored = response
        if request.get("token_policy") == "observed_only":
            stored = {**response, "effective_model": facts["effective_model"],
                      "effective_reasoning_effort": facts["effective_reasoning_effort"],
                      "effective_settings_source": "llm_executions.transport_facts",
                      "effective_settings_evidence_ref": facts["evidence_ref"]}
        tools.store.put("llm_responses", response["request_id"], stored)
        return calls


class BoundedResearchAdapter(ResearchLLMAdapter):
    """Finite opportunity selection and artifact preparation, with no dispatch API.

    v1 finish-reserve research remains unchanged. Human policy chooses the
    operators; the researcher chooses an evidenced opportunity or defers.
    """

    operators = (
        "chunk_steps",
        "finish_requires_native_success",
        "planner_finish_reserve_calls",
        "planner_segment_handoff_hint",
        "planner_task_hint",
        "relation_feedback_update",
    )

    def _hypothesis_minimum(self, tools):
        return _minimum_hypotheses(tools.protocol,
            diagnostic=tools.domain_policy.get("research_mode") == "technical_diagnostic")

    def prepare_request(self, tools, **kwargs):
        # Keep this request distinct from the immutable v1 request contract.
        request_id = kwargs["request_id"]
        allowed = tools.domain_policy.get("rule_operators", [])
        diagnostic = tools.domain_policy.get("research_mode") == "technical_diagnostic"
        from .task_policy import exploration_policy, task_policy
        policy = task_policy(tools.protocol)
        opportunity_only = tools.domain_policy.get("opportunity_only") is True
        relation_only = tools.domain_policy.get("relation_only") is True
        if relation_only and (opportunity_only or allowed != ["relation_feedback_update"]):
            raise ValueError("relation-only requests need only the registered feedback operator")
        fixed_results = tools.domain_policy.get("fixed_relation_result_ids")
        if fixed_results is not None and (not relation_only or list(kwargs["result_ids"]) != fixed_results):
            raise ValueError("relation request differs from its fixed development denominator")
        if opportunity_only and (policy.get("opportunity_discovery") is not True or allowed):
            raise ValueError("opportunity-only needs explicit discovery policy and no trial operators")
        if (not allowed and not diagnostic and not opportunity_only) or not set(allowed) <= set(self.operators):
            raise ValueError("human domain policy must name supported rule operators")
        relation_support = None
        if "relation_feedback_update" in allowed:
            from .relation_learning import preview_relation_feedback
            relation_support = preview_relation_feedback(tools, kwargs["result_ids"])
            if not relation_support["supported_relations"]:
                raise ValueError("relation operator has no supported new development feedback")
        exploration = exploration_policy(tools)
        if "planner_task_hint" in allowed and not policy:
            raise ValueError("task hint requires explicit new-task policy")
        if not kwargs["result_ids"] and not exploration:
            raise ValueError("empty development input requires registered exploration scope")
        if exploration and not set(allowed) <= set(exploration["operators"]):
            raise ValueError("request operators exceed exploration scope")
        request = super().prepare_request(tools, **kwargs)
        envelope = {
            **request,
            "schema": "rsi-bounded-research-request-v2",
            "domain_policy": tools.domain_policy,
            "allowed_operators": allowed,
            "base_bundle": asdict(tools.base),
            "base_origin": "human-selected engineering snapshot; not an autonomous research finding",
            "research_mode": "exploratory" if exploration else "technical_diagnostic" if diagnostic else "behavior_proposal",
            "experience_overview": tools.query_experience(kwargs["result_ids"]),
            "instructions": "Choose evidence-backed normal, failure, technical-diagnostic or capability-boundary opportunities. "
            "Retain competing explanations and a distinguishing prediction. Prepare one finite rule "
            "artifact or defer with missing evidence/operators. No experiments, release or evaluation edits.",
            "autonomy_status": "prepared_only_no_research_execution",
        }
        if relation_support is not None:
            envelope["relation_learning"] = relation_support
            envelope["instructions"] += (
                " For relation_feedback_update choose a nonempty unique subset of supported_relations. "
                "The program recomputes feedback from the selected legal result_ids and freezes rho; "
                "you cannot supply policy values or feedback labels. Keep missing worker supervision explicit.")
        if relation_only:
            envelope["relation_only"] = True
            envelope["remaining_job_budget"] = tools.job.remaining()
            if fixed_results is not None:
                envelope["fixed_relation_result_ids"] = list(fixed_results)
                envelope["instructions"] += " Keep every registered result_id in the fixed denominator; do not drop negative feedback."
        if opportunity_only:
            envelope["opportunity_only"] = True
            envelope["instructions"] += " Propose an unlisted goal or defer; do not prepare a trial or Candidate."
        if "planner_task_hint" in allowed:
            from .rule_updates import TASK_HINT_SEMANTICS
            envelope["operator_semantics"] = {"planner_task_hint": TASK_HINT_SEMANTICS}
        minimum = self._hypothesis_minimum(tools)
        if minimum == 1:
            envelope["minimum_hypotheses"] = minimum
        if policy:
            envelope["task_policy"] = policy
        if exploration:
            envelope["exploration"] = exploration
            envelope["instructions"] = (
                "Test the registered hypothesis within its development slots and budget. "
                "Missing trigger evidence is a research question. Retain competing explanations; "
                "prepare one allowed rule or defer. No experiments or release actions are dispatched here."
            )
        if policy.get("opportunity_discovery") is True and not relation_only:
            envelope["opportunity_discovery"] = True
            envelope["remaining_job_budget"] = tools.job.remaining()
            envelope["instructions"] += (
                " Successful development runs can motivate normal opportunities. "
                "You may propose an unlisted capability in the allowed domain and goal kind, "
                "using a free-text target description and the registered independent value references. "
                "Use propose_goal with source new_goal and no intervention; it is handed to "
                "independent value review before a new experiment protocol is designed. "
                "The consumer checks the proposed next-action cost against the remaining job "
                "budget and preserves the opportunity with a deferral when it does not fit."
            )
        if minimum == 1:
            envelope["instructions"] += " One testable explanation is sufficient; add alternatives when they affect the test."
        from .research_transport import select_input
        envelope = select_input(tools, envelope)
        tools.store.put("bounded_requests", request_id, envelope)
        from .research_transport import pack
        transport = pack(envelope)
        tools.store.put("bounded_transports", request_id, transport)
        return transport

    def response_schema(self, request_id, *, tools=None):
        text = {"type": "string", "minLength": 1}
        hypothesis = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                k: text for k in ("explanation", "prediction", "distinguishing_test")
            },
            "required": ["explanation", "prediction", "distinguishing_test"],
        }
        intervention = {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chunk_steps": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["chunk_steps"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "finish_requires_native_success": {
                            "type": "boolean",
                            "const": True,
                        }
                    },
                    "required": ["finish_requires_native_success"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "planner_finish_reserve_calls": {"type": "integer", "const": 1}
                    },
                    "required": ["planner_finish_reserve_calls"],
                },
                {"type": "object", "additionalProperties": False,
                 "properties": {"planner_segment_handoff_hint": {"type": "boolean", "const": True}},
                 "required": ["planner_segment_handoff_hint"]},
                {"type": "object", "additionalProperties": False,
                 "properties": {"planner_task_hint": {"type": "string", "minLength": 1, "maxLength": 1200}},
                 "required": ["planner_task_hint"]},
                {"type": "null"},
            ]
        }
        if tools is not None:
            request = tools.store.get("bounded_requests", request_id)
            support = request.get("relation_learning")
            if support is not None:
                intervention["anyOf"].insert(0, {"type": "object", "additionalProperties": False,
                    "properties": {"relation_feedback_update": {"type": "array", "minItems": 1,
                        "maxItems": len(support["supported_relations"]),
                        "items": {"type": "string", "enum": support["supported_relations"]}}},
                    "required": ["relation_feedback_update"]})
        properties = {
            "schema": {"type": "string", "const": "rsi-bounded-research-response-v2"},
            "request_id": {"type": "string", "const": request_id},
            "decision": {"type": "string", "enum": ["prepare_rule_trial", "defer"]},
            "source": {
                "type": "string",
                "enum": ["normal", "failure", "technical_diagnostic", "capability_boundary", "exploration"],
            },
            "result_ids": {"type": "array", "minItems": 0, "items": text},
            "reason": text,
            "external_value": text,
            "competing_explanations": {
                "type": "array",
                "minItems": tools.store.get("bounded_requests", request_id).get("minimum_hypotheses", 2) if tools is not None else 2,
                "items": hypothesis,
            },
            "intervention": intervention,
            "distinguishing_prediction": text,
            "stop_condition": text,
            "missing_evidence": {"type": "array", "items": text},
        }
        if tools is not None and tools.store.get("bounded_requests", request_id).get("opportunity_discovery") is True:
            properties["schema"]["const"] = "rsi-bounded-research-response-v3"
            properties["source"]["enum"].append("new_goal")
            properties["decision"]["enum"].append("propose_goal")
            properties["target"] = {"anyOf": [
                {"type": "null"},
                {"type": "object", "additionalProperties": False,
                 "properties": {name: text for name in (
                     "domain", "kind", "description", "value_criterion_ref", "independent_value_test_ref")},
                 "required": ["domain", "kind", "description", "value_criterion_ref", "independent_value_test_ref"]},
            ]}
        if tools is not None:
            fixed = tools.store.get("bounded_requests", request_id).get("fixed_relation_result_ids")
            if fixed is not None:
                properties["result_ids"].update(minItems=len(fixed), maxItems=len(fixed),
                    items={"type": "string", "enum": fixed})
        if tools is not None and tools.store.get("bounded_requests", request_id).get("opportunity_only") is True:
            properties["decision"]["enum"] = ["propose_goal", "defer"]
            properties["intervention"] = {"anyOf": [{"type": "null"}]}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        }

    def apply_response(
        self, tools, response, *, evidence_level, cost_limit, synthetic=False,
        reserved_bundle_id=None, development_states=None
    ):
        """Validate a proposal and prepare an artifact; never execute an experiment.

        Synthetic responses are engineering fixtures, never model execution facts.
        Real responses require already recorded, trusted transport evidence.
        """
        if synthetic and evidence_level != "offline":
            raise ValueError("synthetic responses must remain offline")
        request_id = response.get("request_id")
        request = tools.store.get("bounded_requests", request_id)
        if (
            request["job_id"] != tools.job.state["job_id"]
            or request["bundle_id"] != tools.base.bundle_id
        ):
            raise ValueError("request belongs to another job")
        if not synthetic:
            facts = tools.store.get("llm_executions", request_id)["transport_facts"]
            if (facts["effective_model"], facts["effective_reasoning_effort"]) != (
                self.model,
                self.reasoning_effort,
            ):
                raise ValueError("trusted model/effort evidence required")
        schema = BoundedResearchAdapter.response_schema(self, request_id, tools=tools)
        if (
            set(response) != set(schema["required"])
            or response["schema"] != schema["properties"]["schema"]["const"]
        ):
            raise ValueError("invalid bounded response fields")
        for name in (
            "reason",
            "external_value",
            "distinguishing_prediction",
            "stop_condition",
        ):
            if not isinstance(response[name], str) or not response[name].strip():
                raise ValueError("proposal needs value, prediction and stop condition")
        if not isinstance(response["missing_evidence"], list) or any(
            not isinstance(x, str) or not x for x in response["missing_evidence"]
        ):
            raise ValueError("missing evidence must list concrete gaps")
        _validate_hypotheses(response["competing_explanations"], request.get("minimum_hypotheses", 2))
        legal = {r["result_id"] for r in request["evidence"]}
        exploratory = request.get("research_mode") == "exploratory"
        if response["source"] == "exploration" and not exploratory:
            raise ValueError("exploration source outside prepared request")
        if exploratory and response["source"] != "exploration":
            raise ValueError("exploratory request must retain its hypothesis evidence basis")
        if (not response["result_ids"] and not exploratory) or not set(response["result_ids"]) <= legal:
            raise ValueError("proposal evidence outside supplied development input")
        decision, intervention = response["decision"], response["intervention"]
        fixed = request.get("fixed_relation_result_ids")
        if fixed is not None and (len(response["result_ids"]) != len(fixed)
                                  or set(response["result_ids"]) != set(fixed)):
            raise ValueError("relation proposal must retain its fixed development denominator")
        if decision not in schema["properties"]["decision"]["enum"]:
            raise ValueError("decision outside prepared response schema")
        target = response.get("target")
        if response["source"] not in schema["properties"]["source"]["enum"]:
            raise ValueError("opportunity source outside prepared response schema")
        if response["source"] == "new_goal":
            required_target = {"domain", "kind", "description", "value_criterion_ref", "independent_value_test_ref"}
            if (decision not in {"propose_goal", "defer"} or intervention is not None or
                    not isinstance(target, dict) or set(target) != required_target or
                    any(not isinstance(value, str) or not value.strip() for value in target.values())):
                raise ValueError("new goal requires a concrete target and no rule intervention")
        elif target is not None or decision == "propose_goal":
            raise ValueError("target and propose_goal belong to the new_goal entry")
        if decision == "defer":
            if intervention is not None or not response["missing_evidence"]:
                raise ValueError("defer needs gaps and no intervention")
        elif decision == "propose_goal":
            if request.get("opportunity_discovery") is not True:
                raise ValueError("new goal outside prepared opportunity discovery scope")
        elif decision == "prepare_rule_trial":
            if request.get("research_mode") == "technical_diagnostic":
                raise ValueError("diagnostic request cannot trigger behavior trial")
            if (
                not isinstance(intervention, dict)
                or len(intervention) != 1
                or not set(intervention) <= set(request["allowed_operators"])
            ):
                raise ValueError("intervention outside human operator policy")
        else:
            raise ValueError("unsupported bounded decision")
        opportunity_id = request_id + "-opportunity"
        opportunity = tools.discover_opportunity(
            opportunity_id,
            source=response["source"],
            result_ids=response["result_ids"],
            external_value=response["external_value"],
            competing_explanations=response["competing_explanations"],
            expected_cost=cost_limit,
            target=target,
        )
        admission = opportunity.get("admission")
        tools._save(
            "bounded_proposals",
            request_id,
            {
                "response": response,
                "origin": "synthetic_engineering_fixture"
                if synthetic
                else "research_transport",
                "stop_condition": response["stop_condition"],
                **({"opportunity_admission": admission} if admission is not None else {}),
                "execution_authorized": False,
            },
        )
        if decision == "defer" or (admission is not None and admission["action"] == "defer"):
            gaps = list(response["missing_evidence"])
            if admission is not None:
                gaps.extend(item for item in admission["missing_evidence"] if item not in gaps)
            return tools.defer(
                request_id + "-defer",
                opportunity_id=opportunity_id,
                reason=response["reason"],
                missing_evidence=gaps,
            )
        if decision == "propose_goal":
            return tools._owned("opportunities", opportunity_id)
        if tools.store.get("rho", tools.base.rho_ref)["rhoC"].get("type") != "fixed":
            relation_admission = tools.research_relation_admission(response["result_ids"],
                states=development_states, legal_repairs=list(intervention))
            if relation_admission["action"] == "defer":
                return tools.defer(request_id + "-relation-defer", opportunity_id=opportunity_id,
                    reason="rhoC defers this prepared development action",
                    missing_evidence=["observed screening yield or legal evidence for selected development states"])
        experiment_id = request_id + "-experiment"
        tools.plan_rule_experiment(
            experiment_id,
            opportunity_id=opportunity_id,
            bundle_id=reserved_bundle_id or request_id + "-trial",
            **intervention,
            criterion_ref=tools.protocol["protocol_id"],
            distinguishing_prediction=response["distinguishing_prediction"],
            cost_limit=cost_limit,
            evidence_level=evidence_level,
        )
        return tools.prepare_rule_trial(experiment_id)


class Protocol10ResponseRejected(ValueError):
    """The saved Research response cannot establish a legal protocol10 branch."""


class Protocol10ResearchAdapter(BoundedResearchAdapter):
    """Bind one legal proposal and its seed2 development slots before observation.

    The existing bounded adapter owns evidence, operator and cost validation.
    This contract adds the frozen candidate identity and ordered slot selection.
    """

    operators = tuple(operator for operator in BoundedResearchAdapter.operators if operator != "planner_task_hint")
    candidate_id = "candidate-protocol10-01"

    def _hypothesis_minimum(self, tools):
        return 2

    def response_schema(self, request_id, *, tools=None):
        from .improvement_protocol import TASKS

        schema = super().response_schema(request_id, tools=tools)
        schema["properties"]["intervention"]["anyOf"] = [
            item for item in schema["properties"]["intervention"]["anyOf"]
            if "planner_task_hint" not in item.get("properties", {})
        ]
        schema["properties"]["development_slots"] = {
            "type": "array", "maxItems": 2, "uniqueItems": True,
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"task_key": {"type": "string", "enum": list(TASKS)},
                                     "seed": {"type": "integer", "const": 2}},
                      "required": ["task_key", "seed"]},
        }
        schema["required"].append("development_slots")
        return schema

    def apply_response(self, tools, response, *, evidence_level, cost_limit, synthetic=False):
        """Consume one saved response; malformed proposals terminate the branch."""
        try:
            return self._apply_protocol10_response(tools, response, evidence_level=evidence_level,
                                                  cost_limit=cost_limit, synthetic=synthetic)
        except ValueError as exc:
            raise Protocol10ResponseRejected(str(exc)) from exc

    def _apply_protocol10_response(self, tools, response, *, evidence_level, cost_limit, synthetic):
        from .improvement_protocol import TASKS

        if not isinstance(response, dict):
            raise ValueError("protocol10 response must be an object")

        request_id = response.get("request_id")
        if set(response) != set(self.response_schema(request_id)["required"]):
            raise ValueError("invalid protocol10 response fields")
        slots = response["development_slots"]
        if (not isinstance(slots, list) or len(slots) > 2
                or any(not isinstance(slot, dict) or set(slot) != {"task_key", "seed"}
                       or slot["task_key"] not in TASKS or type(slot["seed"]) is not int
                       or slot["seed"] != 2 for slot in slots)
                or len({slot["task_key"] for slot in slots}) != len(slots)):
            raise ValueError("protocol10 needs at most two distinct ordered seed2 slots")
        if (response["decision"] == "defer" and slots) or (
                response["decision"] == "prepare_rule_trial" and not slots):
            raise ValueError("proposal needs development slots; stop must select none")
        intervention = response.get("intervention")
        if intervention is not None and not isinstance(intervention, dict):
            raise ValueError("protocol10 intervention must be a permitted object or null")
        if (intervention or {}).get("planner_segment_handoff_hint") is True:
            raise ValueError("protocol10 cannot revive the rejected hint intervention")
        if evidence_level != "offline" and "chunk_steps" in (response.get("intervention") or {}):
            raise ValueError("changed control timing requires a newly registered matched baseline")
        selection = {
            "request_id": request_id, "candidate_id": self.candidate_id,
            "bundle_id": self.candidate_id, "development_slots": slots,
            "decision": response["decision"], "response": response,
            "experiment_id": request_id + "-experiment",
        }
        # Freeze the complete response before any trial artifact or development.
        tools._save("protocol10_selections", request_id, selection)
        bounded_response = {key: value for key, value in response.items()
                            if key != "development_slots"}
        # The parent validates its own finite schema, not this extended one.
        return BoundedResearchAdapter.apply_response(
            self, tools, bounded_response,
            evidence_level=evidence_level, cost_limit=cost_limit,
            synthetic=synthetic, reserved_bundle_id=self.candidate_id,
        )
