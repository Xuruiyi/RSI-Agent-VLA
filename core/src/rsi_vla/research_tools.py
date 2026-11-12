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
        bundles = set(bundle_ids or [self.base.bundle_id])
        allowed = {self.base.bundle_id}
        allowed.update(t["bundle_id"] for t in self.store.all("rule_trials")
                       if t["job_id"] == self.job.state["job_id"])
        if not bundles <= allowed:
            raise ValueError("evidence bundle outside this job")
        return [self._validate_result(ExperimentResult(**self.store.get("development_results", key)), bundles)
                for key in result_ids]

    def discover_opportunity(self, opportunity_id, *, source, result_ids, external_value,
                             competing_explanations, expected_cost, target=None):
        if source not in {"failure", "normal", "technical_diagnostic", "capability_boundary", "new_goal"}:
            raise ValueError("unsupported opportunity source")
        evidence = self.query_development(result_ids)
        if not external_value or not evidence or not any(r["data_refs"] for r in evidence):
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
        required = {"explanation", "prediction", "distinguishing_test"}
        if (len(competing_explanations) < 2 or
                any(not required <= h.keys() or not all(h[k] for k in required) for h in competing_explanations) or
                len({h["explanation"] for h in competing_explanations}) != len(competing_explanations)):
            raise ValueError("retain at least two distinct testable competing explanations")
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
        return self._save("opportunities", opportunity_id, {
            "opportunity_id": opportunity_id, "source": source, "result_ids": list(result_ids),
            "external_value": external_value, "hypotheses": competing_explanations,
            "expected_cost": _costs(expected_cost), "target": target, "goal_review": goal_review,
            "experiment_authorized": False, "technical_diagnostic_only": source == "technical_diagnostic"})

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
                             criterion_ref, distinguishing_prediction, cost_limit,
                             evidence_level="offline"):
        opportunity = self._owned("opportunities", opportunity_id)
        if opportunity.get("technical_diagnostic_only"):
            raise ValueError("technical diagnosis does not establish a task behavior intervention prerequisite")
        from .rule_updates import read_phi_chunk, rule_update
        before = read_phi_chunk(self.base.phi_ref)
        after, changes = rule_update(before, chunk_steps=chunk_steps,
                                     finish_requires_native_success=finish_requires_native_success,
                                     planner_finish_reserve_calls=planner_finish_reserve_calls)
        intervention = {name: change["after"] for name, change in changes.items()}
        if not criterion_ref or not distinguishing_prediction:
            raise ValueError("experiment needs a criterion and distinguishing prediction")
        if opportunity["source"] == "new_goal":
            raise ValueError("a finite rule trial cannot establish a new capability's independent value")
        if evidence_level not in {"offline", self.protocol["scope"]}:
            raise ValueError("unsupported experiment evidence level")
        trigger_evidence = []
        if "finish_requires_native_success" in intervention and evidence_level != "offline":
            trigger_evidence = self.inspect_native_finish_opportunity(opportunity["result_ids"])
            if not trigger_evidence:
                raise ValueError("native finish trial needs observed medium development finish/native mismatch")
        if "planner_finish_reserve_calls" in intervention and evidence_level != "offline":
            trigger_evidence = self.inspect_finish_budget_opportunity(opportunity["result_ids"])
            if not trigger_evidence:
                raise ValueError("finish reserve trial needs observed counted-tool budget rejection of finish")
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
            "frozen_theta_ref": self.base.theta_ref,
            "execution_status": "prepared", "dynamic_execution_authorized": False})

    def prepare_rule_trial(self, experiment_id):
        plan = self._owned("research_experiments", experiment_id)
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

    def submit_candidate(self, candidate_id, *, experiment_id):
        plan = self._owned("research_experiments", experiment_id)
        trial = self._owned("rule_trials", experiment_id)
        finding = self._owned("research_findings", experiment_id)
        self.query_development([finding["result_id"]], bundle_ids=[trial["bundle_id"]])
        result = self.store.get("development_results", finding["result_id"])
        candidate = Candidate(candidate_id, self.base.bundle_id, trial["bundle_id"],
            self.protocol["protocol_id"], self.protocol["benchmark_id"],
            {"phi_ref": trial["runtime_rules_ref"], "diff_ref": trial["diff_ref"]},
            [trial["runtime_rules_ref"], trial["diff_ref"]], [finding["result_id"]],
            self._candidate_costs(), {"theta_frozen": True,
                "runtime_rule_schema": trial.get("runtime_rule_schema", "rsi-runtime-rules-v1"),
                "evidence_level": plan["evidence_level"],
                "runtime_loaded": result["intervention"].get("runtime_loaded", False),
                "cost_observation": finding.get("cost_observation"),
                "requires_new_timing_baseline": trial.get("requires_new_timing_baseline", True),
                "cost_scope": ("job_findings_plus_other_job_charges" if
                    _token_policy(self.protocol) == "observed_only"
                    else "job_used_at_submission"),
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
        return self._save("research_decisions", decision_id, {
            "decision_id": decision_id, "opportunity_id": opportunity_id, "action": "defer",
            "reason": reason, "missing_evidence": list(missing_evidence),
            "remaining_budget": self.job.remaining()})


class ResearchToolkit:
    """Explicit mutating tools; RPent readonly scheduling is not an ACL."""
    names = ("query_development", "discover_opportunity", "inspect_native_finish_opportunity",
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
    def __init__(self, *, model, reasoning_effort):
        if model != "gpt-6.1-sol" or reasoning_effort != "medium":
            raise ValueError("ResearchLLM requires explicit gpt-6.1-sol/medium")
        self.model, self.reasoning_effort = model, reasoning_effort

    def prepare_request(self, tools, *, request_id, result_ids, max_input_tokens, max_output_tokens):
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
        views = [tools.store.get("evidence_views", r["intervention"]["evidence_view_id"])
                 for r in evidence if r["intervention"].get("evidence_view_id")]
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
    )

    def prepare_request(self, tools, **kwargs):
        # Keep this request distinct from the immutable v1 request contract.
        request_id = kwargs["request_id"]
        allowed = tools.domain_policy.get("rule_operators", [])
        if not allowed or not set(allowed) <= set(self.operators):
            raise ValueError("human domain policy must name supported rule operators")
        request = super().prepare_request(tools, **kwargs)
        envelope = {
            **request,
            "schema": "rsi-bounded-research-request-v2",
            "domain_policy": tools.domain_policy,
            "allowed_operators": allowed,
            "instructions": "Choose evidence-backed normal, failure, technical-diagnostic or capability-boundary opportunities. "
            "Retain competing explanations and a distinguishing prediction. Prepare one finite rule "
            "artifact or defer with missing evidence/operators. No experiments, release or evaluation edits.",
            "autonomy_status": "prepared_only_no_research_execution",
        }
        tools.store.put("bounded_requests", request_id, envelope)
        from .research_transport import pack
        transport = pack(envelope)
        tools.store.put("bounded_transports", request_id, transport)
        return transport

    def response_schema(self, request_id):
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
                {"type": "null"},
            ]
        }
        properties = {
            "schema": {"type": "string", "const": "rsi-bounded-research-response-v2"},
            "request_id": {"type": "string", "const": request_id},
            "decision": {"type": "string", "enum": ["prepare_rule_trial", "defer"]},
            "source": {
                "type": "string",
                "enum": ["normal", "failure", "technical_diagnostic", "capability_boundary"],
            },
            "result_ids": {"type": "array", "minItems": 1, "items": text},
            "reason": text,
            "external_value": text,
            "competing_explanations": {
                "type": "array",
                "minItems": 2,
                "items": hypothesis,
            },
            "intervention": intervention,
            "distinguishing_prediction": text,
            "stop_condition": text,
            "missing_evidence": {"type": "array", "items": text},
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        }

    def apply_response(
        self, tools, response, *, evidence_level, cost_limit, synthetic=False
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
        schema = self.response_schema(request_id)
        if (
            set(response) != set(schema["required"])
            or response["schema"] != "rsi-bounded-research-response-v2"
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
        legal = {r["result_id"] for r in request["evidence"]}
        if not response["result_ids"] or not set(response["result_ids"]) <= legal:
            raise ValueError("proposal evidence outside supplied development input")
        decision, intervention = response["decision"], response["intervention"]
        if decision == "defer":
            if intervention is not None or not response["missing_evidence"]:
                raise ValueError("defer needs gaps and no intervention")
        elif decision == "prepare_rule_trial":
            if (
                not isinstance(intervention, dict)
                or len(intervention) != 1
                or not set(intervention) <= set(request["allowed_operators"])
            ):
                raise ValueError("intervention outside human operator policy")
        else:
            raise ValueError("unsupported bounded decision")
        opportunity_id = request_id + "-opportunity"
        tools.discover_opportunity(
            opportunity_id,
            source=response["source"],
            result_ids=response["result_ids"],
            external_value=response["external_value"],
            competing_explanations=response["competing_explanations"],
            expected_cost=cost_limit,
        )
        tools._save(
            "bounded_proposals",
            request_id,
            {
                "response": response,
                "origin": "synthetic_engineering_fixture"
                if synthetic
                else "research_transport",
                "stop_condition": response["stop_condition"],
                "execution_authorized": False,
            },
        )
        if decision == "defer":
            return tools.defer(
                request_id + "-defer",
                opportunity_id=opportunity_id,
                reason=response["reason"],
                missing_evidence=response["missing_evidence"],
            )
        experiment_id = request_id + "-experiment"
        tools.plan_rule_experiment(
            experiment_id,
            opportunity_id=opportunity_id,
            bundle_id=request_id + "-trial",
            **intervention,
            criterion_ref=tools.protocol["protocol_id"],
            distinguishing_prediction=response["distinguishing_prediction"],
            cost_limit=cost_limit,
            evidence_level=evidence_level,
        )
        return tools.prepare_rule_trial(experiment_id)
