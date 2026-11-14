"""Trusted release writer, separate from research storage."""
from .contracts import Candidate, ExperimentResult, GateDecision
from .evaluation import Gate, assignment_metrics, critical_old_evidence_matches, score
from .storage import Records, atomic_json


class ReleaseManager:
    def __init__(self, release_root, research_store, evaluation_store):
        self.records = Records(release_root)
        self.research = research_store
        self.evaluations = evaluation_store
        self.active_path = self.records.root / "active.json"

    def bootstrap(self, bundle_id, *, scope="offline"):
        self.research.get("bundles", bundle_id)
        if self.active_path.exists():
            if self.active()["bundle_id"] != bundle_id:
                raise ValueError("active version already initialized")
            return
        atomic_json(self.active_path, {"bundle_id": bundle_id, "scope": scope,
                                      "activation": "bootstrap"})

    def qualify_initial(self, qualification_id, *, result_id, evidence_import_id=None):
        """Establish a usable initial version without claiming self-relative gain."""
        if evidence_import_id is not None:
            return self._qualify_imported_initial(qualification_id, result_id=result_id,
                                                 evidence_import_id=evidence_import_id)
        result = ExperimentResult(**self.evaluations.get("results", result_id))
        bundle = self.research.get("bundles", result.bundle_id)
        protocol = self.evaluations.get("protocols", result.protocol_id)
        _require_deployment_protocol(protocol)
        if protocol.get('deployment_derivation'):
            raise ValueError('derived initial qualification requires the trusted evidence import')
        requirements = _validate_initial_result(protocol, bundle, result)
        value = {"qualification_id": qualification_id, "bundle_id":result.bundle_id,
            "bundle_snapshot":bundle, "protocol_id":result.protocol_id,"scope":protocol["scope"],
            "kind":"initial_effective", "result_refs":[result_id],
            "requirements":requirements, "relative_gain_claimed":False}
        self.records.put("qualifications", qualification_id, value)
        return value

    def _qualify_imported_initial(self, qualification_id, *, result_id, evidence_import_id):
        from .qualification_import import deployment_authorization
        imported = self.evaluations.get('initial_evidence_imports', evidence_import_id)
        authorization = deployment_authorization(imported['authorization_ref'], role='release')
        bundle = self.research.get('bundles', authorization['bundle_id'])
        protocol = self.evaluations.get('protocols', authorization['protocol_id'])
        value = dict(qualification_id=qualification_id, bundle_id=bundle['bundle_id'],
                     bundle_snapshot=bundle, protocol_id=protocol['protocol_id'], scope=protocol['scope'],
                     kind='initial_effective', result_refs=[result_id], relative_gain_claimed=False,
                     evidence_import_id=evidence_import_id, evidence_import_snapshot=imported,
                     protocol_snapshot=protocol,
                     profile_snapshot=_json_reference(bundle['backend_ref']),
                     authorization_ref=imported['authorization_ref'],
                     authorization_snapshot=authorization,
                     requirements=protocol['initial_requirements'])
        self._validate_imported_qualification(value, role='release')
        self.records.put('qualifications', qualification_id, value)
        return value

    def _validate_imported_qualification(self, value, *, role, renewal=None):
        from pathlib import Path
        from .qualification_import import deployment_authorization, validate_compatibility
        from .release_preparation import validate_initial_deployment_derivation
        if renewal is None:
            authorization = deployment_authorization(value['authorization_ref'],
                bundle_id=value['bundle_id'], protocol_id=value['protocol_id'], role=role)
        else:
            from .qualification_import import _historical_authorization_snapshot
            authorization = _historical_authorization_snapshot(
                value['authorization_ref'], value['authorization_snapshot'])
        if (self.records.root.resolve() != Path(authorization['role_domains']['release']).resolve()
                or self.evaluations.root.resolve() != Path(authorization['role_domains']['evaluation']).resolve()
                or self.research.root.resolve() != Path(authorization['role_domains']['research']).resolve()
                or authorization != value['authorization_snapshot']):
            raise ValueError('deployment authorization store/snapshot mismatch')
        imported = self.evaluations.get('initial_evidence_imports', value['evidence_import_id'])
        if imported != value['evidence_import_snapshot']:
            raise ValueError('trusted evidence import changed')
        approval = imported['approval_snapshot']
        if (_json_reference(imported['approval_ref']) != approval
                or imported['authorization_snapshot'] != authorization
                or approval['source_bundle']['bundle_id'] != authorization['source_bundle_id']
                or approval['source_protocol']['protocol_id'] != authorization['source_protocol_id']
                or value['result_refs'] != [authorization['source_result_id']]):
            raise ValueError('deployment authorization source mismatch')
        result = ExperimentResult(**self.evaluations.get('results', authorization['source_result_id']))
        if (self.evaluations.get('protocols', authorization['source_protocol_id']) != approval['source_protocol']
                or result.__dict__ != approval['source_result']
                or not inspect_initial_result(approval['source_protocol'], approval['source_bundle'], result)['passed']):
            raise ValueError('imported initial evidence no longer matches independent qualification')
        bundle = self.research.get('bundles', value['bundle_id'])
        protocol = self.evaluations.get('protocols', value['protocol_id'])
        profile = _json_reference(bundle['backend_ref'])
        if (bundle != value['bundle_snapshot'] or protocol != value['protocol_snapshot']
                or profile != value['profile_snapshot']):
            raise ValueError('deployment qualification target snapshot mismatch')
        origin = protocol['deployment_derivation']
        if (origin['evidence_import_id'] != value['evidence_import_id']
                or origin['authorization_ref'] != value['authorization_ref']
                or protocol['authorization_scope'] != authorization['scope']):
            raise ValueError('deployment qualification derivation/authorization mismatch')
        if not Path(origin['compatibility_ref']).resolve(strict=True).is_relative_to(
                Path(authorization['role_domains']['operator']).resolve()):
            raise ValueError('deployment compatibility must be protected operator input')
        validate_initial_deployment_derivation(approval['source_bundle'], approval['source_protocol'],
            approval['source_profile'], bundle, protocol, profile)
        validate_compatibility(_json_reference(origin['compatibility_ref']),
            source_bundle=approval['source_bundle'], source_protocol=approval['source_protocol'],
            source_profile=approval['source_profile'], bundle=bundle, protocol=protocol, profile=profile)

    def qualification(self, bundle_id, *, scope, protocol_id=None, consumer_role='release'):
        bundle = self.research.get("bundles", bundle_id)
        _require_deployment_protocol(self.evaluations.get("protocols", protocol_id or bundle["protocol_id"]))
        for value in self.records.all("qualifications"):
            if (value["bundle_id"] == bundle_id and value["scope"] == scope
                    and value["protocol_id"] == (protocol_id or bundle["protocol_id"])
                    and value["bundle_snapshot"] == bundle):
                if value.get('upgrade_import_id'):
                    from .upgrade_bridge import validate_upgrade_qualification
                    validate_upgrade_qualification(self, value, role=consumer_role)
                elif value.get('evidence_import_id'):
                    self._validate_imported_qualification(value, role=consumer_role)
                elif self.evaluations.get('protocols', value['protocol_id']).get('deployment_derivation'):
                    raise ValueError('derived initial qualification lacks trusted evidence import')
                return value
        raise ValueError("bundle lacks applicable independent qualification")

    def establish_initial(self, qualification_id, *, safe_point, queues_empty):
        from .runtime import dynamic_job_lock
        with dynamic_job_lock():
            return self._establish_initial(qualification_id, safe_point=safe_point, queues_empty=queues_empty)

    def _establish_initial(self, qualification_id, *, safe_point, queues_empty):
        value = self.records.get("qualifications", qualification_id)
        self.qualification(value["bundle_id"], scope=value["scope"])
        active = self.active()
        if (value.get('evidence_import_id') and active.get('activation') == 'qualified_initial'
                and active.get('qualification_ref') == qualification_id
                and active['bundle_id'] == value['bundle_id'] and active['scope'] == value['scope']
                and safe_point in {'task_boundary', 'approved_safe_point'} and queues_empty):
            return active
        if (value["kind"] != "initial_effective" or active["activation"] != "bootstrap"
                or active["bundle_id"] != value["bundle_id"] or active["scope"] != value["scope"]
                or safe_point not in {"task_boundary", "approved_safe_point"} or not queues_empty):
            raise ValueError("initial binding requires matching bootstrap and safe point")
        atomic_json(self.active_path, {**active, "activation":"qualified_initial",
            "qualification_ref":qualification_id, "revision":active.get("revision",0)+1})
        return self.active()

    def active(self):
        import json
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    def evaluate(self, decision_id, candidate_id, *, candidate_result=None,
                 baseline_result=None, anchor_result=None):
        candidate = Candidate(**self.research.get("candidates", candidate_id))
        protocol = self.evaluations.get("protocols", candidate.protocol_id)
        def result(key):
            return ExperimentResult(**self.evaluations.get("results", key)) if key else None
        decision = Gate(protocol).evaluate(decision_id, candidate,
                    current_bundle_id=self.active()["bundle_id"],
                    candidate_result=result(candidate_result), baseline_result=result(baseline_result),
                    anchor_result=result(anchor_result))
        self.records.put("decisions", decision_id, decision)
        self.records.put("decision_bases", decision_id, self.active())
        if (decision.verdict == "accept" and decision.release_eligible
                and protocol.get("comparison_only") is not True):
            bundle = self.research.get("bundles", decision.bundle_id)
            self.records.put("qualifications", decision_id, {
                "qualification_id":decision_id, "bundle_id":decision.bundle_id,
                "bundle_snapshot":bundle, "protocol_id":decision.protocol_id, "scope":decision.scope,
                "kind":"evaluated_candidate", "result_refs":decision.result_refs,
                "decision_ref":decision_id, "relative_gain_claimed":True})
        return decision

    def activate(self, decision_id, *, safe_point, queues_empty, scope="offline"):
        if self.records.path("upgrade_decisions", decision_id).exists():
            from .upgrade_bridge import activate_upgrade
            return activate_upgrade(self, decision_id, safe_point=safe_point, queues_empty=queues_empty)
        from .runtime import dynamic_job_lock
        with dynamic_job_lock():
            return self._activate(decision_id, safe_point=safe_point, queues_empty=queues_empty, scope=scope)

    def _activate(self, decision_id, *, safe_point, queues_empty, scope="offline"):
        decision = GateDecision(**self.records.get("decisions", decision_id))
        _require_deployment_protocol(self.evaluations.get("protocols", decision.protocol_id))
        if decision.verdict != "accept" or not decision.release_eligible:
            raise ValueError("decision has no release eligibility")
        if decision.scope != scope or self.active()["scope"] != scope:
            raise ValueError("deployment scope mismatch")
        if (self.records.get("decision_bases", decision_id) != self.active() or
                decision.base_bundle_id != self.active()["bundle_id"]):
            raise ValueError("decision baseline is stale")
        if safe_point not in {"task_boundary", "approved_safe_point"} or not queues_empty:
            raise ValueError("activation needs a safe point and cleared queues")
        self.qualification(self.active()["bundle_id"], scope=scope, protocol_id=decision.protocol_id)
        self.qualification(decision.bundle_id, scope=scope, protocol_id=decision.protocol_id)
        self.records.put("activations", decision_id, {
            "bundle_id": decision.bundle_id, "previous": decision.base_bundle_id,
            "decision_id": decision_id, "scope": scope, "safe_point": safe_point})
        atomic_json(self.active_path, {"bundle_id": decision.bundle_id, "scope": scope,
                                      "activation": decision_id, "revision": self.active().get("revision", 0) + 1})
        return self.active()

    def rollback(self, target_bundle_id, *, safe_point, queues_empty, reason):
        from .runtime import dynamic_job_lock
        with dynamic_job_lock():
            return self._rollback(target_bundle_id, safe_point=safe_point, queues_empty=queues_empty, reason=reason)

    def _rollback(self, target_bundle_id, *, safe_point, queues_empty, reason):
        if not reason or safe_point not in {"task_boundary", "approved_safe_point"} or not queues_empty:
            raise ValueError("rollback requires reason and safe point")
        # Only return to a bundle that was in this release's actual ancestry.
        history = self.records.all("activations")
        frontier = [self.active()["bundle_id"]]
        ancestors = set()
        while frontier:
            current = frontier.pop()
            for activation in history:
                if activation["bundle_id"] == current and activation["previous"] not in ancestors:
                    ancestors.add(activation["previous"])
                    frontier.append(activation["previous"])
        if target_bundle_id not in ancestors:
            raise ValueError("rollback target is not an actually activated ancestor")
        previous = self.active()
        if self.records.path('upgrade_decisions', previous['activation']).exists():
            from .upgrade_bridge import validate_upgrade_rollback
            validate_upgrade_rollback(self, target_bundle_id, reason=reason)
        else:
            protocol_id = self.research.get('bundles', previous['bundle_id'])['protocol_id']
            self.qualification(target_bundle_id, scope=previous['scope'], protocol_id=protocol_id)
        self.records.put("rollbacks", "rollback-" + str(len(self.records.all("rollbacks"))), {
            "from": previous["bundle_id"], "to": target_bundle_id, "reason": reason,
            "safe_point": safe_point})
        atomic_json(self.active_path, {"bundle_id": target_bundle_id, "scope": previous["scope"],
                                      "activation": "rollback", "revision": previous.get("revision", 0) + 1})
        return self.active()

    def deployment_bundle(self, *, scope, safe_point, consumer_role='runtime', authorization_ref=None):
        """Read trusted effective state once at a new task boundary."""
        if safe_point not in {"task_boundary", "approved_safe_point"}:
            raise ValueError("deployment selection requires a safe point")
        active = self.active()
        if authorization_ref is not None:
            from .qualification_import import renewal_authorization
            renewal, qualification = renewal_authorization(authorization_ref, self,
                scope=scope, role=consumer_role)
            self._validate_imported_qualification(qualification, role=consumer_role, renewal=renewal)
            return self.research.get('bundles', active['bundle_id']), active
        if active["scope"] != scope:
            raise ValueError("deployment scope mismatch")
        if active["activation"] == "bootstrap":
            raise ValueError("initial qualification must be established at a safe point")
        self.qualification(active["bundle_id"], scope=scope, consumer_role=consumer_role)
        return self.research.get("bundles", active["bundle_id"]), active


def _json_reference(reference):
    import json
    from pathlib import Path
    return json.loads(Path(reference).read_text(encoding='utf-8'))


def _validate_initial_result(protocol, bundle, result):
    requirements = protocol.get("initial_requirements")
    if not requirements:
        raise ValueError("initial qualification requires predeclared requirements")
    if (protocol.get("status") != "frozen" or result.protocol_id != bundle["protocol_id"]
            or result.role != "release" or result.evidence_level != protocol["scope"]
            or result.benchmark_id != protocol["benchmark_id"] or result.bundle_id != protocol["anchor_bundle_id"]
            or result.candidate_id is not None or result.assigned_count <= 0
            or result.unknown_count or result.invalid_count or not result.data_refs
            or any(result.hard_constraints.get(k) is not True for k in protocol["hard_constraints"])
            or protocol["scope"] != "offline" and protocol.get("dynamic_authorized") is not True):
        raise ValueError("initial qualification lacks complete matched independent evidence")
    metrics = assignment_metrics(protocol, result)
    if score(protocol, result) is None:
        raise ValueError("initial score unavailable")
    _validate_initial_capability(requirements, metrics, result)
    return requirements


def _validate_initial_capability(requirements, metrics, result):
    if any(metrics.get(k) is None or metrics[k] < minimum
           for k, minimum in requirements.get("metric_minima", {}).items()):
        raise ValueError("initial performance below predeclared minimum")
    for task in requirements.get("required_old_tasks", []):
        rows = [a for a in result.assignments if a["task_key"] == task]
        if not rows or not all(a["validity"] == "valid" and a["outcome"] == "success" for a in rows):
            raise ValueError("critical old capability not established")


def _require_deployment_protocol(protocol):
    if protocol.get("comparison_only") is True:
        raise ValueError("comparison-only protocol has no deployment authorization")


def _validate_initial_comparison_result(protocol, bundle, result):
    if protocol.get("comparison_only") is not True:
        raise ValueError("initial comparison check requires comparison_only protocol")
    if (protocol.get("status") != "frozen" or result.protocol_id != protocol["protocol_id"]
            or bundle["protocol_id"] != protocol["protocol_id"]
            or result.bundle_id != bundle["bundle_id"]
            or result.bundle_id != protocol["anchor_bundle_id"]
            or result.role != "release" or result.evidence_level != protocol["scope"]
            or result.benchmark_id != protocol["benchmark_id"] or result.candidate_id is not None
            or result.assigned_count <= 0 or result.unknown_count or result.invalid_count
            or not result.data_refs
            or protocol["scope"] != "offline" and protocol.get("dynamic_authorized") is not True):
        raise ValueError("initial comparison lacks complete matched independent evidence")
    assignment_metrics(protocol, result)
    if score(protocol, result) is None:
        raise ValueError("initial comparison score unavailable")
    for key in protocol["hard_constraints"]:
        if key == "critical_old_preserved":
            if not critical_old_evidence_matches(protocol, result):
                raise ValueError("initial comparison old capability evidence unknown/conflicting")
        elif result.hard_constraints.get(key) is not True:
            raise ValueError("initial comparison technical requirement unestablished: " + key)


def inspect_initial_comparison_result(protocol, bundle, result):
    """Check technical comparability independently of initial ability qualification."""
    try:
        _validate_initial_comparison_result(protocol, bundle, result)
    except (ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": str(exc), "failure_kind": "technical",
                "result_id": result.result_id, "qualification_established": False,
                "activation_performed": False}
    return {"passed": True, "reason": "initial technical comparison requirements satisfied",
            "failure_kind": None, "result_id": result.result_id,
            "qualification_established": False, "activation_performed": False}


def inspect_initial_result(protocol, bundle, result):
    """Pure qualification check; does not establish a qualification or active version."""
    if protocol.get("comparison_only") is True:
        technical = inspect_initial_comparison_result(protocol, bundle, result)
        if not technical["passed"]:
            return technical
    try:
        if protocol.get("comparison_only") is True:
            requirements = protocol.get("initial_requirements")
            if not requirements:
                raise ValueError("initial qualification requires predeclared requirements")
            _validate_initial_capability(requirements, assignment_metrics(protocol, result), result)
            if ("critical_old_preserved" in protocol["hard_constraints"]
                    and result.hard_constraints.get("critical_old_preserved") is not True):
                raise ValueError("critical old capability not established")
        else:
            _validate_initial_result(protocol, bundle, result)
    except ValueError as exc:
        value = {"passed": False, "reason": str(exc), "result_id": result.result_id,
                 "qualification_established": False, "activation_performed": False}
        if protocol.get("comparison_only") is True:
            value["failure_kind"] = "capability"
        return value
    value = {"passed": True, "reason": "current initial requirements satisfied",
             "result_id": result.result_id, "qualification_established": False, "activation_performed": False}
    if protocol.get("comparison_only") is True:
        value["failure_kind"] = None
    return value
