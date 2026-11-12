"""Trusted release writer, separate from research storage."""
from .contracts import Candidate, ExperimentResult, GateDecision
from .evaluation import Gate
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
        return decision

    def activate(self, decision_id, *, safe_point, queues_empty, scope="offline"):
        decision = GateDecision(**self.records.get("decisions", decision_id))
        if decision.verdict != "accept" or not decision.release_eligible:
            raise ValueError("decision has no release eligibility")
        if decision.scope != scope or self.active()["scope"] != scope:
            raise ValueError("deployment scope mismatch")
        if (self.records.get("decision_bases", decision_id) != self.active() or
                decision.base_bundle_id != self.active()["bundle_id"]):
            raise ValueError("decision baseline is stale")
        if safe_point not in {"task_boundary", "approved_safe_point"} or not queues_empty:
            raise ValueError("activation needs a safe point and cleared queues")
        self.research.get("bundles", decision.bundle_id)
        self.records.put("activations", decision_id, {
            "bundle_id": decision.bundle_id, "previous": decision.base_bundle_id,
            "decision_id": decision_id, "scope": scope, "safe_point": safe_point})
        atomic_json(self.active_path, {"bundle_id": decision.bundle_id, "scope": scope,
                                      "activation": decision_id, "revision": self.active().get("revision", 0) + 1})
        return self.active()

    def rollback(self, target_bundle_id, *, safe_point, queues_empty, reason):
        if not reason or safe_point not in {"task_boundary", "approved_safe_point"} or not queues_empty:
            raise ValueError("rollback requires reason and safe point")
        # Only return to a bundle that was in this release's actual ancestry.
        history = self.records.all("activations")
        known = {a["previous"] for a in history} | {a["bundle_id"] for a in history}
        if target_bundle_id not in known:
            raise ValueError("rollback target was never active")
        self.research.get("bundles", target_bundle_id)
        previous = self.active()
        self.records.put("rollbacks", "rollback-" + str(len(self.records.all("rollbacks"))), {
            "from": previous["bundle_id"], "to": target_bundle_id, "reason": reason,
            "safe_point": safe_point})
        atomic_json(self.active_path, {"bundle_id": target_bundle_id, "scope": previous["scope"],
                                      "activation": "rollback", "revision": previous.get("revision", 0) + 1})
        return self.active()

    def deployment_bundle(self, *, scope, safe_point):
        """Read trusted effective state once at a new task boundary."""
        if safe_point not in {"task_boundary", "approved_safe_point"}:
            raise ValueError("deployment selection requires a safe point")
        active = self.active()
        if active["scope"] != scope:
            raise ValueError("deployment scope mismatch")
        if active["activation"] == "bootstrap" and scope != "offline":
            raise ValueError("unqualified bootstrap is not a trusted deployment version")
        return self.research.get("bundles", active["bundle_id"]), active
