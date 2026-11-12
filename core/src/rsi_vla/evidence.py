"""Append-only facts and derived current relations."""
from .contracts import EvidenceView, ExecutionEvent, record
from .storage import atomic_json


class EventLedger:
    def __init__(self, store):
        self.store = store

    def append(self, event):
        self.store.get("bundles", event.active_bundle)
        for old in self.store.all("events"):
            if (old["run_id"], old["episode_id"], old["sequence"]) == (event.run_id, event.episode_id, event.sequence) and old["event_id"] != event.event_id:
                raise ValueError("conflicting episode event sequence")
        return self.store.put("events", event.event_id, event)

    def view(self, view_id, bundle_id, selected_media=None, policy_ref="fixed-all"):
        facts = [e for e in self.store.all("events") if e["active_bundle"] == bundle_id]
        facts.sort(key=lambda e: (e["run_id"], e["episode_id"], e["sequence"]))
        available = {media for e in facts for media in e["media"]}
        chosen = sorted(available if selected_media is None else set(selected_media))
        if not set(chosen) <= available:
            raise ValueError("media outside legal recorded facts")
        # The only selection operation is on L1. Every original L0 field remains.
        return EvidenceView(view_id, bundle_id, facts, chosen, policy_ref)


class TaskState:
    def __init__(self, store, episode_id):
        self.store = store
        self.episode_id = episode_id
        self.path = store.path("task_state", episode_id)
        self.state = store.get("task_state", episode_id) if self.path.exists() else {
            "episode_id": episode_id, "relations": {}, "commits": [], "history": []}

    def update(self, relation, value, event_id, *, criterion_ref, evidence_kind):
        event = ExecutionEvent(**self.store.get("events", event_id))
        if event.episode_id != self.episode_id:
            raise ValueError("cross-episode state evidence")
        entry = {"relation": relation, "value": value, "event_id": event_id,
                 "criterion_ref": criterion_ref, "evidence_kind": evidence_kind}
        existing = [e for e in self.state["history"] if e["event_id"] == event_id and e["relation"] == relation]
        if existing:
            if existing[0] != entry:
                raise ValueError("conflicting completion submission")
            return False
        if not criterion_ref or evidence_kind not in {"legal_observation", "independent_criterion"}:
            raise ValueError("tool completion/heuristic is not a state criterion")
        self.state["relations"][relation] = {**entry, "valid": value is not None}
        self.state["history"].append(entry)
        self.state["commits"].append([event_id, relation])
        atomic_json(self.path, self.state)
        return True
