"""Append-only facts and derived current relations."""
from .contracts import EvidenceView, ExecutionEvent
from .storage import atomic_json


class EventLedger:
    def __init__(self, store):
        self.store = store

    def append(self, event):
        return self.append_many([event])[0]

    def append_many(self, events):
        """Append an ordered batch with the same identity checks as single facts.

        Build the episode-sequence index once per batch. The store has one
        writer; a later call rebuilds from disk, including another ledger's
        completed writes. An interrupted prefix remains safe to import again.
        """
        by_sequence = {}
        for old in self.store.all("events"):
            key = (old["run_id"], old["episode_id"], old["sequence"])
            by_sequence.setdefault(key, set()).add(old["event_id"])
        appended = []
        for event in events:
            self.store.get("bundles", event.active_bundle)
            key = (event.run_id, event.episode_id, event.sequence)
            identities = by_sequence.get(key, set())
            if identities and identities != {event.event_id}:
                raise ValueError("conflicting episode event sequence")
            appended.append(self.store.put("events", event.event_id, event))
            by_sequence.setdefault(key, set()).add(event.event_id)
        return appended

    def view(self, view_id, bundle_id, selected_media=None, policy_ref="fixed-all"):
        facts = [e for e in self.store.all("events") if e["active_bundle"] == bundle_id]
        facts.sort(key=lambda e: (e["run_id"], e["episode_id"], e["sequence"]))
        from .relation_learning import select_media
        bundle = self.store.get("bundles", bundle_id)
        rho = self.store.get("rho", bundle["rho_ref"])
        chosen = select_media(rho["rhoA"], facts, selected_media)
        if rho["rhoA"].get("type") != "fixed" and policy_ref == "fixed-all":
            policy_ref = bundle["rho_ref"]
        # The only selection operation is on L1. Every original L0 field remains.
        selection = {
            "target": "references_only", "mechanism": "caller_selection" if selected_media is not None else rho["rhoA"]["type"],
            "policy_ref": policy_ref,
            "probabilities": {ref: (None if selected_media is not None else float(ref in chosen))
                              for ref in sorted({ref for fact in facts for ref in fact["media"]})},
            "probability_basis": "unknown" if selected_media is not None else "deterministic_conditional_on_recorded_media_and_bound_policy",
            "zero_coverage": sorted({ref for fact in facts for ref in fact["media"]} - set(chosen)),
            "population_correction": "not_established; zero coverage cannot be recovered by weighting",
            "persisted_bytes_saved": None,
            "l0_integrity": evidence_integrity(facts),
        }
        return EvidenceView(view_id, bundle_id, facts, chosen, policy_ref, selection)


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


def evidence_integrity(facts):
    """Report gaps and missing facts without rewriting an original event."""
    from collections import defaultdict

    sequences = defaultdict(list)
    for fact in facts:
        sequences[(fact["run_id"], fact["episode_id"])].append(fact["sequence"])
    gaps = []
    for (run_id, episode_id), numbers in sorted(sequences.items()):
        ordered = sorted(set(numbers))
        previous = -1
        for number in ordered:
            if number > previous + 1:
                gaps.append({"run_id": run_id, "episode_id": episode_id,
                             "first_missing_sequence": previous + 1,
                             "last_missing_sequence": number - 1})
            previous = number
    return {"sequence_gaps": gaps,
            "missing_by_event": {fact["event_id"]: fact["missing"] for fact in facts if fact["missing"]},
            "tail_completeness": "unknown_without_independent_closure",
            "facts_modified": False}
