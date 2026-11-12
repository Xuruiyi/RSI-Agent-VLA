"""Single-worker local JSON persistence; separate roots support process permissions."""
import json
import os
import re
from pathlib import Path
from .contracts import record


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class Records:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, kind, key):
        return self.root / identifier(kind) / (identifier(key) + ".json")

    def put(self, kind, key, value):
        """Idempotent equal writes, conflicting re-use is an error."""
        value = record(value) if hasattr(value, "__dataclass_fields__") else value
        path = self.path(kind, key)
        if path.exists():
            if self.get(kind, key) != value:
                raise ValueError(f"conflicting {kind} identity: {key}")
            return False
        atomic_json(path, value)
        return True

    def get(self, kind, key):
        return json.loads(self.path(kind, key).read_text(encoding="utf-8"))

    def all(self, kind):
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((self.root / identifier(kind)).glob("*.json"))]


class ResearchStore(Records):
    """No active-version or formal decision write API."""
    def save_bundle(self, bundle):
        for kind, ref in (("psi", bundle.psi_ref), ("memory", bundle.memory_ref), ("rho", bundle.rho_ref)):
            self.get(kind, ref)
        return self.put("bundles", bundle.bundle_id, bundle)

    def save_candidate(self, candidate):
        from .contracts import RunBundle
        base = RunBundle(**self.get("bundles", candidate.base_bundle_id))
        target = RunBundle(**self.get("bundles", candidate.bundle_id))
        fields = {"theta_ref", "processor_ref", "phi_ref", "rho_ref", "psi_ref", "memory_ref", "backend_ref", "code_ref", "diff_ref"}
        actual = {name: getattr(target, name) for name in fields if getattr(base, name) != getattr(target, name)}
        if not actual or actual != candidate.changes:
            raise ValueError("candidate changes do not match actual bundle difference")
        if target.parent_bundle_id != base.bundle_id or target.protocol_id != candidate.protocol_id:
            raise ValueError("candidate parent/protocol mismatch")
        if not candidate.artifact_refs:
            raise ValueError("candidate requires artifact references")
        for key in candidate.development_results:
            self.get("development_results", key)
        return self.put("candidates", candidate.candidate_id, candidate)

    def save_development_result(self, result):
        if result.role != "development":
            raise ValueError("research store accepts development results only")
        return self.put("development_results", result.result_id, result)


def initialize(store, bundle, *, psi, memory, rho=None):
    """External priors keep provenance; initial snapshots are immutable by ID."""
    if not memory.get("provenance") or "legal_information" not in memory:
        raise ValueError("memory needs provenance and legal_information")
    rho = rho or {
        "rhoA": {"type": "fixed", "retain_l0": "all", "media": "all_available"},
        "rhoB": {"type": "fixed", "route": "existing_authorized_worker"},
        "rhoC": {"type": "fixed", "order": ["inspect", "bounded_experiment", "defer"]},
        "learned": False,
    }
    if not {"rhoA", "rhoB", "rhoC"}.issubset(rho):
        raise ValueError("rhoA/B/C required")
    store.put("psi", bundle.psi_ref, psi)
    store.put("memory", bundle.memory_ref, memory)
    store.put("rho", bundle.rho_ref, rho)
    store.save_bundle(bundle)
