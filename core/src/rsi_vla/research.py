"""Budget-accounted jobs. Recovery consumes the same job ledger."""
from .storage import atomic_json


class BudgetExhausted(RuntimeError):
    pass


class ResearchJob:
    def __init__(self, store, job_id, bundle_id=None, limits=None):
        self.store = store
        self.path = store.path("jobs", job_id)
        if self.path.exists():
            self.state = store.get("jobs", job_id)
            if bundle_id is not None and self.state["bundle_id"] != bundle_id:
                raise ValueError("job start bundle is immutable")
            if limits is not None and self.state["limits"] != limits:
                raise ValueError("job budget cannot reset on recovery")
        else:
            if not bundle_id or not limits or any(v < 0 for v in limits.values()):
                raise ValueError("new jobs require a bundle and nonnegative finite limits")
            from math import isfinite
            if not all(isfinite(v) for v in limits.values()):
                raise ValueError("budget must be finite")
            store.get("bundles", bundle_id)
            self.state = {"job_id": job_id, "bundle_id": bundle_id, "limits": limits,
                          "used": {k: 0 for k in limits},
                          "status": "budget_exhausted" if any(v == 0 for v in limits.values()) else "running",
                          "charges": []}
            self._save()

    def _save(self):
        atomic_json(self.path, self.state)

    def remaining(self):
        return {k: max(0, v - self.state["used"][k]) for k, v in self.state["limits"].items()}

    def consume(self, costs, reason):
        from math import isfinite
        if self.state["status"] != "running":
            raise BudgetExhausted("job has terminated")
        if any(k not in self.state["limits"] or not isfinite(v) or v < 0 for k, v in costs.items()):
            raise ValueError("invalid/unregistered cost")
        self.state["charges"].append({"reason": reason, "costs": costs})
        for key, value in costs.items():
            self.state["used"][key] += value
        if any(self.state["used"][k] >= cap for k, cap in self.state["limits"].items()):
            self.state["status"] = "budget_exhausted"
        self._save()
        return self.state["status"]

    def run(self, function, *, cost, reason):
        """Charge a bounded action before invocation; never start after exhaustion.

        This synchronous helper does not preempt a process. External/GPU backends
        need their own in-process resource supervisor before they can be used.
        """
        if self.state["status"] != "running":
            raise BudgetExhausted("job has terminated")
        if any(v > self.remaining().get(k, -1) for k, v in cost.items()):
            self.state["status"] = "budget_exhausted"
            self._save()
            raise BudgetExhausted("action exceeds remaining budget")
        self.consume(cost, reason)
        return function()

    def unknown_action(self, *, allowed_observation=False, allowed_recovery=False):
        if self.state["status"] != "running":
            return "stop"
        if allowed_observation:
            return "observe"
        if allowed_recovery:
            return "recover"
        return "stop"
