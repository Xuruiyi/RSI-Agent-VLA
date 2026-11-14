"""Stage role measurements and commit them through the single host phase owner."""

import json
import math
import os
from copy import deepcopy
from pathlib import Path

from .storage import atomic_json

_staged = {}
_operation_id = None


def set_operation(operation_id):
    global _operation_id
    _operation_id = operation_id


def read_phase(path):
    key = str(Path(path).resolve())
    return deepcopy(
        _staged[key]["after"] if key in _staged else json.loads(Path(path).read_text())
    )


def write_phase(path, value):
    key = str(Path(path).resolve())
    role = os.environ.get("RSI_ISOLATED_ROLE")
    if role in {"research", "evaluation", "runtime"}:
        if key not in _staged:
            _staged[key] = {
                "phase_ref": key,
                "before": json.loads(Path(path).read_text()),
            }
        _staged[key]["after"] = deepcopy(value)
    elif role == "operator":
        from .protocol10_roles import broker_request

        broker_request(
            {
                "action": "phase_commit",
                "proposal": {
                    "phase_ref": key,
                    "before": json.loads(Path(path).read_text()),
                    "after": value,
                    "operation_id": _operation_id,
                },
            }
        )
    else:
        atomic_json(path, value)


def proposals():
    return deepcopy(list(_staged.values()))


def commit(proposal, phase_ref):
    """Host owner accepts append-only measured costs against the exact prior ledger."""
    if Path(proposal["phase_ref"]).resolve() != Path(phase_ref).resolve():
        raise ValueError("role receipt refers to another phase")
    before, after = proposal["before"], proposal["after"]
    actual = json.loads(Path(phase_ref).read_text())
    if actual == after:
        return
    if actual != before:
        raise ValueError("phase receipt conflicts with current authoritative ledger")
    allowed = {
        "used",
        "remaining",
        "research_charges",
        "development_trial_charges",
        "protocol10_trial_charges",
        "derived_storage_charges",
        "trial_cost_reconciliations",
        "unresolved_costs",
    }
    if any(
        before.get(k) != after.get(k)
        for k in before.keys() | after.keys()
        if k not in allowed
    ):
        raise ValueError("role receipt changes phase authority or limits")
    for key, old in before["used"].items():
        value = after["used"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < old:
            raise ValueError("role receipt removes measured costs")
        if after["remaining"][key] != max(0, after["limits"][key] - value):
            raise ValueError("role receipt has inconsistent phase remaining")
    for key in (
        "research_charges",
        "development_trial_charges",
        "protocol10_trial_charges",
        "trial_cost_reconciliations",
    ):
        old = before.get(key, [])
        new = after.get(key, [])
        if len(new) < len(old):
            raise ValueError("role receipt removes attempt evidence")
        # Reconciliation may only fill missing dimensions of an existing attempt.
        for a, b in zip(old, new):
            if any(b.get(k) != v for k, v in a.items() if k != "costs") or any(
                b.get("costs", {}).get(k) != v for k, v in a.get("costs", {}).items()
            ):
                raise ValueError("role receipt rewrites prior attempt evidence")
    atomic_json(phase_ref, after)
