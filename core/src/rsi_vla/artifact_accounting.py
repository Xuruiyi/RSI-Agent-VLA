"""Book newly materialized research evidence outside the supervised raw output."""

from pathlib import Path

from .phase_receipts import read_phase, write_phase
from .storage import atomic_json

FINANCIAL = {"jobs", "artifact_accounting_inputs", "artifact_accounting"}


def sizes(store, *, exclude_output=None):
    return {
        str(p.relative_to(store.root)): p.stat().st_size
        for p in store.root.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.relative_to(store.root).parts[0] not in FINANCIAL
        and (exclude_output is None or not p.is_relative_to(Path(exclude_output)))
    }


def begin(tools, key):
    """Snapshot preexisting immutable evidence once, never bill its import again."""
    if not tools.store.path("artifact_accounting_inputs", key).exists():
        tools.store.put(
            "artifact_accounting_inputs", key, {"files": sizes(tools.store)}
        )


def finish(tools, key, *, phase_path, output=None, billed_output_bytes=None):
    before = tools.store.get("artifact_accounting_inputs", key)["files"]
    total = sum(
        max(0, count - before.get(path, 0))
        for path, count in sizes(
            tools.store,
            exclude_output=output if billed_output_bytes is not None else None,
        ).items()
    )
    raw_growth = 0
    if output is not None and billed_output_bytes is not None:
        raw_bytes = sum(
            p.stat().st_size
            for p in Path(output).rglob("*")
            if p.is_file() and not p.is_symlink()
        )
        raw_growth = max(0, raw_bytes - billed_output_bytes)
        total += raw_growth
    path = tools.store.path("artifact_accounting", key)
    old = (
        tools.store.get("artifact_accounting", key)
        if path.exists()
        else {"booked_bytes": 0}
    )
    # Financial ledger files themselves follow the existing global-ledger accounting convention.
    delta = max(0, total - old["booked_bytes"])
    reason = "new derived evidence storage " + key + " " + str(total)
    if delta and not any(c["reason"] == reason for c in tools.job.state["charges"]):
        tools.job.record_storage(delta, reason)
    phase_path = Path(phase_path)
    phase = read_phase(phase_path)
    entries = phase.setdefault("derived_storage_charges", [])
    previous = next((e for e in entries if e["key"] == key), None)
    previous_total = previous["storage_bytes"] if previous else 0
    phase_delta = max(0, total - previous_total)
    if previous is None:
        entries.append(
            {"key": key, "storage_bytes": total, "store_ref": str(tools.store.root)}
        )
    elif phase_delta:
        previous["storage_bytes"] = total
    if phase_delta:
        phase["used"]["storage_bytes"] += phase_delta
        phase["remaining"]["storage_bytes"] = max(
            0, phase["limits"]["storage_bytes"] - phase["used"]["storage_bytes"]
        )
    write_phase(phase_path, phase)
    record = {
        "booked_bytes": max(total, old["booked_bytes"]),
        "post_measurement_raw_growth": raw_growth,
        "store_ref": str(tools.store.root),
        "basis": "new logical evidence files and positive growth; excludes financial ledger metadata and reused historical files",
        "job_storage_limit_exceeded": tools.job.state["used"].get("artifact_bytes", 0)
        > tools.job.state["limits"].get("artifact_bytes", 0),
    }
    atomic_json(path, record)
    return record
