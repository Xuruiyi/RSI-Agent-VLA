"""Import one completed development attempt; never dispatch or charge it again."""

import json
from dataclasses import asdict
from pathlib import Path
from .adapters.rpent import read_trajectory
from .contracts import EvidenceView, ExperimentResult
from .evidence import EventLedger

IMPORT_VERSION = "rpent-development-v2-action-origin"


def import_development(store, protocol, run_dir, *, result_id, revision_id=None, audit_path=None):
    """Use a versioned store for changed derivations; equal replay is idempotent."""
    root = Path(run_dir).resolve()

    def read(name):
        return json.loads((root / name).read_text())

    context = read("runtime_context.json")
    result = read("independent_result.json")
    if context.get("information_scope", "development") != "development":
        raise ValueError("only development execution may enter research input")
    identity = {
        k: context[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
    }
    if any(result.get(k) != v for k, v in identity.items()):
        raise ValueError("independent result identity mismatch")
    if identity["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("protocol mismatch")
    bundle = store.get("bundles", identity["bundle_id"])
    if bundle["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("bundle protocol mismatch")
    task = context["task"]
    keys = [
        t["task_key"]
        for t in protocol["tasks"]
        if (t["suite"], t["task_id"]) == (task["suite"], task["task"])
    ]
    split = protocol["task_sets"]["development"]
    if (
        len(keys) != 1
        or keys[0] not in split["task_keys"]
        or task["seed"] not in split["seeds"]
    ):
        raise ValueError("only frozen development assignments may be imported")
    transcripts = list(root.glob("transcript*.json"))
    if len(transcripts) != 1:
        raise ValueError("one transcript required")
    events = read_trajectory(
        root / "states.json",
        transcripts[0],
        bundle_id=identity["bundle_id"],
        run_id=identity["run_id"],
        episode_id=identity["episode_id"],
        protocol_id=identity["protocol_id"],
        independent_result_path=root / "independent_result.json",
        action_events_path=root / "action_events.jsonl",
    )
    cost = read("cost_breakdown.json")
    if cost["identity"] != identity:
        raise ValueError("cost identity mismatch")
    # Preserve null/missing cost detail in L0 references; use the existing measured cost vector.
    audit = json.loads(Path(audit_path).read_text()) if audit_path else read("independent-audit.json")
    if audit["identity"] != identity:
        raise ValueError("audit identity mismatch")
    outcome = events[-1].outcome
    invalid = (
        result.get("validity") == "technical_invalid"
        or events[-1].flags["technical_invalid"]
    )
    view_id = result_id + "-view"
    view = EvidenceView(
        view_id,
        identity["bundle_id"],
        [asdict(e) for e in events],
        sorted({m for e in events for m in e.media}),
        IMPORT_VERSION,
    )
    refs = [
        str(Path(audit_path).resolve() if name == "independent-audit.json" and audit_path else root / name)
        for name in (
            "runtime_context.json",
            "independent_result.json",
            "cost_breakdown.json",
            "independent-audit.json",
            "states.json",
            "action_events.jsonl",
            "model_requests.jsonl",
            "environment_service/action_events.jsonl",
        )
    ]
    refs.append(str(transcripts[0]))
    if audit_path:
        refs.append(str(audit_path))
    if (root / "dataset_manifest.json").exists():
        refs.append(str(root / "dataset_manifest.json"))
    view_ref = str(store.path("evidence_views", view_id).resolve())
    record = ExperimentResult(
        result_id,
        identity["bundle_id"],
        protocol["protocol_id"],
        split.get("benchmark_id", protocol["benchmark_id"]),
        "development",
        protocol["scope"],
        {
            "information_scope": "development",
            "identity": identity,
            "import_version": IMPORT_VERSION,
            "evidence_view_id": view_id,
            "historical_execution_cost_only": True,
        },
        refs + [view_ref],
        protocol["protocol_id"],
        {},
        {},
        audit["costs"],
        assigned_count=1,
        unknown_count=int(outcome == "unknown"),
        invalid_count=int(invalid),
        outcome=outcome,
        assignments=[
            dict(
                task_key=keys[0],
                seed=task["seed"],
                outcome=outcome,
                validity="invalid" if invalid else "valid",
                evidence_ref=str(root / "independent_result.json"),
                outcome_source=events[-1].outcome_source and "native_benchmark",
            )
        ],
        notes=[
            "Historical development import; no new execution, charge, or release evidence."
        ],
    )
    receipt = {
        "identity": identity,
        "import_version": IMPORT_VERSION,
        "result_id": result_id,
        "source_refs": refs,
        "view_id": view_id,
        "cost_booking": "historical_only_no_charge",
    }
    # A run has one derivation per store. Changed classification uses a new versioned store.
    store.put("development_import_revisions" if revision_id else "development_imports",
              identity["run_id"] + "-" + revision_id if revision_id else identity["run_id"],
              {**receipt, "revision_id": revision_id} if revision_id else receipt)
    ledger = EventLedger(store)
    ledger.append_many(events)
    store.put("evidence_views", view_id, view)
    store.save_development_result(record)
    return record
