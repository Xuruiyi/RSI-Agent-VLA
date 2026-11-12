#!/usr/bin/env python3
"""Default dry-run for one finite Research request; execute requires a new matching permit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--permit", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    from rsi_vla.bounded_runner import child, execute, reconcile

    if args.child:
        child(args.child)
        return 0
    from rsi_vla.storage import ResearchStore
    from rsi_vla.research import ResearchJob
    from rsi_vla.research_tools import ResearchTools, BoundedResearchAdapter
    from rsi_vla.research_preflight import operator_preflight
    from rsi_vla.research_transport import unpack

    root = Path(__file__).resolve().parents[1]
    p = json.loads(args.plan.read_text())
    store = ResearchStore(Path(p["artifacts"]) / "research")
    job = ResearchJob(store, p["job_id"])
    tools = ResearchTools(
        store,
        job,
        json.loads(Path(p["protocol"]).read_text()),
        domain_policy=p["domain_policy"],
    )
    adapter = BoundedResearchAdapter(model="gpt-6.1-sol", reasoning_effort="medium")
    transport = store.get("bounded_transports", p["request_id"])
    if args.reconcile:
        print(
            json.dumps(
                reconcile(
                    tools, adapter, p["output"], cost_limit=p["experiment_cost_limit"]
                )
            )
        )
        return 0
    preflight = operator_preflight(
        tools, root, protocol_path=p["protocol"], result_ids=p["result_ids"]
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "request_id": unpack(transport)["request_id"],
                    "preflight": preflight,
                    "dispatch": False,
                    "blockers": ["no trialable operator"]
                    if not any(r["currently_trialable"] for r in preflight)
                    else ["new permit required"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.permit is None:
        raise PermissionError(
            "explicit new permit required; no historical authorization fallback"
        )
    print(
        json.dumps(
            execute(
                tools,
                adapter,
                transport,
                root=root,
                output=p["output"],
                permit_path=args.permit,
                preflight=preflight,
                cost_limit=p["experiment_cost_limit"],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
