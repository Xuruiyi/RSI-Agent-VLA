#!/usr/bin/env python3
"""Prepare one development trial; execution requires its new concrete permit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--permit", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--reconcile", action="store_true")
    mode.add_argument("--reparse", action="store_true", help="derive a versioned audit of existing output; no execution")
    parser.add_argument("--revision-id")
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--read-source-ref", type=Path)
    args = parser.parse_args()
    if args.reparse and (not args.revision_id or args.audit_root is None):
        parser.error("--reparse requires --revision-id and --audit-root")
    from rsi_vla.storage import ResearchStore
    from rsi_vla.research import ResearchJob
    from rsi_vla.research_tools import ResearchTools
    from rsi_vla.research_cycle import prepare_trial, execute_trial, reconcile_trial

    p = json.loads(args.plan.read_text())
    store = ResearchStore(Path(p["artifacts"]) / "research")
    tools = ResearchTools(
        store,
        ResearchJob(store, p["job_id"]),
        json.loads(Path(p["protocol"]).read_text()),
        domain_policy=p["domain_policy"],
    )
    experiment = args.experiment or p["request_id"] + "-experiment"
    if not store.path("rule_trials", experiment).exists():
        if args.execute or args.reconcile or args.reparse:
            raise ValueError("no actual Research trial; defer never starts a task")
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "dispatch": False,
                    "blockers": [
                        "await new authorized Research response and actual trial"
                    ],
                }
            )
        )
        return
    if args.execute or args.reconcile or args.reparse:
        if args.permit is None:
            raise PermissionError("new artifact-bound development permit required")
    if args.reparse:
        from rsi_vla.trial_reparse import reparse_trial
        result = reparse_trial(tools, experiment, output=p["development_output"], permit_path=args.permit,
                               revision_id=args.revision_id, audit_root=args.audit_root, read_source_ref=args.read_source_ref)
    elif args.reconcile:
        result = reconcile_trial(
            tools, experiment, output=p["development_output"], permit_path=args.permit
        )
    else:
        plan = prepare_trial(
            tools,
            experiment,
            root=Path(__file__).resolve().parents[1],
            artifacts=p["artifacts"],
            protocol_path=p["protocol"],
            output=p["development_output"],
            run_id=p["development_run_id"],
            permit_path=args.permit,
        )
        if plan.get("mode") == "already_recorded":
            result = tools._owned("research_cycles", experiment + "-cycle")
        else:
            result = (
                execute_trial(tools, plan, permit_path=args.permit)
                if args.execute
                else plan
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
