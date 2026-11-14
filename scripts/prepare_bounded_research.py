#!/usr/bin/env python3
"""Prepare finite research input or reconcile a supplied response; no model/backend calls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))


def main():
    from rsi_vla.storage import ResearchStore, atomic_json
    from rsi_vla.research import ResearchJob
    from rsi_vla.research_tools import ResearchTools, BoundedResearchAdapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument(
        "--job",
        required=True,
        help="existing separately budgeted job; does not resume execution",
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result-id", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--cost-limit",
        type=Path,
        help="registered experiment cost vector, not new authorization",
    )
    args = parser.parse_args()
    store = ResearchStore(args.artifacts / "research")
    if not store.path("jobs", args.job).exists():
        raise ValueError("existing budgeted research job required")
    tools = ResearchTools(
        store,
        ResearchJob(store, args.job),
        json.loads(args.protocol.read_text()),
        domain_policy=json.loads(args.policy.read_text()),
    )
    adapter = BoundedResearchAdapter()
    request = adapter.prepare_request(
        tools,
        request_id=args.request,
        result_ids=args.result_id,
        max_input_tokens=64000,
        max_output_tokens=4000,
    )
    atomic_json(args.output / "request.json", request)
    atomic_json(
        args.output / "response-schema.json", adapter.response_schema(args.request, tools=tools)
    )
    if args.response:
        if not args.cost_limit:
            raise ValueError("explicit cost-limit vector required")
        result = adapter.apply_response(
            tools,
            json.loads(args.response.read_text()),
            evidence_level="offline" if args.synthetic else tools.protocol["scope"],
            cost_limit=json.loads(args.cost_limit.read_text()),
            synthetic=args.synthetic,
        )
        atomic_json(args.output / "prepared-response.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
