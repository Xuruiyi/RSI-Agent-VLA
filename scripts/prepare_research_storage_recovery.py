#!/usr/bin/env python3
"""Prepare linked new Research identities with complete per-request budgets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.research_supervision import prepare_research_storage_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-configuration", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--request-suffix", required=True)
    parser.add_argument(
        "--research-sdk-projection",
        choices=["terminal_history_v1", "terminal_history_catalog_v2"],
    )
    parser.add_argument("--research-request-limit", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_research_storage_recovery(
                args.previous_configuration,
                args.stage,
                args.request_suffix,
                research_sdk_projection=args.research_sdk_projection,
                research_request_limit=args.research_request_limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
