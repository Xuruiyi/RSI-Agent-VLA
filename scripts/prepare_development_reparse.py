#!/usr/bin/env python3
"""Prepare a versioned development reparse and the original unstarted graph tail."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.development_reparse import prepare_reparse_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-configuration", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--research-result-id", action="append", dest="research_result_ids"
    )
    parser.add_argument(
        "--research-development-task",
        "--research-task-key",
        dest="research_development_task",
    )
    parser.add_argument("--research-media-refs", choices=("selected_events",))
    parser.add_argument("--research-sdk-projection", choices=("terminal_history_v1",))
    args = parser.parse_args()
    result = prepare_reparse_stage(
        args.previous_configuration,
        run_id=args.run,
        revision_id=args.revision,
        stage_id=args.stage,
        research_result_ids=args.research_result_ids,
        research_development_task=args.research_development_task,
        research_media_refs=args.research_media_refs,
        research_sdk_projection=args.research_sdk_projection,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
