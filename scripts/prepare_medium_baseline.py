#!/usr/bin/env python3
"""Emit the frozen medium baseline queue without starting any job."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1] / 'config/pilot_v7.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['runtime_config']['reasoning_effort'] != 'medium':
        raise ValueError('baseline queue requires the medium protocol')
    slots = []
    for split in ('development', 'release'):
        spec = protocol['task_sets'][split]
        for task_key in spec['task_keys']:
            for seed in spec['seeds']:
                slots.append(dict(protocol_id=protocol['protocol_id'], bundle_id=protocol['anchor_bundle_id'],
                                  split=split, task_key=task_key, seed=seed, status='not_started',
                                  independent_result=None, capability=None))
    queue = dict(schema_version='1.0', purpose='medium baseline planning, not runtime evidence',
                 dynamic_authorized=False, concurrent_jobs=1, sealed_scheduled=False,
                 prerequisite='checkpoint 1 dynamic acceptance and explicit matrix resource authorization',
                 initial_current_reuses_anchor_result=True, assignments=slots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(queue, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(assignments=len(slots), dynamic_jobs_started=0, output=str(args.output))))


if __name__ == '__main__':
    main()
