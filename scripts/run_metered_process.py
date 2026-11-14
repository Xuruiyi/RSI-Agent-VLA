"""Run bounded offline preparation with a durable receipt before its side effects."""

import argparse
import json
import os
from pathlib import Path
import time

from rsi_vla.adapters.runtime_supervisor import supervise
from rsi_vla.process_accounting import snapshot, settle_receipt
from rsi_vla.storage import atomic_json, identifier


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--storage-bytes', type=int, required=True)
    parser.add_argument('--write-root', type=Path, action='append', default=[])
    parser.add_argument('--reconcile', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    identifier(args.key)
    receipt_path = args.output / (args.key + '-receipt.json')
    if args.reconcile:
        settle_receipt(args.phase, receipt_path)
        return 0
    if receipt_path.exists():
        raise ValueError('operation already recorded; inspect/reconcile receipt, never repeat launch')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('a process command is required')
    phase = json.loads(args.phase.read_text())
    if (args.wall_seconds <= 0 or args.storage_bytes <= 0
            or phase['remaining']['wall_seconds'] < args.wall_seconds
            or phase['remaining']['storage_bytes'] < args.storage_bytes):
        raise ValueError('process envelope exceeds remaining phase budget')
    roots = list(dict.fromkeys([str(args.output.resolve())] + [str(p.resolve()) for p in args.write_root]))
    baseline = snapshot(roots)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(scope=args.scope, key=args.key, status='started', command=command,
                   started_unix=time.time(), wall_seconds=None, storage=None)
    # Exclusive creation is the submission barrier; uncertain started receipts block repetition.
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream)
    output = args.output / args.key
    context = dict(dynamic_authorized=True, authorization_ref=str(receipt_path),
                   output_dir=str(output), job_started_monotonic=started,
                   storage_write_roots=roots, storage_baseline=baseline,
                   limits=dict(wall_seconds=args.wall_seconds, storage_bytes=args.storage_bytes))
    context_path = args.output / (args.key + '-context.json')
    atomic_json(context_path, context)
    scratch = args.output / (args.key + '-scratch')
    scratch.mkdir()
    settings = dict(TMPDIR=str(scratch.resolve()), XDG_CACHE_HOME=str((scratch / 'cache').resolve()))
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        result = supervise(command, context_path=context_path)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    receipt.update(status='completed', wall_seconds=time.monotonic() - started,
                   exit_code=result['exit_code'], stop_reason=result['stop_reason'],
                   cleanup_confirmed=result['cleanup_confirmed'], storage=result['storage_accounting'])
    atomic_json(receipt_path, receipt)
    settle_receipt(args.phase, receipt_path)
    print(json.dumps(receipt))
    return 0 if result['status'] == 'completed' and result['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
