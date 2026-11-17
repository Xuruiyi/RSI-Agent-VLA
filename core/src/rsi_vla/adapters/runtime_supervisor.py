"""Supervise an authorized backend process group without loading robot/model code.

Storage has a sampled stop threshold, with measured overshoot. GPU device
seconds are an allocated-device wall-time envelope, not utilization telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def _storage_bytes(root):
    total = 0
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass  # Atomic producer renames are permitted.
    return total


def _live_group(pgid):
    """Return Linux group members; zombies cannot execute further work."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return []
    except PermissionError:
        pass  # The group exists; retain the normal membership check.
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail = path.read_text().rsplit(')', 1)[1].split()
            if int(tail[2]) == pgid and tail[0] != 'Z':
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def _stop_group(process, grace, interval, complete_deadline):
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # A live leader already establishes that the group remains. Scan for
    # descendants only after its exit, keeping graceful cleanup out of repeated
    # whole-host /proc scans.
    deadline = min(time.monotonic() + grace, complete_deadline - interval)
    while time.monotonic() < deadline:
        process.poll()
        remaining = [pgid] if process.returncode is None else _live_group(pgid)
        if not remaining:
            # Membership can disappear after the poll above; reap that exit too.
            process.poll()
            return []
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    process.poll()
    if process.returncode is None or _live_group(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Reap before checking membership; an absent group needs no /proc scan.
    while time.monotonic() < complete_deadline:
        process.poll()
        remaining = [pgid] if process.returncode is None else _live_group(pgid)
        if not remaining:
            # Membership can disappear after the poll above; reap that exit too.
            process.poll()
            return []
        time.sleep(min(interval, max(0, complete_deadline - time.monotonic())))
    process.poll()
    return _live_group(pgid)


def _stop_measured_tree(process, meter, grace, interval, complete_deadline):
    """Stop only birth-matched observed descendants; tag-only rows remain unknown."""
    def signal_descendants(sig):
        for row in meter.sample()["current_processes"]:
            if row["pid"] == process.pid or not row.get("ancestor_observed"):
                continue
            try:
                current = meter._process(row["pid"])
                if (current["uid"] == row["uid"] == os.getuid()
                        and current["start_ticks"] == row["start_ticks"]):
                    os.kill(row["pid"], sig)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    signal_descendants(signal.SIGTERM)
    remaining = _stop_group(process, grace, interval, complete_deadline)
    until = min(time.monotonic() + grace, complete_deadline - interval)
    while time.monotonic() < until and meter.sample()["current_processes"]:
        time.sleep(min(interval, max(0, until - time.monotonic())))
    signal_descendants(signal.SIGKILL)
    while time.monotonic() < complete_deadline:
        process.poll()
        rows = meter.sample()["current_processes"]
        if not rows:
            return remaining
        if not any(row.get("ancestor_observed") for row in rows):
            break
        time.sleep(min(interval, max(0, complete_deadline - time.monotonic())))
    return sorted(set(remaining) | {row["pid"] for row in meter.sample()["current_processes"]})


def supervise(command: list[str], *, context_path: str | Path,
              poll_interval_s: float = 0.1, terminate_grace_s: float = 5.0) -> dict:
    """Launch one authorized job and stop its whole group on the first bound.

    The caller validates bundle/profile/protocol and holds the global job lock.
    The context binds dynamic authority or an explicit live CPU model0 permit.
    No retries, deleted artifacts or follow-up jobs are performed.
    """
    from ..storage import atomic_json
    context_path = Path(context_path).resolve()
    context = json.loads(context_path.read_text(encoding='utf-8'))
    if context.get('cpu_authorization_ref'):
        from ..cpu_command import validate_supervision
        validate_supervision(context)
    elif context.get('dynamic_authorized') is not True or not context.get('authorization_ref'):
        raise PermissionError('runtime dynamic job is not authorized')
    if not command or poll_interval_s <= 0 or terminate_grace_s < 0:
        raise ValueError('invalid command or supervision interval')
    limits = context['limits']
    memory_limit = context.get('memory_limit_bytes')
    if memory_limit is not None and (type(memory_limit) is not int or memory_limit <= 0):
        raise ValueError('memory_limit_bytes must be a positive integer')
    memory_meter = None
    memory_record = None
    wall_limit = limits['wall_seconds']
    storage_limit = limits['storage_bytes']
    gpu_limit = limits.get('gpu_seconds')
    gpu_count = context.get('gpu_count')
    if wall_limit <= 0 or storage_limit <= 0:
        raise ValueError('wall_seconds and storage_bytes must be positive')
    if gpu_limit is not None and (type(gpu_count) is not int or gpu_count < 1 or gpu_limit <= 0):
        raise ValueError('finite gpu_seconds requires a positive allocated gpu_count')
    output_dir = Path(context['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'supervisor.json'
    if report_path.exists():
        raise FileExistsError('this run already has supervisor.json')
    started = context.get('job_started_monotonic', time.monotonic())
    complete_seconds = min(wall_limit, gpu_limit / gpu_count) if gpu_limit is not None else wall_limit
    deadline_reason = 'gpu_seconds_envelope' if complete_seconds < wall_limit else 'wall_seconds'
    complete_deadline = started + complete_seconds
    # Reserve stop publication, graceful cleanup and a forced-termination sample inside the
    # whole-job budget. Very short offline fixtures use a proportional grace.
    effective_grace = min(terminate_grace_s, complete_seconds / 3)
    effective_interval = min(poll_interval_s, complete_seconds / 10)
    stop_deadline = complete_deadline - effective_grace - 2 * effective_interval
    from ..process_accounting import StorageMeter
    state_ref = context.get('storage_meter_state_ref')
    storage_state = json.loads(Path(state_ref).read_text()) if state_ref and Path(state_ref).exists() else None
    storage_meter = StorageMeter(context.get('storage_write_roots', [output_dir]),
        baseline=context.get('storage_baseline', {}), state=storage_state)
    business_before = storage_meter.cumulative
    scope_state_ref = context.get('scope_meter_state_ref')
    scope_state = json.loads(Path(scope_state_ref).read_text()) if scope_state_ref and Path(scope_state_ref).exists() else None
    scope_meter = StorageMeter(context['scope_accounting_roots'], state=scope_state) if context.get('scope_accounting_roots') else None
    scope_before = scope_meter.cumulative if scope_meter else 0
    storage_record = storage_meter.sample()
    scope_record = scope_meter.sample() if scope_meter else None
    peak_storage = storage_record['cumulative_new_bytes_lower_bound']
    stop_reason = None
    cleanup_seconds = 0.0
    error = None
    process = None
    remaining = []
    interrupted = threading.Event()
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda _signum, _frame: interrupted.set())
    def stop_owned():
        if memory_meter:
            return _stop_measured_tree(process, memory_meter, effective_grace, effective_interval, complete_deadline)
        return _stop_group(process, effective_grace, effective_interval, complete_deadline)

    def publish_stop(reason):
        from ..storage import atomic_json
        atomic_json(output_dir/'runtime_stop.json', {
            'identity': {key:context.get(key) for key in ('run_id','episode_id','bundle_id','protocol_id')},
            'reason': reason,
        })

    try:
        if peak_storage >= storage_limit:
            stop_reason = 'storage_bytes'
            publish_stop(stop_reason)
        else:
            env = {**os.environ, 'RSI_RUNTIME_CONTEXT': str(context_path)}
            with (output_dir / 'backend.log').open('xb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                if memory_limit is not None:
                    from ..process_accounting import ProcessTreeMemoryMeter
                    memory_meter = ProcessTreeMemoryMeter(process.pid, context_ref=context_path)
                while True:
                    memory_record = memory_meter.sample() if memory_meter else None
                    storage_record = storage_meter.sample()
                    scope_record = scope_meter.sample() if scope_meter else None
                    peak_storage = max(peak_storage, storage_record['cumulative_new_bytes_lower_bound'])
                    from ..storage import atomic_json
                    atomic_json(output_dir / 'storage-progress.json', storage_record)
                    if interrupted.is_set():
                        stop_reason = 'external_interruption'
                    elif time.monotonic() >= stop_deadline:
                        stop_reason = deadline_reason
                    elif memory_record and memory_record['current_rss_bytes'] >= memory_limit:
                        stop_reason = 'memory_bytes'
                    elif peak_storage >= storage_limit:
                        stop_reason = 'storage_bytes'
                    elif scope_meter and scope_record['cumulative_new_bytes_lower_bound'] - scope_before >= context['scope_storage_remaining_bytes']:
                        stop_reason = 'scope_storage_bytes'
                    elif scope_meter and sum({value['inode']:value['size'] for value in scope_meter.previous.values()}.values()) >= context['scope_current_disk_limit_bytes']:
                        stop_reason = 'scope_current_disk_bytes'
                    if stop_reason is not None:
                        publish_stop(stop_reason)
                        cleanup_started = time.monotonic()
                        remaining = stop_owned()
                        cleanup_seconds += time.monotonic()-cleanup_started
                        break
                    if process.poll() is not None:
                        if _live_group(process.pid) or memory_meter and memory_meter.sample()['current_processes']:
                            stop_reason = 'orphaned_backend_processes'
                            publish_stop(stop_reason)
                            cleanup_started = time.monotonic()
                            remaining = stop_owned()
                            cleanup_seconds += time.monotonic()-cleanup_started
                        break
                    time.sleep(min(effective_interval, max(0, stop_deadline - time.monotonic())))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        stop_reason = 'backend_startup_error' if process is None else 'supervision_error'
        publish_stop(stop_reason)
        if process is not None:
            cleanup_started = time.monotonic()
            remaining = stop_owned()
    except BaseException:
        publish_stop("external_interruption")
        if process is not None:
            stop_owned()
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    wall = time.monotonic() - started
    if memory_meter:
        memory_record = memory_meter.sample()
        remaining = sorted(set(remaining) | {row['pid'] for row in memory_record['current_processes']})
    storage_record = storage_meter.sample()
    peak_storage = max(peak_storage, storage_record['cumulative_new_bytes_lower_bound'])
    report = {
        'schema_version': '1.0',
        **{key: context.get(key) for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id', 'profile_ref')},
        'identity': {key: context.get(key) for key in ('run_id', 'episode_id', 'bundle_id', 'protocol_id', 'profile_ref')},
        'status': 'interrupted' if stop_reason else 'completed',
        'stop_reason': stop_reason, 'exit_code': process.returncode if process is not None else None,
        'error': error,
        'wall_seconds': wall, 'wall_overshoot_seconds': max(0, wall - wall_limit),
        'storage_bytes': peak_storage, 'storage_overshoot_bytes': max(0, peak_storage - storage_limit),
        'storage_measurement': 'sampled logical growth across explicit output/scratch/log/cache domains',
        'storage_accounting': storage_record,
        'poll_interval_s': effective_interval, 'terminate_grace_s': effective_grace,
        'complete_budget_seconds': complete_seconds,
        'cleanup_reserved_seconds': effective_grace + 2 * effective_interval,
        'gpu_device_seconds': None,
        'gpu_device_seconds_envelope': wall * gpu_count if type(gpu_count) is int else None,
        'gpu_measurement': 'allocated device wall-time envelope, not utilization telemetry',
        'process_group_cleanup_seconds': cleanup_seconds,
        'cleanup_confirmed': not remaining, 'remaining_process_ids': remaining,
        'missing': ['gpu_utilization_telemetry', 'backend_token_costs:see_runtime_costs.json'],
    }
    if memory_limit is not None:
        report['memory_limit_bytes'] = memory_limit
        report['memory_accounting'] = memory_record
        report['memory_enforcement'] = 'conservative sampled process-tree RSS stop; no kernel cgroup quota; tag-only residue is not signalled'
    report['business_storage_delta_bytes'] = storage_meter.cumulative - business_before
    if scope_meter:
        report['scope_storage_accounting'] = scope_meter.sample()
        report['scope_storage_delta_bytes'] = scope_meter.cumulative - scope_before
        report['control_storage_delta_lower_bound'] = max(0, report['scope_storage_delta_bytes'] - report['business_storage_delta_bytes'])
    # Include the report itself in the final retained byte count.
    for _ in range(3):
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report['storage_accounting'] = storage_meter.sample()
        report['storage_bytes'] = max(report['storage_bytes'], report['storage_accounting']['cumulative_new_bytes_lower_bound'])
        report['storage_overshoot_bytes'] = max(0, report['storage_bytes'] - storage_limit)
        report['business_storage_delta_bytes'] = storage_meter.cumulative - business_before
        if scope_meter:
            report['scope_storage_accounting'] = scope_meter.sample()
            report['scope_storage_delta_bytes'] = scope_meter.cumulative - scope_before
            report['control_storage_delta_lower_bound'] = max(0, report['scope_storage_delta_bytes'] - report['business_storage_delta_bytes'])
    report['storage_output_bytes_at_sample'] = _storage_bytes(output_dir)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if state_ref:
        atomic_json(state_ref, storage_meter.state())
    if scope_meter and scope_state_ref:
        atomic_json(scope_state_ref, scope_meter.state())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = supervise(command, context_path=args.context)
    return 0 if result['status'] == 'completed' and result['exit_code'] == 0 and result['cleanup_confirmed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
