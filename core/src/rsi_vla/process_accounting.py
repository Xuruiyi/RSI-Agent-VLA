"""Persist sampled logical growth for the actual output and scratch write domains."""

import json
import os
import glob
from pathlib import Path

from .storage import atomic_json


def snapshot(roots):
    """Count each regular inode once, including hard-linked files across domains."""
    return {value['inode']: value['size'] for value in logical_snapshot(roots).values()}


def logical_snapshot(roots):
    """Retain logical paths while exposing inode identity for hard-link deduplication."""
    files = {}
    for root in roots:
        matches = glob.glob(str(root)) if glob.has_magic(str(root)) else [str(root)]
        for match in matches:
            _snapshot_root(Path(match), files)
    return files


def _snapshot_root(root, files):
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError('storage write domain must not be a symlink')
    if root.is_file():
        try:
            stat = root.stat()
            files[str(root)] = {'inode': str(stat.st_dev) + ':' + str(stat.st_ino), 'size': stat.st_size}
        except FileNotFoundError:
            pass  # A registered artifact may be removed between samples.
        return
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in names:
            path = Path(directory) / name
            try:
                if not path.is_symlink() and path.is_file():
                    stat = path.stat()
                    files[str(path)] = {'inode': str(stat.st_dev) + ':' + str(stat.st_ino), 'size': stat.st_size}
            except FileNotFoundError:
                pass


def logical_growth(now, before):
    groups = {}
    for path, value in now.items():
        groups.setdefault(value['inode'], []).append((path, value['size']))
    previous_groups = {}
    for path, value in before.items():
        previous_groups.setdefault(value['inode'], []).append((path, value['size']))
    credits = {}
    for inode, members in previous_groups.items():
        # A retained hard link owns its old credit. Otherwise one replacement
        # at the same logical path may inherit it, never multiple new inodes.
        target = inode if inode in groups else next((now[path]['inode'] for path, _ in members if path in now), None)
        if target is not None:
            credits[target] = max(credits.get(target, 0), max(size for _, size in members))
    return sum(max(0, max(size for _, size in members)-credits.get(inode, 0))
               for inode, members in groups.items())



class StorageMeter:
    """Sampled lower bound on materialization; deletion never refunds prior cost.

    Current occupancy, sampled peak and cumulative newly observed logical bytes
    are separate. Files created and deleted wholly between samples are unknown.
    Rewriting existing bytes is not additional logical storage. Preexisting
    shared files are a baseline, and overlapping roots/hard links count once.
    """

    def __init__(self, roots, *, baseline=None, state=None):
        self.roots = list(dict.fromkeys(str(Path(root).absolute()) for root in roots))
        now = logical_snapshot(self.roots)
        if state is not None:
            if state.get('schema') != 'logical-storage-state-v2' or state['roots'] != self.roots:
                raise ValueError('storage continuation state differs from registered roots/version')
            self.baseline = state['baseline']
            self.previous = state['previous']
            self.cumulative, self.current, self.peak = (state[key] for key in ('cumulative', 'current', 'peak'))
        else:
            if baseline is None:
                files = now
            elif baseline.get('schema') == 'logical-storage-baseline-v2':
                files = baseline['files']
            else:
                files = {path: value for path, value in now.items() if value['inode'] in baseline}
                files = {path: {**value, 'size': baseline[value['inode']]} for path, value in files.items()}
            self.baseline = {'schema': 'logical-storage-baseline-v2', 'files': files}
            self.previous = dict(files)
            self.cumulative = self.current = self.peak = 0

    def sample(self):
        now = logical_snapshot(self.roots)
        self.cumulative += logical_growth(now, self.previous)
        self.current = logical_growth(now, self.baseline['files'])
        self.peak = max(self.peak, self.current)
        self.previous = now
        return self.record()

    def record(self):
        return dict(meter_version='logical-path-v2', roots=self.roots, cumulative_new_bytes_lower_bound=self.cumulative,
                    current_new_bytes=self.current, sampled_peak_new_bytes=self.peak,
                    exact_cumulative_bytes=None,
                    basis='sampled positive size growth by logical path; hard links deduplicated; atomic replacement is not new bytes; cleanup does not refund',
                    gap='files materialized and removed between samples are not observed')

    def state(self):
        return dict(schema='logical-storage-state-v2', roots=self.roots, baseline=self.baseline,
                    previous=self.previous, cumulative=self.cumulative, current=self.current, peak=self.peak)


def settle_receipt(phase_path, receipt_path):
    """Import one completed process receipt idempotently before evaluation.

    The caller serializes phase writers. Receipt persistence precedes this
    operation, so a crash after the atomic phase write is safe to reconcile.
    """
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('status') != 'completed' or receipt.get('wall_seconds') is None:
        raise ValueError('submission status unknown; inspect receipt before repeating')
    costs = dict(wall_seconds=receipt['wall_seconds'],
                 storage_bytes=receipt['storage']['cumulative_new_bytes_lower_bound'])
    phase_path = Path(phase_path)
    phase = json.loads(phase_path.read_text())
    entry = dict(scope=receipt['scope'], key=receipt['key'], costs=costs,
                 receipt_ref=str(Path(receipt_path).resolve()))
    entries = phase.setdefault('metered_process_charges', [])
    existing = [item for item in entries if (item['scope'], item['key']) == (entry['scope'], entry['key'])]
    if existing:
        if existing != [entry]:
            raise ValueError('process receipt conflicts with existing charge')
        return entry
    entries.append(entry)
    for dimension, amount in costs.items():
        phase['used'][dimension] += amount
        phase['remaining'][dimension] = phase['limits'][dimension] - phase['used'][dimension]
    atomic_json(phase_path, phase)
    return entry


def accounted_storage_bytes(output):
    """Retain sampled scratch cost and add output growth after supervision."""
    output = Path(output)
    raw = sum(snapshot([output]).values())
    report_path = output / 'supervisor.json'
    accounted = raw
    if report_path.exists():
        report = json.loads(report_path.read_text())
        accounted = max(raw, report.get('storage_bytes', 0))
        if 'storage_output_bytes_at_sample' in report:
            accounted = max(accounted, report.get('storage_bytes', 0)
                            + max(0, raw - report['storage_output_bytes_at_sample']))
    if (output / 'artifact_retention.json').exists():
        from .artifact_retention import certified_retention_floor
        floor, retained = certified_retention_floor(output)
        accounted = max(accounted, floor + max(0, raw - retained))
    return accounted
