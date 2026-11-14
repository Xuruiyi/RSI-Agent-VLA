"""Persist sampled logical growth for the actual output and scratch write domains."""

import json
import os
from pathlib import Path

from .storage import atomic_json


def snapshot(roots):
    """Count each regular inode once, including hard-linked files across domains."""
    files = {}
    for root in roots:
        root = Path(root)
        if root.is_symlink():
            raise ValueError('storage write domain must not be a symlink')
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                try:
                    if path.is_symlink():
                        continue
                    stat = path.stat()
                    if path.is_file():
                        files[str(stat.st_dev) + ':' + str(stat.st_ino)] = stat.st_size
                except FileNotFoundError:
                    pass
    return files


class StorageMeter:
    """Sampled lower bound on materialization; deletion never refunds prior cost.

    Current occupancy, sampled peak and cumulative newly observed logical bytes
    are separate. Files created and deleted wholly between samples are unknown.
    Rewriting existing bytes is not additional logical storage. Preexisting
    shared files are a baseline, and overlapping roots/hard links count once.
    """

    def __init__(self, roots, *, baseline=None):
        self.roots = list(dict.fromkeys(str(Path(root).absolute()) for root in roots))
        self.baseline = snapshot(self.roots) if baseline is None else baseline
        self.previous = dict(self.baseline)
        self.cumulative = 0
        self.current = 0
        self.peak = 0

    def sample(self):
        now = snapshot(self.roots)
        self.cumulative += sum(max(0, size - self.previous.get(key, 0)) for key, size in now.items())
        self.current = sum(max(0, size - self.baseline.get(key, 0)) for key, size in now.items())
        self.peak = max(self.peak, self.current)
        self.previous = now
        return self.record()

    def record(self):
        return dict(roots=self.roots, cumulative_new_bytes_lower_bound=self.cumulative,
                    current_new_bytes=self.current, sampled_peak_new_bytes=self.peak,
                    exact_cumulative_bytes=None,
                    basis='sampled logical positive growth; inode deduplication; cleanup does not refund',
                    gap='files materialized and removed between samples are not observed')


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
    if not report_path.exists():
        return raw
    report = json.loads(report_path.read_text())
    if 'storage_output_bytes_at_sample' not in report:
        return max(raw, report.get('storage_bytes', 0))
    return max(raw, report.get('storage_bytes', 0)
               + max(0, raw - report['storage_output_bytes_at_sample']))
