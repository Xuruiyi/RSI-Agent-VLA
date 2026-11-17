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


class ProcessTreeMemoryMeter:
    """Read RSS for the owned leader, descendants and exact inherited job tags.

    PID birth and UID bind each observation. Shared resident pages count per
    process, so this can overestimate physical RAM; exact peaks between samples
    remain unknown. This meter only reads process metadata.
    """

    def __init__(self, leader_pid: int, *, context_ref: str | Path) -> None:
        self.leader_pid = leader_pid
        self.uid = os.getuid()
        self.markers = {"RSI_RUNTIME_CONTEXT": {str(Path(context_ref).resolve())}}
        self.identities = {}
        self.current = {}
        self.peak = 0
        leader = self._process(leader_pid)
        if leader["uid"] != self.uid:
            raise PermissionError("memory meter leader belongs to another user")
        self.identities[leader_pid] = leader["start_ticks"]
        self.ancestor_identities = {(leader_pid, leader["start_ticks"])}

    @staticmethod
    def _process(pid: int) -> dict:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "uid": path.stat().st_uid, "state": fields[0],
                "ppid": int(fields[1]), "pgid": int(fields[2]),
                "start_ticks": int(fields[19]),
                "rss_bytes": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE")}

    @staticmethod
    def _markers(pid: int) -> dict:
        try:
            values = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return {}
        result = {}
        for value in values:
            name, _, content = value.partition(b"=")
            if name in {b"RSI_RUNTIME_CONTEXT", b"RSI_TRAINING_JOB_ROOT"}:
                result[name.decode()] = content.decode(errors="replace")
        return result

    def sample(self) -> dict:
        processes, markers = {}, {}
        for path in Path("/proc").glob("[0-9]*"):
            try:
                item = self._process(int(path.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if item["uid"] == self.uid and item["state"] != "Z":
                processes[item["pid"]] = item
                markers[item["pid"]] = self._markers(item["pid"])
        selected = {pid for pid, item in processes.items()
                    if self.identities.get(pid) == item["start_ticks"]}
        while True:
            before = set(selected)
            selected.update(pid for pid, item in processes.items()
                            if item["ppid"] in selected or any(
                                value in self.markers.get(name, set())
                                for name, value in markers[pid].items()))
            for pid in selected:
                for name, value in markers[pid].items():
                    self.markers.setdefault(name, set()).add(value)
            if before == selected:
                break
        proven = {pid for pid, item in processes.items()
                  if (pid, item["start_ticks"]) in self.ancestor_identities}
        while True:
            before = set(proven)
            proven.update(pid for pid, item in processes.items() if item["ppid"] in proven)
            if before == proven:
                break
        self.ancestor_identities.update((pid, processes[pid]["start_ticks"]) for pid in proven)
        self.current = {pid: {**processes[pid], "ancestor_observed": pid in proven} for pid in selected}
        self.identities = {pid: item["start_ticks"] for pid, item in self.current.items()}
        rss = sum(item["rss_bytes"] for item in self.current.values())
        self.peak = max(self.peak, rss)
        return {"current_rss_bytes": rss, "sampled_peak_rss_bytes": self.peak,
                "current_processes": list(self.current.values()), "exact_peak_rss_bytes": None,
                "basis": "same-UID leader/observed descendants/exact inherited job markers; PID birth bound; shared pages counted per process",
                "gap": "polling may miss short peaks and can temporarily exceed the stop threshold"}
