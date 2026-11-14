"""Audit the one supported ancillary file read, from SDK and source evidence."""

import ast
import json
import subprocess
from pathlib import Path

from .adapters.runtime_acceptance import audit_sdk_tools
from .adapters.runtime_closure import local_stop_audit

READ_PATH = '/home/xzh/.codex/skills/stop-that-shit/SKILL.md'
READ_SOURCES = (
    'rpent/tools/common.py', 'rpent/tools/base.py', 'rpent/tools/toolkit.py',
    'rpent/memory/manager.py', 'rpent/memory/__init__.py',
    'rpent/planner/utils/http_mcp_server.py', 'rpent/planner/base.py',
    'rpent/utils/config.py',
)
# This is the reviewed implementation, not a claim that these modules were frozen.
REVIEWED_REVISION = 'c57b134577dde2cc62cbdf967b28d839fc11cb14'


def _read(path):
    return json.loads(Path(path).read_text())


def audit_read(root, identity, calls, ends, *, source_ref=None):
    """Keep unknown on missing provenance; reject mismatched SDK identity/content."""
    root = Path(root)
    report = {'value': None, 'checks': {}, 'missing': [], 'evidence_refs': [],
              'source_basis': None, 'reads': []}
    checks = report['checks']
    try:
        context = _read(root / 'runtime_context.json')
        if any(context.get(k) != v for k, v in identity.items()):
            raise ValueError('read runtime identity mismatch')
        sdk_paths = list(root.glob('codex_*.stream.jsonl'))
        if len(sdk_paths) != 1:
            raise OSError('one complete SDK stream required')
        started, completed = [], []
        for number, line in enumerate(sdk_paths[0].read_text().splitlines(), 1):
            event = json.loads(line)
            item = (event.get('payload') or {}).get('item', {})
            if item.get('type') != 'mcpToolCall':
                continue
            if event.get('method') == 'item/started':
                started.append((number, item))
            elif event.get('method') == 'item/completed':
                completed.append((number, item))
        events = [json.loads(line) for line in (root / 'action_events.jsonl').read_text().splitlines()]
        tool_audit = audit_sdk_tools(root, identity, events=events)
        checks['sdk_l0_correspondence'] = not tool_audit['issues']
        report['tool_audit'] = tool_audit
        read_starts = [(number, item) for number, item in started if item['tool'] == 'read_text_file']
        read_ends = [(number, item) for number, item in completed if item['tool'] == 'read_text_file']
        read_ids = {item['id'] for _, item in read_starts}
        checks['sdk_read_terminal_pair'] = (bool(read_starts)
            and len(read_ids) == len(read_starts) == len(read_ends)
            and {item['id'] for _, item in read_ends} == read_ids)
        for _, item in read_ends:
            starts_for_id = [entry for _, entry in read_starts if entry['id'] == item['id']]
            associated = [entry for entry in tool_audit['tools'] if entry['id'] == item['id']]
            if len(starts_for_id) != 1 or len(associated) != 1:
                checks['sdk_read_terminal_pair'] = False
                continue
            start_item, association = starts_for_id[0], associated[0]
            l0_starts = [event for event in calls if event.get('sequence') == association.get('l0_start_sequence')]
            l0_ends = [event for event in ends if event.get('sequence') == association.get('l0_terminal_sequence')]
            paired = (len(l0_starts) == len(l0_ends) == 1
                and start_item['server'] == item['server'] == 'rpent'
                and start_item.get('arguments') == item.get('arguments')
                and item.get('status') == 'completed')
            if paired:
                l0_start, l0_end = l0_starts[0], l0_ends[0]
                paired = (l0_start['tool'] == l0_end['tool'] == 'read_text_file'
                    and l0_end.get('status') == 'completed'
                    and start_item['started_at_ms'] <= l0_start['utc_ns'] / 1e6
                        <= l0_end['utc_ns'] / 1e6 <= item['completed_at_ms'])
            checks['sdk_read_terminal_pair'] &= paired
        missing_nonreads = [item for item in tool_audit['tools']
                            if not item['sdk_completed'] and item.get('tool') != 'read_text_file']
        if missing_nonreads:
            model_events = [json.loads(line) for line in (root / 'model_requests.jsonl').read_text().splitlines()]
            stop = local_stop_audit(root, identity, model_events, events, _read(root / 'result.json'))
            report['local_stop_audit'] = stop
            checks['nonread_sdk_delivery_closed'] = stop['applicable'] and not stop['issues']
            report['evidence_refs'].extend(str(root / name) for name in
                ('action_events.jsonl', 'model_requests.jsonl', 'result.json', 'local_model_stop.json'))
        checks['counted_read'] = _read(root / 'runtime_costs.json')['tool_calls'] == len(calls)
        for number, item in completed:
            if item['tool'] != 'read_text_file':
                continue
            args = item['arguments']
            max_chars = args.get('max_chars')
            if (args.get('path') != READ_PATH
                    or 'max_chars' in args and (type(max_chars) is not int or max_chars <= 0)):
                raise ValueError('file read outside reviewed path/range')
            result = item['result']
            if result.get('isError', False) is not False or len(result['content']) != 1 or result['content'][0]['type'] != 'text':
                raise ValueError('file read returned error/nontext content')
            data = json.loads(result['content'][0]['text'])
            if data.get('error') is not None:
                raise ValueError('file read payload reports an error')
            content = data['content']
            # SDK JSON must be complete; never fill a historical read from today's file.
            full = (data['path'] == READ_PATH and type(data['size']) is int
                    and data['size'] == len(content) and (max_chars is None or len(content) <= max_chars)
                    and '[TRUNCATED' not in content and '[truncated]' not in content
                    and content.startswith('---\nname: stop-that-shit\n')
                    and 'Meet the task' in content)
            checks['returned_complete_skill_text'] = checks.get('returned_complete_skill_text', True) and full
            report['reads'].append({'SDK_ref': str(sdk_paths[0]), 'line': number,
                                    'call_id': item['id'], 'arguments': args, 'effective_max_chars': max_chars,
                                    'returned_chars': data['size'], 'truncated': not full,
                                    'content_ref': f'{sdk_paths[0]}:{number}'})
        if not report['reads']:
            raise OSError('no actual read in SDK')
        report['evidence_refs'].append(str(sdk_paths[0]))
        # Frozen concrete construction binds the plain manager, not an arbitrary subclass.
        spec = (root / 'code_snapshot/RPent/robots/libero/robot_spec.py').read_text()
        toolkit = (root / 'code_snapshot/RPent/robots/libero/toolkit.py').read_text()
        checks['concrete_dispatch_consumer'] = all(s in spec for s in (
            'memory = MemoryManager(', 'memory=memory,', 'return LiberoToolkit(')) and all(s in toolkit for s in (
            'journal.admit_tool(name)', 'super().execute_tool(name, arguments)', '"tool_end"'))
        if source_ref is None:
            repo = Path(_read(root / 'launch_plan.json')['cwd'])
            revision = REVIEWED_REVISION
            origin = root / 'code_snapshot/RPent'
            source_kind = 'prelaunch_frozen_reviewed_sources'
            provenance = {'git_state_before_ref': None}
        else:
            provenance = _read(source_ref)
            if provenance['identity'] != identity or Path(provenance['run_root']).resolve() != root.resolve():
                raise ValueError('read source provenance belongs to another run')
            before = _read(provenance['git_state_before_ref'])
            revision = before['RPent_HEAD_and_origin'].splitlines()[0]
            if revision != REVIEWED_REVISION or provenance['revision'] != revision:
                raise ValueError('unreviewed historical read implementation revision')
            repo = Path(provenance['repository'])
            if set(provenance['files']) != set(READ_SOURCES):
                raise OSError('incomplete read producer/consumer sources')
            for name in READ_SOURCES:
                if any(line[3:].strip('"') == name for line in before['RPent'].splitlines()[1:]):
                    raise OSError('historical dirty read source lacks frozen bytes: ' + name)
            origin = Path(source_ref).parent / 'source-origin'
            source_kind = 'posthoc_git_origin_not_runtime_frozen'
        launch = _read(root / 'launch_plan.json')
        if Path(launch['cwd']).resolve() != repo.resolve() or launch['environment']['RPENT_REPO_ROOT'] != str(repo):
            raise ValueError('saved runtime source root differs from read provenance')
        for name in READ_SOURCES:
            expected = subprocess.run(['git', '-C', str(repo), 'show', revision + ':' + name],
                                      capture_output=True, check=True).stdout
            copied = origin / name
            if copied.read_bytes() != expected:
                raise ValueError('read source copy differs from reviewed Git: ' + name)
            report['evidence_refs'].append(str(copied))
        common = (origin / 'rpent/tools/common.py').read_text()
        manager = (origin / 'rpent/memory/manager.py').read_text()
        # Exact Git bytes above bind the manually reviewed semantics of this specific revision.
        checks['read_permission_and_effects'] = all(s in common for s in (
            'self._memory.authorize_read(path)', 'resolved.read_text(encoding="utf-8", errors="replace")',
            'self._memory.record_read(resolved, content)')) and all(s in manager for s in (
            'return self._authorize(path, write=False)', 'resolved = (repo_root / path).resolve()',
            'if not resolved.is_relative_to(self._root):', 'return resolved'))
        report['source_basis'] = {
            'kind': source_kind, 'revision': revision,
            'git_state_before_ref': provenance['git_state_before_ref'], 'launch_ref': str(root / 'launch_plan.json'),
            'permission': 'absolute path outside run memory and RPent/memory; _authorize returns resolved path',
            'effects': 'UTF-8 file read under lock; plain MemoryManager.record_read is a docstring-only no-op; no write/exec/control consumer',
            'limits': 'saved SDK proves delivered text; Git/state/launch establish source relationship, not a prelaunch module copy or OS access isolation',
        }
        if source_ref:
            report['evidence_refs'].append(str(source_ref))
        tree = ast.parse(common)
        common_tools = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CommonTools')
        read_method = next(node for node in common_tools.body if isinstance(node, ast.FunctionDef) and node.name == 'read_text_file')
        parameters = read_method.args.posonlyargs + read_method.args.args
        defaults = dict(zip((arg.arg for arg in parameters[-len(read_method.args.defaults):]), read_method.args.defaults))
        default_max_chars = ast.literal_eval(defaults['max_chars'])
        if type(default_max_chars) is not int or default_max_chars <= 0:
            raise ValueError('reviewed read default is not a positive integer')
        report['source_basis']['read_max_chars_default'] = default_max_chars
        report['source_basis']['read_default_basis'] = 'CommonTools.read_text_file AST in exact reviewed source bytes'
        for reading in report['reads']:
            if reading['effective_max_chars'] is None:
                reading['effective_max_chars'] = default_max_chars
                within_default = reading['returned_chars'] <= default_max_chars
                reading['truncated'] |= not within_default
                checks['returned_complete_skill_text'] &= within_default
    except (OSError, KeyError, TypeError, StopIteration, subprocess.CalledProcessError) as exc:
        report['missing'].append(str(exc))
    except ValueError as exc:
        checks['read_evidence_matches'] = False
        report['missing'].append(str(exc))
    values = list(checks.values())
    report['value'] = False if False in values else True if values and not report['missing'] and all(v is True for v in values) else None
    return report
