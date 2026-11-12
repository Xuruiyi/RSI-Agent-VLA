#!/usr/bin/env python3
"""Inspect the matched installation without importing model/simulator packages."""
import argparse
import ast
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def inspect(root: Path) -> dict:
    packages = {}
    for name in ('torch', 'torchvision', 'rpent', 'rlinf', 'rpent-openpi',
                 'rpent-libero', 'rpent-liberopro', 'sam3', 'openai-codex',
                 'sentencepiece', 'safetensors', 'omegaconf', 'transformers',
                 'mujoco', 'robosuite', 'lerobot'):
        dist = metadata.distribution(name)
        direct = dist.read_text('direct_url.json')
        packages[name] = {'version': dist.version,
                          'source': json.loads(direct) if direct else None}
    origins = {}
    for name in ('rpent', 'rlinf', 'openpi', 'liberopro', 'sam3'):
        spec = importlib.util.find_spec(name)
        origins[name] = None if spec is None else (spec.origin or list(spec.submodule_search_locations or []))
    source = root / 'RPent/rpent/robots/components/pi05_vla_server.py'
    tree = ast.parse(source.read_text())
    presets = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '') == 'PI05_EMBODIMENTS')
    return {'python': sys.version.split()[0], 'executable': sys.executable,
            'packages': packages, 'module_origins': origins,
            'libero_model_preset': presets['libero'],
            'rpent_head': subprocess.check_output(['git', '-C', str(root / 'RPent'), 'rev-parse', 'HEAD'], text=True).strip(),
            'rpent_status': subprocess.check_output(['git', '-C', str(root / 'RPent'), 'status', '--short'], text=True),
            'evidence_kind': 'offline_metadata_and_static_source'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    actual = inspect(args.root.resolve())
    if args.expected:
        expected = json.loads(args.expected.read_text())
        for key in ('python', 'packages', 'libero_model_preset'):
            if actual[key] != expected[key]:
                raise SystemExit(f'matched installation changed: {key}')
        for name in ('rlinf', 'openpi', 'liberopro', 'sam3'):
            if actual['module_origins'][name] != expected['module_origins'][name]:
                raise SystemExit(f'module origin changed: {name}')
    print(json.dumps(actual, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
