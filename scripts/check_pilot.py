"""Static stdlib-only verification of frozen pilot against installed task assets."""
import argparse
import ast
import json
import pickletools
import re
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--protocol', type=Path, help='defaults to the current medium protocol')
args = parser.parse_args()
WORK = args.root.resolve()
ROOT = WORK / 'RPent/.venv/lib/python3.11/site-packages/liberopro/liberopro'
p = json.loads((args.protocol or WORK / 'config/pilot_v7.json').read_text())
source = (ROOT / 'benchmark/libero_suite_task_map.py').read_text()
tree = ast.parse(source)
mapping = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'libero_task_map' for target in node.targets)))
assert 'libero_task_map[f"{_base}_{_perturb}"] = libero_task_map[_base]' in source
benchmark = (ROOT / 'benchmark/__init__.py').read_text()
btree = ast.parse(benchmark)
orders = ast.literal_eval(next(node.value for node in btree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'task_orders' for target in node.targets)))
assert orders[0] == list(range(10))
assert p['status'] == 'frozen' and p['scope'] == 'simulation'
assert type(p['dynamic_authorized']) is bool
profile = json.loads((WORK / p['runtime_config']['profile_reference']).read_text())
assert profile['planner']['model'] == p['runtime_config']['model'] == 'gpt-6.1-sol'
assert profile['planner']['reasoning_effort'] == 'medium'
assert p['runtime_config']['reasoning_effort'] == 'medium'
assert p['runtime_config']['research_llm'] == {'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium'}
assert len(p['tasks']) == 4 and len({t['task_key'] for t in p['tasks']}) == 4
for task in p['tasks']:
    suite = task['suite']
    base = suite[:-5] if suite.endswith('_swap') else suite
    assert mapping[base][task['task_id']] == task['task_name']
    assert task['task_order_index'] == 0
    bddl = ROOT / task['bddl_relative_path']
    init = ROOT / task['init_relative_path']
    assert bddl.is_file() and init.is_file()
    assert re.search(r'\(:language\s+([^)]+)\)', bddl.read_text()).group(1).strip() == task['language']
    with zipfile.ZipFile(init) as archive:
        blob = archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl')))
    ops = list(pickletools.genops(blob))
    shapes = [list((ops[i+2][1], ops[i+3][1])) for i, (op, arg, pos) in enumerate(ops)
              if op.name == 'MARK' and i + 4 < len(ops)
              and ops[i+1][1] == 1 and ops[i+2][0].name.startswith('BININT')
              and ops[i+3][0].name.startswith('BININT') and ops[i+4][0].name == 'TUPLE2']
    assert shapes == [task['init_shape_static']]
    assert shapes[0][0] == 50
    slots = []
    for split in ('development', 'release', 'sealed'):
        spec = p['task_sets'][split]
        assert spec['assignments_per_bundle'] == len(spec['task_keys']) * len(spec['seeds']) == 8
        assert spec['init_indices'] == [seed % shapes[0][0] for seed in spec['seeds']]
        slots.extend(spec['init_indices'])
    assert len(set(slots)) == len(slots) == 6
    print('PASS task map + BDDL + init metadata:', task['task_key'], suite, task['task_id'], shapes[0])
base = (ROOT / p['tasks'][0]['bddl_relative_path']).read_text()
swap = (ROOT / p['tasks'][1]['bddl_relative_path']).read_text()
assert base.replace('(On salad_dressing_1 floor_target_object_region)', '(On salad_dressing_1 floor_other_object_region_4)').replace('(On tomato_sauce_1 floor_other_object_region_4)', '(On tomato_sauce_1 floor_target_object_region)') == swap
assert abs(sum(m['weight'] for m in p['metrics'].values()) - 1) < 1e-12
assert all(m['min'] == 0 and m['max'] == 1 and m['direction'] == 'max' for m in p['metrics'].values())
assert not p['cost_weights']
assert p['min_gain'] == p['anchor_max_drop'] == 0.02
if p['dynamic_authorized']:
    authorization = json.loads((WORK / p['authorization_ref']).read_text())
    assert authorization['authorized'] is True
    assert p['budget']['authorized_now'] == dict(simulation_episodes=16, gpu_seconds=14400, llm_requests=320, training_steps=0, downloads_bytes=0, hardware_actions=0)
    assert p['task_sets']['sealed']['authorized'] is False
    assert p['runtime_config']['token_policy'] == profile['token_policy'] == 'observed_only'
else:
    assert p['budget']['authorized_now'] == dict.fromkeys(p['budget']['authorized_now'], 0)
assert p['budget']['first_release_cycle_proposal']['simulation_episodes_at_most'] == 16
assert p['next_step_proposal']['episodes'] == 1
assert type(p['next_step_proposal']['authorized']) is bool
assert not p['next_step_proposal']['authorized'] or p['dynamic_authorized']
assert p['next_step_proposal']['counts_as_formal_release_result'] is False
assert p['recovery']['reset_retries_per_assignment'] == 0
assert p['task_sets']['release']['benchmark_id'] == p['benchmark_id']
assert 'critical_progress' in p['hard_constraints']
print('PASS split identity, score normalization/weights, explicit authorization scope, recovery and cycle/next-step budgets')
print('PASS static swap BDDL difference is exactly two exchanged initial regions')
print('PASS JSON parses with stdlib; simulator/Torch/GPU/LLM/download not invoked')
