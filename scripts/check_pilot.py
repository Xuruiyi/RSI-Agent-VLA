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
assert p['runtime_config']['research_llm']['model'] == profile['research_llm']['model'] == 'gpt-6.1-sol'
assert p['runtime_config']['research_llm']['reasoning_effort'] == profile['research_llm']['reasoning_effort']
assert p['runtime_config']['research_llm']['reasoning_effort'] in {'medium', 'high'}
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
        assert spec['task_keys'] and set(spec['task_keys']) <= {t['task_key'] for t in p['tasks']}
        assert len(set(spec['task_keys'])) == len(spec['task_keys'])
        assert len(set(spec['seeds'])) == len(spec['seeds'])
        assert spec['assignments_per_bundle'] == len(spec['task_keys']) * len(spec['seeds'])
        assert spec['init_indices'] == [seed % shapes[0][0] for seed in spec['seeds']]
        if task['task_key'] in spec['task_keys']:
            assert len(set(spec['init_indices'])) == len(spec['seeds'])
            slots.extend(spec['init_indices'])
    assert len(set(slots)) == len(slots)
    print('PASS task map + BDDL + init metadata:', task['task_key'], suite, task['task_id'], shapes[0])
base = (ROOT / p['tasks'][0]['bddl_relative_path']).read_text()
swap = (ROOT / p['tasks'][1]['bddl_relative_path']).read_text()
assert base.replace('(On salad_dressing_1 floor_target_object_region)', '(On salad_dressing_1 floor_other_object_region_4)').replace('(On tomato_sauce_1 floor_other_object_region_4)', '(On tomato_sauce_1 floor_target_object_region)') == swap
assert abs(sum(m['weight'] for m in p['metrics'].values()) - 1) < 1e-12
assert all(m['min'] == 0 and m['max'] == 1 and m['direction'] == 'max' for m in p['metrics'].values())
assert not p['cost_weights']
assert p['min_gain'] == p['anchor_max_drop'] == 0.02
release08_scope = 'repair05-fresh-release08-eight-slots-two-recoveries-only'
if (p['protocol_id'] == 'rsi-libero-pro-5a-release08-four-seed3'
        or p['anchor_bundle_id'] == 'initial-medium-5a-release08'
        or profile['profile_id'] == 'matched-5a-release08-20261113'
        or p.get('comparison_only') is True):
    assert p['dynamic_authorized'] is True
    assert p.get('authorization_scope') == release08_scope
if (p['protocol_id'] == 'rsi-libero-pro-5a-release07-four-seed3'
        or p['anchor_bundle_id'] == 'initial-medium-5a-release07'
        or profile['profile_id'] == 'matched-5a-release07-20261113'):
    assert p['dynamic_authorized'] is True
    assert p.get('authorization_scope') == 'repair05-fresh-release07-eight-slots-only'
if p['dynamic_authorized']:
    authorization = json.loads((WORK / p['authorization_ref']).read_text())
    assert authorization['authorized'] is True
    if p.get('authorization_scope') == 'bounded-5a-research-development-only':
        assert authorization['scope'] == p['authorization_scope']
        assert authorization['max_research_dispatches'] in (0, 1)
        assert authorization['protocol_budget'] == p['budget']['authorized_now'] == dict(simulation_episodes=1, gpu_seconds=900, llm_requests=20 + authorization['max_research_dispatches'], training_steps=0, downloads_bytes=0, hardware_actions=0)
        assert authorization['allowed_operators'] == ['planner_segment_handoff_hint']
        assert authorization['release_authorized'] is False
        assert p['release_authorized'] is profile['release_authorized'] is False
        assert p['allowed_runtime_splits'] == ['development']
    elif p.get('authorization_scope') == release08_scope:
        assert authorization['scope'] == p['authorization_scope']
        assert p['protocol_id'] == authorization['protocol_id'] == 'rsi-libero-pro-5a-release08-four-seed3'
        assert p['anchor_bundle_id'] == 'initial-medium-5a-release08'
        assert profile['profile_id'] == 'matched-5a-release08-20261113'
        assert authorization['max_research_dispatches'] == 0
        assert authorization['formal_assignments'] == 8
        assert authorization['maximum_starts'] == 10
        assert authorization['max_technical_recoveries'] == 2
        assert authorization['max_recoveries_per_slot'] == 1
        assert authorization['automatic_retries'] == 0
        assert p['comparison_only'] is authorization['comparison_only'] is True
        assert p['deployment_authorized'] is authorization['deployment_authorized'] is False
        assert p['qualification_establishment_authorized'] is authorization['qualification_establishment_authorized'] is False
        assert authorization['inheritance_authorized'] is False
        assert authorization['release_authorized'] is p['release_authorized'] is profile['release_authorized'] is True
        assert p['allowed_runtime_splits'] == ['release']
        assert p['task_sets']['development']['authorized'] is False
        assert p['task_sets']['release']['authorized'] is True
        assert p['task_sets']['release']['seeds'] == [3]
        assert p['task_sets']['release']['task_keys'] == ['base_salad', 'position_swap_salad', 'old_milk', 'old_drawer']
        assert p['development_derivation']['source_candidate_id'] == 'development-5a-repair05-experiment-finish-audit-v4-candidate'
        assert p['development_derivation']['development_results_relabelled'] is False
        assert authorization['protocol_budget'] == p['budget']['authorized_now'] == dict(simulation_episodes=10, gpu_seconds=9000, llm_requests=200, training_steps=0, downloads_bytes=0, hardware_actions=0)
        slot = dict(model_requests=20, tool_calls=60, simulation_episodes=1, environment_steps=2000, wall_seconds=900, gpu_seconds=900, storage_bytes=1073741824)
        assert authorization['per_slot_envelope'] == authorization['inheritance_reserve'] == slot
        assert authorization['total_dynamic_envelope'] == {k: v * 10 for k, v in slot.items()}
        assert authorization['offline_reserve'] == dict(wall_seconds=600, storage_bytes=268435456)
        assert authorization['admission_envelope'] == dict(model_requests=220, tool_calls=660, simulation_episodes=11, environment_steps=22000, wall_seconds=10500, gpu_seconds=9900, storage_bytes=12079595520)
    elif p.get('authorization_scope') in {'repair05-first-release-eight-slots-only', 'repair05-fresh-release06-eight-slots-only', 'repair05-fresh-release07-eight-slots-only'}:
        assert authorization['scope'] == p['authorization_scope']
        assert authorization['max_research_dispatches'] == 0
        assert authorization['maximum_starts'] == 8 and authorization['automatic_retries'] == 0
        assert authorization['release_authorized'] is p['release_authorized'] is profile['release_authorized'] is True
        assert p['allowed_runtime_splits'] == ['release']
        assert p['task_sets']['development']['authorized'] is False
        assert p['task_sets']['release']['authorized'] is True
        assert p['task_sets']['release']['seeds'] == [3]
        assert p['task_sets']['release']['task_keys'] == ['base_salad', 'position_swap_salad', 'old_milk', 'old_drawer']
        assert p['development_derivation']['source_candidate_id']
        assert p['development_derivation']['development_results_relabelled'] is False
        if p['authorization_scope'] == 'repair05-fresh-release07-eight-slots-only':
            assert p['protocol_id'] == authorization['protocol_id'] == 'rsi-libero-pro-5a-release07-four-seed3'
            assert p['anchor_bundle_id'] == 'initial-medium-5a-release07'
            assert profile['profile_id'] == 'matched-5a-release07-20261113'
            assert p['development_derivation']['source_candidate_id'] == 'development-5a-repair05-experiment-finish-audit-v4-candidate'
        assert authorization['protocol_budget'] == p['budget']['authorized_now'] == dict(simulation_episodes=8, gpu_seconds=7200, llm_requests=160, training_steps=0, downloads_bytes=0, hardware_actions=0)
    else:
        assert p['budget']['authorized_now'] == dict(simulation_episodes=16, gpu_seconds=14400, llm_requests=320, training_steps=0, downloads_bytes=0, hardware_actions=0)
    assert p['task_sets']['sealed']['authorized'] is False
    assert p['runtime_config']['token_policy'] == profile['token_policy'] == 'observed_only'
else:
    assert p['budget']['authorized_now'] == dict.fromkeys(p['budget']['authorized_now'], 0)
release_slots = p['task_sets']['release']['assignments_per_bundle']
assert p['budget']['first_release_cycle_proposal']['assignments_per_bundle'] == release_slots
assert p['budget']['first_release_cycle_proposal']['simulation_episodes_at_most'] == 2 * release_slots
assert p['next_step_proposal']['episodes'] == (0 if p.get('authorization_scope') in {'repair05-first-release-eight-slots-only', 'repair05-fresh-release06-eight-slots-only', 'repair05-fresh-release07-eight-slots-only', release08_scope} else 1)
assert type(p['next_step_proposal']['authorized']) is bool
assert not p['next_step_proposal']['authorized'] or p['dynamic_authorized']
assert p['next_step_proposal']['counts_as_formal_release_result'] is False
assert p['recovery']['reset_retries_per_assignment'] == 0
assert p['task_sets']['release']['benchmark_id'] == p['benchmark_id']
assert 'critical_progress' in p['hard_constraints']
print('PASS split identity, score normalization/weights, explicit authorization scope, recovery and cycle/next-step budgets')
print('PASS static swap BDDL difference is exactly two exchanged initial regions')
print('PASS JSON parses with stdlib; simulator/Torch/GPU/LLM/download not invoked')
