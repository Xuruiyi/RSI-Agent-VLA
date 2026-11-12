"""Prepare bound runtime jobs; execute only with explicit resource authorization."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .evaluation import load_protocol
from .storage import ResearchStore, atomic_json, identifier


DYNAMIC_LOCK = Path('/tmp/rsi-agent-vla-dynamic.lock')


@contextmanager
def dynamic_job_lock():
    """Runtime, research calls and training share the host's one-job queue."""
    with DYNAMIC_LOCK.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another RSI dynamic job holds the shared resource slot') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_reference(reference):
    path, _, fragment = reference.partition('#')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in fragment.strip('/').split('/') if fragment else []:
        value = value[key]
    return value


def prepare_runtime(root, artifacts, *, protocol_path, run_id, bundle_id, task_key,
                    seed, split, output_dir, authorization_path=None, cuda_device=0, deployment_binding=None):
    """No service imports, network, model loading or output creation."""
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    identifier(run_id)
    identifier(bundle_id)
    if type(cuda_device) is not int or cuda_device < 0:
        raise ValueError('one nonnegative physical CUDA device ID is required')
    store = ResearchStore(Path(artifacts) / 'research')
    bundle = store.get('bundles', bundle_id)
    protocol = load_protocol(protocol_path)
    if bundle['protocol_id'] != protocol['protocol_id']:
        raise ValueError('bundle/protocol identity mismatch')
    profile = resolve_reference(bundle['backend_ref'])
    for role in ('planner', 'research_llm'):
        if (profile[role]['model'], profile[role]['reasoning_effort']) != ('gpt-6.1-sol', 'medium'):
            raise ValueError(role + ' must explicitly use gpt-6.1-sol/medium')
    intended = protocol['runtime_config']
    if any(profile['planner'][key] != intended[key] for key in ('model', 'reasoning_effort')):
        raise ValueError('profile/protocol runtime configuration mismatch')
    tasks = {task['task_key']: task for task in protocol['tasks']}
    task = tasks[task_key]
    if split == 'smoke':
        proposal = protocol['next_step_proposal']
        if task_key != proposal['task'] or seed not in proposal['seeds']:
            raise ValueError('smoke must match the registered single-episode proposal')
    elif split in ('development', 'release'):
        matrix = protocol['task_sets'][split]
        if task_key not in matrix['task_keys'] or seed not in matrix['seeds']:
            raise ValueError('task/seed outside frozen assignment matrix')
    elif split == 'deployment':
        if not deployment_binding or deployment_binding['bundle_id'] != bundle_id:
            raise ValueError('deployment requires a trusted effective-version binding')
        registered = [protocol['task_sets'][name] for name in ('development', 'release')]
        if not any(task_key in row['task_keys'] and seed in row['seeds'] for row in registered):
            raise ValueError('deployment outside registered nonsealed task scope')
    else:
        raise ValueError('sealed results cannot enter this development/release launcher')
    budgets = protocol['budget']['per_attempt_proposal']
    if profile['control']['max_episode_steps'] != budgets['max_episode_steps']:
        raise ValueError('effective profile must already contain the common pilot step bound')
    model = resolve_reference(bundle['theta_ref'])
    processor = resolve_reference(bundle['processor_ref']) if '#' in bundle['processor_ref'] else bundle['processor_ref']
    if model.get('kind') == 'pi05_libero_native_full_weights':
        from .adapters.pi05_training import ACTION_CONTRACT
        if model['action_contract'] != ACTION_CONTRACT or model['config_name'] != 'pi05_libero':
            raise ValueError('trained model descriptor does not match the actual runtime contract')
        checkpoint = Path(model['model_path'])
    else:
        checkpoint = Path(model['checkpoint'])
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
    normalization = Path(processor)
    normalization = normalization if normalization.is_absolute() else checkpoint / normalization
    memory_record = store.get('memory', bundle['memory_ref'])
    memory = memory_record.get('upstream')
    if not memory or not memory_record.get('legal_information'):
        raise ValueError('runtime needs an explicit legal upstream memory snapshot')
    command = [str(root / 'RPent/.venv/bin/python'), '-m', 'rpent.cli.main',
               '--robot', 'libero', '--libero-type', profile['control']['libero_type'],
               '--planner', profile['planner']['backend'], '--model', profile['planner']['model'],
               '--reasoning-effort', profile['planner']['reasoning_effort'], '--am-backend', 'pi05',
               '--cuda-device', str(cuda_device), '--no-auto-merge-memory',
               '--memory-profile', memory['profile'], '--memory-version', memory['version'],
               '--memory-repo', memory['repo'] + '@' + memory['revision'],
               '--max-episode-steps', str(budgets['max_episode_steps']),
               '--max-turns', str(budgets['max_turns']),
               '--planner-timeout-s', str(budgets['planner_timeout_seconds']),
               '--suite', task['suite'], '--task', str(task['task_id']), '--seed', str(seed),
               '--output-dir', str(output)]
    context = dict(schema_version='1.0', run_id=run_id, episode_id=f'{task_key}-s{seed}',
                   bundle_id=bundle_id, protocol_id=protocol['protocol_id'],
                   profile_ref=bundle['backend_ref'], parent_decision_id=None,
                   theta_ref=bundle['theta_ref'], processor_ref=bundle['processor_ref'],
                   memory_ref=bundle['memory_ref'], memory=memory,
                   output_dir=str(output), planner=profile['planner'], gpu_count=1,
                   dynamic_authorized=False, authorization_ref=None,
                   gpu_devices=[cuda_device],
                   runtime_rules_ref=None,
                   budget_stop_ref=str(output / 'budget_stop.json'),
                   code_snapshot_ref=str(output / 'code_snapshot/manifest.json'),
                   limits=dict(wall_seconds=budgets['full_job_wall_seconds'],
                               planner_seconds=budgets['planner_timeout_seconds'],
                               tool_calls=budgets['tool_calls'], environment_steps=budgets['max_episode_steps'],
                               input_tokens=budgets['llm_input_tokens_including_cached'],
                               output_tokens=budgets['llm_output_tokens'], storage_bytes=budgets['artifact_bytes'],
                               gpu_seconds=budgets['full_job_wall_seconds']))
    if profile.get('token_policy') == 'observed_only':
        context['token_policy'] = 'observed_only'
        context['limits'].pop('input_tokens')
        context['limits'].pop('output_tokens')
        context['limits']['model_requests'] = budgets['model_requests']
        context['token_reference_only'] = {'input': budgets['llm_input_tokens_including_cached'],
                                         'output': budgets['llm_output_tokens']}
    if '#' not in bundle['phi_ref']:
        # Candidate rules remain frozen artifacts, consumed by the concrete toolkit.
        from .rule_updates import load_runtime_rules
        rules = load_runtime_rules(bundle['phi_ref'])
        if rules['baseline_chunk_steps'] != profile['control']['action_chunk']:
            raise ValueError('runtime rules baseline cap differs from the actual model chunk')
        expected_prefix = profile['control'].get('executed_prefix_steps', profile['control']['action_chunk'])
        if rules['chunk_steps'] != expected_prefix:
            raise ValueError('changed action timing requires a new matched protocol/profile baseline')
        context['runtime_rules_ref'] = str(Path(bundle['phi_ref']).resolve())
    task_input = profile.get('task_inputs', {}).get(task_key, profile.get('task_input'))
    if task_input:
        if task_input['task_scope'] != {'suite':task['suite'], 'task':task['task_id']}:
            raise ValueError('startup input not defined for selected task')
        context['task_input'] = task_input
    for key in ('execution_version', 'planner_observation'):
        if key in profile:
            context[key] = profile[key]
    context['task'] = {'suite': task['suite'], 'task': task['task_id'], 'seed': seed}
    context['code_ref'] = bundle['code_ref']
    context['information_scope'] = split
    if deployment_binding:
        context['release_binding'] = deployment_binding
    blockers = []
    if authorization_path is None:
        blockers.append('explicit single-job GPU/LLM/simulation authorization missing')
    else:
        authorization = json.loads(Path(authorization_path).read_text(encoding='utf-8'))
        for key, expected in dict(run_id=run_id, bundle_id=bundle_id, protocol_id=protocol['protocol_id'],
                                  task_key=task_key, seed=seed, simulation_episodes=1,
                                  gpu_devices=[cuda_device]).items():
            if authorization.get(key) != expected:
                raise ValueError('authorization identity/scope mismatch: ' + key)
        if authorization.get('authorized') is not True:
            raise ValueError('authorization is not affirmative')
        for key, value in context['limits'].items():
            granted = authorization.get('limits', {}).get(key, -1)
            if type(granted) not in (int, float) or granted <= 0:
                raise ValueError('authorization does not cover proposed bound: ' + key)
            context['limits'][key] = min(value, granted)
        context['limits']['planner_seconds'] = min(context['limits']['planner_seconds'], context['limits']['wall_seconds'])
        command[command.index('--planner-timeout-s')+1] = str(int(context['limits']['planner_seconds']))
        command[command.index('--max-episode-steps')+1] = str(int(context['limits']['environment_steps']))
        context.update(dynamic_authorized=True, authorization_ref=str(Path(authorization_path).resolve()))
    if profile['dynamic_authorized'] is not True or protocol.get('dynamic_authorized') is not True:
        blockers.append('authoritative profile/protocol dynamic authorization remains false')
    if profile['planner']['backend'] == 'codex' and profile.get('token_policy') != 'observed_only':
        blockers.append('Codex SDK has no verified per-request input/output token admission bound')
    environment=dict(PI05_CHECKPOINT_PATH=str(checkpoint), PI05_NORM_STATS_PATH=str(normalization),
                     RPENT_REPO_ROOT=str(root/'RPent'),RPENT_RLINF_ROOT='',RLINF_REPO_PATH='',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(cuda_device))
    auxiliary=profile.get('auxiliary_models')
    if auxiliary:
        environment['SAM3_CHECKPOINT_PATH']=auxiliary['sam3_checkpoint']
        environment['OPENPI_DATA_HOME']=auxiliary['openpi_data_home']
        required={'SAM3':Path(environment['SAM3_CHECKPOINT_PATH']),
                  'tokenizer':Path(environment['OPENPI_DATA_HOME'])/'big_vision/paligemma_tokenizer.model',
                  'normalization':normalization/'norm_stats.json' if normalization.is_dir() else normalization}
        for name,path in required.items():
            if not path.is_file():blockers.append('required runtime asset unavailable before launch: '+name)
    return dict(context=context, command=command, cwd=str(root / 'RPent'), split=split,
                benchmark_id=protocol['benchmark_id'] if split == 'release' else None,
                counts_as_release=split == 'release', blockers=blockers,
                environment=environment,
                model_configuration=dict(requested=profile['planner'], effective=None,
                                         theta_ref=bundle['theta_ref'], processor_ref=bundle['processor_ref'],
                                         historical_low_is_medium_baseline=False))


def execute_runtime(plan):
    if plan['blockers']:
        raise RuntimeError('; '.join(plan['blockers']))
    from .adapters.runtime_supervisor import supervise
    output = Path(plan['context']['output_dir'])
    with dynamic_job_lock():
        started = time.monotonic()
        plan['context']['job_started_monotonic'] = started
        output.mkdir(parents=True, exist_ok=False)
        root=Path(plan['cwd']).parent
        files=('core/src/rsi_vla/evaluator.py','core/src/rsi_vla/runtime.py','core/src/rsi_vla/adapters/request_gate.py',
               'core/src/rsi_vla/adapters/runtime_supervisor.py','core/src/rsi_vla/adapters/runtime_acceptance.py',
               'core/src/rsi_vla/adapters/rpent.py','RPent/rpent/session/runtime.py',
               'RPent/rpent/planner/codex.py','RPent/robots/libero/result.py',
               'RPent/robots/libero/task_input.py','RPent/robots/libero/robot_spec.py','RPent/rpent/cli/main.py',
               'core/src/rsi_vla/adapters/runtime_closure.py',
               'RPent/robots/libero/runtime_evidence.py','RPent/robots/libero/toolkit.py',
               'RPent/robots/libero/env_client.py','RPent/robots/libero/env_server.py',
               'RPent/robots/libero/tools.py','RPent/rpent/robots/components/pi05_vla_server.py')
        for name in files:
            target=output/'code_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/name,target)
        atomic_json(output/'code_snapshot/manifest.json',dict(
            identity={key:plan['context'][key] for key in ('run_id','episode_id','bundle_id','protocol_id')},
            source_root=str(root),files=list(files),
            scope='actual uncommitted runtime sources captured before launch; installed dependencies bound by profile'))
        atomic_json(output / 'launch_plan.json', plan)
        context_path = output / 'runtime_context.json'
        atomic_json(context_path, plan['context'])
        previous_env = {key:os.environ.get(key) for key in (*plan['environment'], 'RSI_RUNTIME_CONTEXT', 'RSI_REQUEST_GATE_URL')}
        os.environ.update(plan['environment'])
        os.environ['RSI_RUNTIME_CONTEXT'] = str(context_path)
        previous = Path.cwd()
        try:
            if plan['context'].get('task_input'):
                from robots.libero.task_input import build_task_input
                memory=plan['context']['memory']
                matches=list((root/'RPent/memory/libero/.versions').glob(
                    '*/snapshots/'+memory['revision']+'/'+memory['version']))
                if len(matches)!=1:
                    raise ValueError('one frozen local memory snapshot required')
                task_input=build_task_input(plan['context'],matches[0],plan['context']['task'])
                atomic_json(output/'task_input.json',task_input)
            os.chdir(plan['cwd'])
            if plan['context'].get('token_policy') == 'observed_only':
                from .adapters.request_gate import RequestGate
                with RequestGate(plan['context']) as gate:
                    os.environ['RSI_REQUEST_GATE_URL'] = gate.url
                    result = supervise(plan['command'], context_path=context_path)
            else:
                result = supervise(plan['command'], context_path=context_path)
            return result
        except BaseException as exc:
            atomic_json(output/'launcher_error.json', {'identity': {
                key:plan['context'][key] for key in ('run_id','episode_id','bundle_id','protocol_id')},
                'error_type':type(exc).__name__})
            raise
        finally:
            wall = time.monotonic() - started
            atomic_json(output / 'launcher_costs.json', {
                'identity': {key: plan['context'][key] for key in ('run_id','episode_id','bundle_id','protocol_id')},
                'whole_job_wall_seconds': wall,
                'gpu_device_seconds_envelope': wall * plan['context']['gpu_count'],
                'wall_limit': plan['context']['limits']['wall_seconds'],
                'training_seconds': 0, 'training_source': 'runtime-only command; no training worker',
                'manual_seconds': 0, 'manual_source': 'noninteractive scheduled attempt; intervention recorded as interruption',
                'artifact_bytes_before_final_report': sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink()),
            })
            from .adapters.runtime_closure import finalize_runtime
            try:
                finalize_runtime(plan['context'])
            finally:
                for key,value in previous_env.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
                os.chdir(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--protocol', type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--bundle', help='explicit experiment/evaluation/reproduction bundle')
    selection.add_argument('--deployment-release', type=Path, help='trusted release root; read at task boundary')
    parser.add_argument('--run', required=True)
    parser.add_argument('--task-key', default='position_swap_salad')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=('smoke', 'development', 'release', 'deployment'), default='smoke')
    parser.add_argument('--output', required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--cuda-device', type=int, default=0, help='physical device ID; only this device is exposed')
    parser.add_argument('--execute', action='store_true', help='requires affirmative authorization and supported budgets')
    args = parser.parse_args(argv)
    protocol_path = args.protocol or args.root / 'config/pilot_v7.json'
    binding = None
    if args.deployment_release:
        if args.split not in {'smoke', 'deployment'}:
            raise ValueError('experiments/evaluation require an explicit bundle')
        args.split = 'deployment'
        from .release import ReleaseManager
        from .evaluation import EvaluationStore
        protocol = load_protocol(protocol_path)
        manager = ReleaseManager(args.deployment_release, ResearchStore(Path(args.artifacts) / 'research'),
                                 EvaluationStore(Path(args.artifacts) / 'evaluation'))
        selected, binding = manager.deployment_bundle(scope=protocol['scope'], safe_point='task_boundary')
        bundle_id = selected['bundle_id']
    else:
        bundle_id = args.bundle or load_protocol(protocol_path)['anchor_bundle_id']
    plan = prepare_runtime(args.root, args.artifacts, protocol_path=protocol_path,
                           run_id=args.run, bundle_id=bundle_id, task_key=args.task_key, seed=args.seed,
                           split=args.split, output_dir=args.output, authorization_path=args.authorization,
                           cuda_device=args.cuda_device, deployment_binding=binding)
    plan['selection'] = {'mode': 'deployment' if binding else 'explicit_experiment', 'release_binding': binding}
    plan['context']['selection'] = plan['selection']
    if args.execute:
        result = execute_runtime(plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] == 'completed' and result['exit_code'] == 0 else 1
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
