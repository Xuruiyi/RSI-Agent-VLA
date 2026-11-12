#!/usr/bin/env python3
"""Run the frozen medium baseline serially, stopping on technical evidence gaps."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core/src'))
from rsi_vla.runtime import prepare_runtime,execute_runtime
from rsi_vla.adapters.runtime_acceptance import inspect_attempt
from rsi_vla.storage import atomic_json


def schedule(protocol):
    return [dict(split=split,task_key=task,seed=seed,status='not_started',outcome='unknown')
        for split in ('development','release') for task in protocol['task_sets'][split]['task_keys']
        for seed in protocol['task_sets'][split]['seeds']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--cuda-device',type=int,default=1)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--prior-campaign',type=Path,help='retain and debit a stopped technical cohort; never replace its slots')
    args=parser.parse_args()
    p=json.loads(args.protocol.read_text());rows=schedule(p)
    if not args.execute:
        print(json.dumps(dict(assignments=rows,jobs_started=0),indent=2));return 0
    authority=args.root/'artifacts/authorized-20261110/authorization.json'
    if json.loads(authority.read_text()).get('authorized') is not True:
        raise PermissionError('phase authorization missing')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();used_gpu=used_requests=used_tools=used_steps=0;prior_wall=0
    if args.prior_campaign:
        prior=json.loads(args.prior_campaign.read_text())
        prior_wall=prior['wall_seconds'];used_gpu=prior['gpu_device_seconds_envelope'];used_requests=prior['model_requests']
        for previous in prior['assignments']:
            if previous.get('output_dir'):
                native=Path(previous['output_dir'])/'environment_service/runtime_costs.json'
                if native.exists():used_steps+=json.loads(native.read_text())['environment_steps_confirmed']
    report=dict(protocol_id=p['protocol_id'],bundle_id=args.bundle,assignments=rows,
                wall_limit=14400,gpu_seconds_limit=14400,model_requests_limit=320,
                sealed_scheduled=False,stop_reason=None,complete=False)
    report['prior_technical_campaign']=str(args.prior_campaign) if args.prior_campaign else None
    for index,row in enumerate(rows):
        remaining=14400-prior_wall-(time.monotonic()-started)
        if remaining<=0 or used_gpu>=14400 or used_requests>=320:
            report['stop_reason']='phase budget exhausted';break
        run=f"baseline-{p['protocol_id']}-{row['split']}-{row['task_key']}-s{row['seed']}"
        output=args.output/run
        plan=prepare_runtime(args.root,args.artifacts,protocol_path=args.protocol,run_id=run,
            bundle_id=args.bundle,task_key=row['task_key'],seed=row['seed'],split=row['split'],
            output_dir=output,cuda_device=args.cuda_device)
        limits=dict(plan['context']['limits'])
        limits.update(wall_seconds=min(900,remaining),gpu_seconds=min(900,14400-used_gpu),
                      model_requests=min(20,320-used_requests),tool_calls=min(p['budget']['per_attempt_proposal']['tool_calls'],640-used_tools),
                      environment_steps=min(2000,32000-used_steps))
        permit={k:plan['context'][k] for k in ('run_id','bundle_id','protocol_id')}
        permit.update(authorized=True,task_key=row['task_key'],seed=row['seed'],simulation_episodes=1,
                      gpu_devices=[args.cuda_device],limits=limits,authorization_ref=str(authority))
        permit_path=args.output/(run+'-permit.json');atomic_json(permit_path,permit)
        plan=prepare_runtime(args.root,args.artifacts,protocol_path=args.protocol,run_id=run,
            bundle_id=args.bundle,task_key=row['task_key'],seed=row['seed'],split=row['split'],
            output_dir=output,cuda_device=args.cuda_device,authorization_path=permit_path)
        row.update(status='running',run_id=run,output_dir=str(output));atomic_json(args.output/'campaign.json',report)
        execute_runtime(plan)
        audit=inspect_attempt(output,require_vla=index==0)
        atomic_json(output/'independent-audit.json',audit)
        row.update(status='completed' if audit['passed'] else 'technical_gap',outcome=audit['outcome'],
                   audit_ref=str(output/'independent-audit.json'))
        c=audit['costs']
        if not c:
            report['costs_incomplete']=True
            for filename,key,target in [('launcher_costs.json','gpu_device_seconds_envelope','gpu_device_seconds_envelope'),
                                        ('request_gate.json','requests_forwarded','model_requests')]:
                source=output/filename
                if source.exists():c[target]=json.loads(source.read_text()).get(key,0)
        used_gpu+=c.get('gpu_device_seconds_envelope',0)
        used_requests+=c.get('model_requests',0);used_tools+=c.get('tool_calls',0);used_steps+=c.get('environment_steps',0)
        print(json.dumps(dict(slot=index+1,task=row['task_key'],seed=row['seed'],split=row['split'],
              outcome=row['outcome'],infrastructure_passed=audit['passed'])),flush=True)
        if not audit['passed']:
            report['stop_reason']='required identity/outcome/cost/budget evidence gap; no automatic retry'
            atomic_json(args.output/'campaign.json',report);break
        if index==0:
            atomic_json(args.output/'checkpoint1.json',dict(passed=True,run_id=run,
                audit_ref=row['audit_ref'],scope='full-task Runtime infrastructure, not baseline matrix acceptance'))
        atomic_json(args.output/'campaign.json',report)
    report.update(complete=all(row['status']=='completed' for row in rows),wall_seconds=prior_wall+time.monotonic()-started,
                  gpu_device_seconds_envelope=used_gpu,model_requests=used_requests)
    atomic_json(args.output/'campaign.json',report)
    return 0 if report['complete'] else 1


if __name__=='__main__':
    sys.exit(main())
