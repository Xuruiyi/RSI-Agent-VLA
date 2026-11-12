#!/usr/bin/env python3
"""One supervised medium research request, limited to legal development evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'core/src'))


def child(output):
    import openai_codex
    from rpent.planner.codex import ReasoningEffort, build_codex_config
    prepared=json.loads((output/'request.json').read_text())
    schema=json.loads((output/'response-schema.json').read_text())
    prompt=('Review only the supplied development evidence. Propose one bounded finish-reserve hint or defer. '
        'No tools are available. Do not inspect release or sealed data. The native finish guard lacks its '
        'required mismatch evidence. A hint can only advise retaining the last admitted tool call for honest '
        'finish; it does not change tool admission, task success, control timing, weights or the 40-call cap. '
        'The stop-propagation implementation is being repaired equally for both bundles; do not call that '
        'repair a candidate gain. Missing old-capability/release anchors imply an inconclusive formal Gate. '
        'Return the requested structured proposal, including competing explanations; acceptance is not required.\n'
        +json.dumps(prepared,ensure_ascii=False))
    config=build_codex_config(mcp_url=None,base_url=None,api_key=None,cwd=str(Path(__file__).resolve().parents[1]/'RPent'))
    with openai_codex.Codex(config=config) as client:
        thread=client.thread_start(model='gpt-6.1-sol',approval_mode=openai_codex.ApprovalMode.deny_all,
                                   sandbox=openai_codex.Sandbox.read_only)
        result=thread.turn(prompt,model='gpt-6.1-sol',effort=ReasoningEffort.medium,output_schema=schema).run()
    (output/'sdk-result.json').write_text(json.dumps({'status':getattr(result.status,'value',str(result.status)),
        'final_response':result.final_response},ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',type=Path)
    parser.add_argument('--artifacts',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--result-id',default='baseline-v6-partial-development')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--reconcile',action='store_true',help='validate existing transport and SDK output; never call a model')
    args=parser.parse_args()
    if args.child:
        child(args.child);return 0
    from rsi_vla.storage import ResearchStore,atomic_json
    from rsi_vla.research import ResearchJob
    from rsi_vla.research_tools import ResearchTools,ResearchLLMAdapter
    from rsi_vla.runtime import dynamic_job_lock
    from rsi_vla.adapters.request_gate import RequestGate
    from rsi_vla.adapters.runtime_supervisor import supervise
    store=ResearchStore(args.artifacts/'research')
    protocol=json.loads(args.protocol.read_text())
    job=ResearchJob(store,'research-v6-finish-reserve',bundle_id='initial-medium-v6',limits={
        'whole_job_wall_seconds':1200,'model_requests':21,'tool_calls':48,
        'environment_steps':2000,'gpu_device_seconds_envelope':900,'artifact_bytes':1084227584})
    tools=ResearchTools(store,job,protocol,domain_policy={'domain':'libero_tabletop',
        'allowed_goal_kinds':[],'policy_ref':str(args.protocol.resolve())})
    adapter=ResearchLLMAdapter(model='gpt-6.1-sol',reasoning_effort='medium')
    request_id='research-v6-finish-reserve-001'
    request=adapter.prepare_request(tools,request_id=request_id,result_ids=[args.result_id],
                                    max_input_tokens=64000,max_output_tokens=4000)
    request['finish_budget_observations']=tools.inspect_finish_budget_opportunity([args.result_id])
    request['native_finish_observations']=tools.inspect_native_finish_opportunity([args.result_id])
    if not args.execute and not args.reconcile:
        print(json.dumps(request,ensure_ascii=False,indent=2));return 0
    output=args.output.resolve()
    if not args.reconcile:output.mkdir(parents=True,exist_ok=False)
    authority=Path(__file__).resolve().parents[1]/'artifacts/authorized-20261110/authorization.json'
    if json.loads(authority.read_text()).get('authorized') is not True:
        raise PermissionError('research authorization absent')
    if not args.reconcile:
        atomic_json(output/'request.json',request)
        atomic_json(output/'response-schema.json',adapter.response_schema(request_id))
    context={'run_id':output.name,'episode_id':'research-only','bundle_id':tools.base.bundle_id,
        'protocol_id':protocol['protocol_id'],'output_dir':str(output),'gpu_count':0,
        'dynamic_authorized':True,'authorization_ref':str(authority),
        'planner':{'model':'gpt-6.1-sol','reasoning_effort':'medium'},
        'limits':{'model_requests':1,'wall_seconds':295,'storage_bytes':10485760}}
    if args.reconcile:
        if json.loads((output/'runtime_context.json').read_text())!=context:
            raise ValueError('existing research context mismatch')
        supervision=json.loads((output/'supervisor.json').read_text())
        wall=json.loads((output/'execution-costs.json').read_text())['costs']['whole_job_wall_seconds']
    else:
        atomic_json(output/'runtime_context.json',context)
        with dynamic_job_lock():
            started=time.monotonic()
            with RequestGate(context) as gate:
                os.environ['RSI_REQUEST_GATE_URL']=gate.url
                try:
                    supervision=supervise([sys.executable,str(Path(__file__).resolve()),'--child',str(output)],
                                          context_path=output/'runtime_context.json')
                finally:os.environ.pop('RSI_REQUEST_GATE_URL',None)
            wall=time.monotonic()-started
    events=[json.loads(s) for s in (output/'model_requests.jsonl').read_text().splitlines()]
    admitted=[e for e in events if e['kind']=='request_admitted']
    terminals=[e for e in events if e['kind']=='response_terminal']
    costs={'whole_job_wall_seconds':wall,'model_requests':len(admitted),'tool_calls':0,
        'input_tokens':sum((e.get('usage') or {}).get('input_tokens',0) for e in terminals),
        'output_tokens':sum((e.get('usage') or {}).get('output_tokens',0) for e in terminals),
        'cached_input_tokens':sum(((e.get('usage') or {}).get('input_tokens_details') or {}).get('cached_tokens',0) for e in terminals)}
    complete_usage=len(terminals)==len(admitted) and all(
        isinstance(e.get('usage'),dict) and type(e['usage'].get('input_tokens')) is int
        and type(e['usage'].get('output_tokens')) is int for e in terminals)
    if not complete_usage:
        for key in ('input_tokens','output_tokens','cached_input_tokens'):costs.pop(key)
    atomic_json(output/'execution-costs.json',{'costs':costs,'complete_usage':complete_usage})
    if admitted:
        adapter.record_execution(tools,request_id,costs,{'effective_model':admitted[0]['model'],
            'effective_reasoning_effort':admitted[0]['reasoning_effort'],
            'evidence_ref':str(output/'model_requests.jsonl')})
    else:
        job.consume({'whole_job_wall_seconds':wall,'model_requests':0},'failed LLM startup '+request_id)
    if (supervision['status']!='completed' or supervision['exit_code']!=0 or
        not supervision['cleanup_confirmed'] or wall>300 or len(admitted)!=1 or len(terminals)!=1 or
        terminals[0].get('backend_model')!='gpt-6.1-sol' or not complete_usage):
        raise RuntimeError('research transport did not complete within its registered bounds; evidence retained')
    sdk=json.loads((output/'sdk-result.json').read_text())
    if sdk['status'].rsplit('.',1)[-1]!='completed':raise RuntimeError('research SDK status is not completed')
    response=json.loads(sdk['final_response'])
    adapter.validate_response(tools,response)
    atomic_json(output/'validated-response.json',response)
    print(json.dumps({'decision':response['decision'],'costs':costs},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
