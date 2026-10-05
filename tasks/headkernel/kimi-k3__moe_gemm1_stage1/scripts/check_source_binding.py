"""Short non-scoreable source-binding probes; no performance samples are collected."""
import argparse
import json
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import canonical,strict_json,validate_manifest
from fresh_runner import cpu_copy
from runtime_adapter import Prepared,create_runtime,file_sha
from source_guard import validate_sources


class ReferenceCalibrationError(RuntimeError):
    pass


def prepare_for_probe(runtime,case):
    """Compile invalid candidates without treating their warmup outputs as a pass.

    The frozen reference's captured-golden check still runs. Candidate warmup
    failures are recorded, then the chosen eager/graph probe runs the unchanged
    oracle. This diagnostic path never writes a task-result/performance report.
    """
    original=Prepared.compare;observations=[]
    def deferred(self,actual,expected):
        index=len(observations)
        try:
            result=original(self,actual,expected);observations.append({'index':index,'correct':True})
            return result
        except AssertionError as error:
            observations.append({'index':index,'correct':False,'error':str(error)})
            if index==1:
                raise ReferenceCalibrationError('Independent reference failed captured-golden calibration: '+str(error)) from error
            return True
    Prepared.compare=deferred
    try:prepared=runtime.prepare_case(case)
    finally:Prepared.compare=original
    if len(observations)!=3:raise RuntimeError('Captured-parity preparation protocol changed')
    prepared.callbacks._compare=original.__get__(prepared,Prepared)
    prepared.captured_parity=all(row['correct'] for row in observations)
    return prepared,observations


def graph_probe(prepared,seed,torch,progress):
    callbacks=prepared.callbacks
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(3):
            truth=callbacks.reset_inputs(seed+index);callbacks.initialize_outputs()
            prepared.invoke_candidate();torch.cuda.synchronize();prepared.validate_metadata()
            prepared.assert_immutable(cpu_copy(prepared.snapshot_inputs(),torch),truth.value)
    torch.cuda.current_stream().wait_stream(stream)
    truth=callbacks.reset_inputs(seed+3);callbacks.initialize_outputs()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):prepared.invoke_candidate()
    progress['graph_captured']=True
    # The captured call allocates new outputs; poison those exact graph buffers.
    callbacks.initialize_outputs()
    graph.replay();progress['graph_replayed']=True
    return callbacks.verify(truth)


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Use a new source-binding diagnostic output file')
    output.parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'build').mkdir(exist_ok=True)
    report={'schema':'kimi-stage1-fast-source-binding-v1','scoreable':False,'mode':args.mode,
            'performance_samples':0,'tokens':args.tokens,'seed':args.seed}
    failure_phase='source_guard'
    try:
        validate_sources(ROOT,ROOT)
        manifest=strict_json((ROOT/'cases.json').read_text());validate_manifest(manifest)
        case=next(case for case in manifest['cases'] if case['tensors']['inputs.a']['shape'][0]==args.tokens)
        policy=strict_json((ROOT/'ut/source_guard_policy.json').read_text())
        request={'request_id':'source-binding-'+secrets.token_hex(16),'challenge_seed':args.seed,
                 'source_sha256':{name:file_sha(ROOT/name) for name in policy['sources']}}
        report.update(case_id=case['case_id'],source_sha256=request['source_sha256'])
        failure_phase='compile'
        runtime=create_runtime(ROOT,manifest,request)
        prepared,observations=prepare_for_probe(runtime,case)
        report.update(compiled=True,captured_warmup_checks=observations,native_binding=prepared.native_engagement())
        if args.mode=='eager':
            failure_phase='eager';prepared.callbacks.check_once(args.seed)
        elif args.mode=='graph':
            failure_phase='graph';graph_probe(prepared,args.seed,runtime.torch,report)
        report['status']='compiled' if args.mode=='compile' else 'candidate_accepted'
        code=0
    except Exception as error:
        report.update(status='invalid_reference' if isinstance(error,ReferenceCalibrationError) else 'candidate_rejected',failure_phase=failure_phase,
                      error_type=type(error).__name__,error=str(error))
        code=1
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(canonical(report));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('compile','eager','graph'),required=True)
    parser.add_argument('--tokens',type=int,choices=(64,8192,16384),default=64)
    parser.add_argument('--seed',type=int,default=734170854);parser.add_argument('--output',required=True)
    raise SystemExit(run(parser.parse_args()))
