"""Protected whole-MoE port evaluation. Native fixtures/oracles remain on CPU."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
sys.path.insert(0,str(ROOT/'scripts'))
from evaluation_contract import fingerprint,strict_json,validate_manifest,observe_case,checked_replays,finalize_report
from source_guard import validate_sources
from fixture_codec import restore_phase,file_sha,fresh_numeric_fixture
from native_enums import restore_native_enum


def source_hash():return hashlib.sha256((ROOT/'source/kernels.py').read_bytes()).hexdigest()
def package_hash():
    h=hashlib.sha256()
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and not any(x in path.relative_to(ROOT).parts for x in ('build','__pycache__')) and path.suffix!='.pyc':
            if path.is_symlink():raise ValueError('Protected package contains a symlink')
            h.update(str(path.relative_to(ROOT)).encode()+b'\0')
            # Raw fixture blobs are separately streamed and verified at restoration.
            h.update(file_sha(path).encode())
    return h.hexdigest()


def load_source():
    validate_sources(ROOT,ROOT)
    spec=importlib.util.spec_from_file_location('guarded_moe_source',ROOT/'source/kernels.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_state(case,module,fixture_root=None,diagnostic=None):
    import torch
    origin=ROOT if fixture_root is None else Path(fixture_root).resolve()
    fixture_path=(origin/case['fixture']['path']).resolve()
    if not fixture_path.is_relative_to(origin) or file_sha(fixture_path)!=case['fixture']['sha256']:raise ValueError('Fixture metadata changed')
    fixture=strict_json(fixture_path.read_text());cpu=restore_phase(fixture_path.parent,fixture,'inputs');captured_output=restore_phase(fixture_path.parent,fixture,'outputs')['result']
    if captured_output.device.type!='cpu':raise ValueError('Captured output evidence must remain on CPU')
    # The importer froze the source signature, actual controls and packed layouts.
    import import_fixtures
    import_fixtures.validate_controls(fixture['controls'])
    tensors=restore_phase(fixture_path.parent,fixture,'inputs',device='cuda')
    output_spec=case['tensors']['result']
    tensors['result']=torch.empty_strided(output_spec['shape'],output_spec['strides'],dtype=torch.bfloat16,device='cuda')
    m,h=tensors['hidden_states'].shape;e,_,i=tensors['w2'].shape;topk=tensors['topk_ids'].shape[1]
    sort_module=importlib.import_module('aiter.ops.moe_sorting_opus')
    provenance=strict_json((ROOT/'provenance/NATIVE-SOURCES.json').read_text())
    if file_sha(sort_module.__file__)!=provenance['sort_module_sha256']:raise ValueError('Native frozen sorting module differs from pinned image')
    native_module=importlib.import_module('aiter.fused_moe')
    if file_sha(native_module.__file__)!=provenance['files']['aiter/fused_moe.py']:raise ValueError('Native reference source differs from pinned image')
    from aiter import ActivationType,QuantType
    native_controls=dict(fixture['controls']);native_controls.pop('tensor_attributes')
    for name,cls in [('activation',ActivationType),('quant_type',QuantType)]:
        value=native_controls[name]
        native_controls[name]=restore_native_enum(value,cls.__qualname__,cls) if isinstance(value,dict) else cls(value)
    if isinstance(native_controls.get('dtype'),dict):native_controls['dtype']=getattr(torch,native_controls['dtype']['name'])

    padded=m*topk+e*32-topk;blocks=math.ceil(padded/32)
    def empty(shape,dtype):return torch.empty(shape,dtype=dtype,device='cuda')
    q1=empty((m,h),torch.float8_e4m3fn);s1=empty((m,h//128),torch.float32)
    g=empty((m*topk,2*i),torch.float32);q2=empty((m*topk,i),torch.float8_e4m3fn);s2=empty((m*topk,i//128),torch.float32)
    partial=empty((m*topk,h),torch.float32)
    ids=empty((padded,),torch.int32);weights=empty((padded,),torch.float32);experts=empty((blocks,),torch.int32);valid=empty((2,),torch.int32)
    ws_size=sort_module.moe_sorting_opus_get_workspace_size(m,e,topk,0)
    workspace=empty((ws_size,),torch.uint8) if ws_size else None
    def invoke():
        # Whole pipeline: native routing sort, input quant, two matrix products,
        # activation/intermediate quant and weighted expert reduction.
        sort_module.moe_sorting_opus_fwd(tensors['topk_ids'],tensors['topk_weight'],ids,weights,experts,valid,tensors['result'],e,32,None,None,workspace,0,None,None,None)
        kernels=[]
        kernels.append(module.quantize_input[(m,h//128)](tensors['hidden_states'],q1,s1,m,h,num_warps=4))
        kernels.append(module.stage1[(blocks,math.ceil(2*i/64))](q1,s1,tensors['w1'],tensors['w1_scale'],ids,experts,valid,g,m,h,i,topk,32,64,128,num_warps=4))
        kernels.append(module.activate_quantize[(m*topk,i//128)](g,q2,s2,m,i,topk,num_warps=4))
        kernels.append(module.stage2[(blocks,math.ceil(h/64))](q2,s2,tensors['w2'],tensors['w2_scale'],ids,experts,valid,partial,m,h,i,topk,32,64,128,num_warps=4))
        kernels.append(module.reduce_routes[(m,math.ceil(h/256))](partial,tensors['topk_weight'],tensors['result'],m,h,topk,num_warps=4))
        return kernels
    def observe():
        controls=dict(fixture['controls']);controls['port_launch']={'BM':32,'BN':64,'BK':128,'input_quant_block':128,'intermediate_quant_block':128,'output_reduce_block':256}
        return observe_case(case,tensors,controls)
    def reset(seed):
        truth=fresh_numeric_fixture(cpu,seed)
        for name in ('hidden_states','topk_ids','topk_weight'):tensors[name].copy_(truth['inputs'][name])
        return truth
    def initialize():
        tensors['result'].fill_(float('nan'));g.fill_(float('nan'));partial.fill_(float('nan'))
        q1.fill_(float('nan'));q2.fill_(float('nan'));s1.fill_(float('nan'));s2.fill_(float('nan'))
    def verify(truth):
        # Snapshot all candidate observations first. No native golden output for
        # the fresh activation values exists on the GPU before this point.
        torch.cuda.synchronize();actual=tensors['result'].detach().cpu().clone()
        observed_inputs={name:tensors[name].detach().cpu().clone() for name in truth['inputs']}
        if not torch.isfinite(actual.float()).all():raise AssertionError('Nonfinite or unwritten whole-MoE output')
        for name,before in truth['inputs'].items():
            after=observed_inputs[name]
            if not torch.equal(after.contiguous().view(torch.uint8),before.contiguous().view(torch.uint8)):raise AssertionError('Input mutation: '+name)
        arguments={name:tensors[name] for name in truth['inputs']};arguments.update(native_controls)
        native_output=native_module.fused_moe(**arguments)
        expected=native_output.detach().cpu().clone()
        del native_output
        if diagnostic is not None:
            repeated_native=native_module.fused_moe(**arguments).detach().cpu().clone()
            from aiter.ops.quant import get_hip_quant
            native_q1,native_scale=get_hip_quant(QuantType.per_1x128)(tensors['hidden_states'],quant_dtype=torch.float8_e4m3fn,transpose_scale=True)
            native_scale=torch.as_strided(native_scale,native_scale.shape,(1,m))
            snapshots={'candidate':actual,'native':expected,'native_repeat':repeated_native,'hidden_states':observed_inputs['hidden_states'],'topk_ids':observed_inputs['topk_ids'],'topk_weight':observed_inputs['topk_weight']}
            for name,value in {'q1':q1,'s1':s1,'gate_up':g,'q2':q2,'s2':s2,'route_partials':partial,'sorted_ids':ids,'sorted_experts':experts,'valid_ids':valid,'native_q1':native_q1,'native_s1':native_scale}.items():snapshots[name]=value.detach().cpu().clone()
            diagnostic(truth['seed'],snapshots)
            del native_q1,native_scale,snapshots
        torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.02)
    observe();return tensors,invoke,observe,reset,initialize,verify


def negative_controls(tensors,initialize,verify,reference):
    initialize();controls={}
    try:verify(reference)
    except AssertionError:controls['no_op']=True
    else:raise AssertionError('No-op accepted')
    tensors['result'].fill_(12345.0)
    try:verify(reference)
    except AssertionError:controls['wrong_output']=True
    else:raise AssertionError('Wrong output accepted')
    return controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['compile','correctness','performance']);ap.add_argument('--request');args=ap.parse_args()
    path=ROOT/'build'/(args.phase+'_report.json');path.parent.mkdir(exist_ok=True);path.unlink(missing_ok=True)
    if not (ROOT/'cases.json').is_file():raise RuntimeError('Whole-MoE native served fixtures not imported. Run scripts/import_fixtures.py with the completed GLM capture manifest.')
    manifest=validate_manifest(strict_json((ROOT/'cases.json').read_text()))
    request=strict_json(Path(args.request).read_text()) if args.request else {'schema_version':1,'request_id':secrets.token_hex(24),'phase':args.phase,'manifest_sha256':fingerprint(manifest),'package_sha256':package_hash(),'source_sha256':{'source/kernels.py':source_hash()},'challenge_seed':secrets.randbelow(2**30)}
    if request['phase']!=args.phase or request['manifest_sha256']!=fingerprint(manifest) or request['source_sha256']!={'source/kernels.py':source_hash()}:raise ValueError('Request source/case/phase mismatch')
    before=package_hash();module=load_source();import torch
    if not torch.cuda.is_available() or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('Requires gfx950 ROCm')
    torch.set_num_threads(16)
    results=[];compiled=[];policy=manifest['measurement']
    for case in manifest['cases']:
        tensors,invoke,observe,reset,initialize,verify=build_state(case,module)
        reference=reset(policy['correctness_seeds'][0]);initialize();kernels=invoke();torch.cuda.synchronize();verify(reference)
        compiled.append({'case_id':case['case_id'],'kernels':[{'name':k.name,'hash':k.hash} for k in kernels]})
        if args.phase=='correctness':
            for seed in policy['correctness_seeds']:
                reference=reset(seed);initialize();invoke();torch.cuda.synchronize();verify(reference)
                controls=negative_controls(tensors,initialize,verify,reference)
            results.append({'case':observe(),'correct':True,'seeds':policy['correctness_seeds'],'negative_controls':controls})
        elif args.phase=='performance':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):invoke()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(case['calls_per_sample']):invoke()
            def measure(call):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();call();end.record();end.synchronize();return begin.elapsed_time(end)
            results.append(checked_replays(case,policy,reset_inputs=reset,initialize_outputs=initialize,replay=graph.replay,verify=verify,measure=measure,observe=observe,seed=request['challenge_seed']))
    if package_hash()!=before:raise ValueError('Task package changed during evaluation')
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'compiled':True,'cases':results,'compiled_kernels':compiled,'oracle_device':'cpu','reference_policy':'fresh_numeric_inputs_native_reference_after_candidate_CPU_snapshot','implementation':'whole_moe_replacement_port','comparison_baseline':'frozen_same_port','original_native_kernel_source':False},manifest,request)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(path)
    if args.phase=='performance':
        import production_comparison
        production_comparison.main()
    print(args.phase+': PASS')
if __name__=='__main__':main()
