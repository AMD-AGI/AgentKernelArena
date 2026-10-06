"""Protected complete-case evaluation; GPU reference follows frozen CPU observations."""
import argparse
import hashlib
import importlib.util
import json
import math
import secrets
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import canonical,fingerprint,strict_json,validate_manifest,observe_case,checked_replays,finalize_report
from source_guard import validate_sources
import storage_guard


def source_hash():return hashlib.sha256((ROOT/'source/kernels.py').read_bytes()).hexdigest()
def package_hash():
    h=hashlib.sha256()
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and not any(x in path.relative_to(ROOT).parts for x in ('build','__pycache__')) and path.suffix!='.pyc':
            if path.is_symlink():raise ValueError('Protected package contains a symlink')
            h.update(str(path.relative_to(ROOT)).encode()+b'\0'+path.read_bytes())
    return h.hexdigest()


def load_source():
    validate_sources(ROOT,ROOT)
    spec=importlib.util.spec_from_file_location('guarded_gpu_source',ROOT/'source/kernels.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def pack(weight):
    n,k=weight.shape
    return weight.reshape(n//16,16,k//32,2,16).permute(0,2,3,1,4).contiguous().reshape(n,k)


def generate(case,seed,compute_reference=True):
    import torch
    if case.get('live_fixture') and seed%2:
        from live_operands import generate_live
        return generate_live(case,seed,compute_reference=compute_reference)
    m,n,k=(case['scalars'][x] for x in ('M','N','K'));g=torch.Generator(device='cpu').manual_seed(seed)
    a=torch.randn((m,k),generator=g,dtype=torch.float32)/math.sqrt(k)
    b=torch.randn((n,k),generator=g,dtype=torch.float32)
    if m>1:
        a[0].zero_();a[1].fill_(1/math.sqrt(k))
    if case['scalars']['fp8']:
        aq=(a*16).clamp(-448,448).to(torch.float8_e4m3fn)
        bq=(b*8).clamp(-448,448).to(torch.float8_e4m3fn)
        sa=torch.exp2(torch.randint(-5,0,(k//128,m),generator=g).float()).t()
        sb=torch.exp2(torch.randint(-5,0,(n//128,k//128),generator=g).float())
        ad=aq.float()*sa.repeat_interleave(128,dim=1)
        bd=bq.float()*sb.repeat_interleave(128,dim=0).repeat_interleave(128,dim=1)
        expected=(ad@bd.t()).to(torch.bfloat16)
        return {'A':aq,'B':pack(bq),'SA':sa,'SB':sb},expected
    dtype=getattr(torch,case['tensors']['A']['dtype'])
    a=a.to(dtype);b=b.to(dtype)
    values=storage_guard.fresh_input_storage({'A':a,'B':b.t()},case,seed)
    expected=(values['A'].float()@values['B'].float()).to(dtype) if compute_reference else None
    return values,expected


def build_state(case,seed,module,defer_reference=False):
    import torch
    fresh,reference=generate(case,seed)
    sizes=storage_guard.extents(case)
    tensors={name:storage_guard.allocate_view(spec,sizes[name],'cuda') for name,spec in case['tensors'].items()}
    for name,value in fresh.items():storage_guard.copy_complete(tensors[name],value)
    native_events=[]
    scalars=case['scalars'];m,n,k=(scalars[x] for x in ('M','N','K'));fp8=scalars['fp8']
    def launch_port():
        return module.gemm_kernel[(math.ceil(m/32),math.ceil(n/64))](tensors['A'],tensors['B'],tensors['A'],tensors['B'],tensors['C'],m,n,k,False,32,64,128,tensors['A'].stride(0),num_warps=4)
    family=case['live_fixture']['capture_family']
    if family=='bf16_gemm':
        from native_dispatch import load_native,describe_dispatch,require_dispatch,bind_solutions
        native=load_native();otype=case['capture_controls']['otype']
        otype=None if otype is None else getattr(torch,otype['name'])
        observed_dispatch=describe_dispatch(native,tensors['A'],tensors['B'].t(),otype=otype)
        require_dispatch(case,observed_dispatch)
    def invoke():
        if family=='aten_bf16_mm':return launch_port()
        if family!='bf16_gemm':raise ValueError('Unknown captured dense family')
        compiled=[]
        def solution(A,B,solidx,bias,output_dtype,scale_a,scale_b,scale_c,bpreshuffle,config=None):
            if A.data_ptr()!=tensors['A'].data_ptr() or B.data_ptr()!=tensors['B'].data_ptr() or any(v is not None for v in (bias,scale_a,scale_b,scale_c)) or bpreshuffle:raise ValueError('Wrapper changed the frozen replacement ABI')
            compiled.append(launch_port())
            return tensors['C']
        with bind_solutions(native,solution) as calls:
            result=native.gemm_a16w16(tensors['A'],tensors['B'].t(),bias=None,otype=otype,scale_a=None,scale_b=None,scale_c=None)
        if calls!=[observed_dispatch['libtype']] or len(compiled)!=1 or result.data_ptr()!=tensors['C'].data_ptr():raise ValueError('Submitted port was not reached exactly once through native solMap')
        return compiled[0]
    def observe():
        if any(tensors[name].untyped_storage().nbytes()!=size for name,size in sizes.items()):raise AssertionError('Runtime backing storage extent changed')
        actual={'M':tensors['A'].shape[0],'N':tensors['C'].shape[1],'K':tensors['A'].shape[1],
                'fp8':tensors['A'].dtype==torch.float8_e4m3fn,'BM':32,'BN':64,'BK':128,'A_ROW_STRIDE':tensors['A'].stride(0)}
        return observe_case(case,tensors,actual)
    def initialize():storage_guard.initialize_output(tensors['C'])
    def verify(ref,*,native_calibration=False):
        expected,inputs=ref
        torch.cuda.synchronize()
        output_storage=storage_guard.snapshot_complete(tensors['C'])
        actual=storage_guard.view_snapshot(output_storage,case['tensors']['C'])
        storage_guard.assert_output_guards(output_storage,case['tensors']['C'])
        storage_guard.assert_inputs_unchanged(tensors,inputs)
        if not torch.isfinite(actual.float()).all():raise AssertionError('Unwritten/nonfinite output')
        if tensors['C'].untyped_storage().data_ptr() in [x.untyped_storage().data_ptr() for name,x in tensors.items() if name!='C']:raise AssertionError('Output aliases input')
        # Candidate output and immutable inputs are already observed on CPU.
        # Only now may an independent FP32 GPU reference exist. Correctness and
        # initial case checks also bind this fast reference to CPU FP32 truth.
        prior_tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            gpu_reference=(tensors['A'].float()@tensors['B'].float()).to(tensors['C'].dtype)
            reference_cpu=gpu_reference.detach().cpu().clone()
            del gpu_reference
        finally:torch.backends.cuda.matmul.allow_tf32=prior_tf32
        if expected is not None:torch.testing.assert_close(reference_cpu,expected,rtol=0.01,atol=0.02)
        if not torch.isfinite(reference_cpu.float()).all():raise AssertionError('Nonfinite independent reference')
        mathematical=reference_cpu if expected is None else expected
        if native_calibration:
            from native_precision import calibrate
            native_events.append(calibrate(case,inputs,actual,mathematical))
        else:torch.testing.assert_close(actual,mathematical,rtol=0.01,atol=0.02)
    def reset(seed):
        values,expected=generate(case,seed,compute_reference=not defer_reference)
        for name,value in values.items():storage_guard.copy_complete(tensors[name],value)
        return expected,values
    verify.native_calibration_events=native_events
    observe();return tensors,invoke,observe,initialize,verify,reset,(reference,fresh)


def negative_controls(tensors,initialize,verify,reference):
    import torch
    result={}
    valid_output=tensors['C'].detach().cpu().clone()
    initialize()
    try:verify(reference)
    except AssertionError:result['no_op']=True
    else:raise AssertionError('No-op accepted')
    tensors['C'].copy_(valid_output);tensors['C'][0,0]=float(valid_output[0,0])+max(1.0,abs(float(valid_output[0,0]))*0.5)
    try:verify(reference)
    except AssertionError:result['wrong_output']=True
    else:raise AssertionError('Wrong output accepted')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['compile','correctness','performance']);p.add_argument('--request');a=p.parse_args()
    if (ROOT/'NOT_BUILT').exists() or not (ROOT/'cases.json').is_file():raise RuntimeError('New actual Kimi dense capture and count-backed fixture intake required; sampled profiles cannot qualify this task')
    build=ROOT/'build';build.mkdir(exist_ok=True);path=build/(a.phase+'_report.json');path.unlink(missing_ok=True)
    manifest=validate_manifest(strict_json((ROOT/'cases.json').read_text()))
    if a.phase!='compile' and any('live_fixture' not in case for case in manifest['cases']):raise RuntimeError('Missing native live operand representatives. Import the completed Kimi dense capture with scripts/import_live_operands.py; generator-only checks are diagnostics.')
    request=strict_json(Path(a.request).read_text()) if a.request else {'schema_version':1,'request_id':secrets.token_hex(24),'phase':a.phase,'manifest_sha256':fingerprint(manifest),'package_sha256':package_hash(),'source_sha256':{'source/kernels.py':source_hash()},'challenge_seed':secrets.randbelow(2**30)}
    if request['phase']!=a.phase or request['manifest_sha256']!=fingerprint(manifest) or request['source_sha256']!={'source/kernels.py':source_hash()}:raise ValueError('Request does not match current cases/source')
    before=package_hash();module=load_source()
    import torch
    if not torch.cuda.is_available() or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('Requires gfx950 ROCm')
    torch.set_num_threads(16)
    results=[];compiled=[];policy=manifest['measurement']
    for case in manifest['cases']:
        tensors,invoke,observe,initialize,verify,reset,reference=build_state(case,policy['correctness_seeds'][0],module,defer_reference=a.phase=='performance')
        initialize();kernel=invoke();torch.cuda.synchronize();verify(reference)
        compiled.append({'case_id':case['case_id'],'kernel_name':kernel.name,'kernel_hash':kernel.hash})
        if a.phase=='compile':continue
        if a.phase=='correctness':
            controls={}
            for seed in policy['correctness_seeds']:
                reference=reset(seed);initialize();invoke();torch.cuda.synchronize();verify(reference)
                controls=negative_controls(tensors,initialize,verify,reference)
            results.append({'case':observe(),'correct':True,'seeds':policy['correctness_seeds'],'negative_controls':controls})
        else:
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
        del tensors,invoke,observe,initialize,verify,reset,reference
    if package_hash()!=before:raise ValueError('Protected package changed during evaluation')
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'compiled':True,'cases':results,'compiled_kernels':compiled,'oracle_device':'cpu','input_immutability':'entire_backing_storage_including_prefix_padding_and_tail','output_guards':'all_bytes_outside_declared_output_view','reference_policy':'CPU FP32 correctness plus independent GPU FP32 after candidate CPU observations for timed replay; TF32 disabled','runtime_source_sha256':source_hash(),'implementation':'submitted_triton_port','comparison_baseline':'frozen_triton_port','stock_source_equivalence':False},manifest,request)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(path)
    if a.phase=='performance':
        import production_comparison
        production_comparison.main()
    print(a.phase+': PASS')
if __name__=='__main__':main()
