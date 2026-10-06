"""Protected graph replay/oracle protocol for three current DeepSeek native seams."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from native import load_native, sha256
from source_guard import validate_sources
from evaluation_contract import (canonical, fingerprint, strict_json, validate_manifest,
    observe_case, checked_replays, finalize_report)
from abi import runtime_abi
from dispatch_contract import validate_dispatch, validate_runtime_dispatch
from mla_decode_distribution import KIND, RuntimeRecipe, load_policy


def write(phase,report):
    target=ROOT/'build'/(phase+'_report.json'); target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.tmp'); temporary.write_text(canonical(report)+'\n'); temporary.replace(target)


def compare(actual,golden,tol,path='output'):
    import torch
    if torch.is_tensor(golden):
        if not torch.is_tensor(actual) or actual.shape!=golden.shape or actual.dtype!=golden.dtype:
            raise AssertionError(path+': shape/dtype differs')
        if actual.device.type!='cpu' or golden.device.type!='cpu':
            raise AssertionError(path+': comparisons require owned CPU snapshots')
        a,g=actual.float(),golden.float()
        for predicate in (torch.isnan,torch.isposinf,torch.isneginf):
            if not torch.equal(predicate(a),predicate(g)): raise AssertionError(path+': nonfinite pattern differs')
        finite=torch.isfinite(g); a,g=a[finite],g[finite]
        if not golden.is_floating_point() or 'e8m0' in str(golden.dtype):
            if not torch.equal(actual.view(torch.uint8),golden.view(torch.uint8)): raise AssertionError(path+': exact output differs')
        elif a.numel():
            atol=tol*g.square().mean().sqrt().clamp_min(1e-6)
            if not bool(((a-g).abs() <= atol+tol*g.abs()).all()): raise AssertionError(path+': values differ')
        return
    if isinstance(golden,(list,tuple)):
        if type(actual)!=type(golden) or len(actual)!=len(golden): raise AssertionError(path+': structure differs')
        for i,(a,g) in enumerate(zip(actual,golden)): compare(a,g,tol,path+'.'+str(i))
        return
    if isinstance(golden,dict):
        if not isinstance(actual,dict) or set(actual)!=set(golden): raise AssertionError(path+': keys differ')
        for k in golden: compare(actual[k],golden[k],tol,path+'.'+k)
        return
    if actual!=golden: raise AssertionError(path+': scalar differs')


def cpu_clone(value):
    import torch
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,tuple): return tuple(cpu_clone(x) for x in value)
    if isinstance(value,list): return [cpu_clone(x) for x in value]
    if isinstance(value,dict): return {k:cpu_clone(v) for k,v in value.items()}
    return value


def leaves(value):
    import torch
    if torch.is_tensor(value): return [value]
    if isinstance(value,(tuple,list)): return sum((leaves(x) for x in value),[])
    if isinstance(value,dict): return sum((leaves(x) for x in value.values()),[])
    return []


def invoke(fn,inputs):
    inputs=dict(inputs); extra=inputs.pop('_kwargs',{})
    if set(inputs)&set(extra): raise RuntimeError('Duplicate native keyword')
    return fn(**inputs,**extra)


def storage_snapshots(inputs, *, immutable_only=False):
    """Own complete input-storage bytes on CPU, including padding and aliases."""
    from snapshots import raw_storage
    tensors,_=runtime_abi(inputs,None)
    mutable={value.untyped_storage().data_ptr() for value in leaves(inputs.get('out'))}
    seen=set(); snapshots={}
    for name,value in tensors.items():
        address=value.untyped_storage().data_ptr()
        if address in seen or (immutable_only and address in mutable): continue
        seen.add(address); snapshots[name]=cpu_clone(raw_storage(value))
    return snapshots


def restore_storages(inputs, snapshots):
    from snapshots import raw_storage
    tensors,_=runtime_abi(inputs,None)
    for name,expected in snapshots.items():
        if name not in tensors: raise AssertionError('Native input binding changed: '+name)
        actual=raw_storage(tensors[name])
        if actual.numel()!=expected.numel(): raise AssertionError('Native input storage size changed: '+name)
        actual.copy_(expected)


def assert_immutable_inputs(inputs, expected):
    import torch
    actual=storage_snapshots(inputs,immutable_only=True)
    for name,value in actual.items():
        if name not in expected or not torch.equal(value,expected[name]):
            raise AssertionError('Native kernel mutated input storage: '+name)


def verify_after_snapshot(output, inputs, expected_inputs, reference_fn, reference_inputs, tol):
    """Freeze observations before a reference launch can expose golden GPU data."""
    import torch
    torch.cuda.synchronize()
    actual_cpu=cpu_clone(output)
    assert_immutable_inputs(inputs,expected_inputs)
    restore_storages(reference_inputs,expected_inputs)
    reference_output=invoke(reference_fn,reference_inputs)
    torch.cuda.synchronize()
    expected_cpu=cpu_clone(reference_output)
    assert_immutable_inputs(reference_inputs,expected_inputs)
    del reference_output
    compare(actual_cpu,expected_cpu,tol)


def engage_specialization(fn,inputs,golden,tol,label):
    """Invoke the actual specialization, synchronize and validate its outputs."""
    import torch
    before=storage_snapshots(inputs)
    output=invoke(fn,inputs)
    torch.cuda.synchronize()
    actual=cpu_clone(output)
    assert_immutable_inputs(inputs,before)
    compare(actual,golden,tol,label)
    return output


def fixture(case,manifest,module):
    from runtime_capture import restore_phase
    from snapshots import restore
    reference=case['fixture']
    relative=reference['path'] if isinstance(reference,dict) else reference
    expected_sha=reference['sha256'] if isinstance(reference,dict) else case['fixture_sha256']
    original_path=ROOT/relative;path=original_path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or original_path.is_symlink() or sha256(path)!=expected_sha:
        raise RuntimeError('Untrusted or changed fixture')
    record=strict_json(path.read_text())
    if record['provenance']['run_id']!=manifest['run_id'] or record['source_sha256']!=manifest['native_source_sha256']:
        raise RuntimeError('Stale fixture source/run identity')
    if record['served']['stage']=='decode' and record['origin']!='served_graph':
        raise RuntimeError('Decode fixture must follow actual served graph replay')
    if record.get('startup_values') is not False: raise RuntimeError('Startup data are not served fixtures')
    tensors=restore_phase(path.parent,record,'inputs',device='cuda',max_storage_bytes=64<<30)
    controls=record['controls']; inputs={}
    cfg=strict_json((ROOT/'provenance/SOURCE.json').read_text())
    for key in cfg['signature_parameters']:
        if key in tensors: inputs[key]=tensors[key]
        elif key in controls: inputs[key]=restore({'tree':controls[key],'storages':{}},'cuda',module)
        else: raise RuntimeError('Captured native argument missing: '+key)
    if cfg.get('has_var_kwargs'):
        inputs['_kwargs']=restore({'tree':controls.get('_kwargs',{}),'storages':{}},'cuda',module)
    for name,tensor in inputs.items():
        if name in tensors:
            for key,value in controls.get(name+'_attributes',{}).items():
                attr=tensors[value['tensor_binding']] if isinstance(value,dict) and 'tensor_binding' in value else restore({'tree':value,'storages':{}},'cuda',module)
                setattr(tensor,key,attr)
    # Captured golden outputs must never be resident on the GPU when a
    # candidate executes. Inputs keep their captured GPU ABI and storage.
    outputs=restore_phase(path.parent,record,'outputs',device='cpu',max_storage_bytes=64<<30)
    golden=(outputs['output'],outputs['lse']) if manifest['seam']=='mla' else (outputs['output'],outputs['scale']) if 'scale' in outputs else outputs['output']
    return inputs,golden


def request_for(phase,manifest,path):
    cfg=strict_json((ROOT/'provenance/SOURCE.json').read_text())
    actual={relative:sha256(ROOT/relative) for relative in cfg['editable_sources']}
    if path:
        request=strict_json(Path(path).read_text())
        if request['phase']!=phase or request['manifest_sha256']!=fingerprint(manifest) or request['source_sha256']!=actual:
            raise RuntimeError('Fresh request does not bind current manifest/source/phase')
        return request
    package={str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*')) if p.is_file()
             and not p.is_symlink() and 'build' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts}
    return {'schema_version':1,'request_id':secrets.token_hex(24),'phase':phase,
        'manifest_sha256':fingerprint(manifest),'package_sha256':fingerprint(package),'source_sha256':actual,
        'challenge_seed':secrets.randbelow(2**30),'origin':'protected-task-runner'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('compile','correctness','performance')); p.add_argument('--request')
    a=p.parse_args()
    (ROOT/'build').mkdir(exist_ok=True)
    (ROOT/'build'/(a.phase+'_report.json')).unlink(missing_ok=True)
    manifest=strict_json((ROOT/'cases.json').read_text()); validate_manifest(manifest)
    validate_dispatch(manifest)
    distribution_policy=load_policy(ROOT,manifest)
    request=request_for(a.phase,manifest,a.request); validate_sources(ROOT,ROOT)
    report={'schema_version':1,'status':'ok','request':request,'cases':[]}
    import torch
    from snapshots import raw_storage
    if not torch.version.hip or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('ROCm gfx950 GPU required')
    module,fn,identity=load_native(ROOT,'candidate')
    reference_module,reference_fn,reference_identity=load_native(ROOT,'reference')
    policy=manifest['measurement']; tol=manifest['tolerance']
    compiled=[]
    for case in manifest['cases']:
        if case['calls_per_sample']!=1: raise RuntimeError('This seam graph represents one native call per replay')
        inputs,golden=fixture(case,manifest,module); reference_inputs,_=fixture(case,manifest,reference_module)
        dispatch=validate_runtime_dispatch(module,inputs)
        if validate_runtime_dispatch(reference_module,reference_inputs)!=dispatch:
            raise RuntimeError('Candidate and reference dispatch controls differ')
        pristine_inputs=storage_snapshots(inputs)
        recipe=(RuntimeRecipe(case,distribution_policy,inputs,module,reference_module)
                if case.get('provenance_kind')==KIND else None)
        initial_out=cpu_clone(inputs.get('out'))
        primary='q' if manifest['seam'].startswith('mla') else 'a' if manifest['seam']=='moe1' else 'hidden_states' if manifest['seam']=='moe1_prefill' else 'inter_states'
        # Current candidate and immutable reference specializations are both
        # invoked with legitimate captured arguments in every phase, including
        # compile. The candidate's CPU snapshot precedes any reference launch.
        output=engage_specialization(fn,inputs,golden,tol,'served-fixture')
        tensors,scalars=runtime_abi(inputs,output); observe_case(case,tensors,scalars)
        restore_storages(reference_inputs,pristine_inputs)
        reference_output=engage_specialization(reference_fn,reference_inputs,golden,tol,'reference-served-fixture')
        del reference_output
        compiled.append({'case_id':case['case_id'],'candidate_binding':identity,
                         'reference_binding':reference_identity,'dispatch':dispatch,'invoked_and_synchronized':True})
        if a.phase=='compile':
            if recipe is not None:
                compiled[-1]['control_distribution']=recipe.proof('compile',policy)
                recipe.close()
            del inputs,reference_inputs,golden,output,pristine_inputs,initial_out
            torch.cuda.empty_cache()
            continue
        for _ in range(3):
            restore_storages(inputs,pristine_inputs)
            invoke(fn,inputs)
        torch.cuda.synchronize(); graph=torch.cuda.CUDAGraph()
        restore_storages(inputs,pristine_inputs)
        with torch.cuda.graph(graph): output=invoke(fn,inputs)
        def reset_inputs(seed,forced_length=None):
            restore_storages(inputs,pristine_inputs)
            if recipe is not None:recipe.apply(inputs,seed,forced_length)
            generator=torch.Generator(device='cuda'); generator.manual_seed(seed)
            values=torch.randn(inputs[primary].shape,dtype=torch.float32,device='cuda',generator=generator).to(inputs[primary].dtype)
            inputs[primary].copy_(values)
            # Return an immutable CPU input token, not a live GPU golden. The
            # reference is evaluated only after replay observations are frozen.
            return storage_snapshots(inputs)
        def initialize_outputs():
            if initial_out is not None: inputs['out'].copy_(initial_out)
            mutable_ptr=inputs['out'].untyped_storage().data_ptr() if initial_out is not None else None
            for value in leaves(output):
                if value.untyped_storage().data_ptr()!=mutable_ptr: raw_storage(value).fill_(0xAA)
        def observe():
            if recipe is not None:recipe.verify_controls(inputs,output)
            tensors,scalars=runtime_abi(inputs,output)
            return observe_case(case,tensors,scalars)
        def verify(expected):
            verify_after_snapshot(output,inputs,expected,reference_fn,reference_inputs,tol)
        def measure(call):
            start=torch.cuda.Event(enable_timing=True); stop=torch.cuda.Event(enable_timing=True)
            start.record(); call(); stop.record(); stop.synchronize(); return float(start.elapsed_time(stop))
        if a.phase=='correctness':
            settings=distribution_policy['lengths'] if recipe is not None else [None]
            all_controls={name:True for name in policy['negative_controls']}
            exhaustive=[]
            for length in settings:
                for seed in policy['correctness_seeds']:
                    expected=reset_inputs(seed,length); initialize_outputs(); observe(); graph.replay(); verify(expected)
                controls={}
                for control in policy['negative_controls']:
                    expected=reset_inputs(request['challenge_seed'],length); initialize_outputs()
                    if control=='wrong_output':
                        graph.replay(); torch.cuda.synchronize()
                        for value in leaves(output): raw_storage(value).zero_()
                    elif control!='no_op': raise RuntimeError('Unknown required negative control')
                    try: verify(expected)
                    except AssertionError: controls[control]=True
                    else: raise RuntimeError('Required negative control escaped: '+control)
                if recipe is not None:
                    exhaustive.append({'length':length,'seeds':policy['correctness_seeds'],'negative_controls':controls})
                for name in all_controls:all_controls[name] &= controls.get(name) is True
            row={'case':observe(),'correct':True,'seeds':policy['correctness_seeds'],
                 'negative_controls':all_controls,'negative_control_scope':'protected no-op replay and output corruption; submitted-source mutation retest is separate'}
            if recipe is not None:row['exhaustive_control_settings']=exhaustive
            report['cases'].append(row)
        else:
            row=checked_replays(case,policy,reset_inputs=reset_inputs,initialize_outputs=initialize_outputs,
                replay=graph.replay,verify=verify,measure=measure,observe=observe,seed=request['challenge_seed'])
            report['cases'].append(row)
        if recipe is not None:
            report['cases'][-1]['control_distribution']=recipe.proof(a.phase,policy)
            recipe.close()
        del graph,inputs,reference_inputs,golden,output,pristine_inputs,initial_out
        torch.cuda.empty_cache()
    report.update(compiled=True,compiled_specializations=compiled,
                  compilation_kind='current native implementations invoked and synchronized for every frozen case',
                  oracle_order='candidate output/input CPU snapshots before reference GPU computation',
                  native_binding=identity,unresolved_legacy_m=manifest.get('unresolved_legacy_m',[]),
                  full_legacy_coverage=not manifest.get('unresolved_legacy_m'))
    write(a.phase,finalize_report(report,manifest,request))


if __name__=='__main__':
    try: main()
    except BaseException as error:
        target=ROOT/'build'/'FAILURE.json'; target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'status':'error','type':type(error).__name__,'error':str(error)},indent=2)+'\n')
        raise
