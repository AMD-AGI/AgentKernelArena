"""Import every notified dense or FP8 GEMM record from a finalized native capture."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import strict_json,validate_manifest
from fixture_codec import file_sha


def comparable_bindings(f):
    family=f['family'];controls=dict(f['controls']);controls.pop('capture_native_dispatch',None)
    if family=='fp8_gemm':
        names={'A':'XQ','B':'WQ','SA':'x_scale','SB':'w_scale'}
        if controls!={'dtype':{'kind':'dtype','name':'bfloat16'},'out':None,'tensor_attributes':{'WQ':{'is_shuffled':True},'XQ':{},'x_scale':{},'w_scale':{}}}:raise ValueError('Unsupported observed FP8 controls')
    elif family=='bf16_gemm':
        names={'A':'A','B':'B'}
        dtype=f['inputs']['A']['dtype'].removeprefix('torch.')
        if dtype not in ('bfloat16','float32'):raise ValueError('Unsupported observed dense dtype')
        expected={'bias':None,'scale_a':None,'scale_b':None,'scale_c':None,'otype':{'kind':'dtype','name':dtype},'tensor_attributes':{'A':{},'B':{}}}
        if controls.get('otype') is None:expected['otype']=None
        if controls!=expected:raise ValueError('Unsupported observed dense controls')
    elif family=='aten_bf16_mm':
        names={'A':'A','B':'B'}
        expected={'out':controls.get('out'),'tensor_attributes':{'A':{},'B':{}}}
        if controls!=expected or controls['out'] not in (None,{'kind':'output_binding','name':'result'}):raise ValueError('Unsupported observed aten controls')
    else:raise ValueError('Not a GEMM family')
    if set(f['inputs'])!=set(names.values()) or set(f['outputs'])!={'result'}:raise ValueError('Unexpected tensor bindings')
    result={name:dict(f['inputs'][actual]) for name,actual in names.items()};result['C']=dict(f['outputs']['result'])
    if family=='bf16_gemm':
        # Native W[N,K] becomes its same-storage view B[K,N], without repacking.
        result['B']['shape']=list(reversed(result['B']['shape']));result['B']['stride']=list(reversed(result['B']['stride']))
    m,k=result['A']['shape'];n=result['C']['shape'][1];fp8=family=='fp8_gemm';dtype=result['A']['dtype'].removeprefix('torch.')
    if min(m,n,k)<=0:raise ValueError('Empty GEMM dimensions need an explicit disposition')
    layouts={'A':([m,k],[k,1],dtype),'B':([n,k],[k,1],dtype) if fp8 else ([k,n],[1,k],dtype),'C':([m,n],[n,1],'bfloat16' if fp8 else dtype)}
    if fp8:
        if dtype!='float8_e4m3fn' or k%128 or n%128:raise ValueError('Unsupported FP8 block geometry')
        layouts.update(SA=([m,k//128],[1,m],'float32'),SB=([n//128,k//128],[k//128,1],'float32'))
    for name,(shape,stride,dtype) in layouts.items():
        actual=result[name]
        if actual['shape']!=shape or actual['stride']!=stride or type(actual['storage_offset']) is not int or actual['storage_offset']<0 or actual['dtype'].removeprefix('torch.')!=dtype:raise ValueError('Unsupported observed ABI: '+name)
    if len({meta['alias'] for meta in result.values()})!=len(result):raise ValueError('Observed aliased GEMM needs an explicit contract')
    return result


def pinned(path,checksum):
    if not path.is_file() or file_sha(path)!=checksum:raise ValueError('Frozen capture receipt changed: '+path.name)
    return strict_json(path.read_text())


def collect(path,fp8,native,ready_path):
    origin=path.parent;rank=strict_json(path.read_text());run=ready_path.parent
    ready=strict_json(ready_path.read_text())
    if ready['image']!=native['runtime_image']:raise ValueError('Capture image differs from pinned task image')
    if not ready.get('source_capture_complete') or (ready['completed_requests'],ready['total_input_tokens'],ready['total_output_tokens'])!=(64,524288,65536):raise ValueError('Finalized exact64-request capture required')
    pin=ready['source_capture_verified_ref'];verified=pinned(run/'SOURCE-CAPTURE-VERIFIED.json',pin['sha256'])
    if not verified.get('complete') or set(verified['ranks'])!={str(i) for i in range(8)}:raise ValueError('Complete all-rank receipts required')
    ranks={}
    for r,pin in verified['ranks'].items():
        p=Path(pin['path']);p=p if p.is_absolute() else run/p
        if not p.resolve().is_relative_to(run):raise ValueError('Rank receipt escapes capture')
        rr=pinned(p,pin['sha256'])
        if not rr['sealed'] or (not rr['complete'] if r=='0' else not rr['metadata_only']) or rr['failures'] or rr.get('additional_failure_count',0) or not rr['required_cases_supplied'] or rr['checkpoint_reason'] not in ('profile_stop','native_stop_profile'):raise ValueError('Unsealed/incomplete rank')
        if rr['provenance']['tp_rank']!=int(r) or rr['provenance']['image']!=ready['image'] or rr['provenance']['run_id']!=verified['run_id']:raise ValueError('Rank identity changed')
        if set(rr['required_cases'])!=set(rr['case_schemas']) or set(rr['runtime_case_counts'])!=set(rr['case_schemas']):raise ValueError('Incomplete notification inventory')
        ranks[r]=rr
    if rank!=ranks['0']:raise ValueError('Importer requires finalized rank0 receipt')
    if rank['provenance'].get('model') not in ('Kimi-K3','Kimi-K3-Instruct','kimi-k3'):raise ValueError('This is not a finalized Kimi capture')
    families={'fp8_gemm'} if fp8 else {'bf16_gemm','aten_bf16_mm'}
    keys={key for key,schema in rank['case_schemas'].items() if schema['family'] in families}
    if not keys or not keys<=set(rank['cases']):raise ValueError('A notified GEMM has no actual operand fixture')
    for r,rr in ranks.items():
        if {k for k,s in rr['case_schemas'].items() if s['family'] in families}!=keys:raise ValueError('Rank-specific GEMM ABI requires additional fixtures: '+r)
    selected=[]
    for key in sorted(keys):
        ref=rank['cases'][key];file=(origin/ref['path']).resolve()
        if not file.is_relative_to(origin):raise ValueError('Fixture escapes source')
        f=pinned(file,ref['sha256']);family=f['family'];served=f['served']
        if f['case_key']!=key or f['startup_values'] or f['origin'] not in ('served_eager','served_graph') or f['provenance']!=rank['provenance'] or served['tp_rank']!=0 or served['run_id']!=verified['run_id']:raise ValueError('Wrong actual served source')
        if served['stage'] not in ('prefill','decode') or not 1<=served['active_requests']<=64:raise ValueError('Unsupported actual stage')
        accepted=native['source_hashes_by_family'][family];accepted=accepted if isinstance(accepted,list) else [accepted]
        if f['source_sha256'] not in accepted:raise ValueError('Pinned native module/schema differs')
        bindings=comparable_bindings(f)
        if family=='bf16_gemm':
            dispatch=f['controls'].get('capture_native_dispatch')
            if not isinstance(dispatch,dict) or dispatch.get('schema')!='aiter-solmap-dispatch-v1' or dispatch.get('libtype') not in ('torch','hipblaslt','skinny','asm','triton','flydsl','opus') or 'solidx' not in dispatch or not isinstance(dispatch.get('config'),dict) or set(dispatch.get('callable',{}))!={'module','qualname'}:raise ValueError('Missing complete actual solMap dispatch')
        counts={r:rr['runtime_case_counts'][key] for r,rr in ranks.items()}
        if any(type(v) is not int or v<=0 for v in counts.values()):raise ValueError('Invalid actual count')
        selected.append((file,f,bindings,counts))
    stages={f['served']['stage'] for _,f,_,_ in selected}
    if stages!={'prefill','decode'}:raise ValueError('Both actual served stages are required')
    if {f['served']['stage'] for _,f,_,_ in selected if f['family']=='bf16_gemm'}!={'prefill','decode'}:raise ValueError('Native wrapper must be captured in both stages')
    return rank,selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('capture_manifest');parser.add_argument('--capture-ready',required=True);args=parser.parse_args()
    path=Path(args.capture_manifest).resolve();origin=path.parent
    requirements=strict_json((ROOT/'CASE-REQUIREMENTS.json').read_text());native=strict_json((ROOT/'provenance/NATIVE-BASELINE.json').read_text())
    fp8=False;rank,selected=collect(path,fp8,native,Path(args.capture_ready).resolve())
    destination=ROOT/'fixtures';destination.mkdir(exist_ok=True);cases=[]
    def copy(source,target):subprocess.run(['rclone','copyto','--transfers','64000','--progress','--buffer-size','0',str(source),str(target)],check=True)
    for file,f,bindings,counts in selected:
        m,k=bindings['A']['shape'];n=bindings['C']['shape'][1];dtype=bindings['A']['dtype'].removeprefix('torch.')
        case_id=('fp8' if fp8 else dtype)+'-'+f['served']['stage']+'-m'+str(m)+'-n'+str(n)+'-k'+str(k)+'-'+f['case_key']
        target=destination/(case_id+'.json');copy(file,target)
        for phase in f['payload'].values():
            for group in phase.values():
                for part in group['segments']:
                    source=(origin/part['blob']).resolve();output=destination/part['blob']
                    if not source.is_relative_to(origin) or file_sha(source)!=part['sha256']:raise ValueError('Native blob changed')
                    if not output.exists():copy(source,output)
                    if file_sha(output)!=part['sha256']:raise ValueError('Staged blob checksum mismatch')
        tensors={name:{'role':'output' if name=='C' else 'input','shape':meta['shape'],'strides':meta['stride'],'storage_offset':meta['storage_offset'],'dtype':meta['dtype'].removeprefix('torch.'),'device_type':'cuda'} for name,meta in bindings.items()}
        cases.append({'case_id':case_id,'occurrences':counts['0'],'calls_per_sample':1,'scalars':{'M':m,'N':n,'K':k,'fp8':fp8,'BM':32,'BN':64,'BK':128},'tensors':tensors,
            'live_fixture':{'path':'fixtures/'+target.name,'sha256':file_sha(target),'native_capture_run':rank['provenance']['run_id'],'source_case_key':f['case_key'],'ABI_preserved':True,'capture_family':f['family'],'source_sha256':f['source_sha256']},
            'capture_controls':f['controls'],'served_geometry':f['served'],'frequency_evidence':{'source_case_key':f['case_key'],'per_rank_counts':counts,'scope':'all notified actual served calls in the complete64-request workload; one numerical representative per structural record','capture_origin':f['origin'],'graph_bucket':f['graph_bucket']}})
    manifest=validate_manifest({'schema_version':1,'runtime_image':rank['provenance']['image'],'source_run':rank['provenance']['run_id'],'cases':cases,'measurement':requirements['measurement'],
        'capture_stop_manifest_sha256':file_sha(path),'coverage':'Every actual notified GEMM record in this task family is retained, including decode, FP32 dense records and distinct equal-shape source records. Unsupported controls or ABIs fail intake.',
        'live_operand_policy':{'required':True,'case_count':len(cases),'selection':'Even seeds use generated numeric challenges; odd seeds perturb each actual activation numerically and preserve real weights/scales/layout. Native diagnostics use the same fresh numeric sequence. CPU FP32 oracle only; fixed captured output is never the replay oracle.'}})
    (ROOT/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'NOT_BUILT').unlink(missing_ok=True)
    print(str(len(cases))+' actual GEMM records imported without stage/dtype filtering; fresh GPU/framework qualification required')
if __name__=='__main__':main()
