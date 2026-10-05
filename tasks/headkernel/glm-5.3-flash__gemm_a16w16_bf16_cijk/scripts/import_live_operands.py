"""Attach one exact-ABI native served operand representative to every GEMM case."""
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
    family=f['family'];controls=f['controls']
    if family=='fp8_gemm':
        names={'A':'XQ','B':'WQ','SA':'x_scale','SB':'w_scale'}
        dtype=controls.get('dtype')
        if dtype!={'kind':'dtype','name':'bfloat16'}:return None
    elif family=='bf16_gemm':
        if any(controls.get(k) is not None for k in ('bias','scale_a','scale_b','scale_c')):return None
        if controls.get('otype') not in (None,{'kind':'dtype','name':'bfloat16'}):return None
        if controls.get('tensor_attributes',{}).get('B',{}).get('is_shuffled',False):return None
        names={'A':'A','B':'B'}
    elif family=='aten_bf16_mm':names={'A':'A','B':'B'}
    else:return None
    result={name:dict(f['inputs'][actual]) for name,actual in names.items()};result['C']=dict(f['outputs']['result'])
    if family=='bf16_gemm':
        # The wrapper accepts row-major W[N,K]. The observed aten::mm ABI
        # accepts its same-storage transpose B[K,N]; no data repacking occurs.
        result['B']['shape']=list(reversed(result['B']['shape']));result['B']['stride']=list(reversed(result['B']['stride']))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('capture_manifest');args=parser.parse_args()
    path=Path(args.capture_manifest).resolve();origin=path.parent;rank=strict_json(path.read_text())
    manifest=validate_manifest(strict_json((ROOT/'cases.json').read_text()));native=strict_json((ROOT/'provenance/NATIVE-BASELINE.json').read_text())
    if not rank.get('sealed') or not rank.get('complete') or rank.get('failures') or rank.get('checkpoint_reason') not in ('profile_stop','native_stop_profile'):raise ValueError('Completed native rank0 stop receipt required')
    if rank['provenance']['tp_rank']!=0 or rank['provenance']['image']!=manifest['runtime_image']:raise ValueError('Wrong image/rank')
    selected={};fp8=manifest['cases'][0]['scalars']['fp8']
    for key,ref in sorted(rank['cases'].items()):
        fixture_path=(origin/ref['path']).resolve()
        if not fixture_path.is_relative_to(origin) or file_sha(fixture_path)!=ref['sha256']:raise ValueError('Native fixture changed')
        f=strict_json(fixture_path.read_text())
        if f['served']['stage']!='prefill' or (fp8 and f['family']!='fp8_gemm') or (not fp8 and f['family'] not in ('bf16_gemm','aten_bf16_mm')):continue
        if f['startup_values'] or f['origin']!='served_eager' or f['provenance']['run_id']!=rank['provenance']['run_id']:raise ValueError('Wrong native source/origin')
        if fp8 and f['source_sha256']!=native['aiter_gemm_module_sha256']:raise ValueError('Native FP8 source hash differs')
        bindings=comparable_bindings(f)
        if bindings is None:continue
        for case in manifest['cases']:
            if case['case_id'] in selected:continue
            match=set(bindings)==set(case['tensors'])
            for name,meta in bindings.items():
                expected=case['tensors'][name]
                if meta['shape']!=expected['shape'] or meta['stride']!=expected['strides'] or meta['storage_offset']!=expected['storage_offset'] or meta['dtype'].removeprefix('torch.')!=expected['dtype']:match=False
            if match:selected[case['case_id']]=(fixture_path,f)
    missing={case['case_id'] for case in manifest['cases']}-set(selected)
    if missing:raise ValueError('Missing exact-ABI native operand representatives: '+str(sorted(missing)))
    destination=ROOT/'fixtures';destination.mkdir(exist_ok=True)
    for case in manifest['cases']:
        file,f=selected[case['case_id']];target=destination/(case['case_id']+'.json')
        subprocess.run(['rclone','copyto','--transfers','64000','--progress',str(file),str(target)],check=True)
        for phase in f['payload'].values():
            for group in phase.values():
                for part in group['segments']:
                    source=(origin/part['blob']).resolve();output=destination/part['blob']
                    if not source.is_relative_to(origin) or file_sha(source)!=part['sha256']:raise ValueError('Native blob changed')
                    if not output.exists():subprocess.run(['rclone','copyto','--transfers','64000','--progress',str(source),str(output)],check=True)
                    if file_sha(output)!=part['sha256']:raise ValueError('Staged blob checksum mismatch')
        case['live_fixture']={'path':'fixtures/'+target.name,'sha256':file_sha(target),'native_capture_run':rank['provenance']['run_id'],'source_case_key':f['case_key'],'ABI_preserved':True,'capture_family':f['family']}
    manifest['live_operand_policy']={'required':True,'case_count':len(selected),'selection':'Correctness includes generated even seeds and native captured odd seeds; performance alternates through fresh challenge seeds. Matched native/port diagnostics use only captured values with identical token-row permutations. Expected outputs stay CPU-only.'}
    (ROOT/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(str(len(selected))+' exact-ABI native operand representatives attached; GPU/framework requalification required')
if __name__=='__main__':main()
