"""Unscored smoke-fixture diagnostic using the exact protected whole-MoE runner."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import task_runner as task
import import_fixtures as importer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--variant-root');args=parser.parse_args()
    root=Path(args.capture_root).resolve();fixtures=[]
    for path in root.rglob('fmoe-*.json'):
        f=task.strict_json(path.read_text())
        if f.get('schema')=='served-tensor-fixture-v1' and f.get('family')=='fmoe':fixtures.append((path,f))
    if not fixtures:raise RuntimeError('No actual native whole-MoE smoke fixture found')
    import torch
    torch.set_num_threads(16)
    if args.variant_root:
        import importlib.util
        variant=Path(args.variant_root).resolve();task.validate_sources(variant,task.ROOT)
        spec=importlib.util.spec_from_file_location('diagnostic_variant',variant/'source/kernels.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:module=task.load_source()
    from diagnostic_arrays import save
    array_root=Path(args.output).with_suffix('');array_root.mkdir(parents=True,exist_ok=False)
    sources=task.strict_json((task.ROOT/'provenance/NATIVE-SOURCES.json').read_text());results=[]
    for path,f in fixtures:
        importer.validate_fixture(f,{'provenance':{'run_id':f['provenance']['run_id']}},sources)
        tensors={}
        for phase,role in [('inputs','input'),('outputs','output')]:
            for name,meta in f[phase].items():tensors[name]={'role':role,'shape':meta['shape'],'strides':meta['stride'],'storage_offset':meta['storage_offset'],'dtype':meta['dtype'].removeprefix('torch.'),'device_type':'cuda'}
        controls=dict(f['controls']);controls['port_launch']={'BM':32,'BN':64,'BK':128,'input_quant_block':128,'intermediate_quant_block':128,'output_reduce_block':256}
        case={'case_id':f['case_key'],'occurrences':1,'calls_per_sample':1,'scalars':controls,'tensors':tensors,
              'fixture':{'path':path.name,'sha256':task.file_sha(path)}}
        try:
            diagnostics=[];verification_errors=[]
            def collect(seed,snapshots):diagnostics.append(save(array_root/case['case_id'],seed,snapshots))
            state=task.build_state(case,module,fixture_root=path.parent,diagnostic=collect)
            values,invoke,observe,reset,initialize,verify=state
            compiled=[]
            for seed in (0,1):
                reference=reset(seed);initialize();kernels=invoke();torch.cuda.synchronize()
                try:verify(reference)
                except AssertionError as error:verification_errors.append({'seed':seed,'error':str(error)})
                compiled=[{'name':k.name,'hash':k.hash} for k in kernels]
            results.append({'case':observe(),'status':'PASS' if len(diagnostics)==2 and not verification_errors and all(d['mismatched_outputs']==0 for d in diagnostics) else 'FAIL','compiled_kernels':compiled,'diagnostics':diagnostics,'verification_errors':verification_errors})
        except Exception as error:
            results.append({'case_id':case['case_id'],'status':'FAIL','exception':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
            break
    record={'status':'PASS' if len(results)==len(fixtures) and all(r['status']=='PASS' for r in results) else 'FAIL',
        'synthetic_smoke_only':True,'task_qualified':False,'score_claim':False,'source_sha256':hashlib.sha256((Path(args.variant_root)/'source/kernels.py').read_bytes()).hexdigest() if args.variant_root else task.source_hash(),'array_root':str(array_root),'cases':results}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if record['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
