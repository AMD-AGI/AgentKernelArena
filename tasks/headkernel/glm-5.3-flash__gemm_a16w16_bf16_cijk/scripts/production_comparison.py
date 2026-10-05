"""Protected matched native-versus-candidate diagnostics; never the Arena score."""
import hashlib
import importlib
import json
from pathlib import Path
import secrets
import task_runner as task


def main():
    import torch
    manifest=task.validate_manifest(task.strict_json((task.ROOT/'cases.json').read_text()))
    if any('live_fixture' not in case for case in manifest['cases']):raise RuntimeError('Matched native diagnostics require exact-ABI native operands for every case')
    if not torch.cuda.is_available() or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('Requires gfx950 ROCm')
    torch.set_num_threads(16)
    module=task.load_source();native=None;comparisons=[];policy=manifest['measurement'];challenge=secrets.randbelow(2**29)
    before=task.package_hash()
    for case in manifest['cases']:
        tensors,port,observe,initialize,verify,reset,reference=task.build_state(case,1,module)
        def reset_live(seed):return reset(seed*2+1)
        family=case['live_fixture']['capture_family'];provenance=task.strict_json((task.ROOT/'provenance/NATIVE-BASELINE.json').read_text())
        if family=='fp8_gemm':
            native_module=importlib.import_module('aiter.ops.gemm_op_a8w8')
            if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()!=provenance['source_hashes_by_family'][family]:raise ValueError('Production source differs from pinned image')
            tensors['B'].is_shuffled=True
            def production():
                tensors['C']=native_module.gemm_a8w8_blockscale_bpreshuffle(tensors['A'],tensors['B'],tensors['SA'],tensors['SB'],dtype=torch.bfloat16,out=None)
                return tensors['C']
        elif family=='bf16_gemm':
            native_module=importlib.import_module('aiter.tuned_gemm')
            if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()!=provenance['source_hashes_by_family'][family]:raise ValueError('Production source differs from pinned image')
            def production():
                tensors['C']=native_module.gemm_a16w16(tensors['A'],tensors['B'].t(),bias=None,otype=tensors['A'].dtype,scale_a=None,scale_b=None,scale_c=None)
                return tensors['C']
        elif family=='aten_bf16_mm':
            if hashlib.sha256(str(torch.ops.aten.mm.default._schema).encode()).hexdigest()!=provenance['source_hashes_by_family'][family]:raise ValueError('Pinned ATen schema changed')
            def production():
                tensors['C']=torch.mm(tensors['A'],tensors['B'])
                return tensors['C']
        else:raise ValueError('No native production callable for observed family')
        legs={}
        for label,invoke in [('candidate_port',port),('native_production',production)]:
            reference=reset_live(challenge);initialize();invoke();torch.cuda.synchronize();verify(reference)
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
            legs[label]=task.checked_replays(case,policy,reset_inputs=reset_live,initialize_outputs=initialize,replay=graph.replay,verify=verify,measure=measure,observe=observe,seed=challenge)
        means={label:sum(row['samples_ms'])/len(row['samples_ms']) for label,row in legs.items()}
        ratio=means['native_production']/means['candidate_port']
        comparisons.append({'case_id':case['case_id'],'live_fixture':case['live_fixture'],'native_output_parity':True,
            'identical_captured_ABI_and_fresh_numeric_challenge_sequence':True,'mean_ms':means,'speedup_vs_native':ratio,
            'candidate_faster_than_native':ratio>1.0,'legs':legs})
    if task.package_hash()!=before:raise ValueError('Task changed during native comparison')
    record={'schema_version':1,'status':'ok','diagnostic_only':True,'score_input':False,'source_sha256':task.source_hash(),
        'manifest_sha256':task.fingerprint(manifest),'comparison':'native_mean_ms / candidate_port_mean_ms on identical fresh numerical challenges derived from the actual captured operands; each graph retains its own capture-time output, with no added output copy',
        'case_count':len(comparisons),'cases':comparisons,'all_cases_have_native_parity':True,
        'all_cases_faster_than_native':all(row['candidate_faster_than_native'] for row in comparisons),
        'claim_scope':'Arena port-vs-port speedup measures local optimization only. A ratio greater than1 here is measured isolated native-operator improvement; serving gain still requires end-to-end validation.'}
    path=task.ROOT/'build/native_production_comparison.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    performance=task.ROOT/'build/performance_report.json'
    if performance.is_file():
        report=task.strict_json(performance.read_text());report['native_production_diagnostic']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_cases_have_native_parity':True,'speedup_vs_native_by_case':{row['case_id']:row['speedup_vs_native'] for row in comparisons},'score_input':False}
        performance.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Matched native production comparison complete; raw samples are diagnostic, excluded from Arena scoring')
if __name__=='__main__':main()
