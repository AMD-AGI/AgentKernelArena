"""Protected complete native-MoE versus submitted-port comparison on live fixtures."""
import hashlib
import importlib
import json
from pathlib import Path
import secrets
import task_runner as task
from native_enums import restore_native_enum
from replay_receipts import ReplayReceipts


def main():
    import torch
    manifest=task.validate_manifest(task.strict_json((task.ROOT/'cases.json').read_text()));policy=manifest['measurement']
    if not torch.cuda.is_available() or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('Requires gfx950 ROCm')
    torch.set_num_threads(16);module=task.load_source()
    native=importlib.import_module('aiter.fused_moe');sources=task.strict_json((task.ROOT/'provenance/NATIVE-SOURCES.json').read_text())
    if task.file_sha(native.__file__)!=sources['files']['aiter/fused_moe.py']:raise ValueError('Native whole-MoE source differs from pinned image')
    from aiter import ActivationType,QuantType
    before=task.package_hash();challenge=secrets.randbelow(2**29);comparisons=[]
    receipts=ReplayReceipts(task.ROOT/'build',challenge,task.source_hash())
    for case in manifest['cases']:
        tensors,port,observe,reset,initialize,verify=task.build_state(case,module)
        port_output=tensors['result'];controls=dict(case['scalars']);controls.pop('port_launch');controls.pop('tensor_attributes')
        for name,cls in [('activation',ActivationType),('quant_type',QuantType)]:
            value=controls[name];controls[name]=restore_native_enum(value,cls.__qualname__,cls) if isinstance(value,dict) else cls(value)
        if isinstance(controls.get('dtype'),dict):controls['dtype']=getattr(torch,controls['dtype']['name'])
        def production():
            arguments={name:value for name,value in tensors.items() if name!='result'};arguments.update(controls)
            tensors['result']=native.fused_moe(**arguments)
            return tensors['result']
        legs={}
        for label,invoke in [('candidate_port',port),('native_production',production)]:
            if label=='candidate_port':tensors['result']=port_output
            replay_reset,replay_verify=receipts.leg(case['case_id'],label,reset,verify)
            reference=reset(challenge);initialize();invoke();torch.cuda.synchronize();replay_verify(reference)
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
            legs[label]=task.checked_replays(case,policy,reset_inputs=replay_reset,initialize_outputs=initialize,replay=graph.replay,verify=replay_verify,measure=measure,observe=observe,seed=challenge)
        means={name:sum(row['samples_ms'])/len(row['samples_ms']) for name,row in legs.items()};ratio=means['native_production']/means['candidate_port']
        comparisons.append({'case_id':case['case_id'],'live_fixture':case['fixture'],'native_output_parity':True,
            'identical_captured_ABI_and_fresh_numeric_challenge_sequence':True,'mean_ms':means,'speedup_vs_native':ratio,'candidate_faster_than_native':ratio>1.0,'legs':legs})
    if task.package_hash()!=before:raise ValueError('Task changed during native comparison')
    record={'schema_version':1,'status':'ok','diagnostic_only':True,'score_input':False,'source_sha256':task.source_hash(),
        'manifest_sha256':task.fingerprint(manifest),'cases':comparisons,'all_cases_have_native_parity':True,
        'challenge_seed':challenge,'replay_receipts':receipts.path.name,
        'replay_receipts_sha256':task.file_sha(receipts.path),
        'all_cases_faster_than_native':all(row['candidate_faster_than_native'] for row in comparisons),
        'allocation_boundary':'Both graph legs retain their own preallocated capture-time output and workspace. Native fused_moe has no out argument; no output copy is added to its timed graph.',
        'claim_scope':'The Arena score compares frozen and edited versions of the replacement port. It is local optimization only. native_mean_ms/candidate_mean_ms greater than1 demonstrates isolated native-operator improvement; serving gain requires a model rerun.'}
    path=task.ROOT/'build/native_production_comparison.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    performance=task.ROOT/'build/performance_report.json'
    if performance.is_file():
        report=task.strict_json(performance.read_text());report['native_production_diagnostic']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_cases_have_native_parity':True,'speedup_vs_native_by_case':{row['case_id']:row['speedup_vs_native'] for row in comparisons},'score_input':False}
        performance.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Matched whole-native-MoE comparison complete; all samples and native ratios reported separately from Arena score')
if __name__=='__main__':main()
