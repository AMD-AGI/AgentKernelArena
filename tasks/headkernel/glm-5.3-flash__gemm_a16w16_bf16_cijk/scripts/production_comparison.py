"""Production-operator diagnostics; not a scored Arena phase or candidate selector."""
import importlib
import json
import secrets
from pathlib import Path
import task_runner as task


def main():
    import torch
    manifest=task.validate_manifest(task.strict_json((task.ROOT/'cases.json').read_text()))
    if not torch.cuda.is_available() or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('Requires gfx950 ROCm')
    torch.set_num_threads(16)
    native=None;source=task.load_source();rows=[];policy=manifest['measurement']
    for case in manifest['cases']:
        tensors,_,observe,initialize,verify,reset,reference=task.build_state(case,policy['correctness_seeds'][0],source)
        if case['scalars']['fp8']:
            if native is None:
                module=importlib.import_module('aiter.ops.gemm_op_a8w8')
                provenance=task.strict_json((task.ROOT/'provenance/NATIVE-BASELINE.json').read_text())
                if task.hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=provenance['aiter_gemm_module_sha256']:raise ValueError('Production source differs from pinned image')
                native=module.gemm_a8w8_blockscale_bpreshuffle
            def invoke():return native(tensors['A'],tensors['B'],tensors['SA'],tensors['SB'],dtype=torch.bfloat16,out=tensors['C'])
        else:
            def invoke():return torch.mm(tensors['A'],tensors['B'],out=tensors['C'])
        initialize();invoke();torch.cuda.synchronize();verify(reference)
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
        rows.append(task.checked_replays(case,policy,reset_inputs=reset,initialize_outputs=initialize,replay=graph.replay,verify=verify,measure=measure,observe=observe,seed=secrets.randbelow(2**30)))
    out=task.ROOT/'build/production_comparison.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'schema_version':1,'diagnostic_only':True,'score_input':False,'implementation':'native_production_operator','cases':rows},indent=2,allow_nan=False)+'\n')
    print('Production diagnostics written; excluded from Arena scoring')

if __name__=='__main__':main()
