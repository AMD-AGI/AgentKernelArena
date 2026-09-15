"""CPU contract regressions; synthetic measurements below are not GPU validation."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
VLLM = sorted((ROOT/'tasks/triton2triton/vllm').glob('*/config.yaml'))
BASE = '5c9f8ef2'


def module_at(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    spec = importlib.util.spec_from_file_location('_isolated_task_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_record(result):
    return parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result), role=result['role'],
                                action=result['action'], returncode=0 if result['status']=='PASS' else 1)


@pytest.mark.parametrize('path', VLLM, ids=lambda p:p.parent.name)
def test_vllm_v2_preserves_all_original_cases_checks_sources_and_helpers(path):
    task = path.parent
    spec = load_task_spec(path, task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state == 'implemented'
    assert spec.candidate.language == 'triton' and spec.baseline.kind == 'initial_candidate'
    relative = (task/'scripts/task_runner.py').relative_to(ROOT).as_posix()
    before = subprocess.check_output(['git','show',f'{BASE}:{relative}'], cwd=ROOT, text=True)
    after = (ROOT/relative).read_text()
    bt, at = ast.parse(before), ast.parse(after)
    bf = {n.name:n for n in bt.body if isinstance(n,ast.FunctionDef)}
    af = {n.name:n for n in at.body if isinstance(n,ast.FunctionDef)}
    correction = af['run_correctness']
    correction.args = deepcopy(bf['run_correctness'].args)
    loop = next(n for n in ast.walk(correction) if isinstance(n,ast.For))
    assert ast.unparse(loop.body[0].test).startswith('case_index is not None')
    loop.body.pop(0)
    assert ast.dump(correction, include_attributes=False) == ast.dump(bf['run_correctness'], include_attributes=False)
    for name in bf.keys()-{'run_correctness'}:
        assert ast.get_source_segment(before,bf[name]) == ast.get_source_segment(after,af[name])
    manifest = json.loads((task/'workloads.json').read_text())
    assert sum('performance' in row['checks'] for row in manifest['cases']) == 5
    assert all('correctness' in row['checks'] for row in manifest['cases'])
    assert manifest['migration']['original_harness_sha256'] == hashlib.sha256(before.encode()).hexdigest()
    for edit in spec.candidate.editable:
        source = task/edit.path
        original = subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
        assert source.read_bytes() == original
    assert 'Evaluation contract' in (task/'README.md').read_text()


def test_vllm_per_case_failure_and_incomplete_measurement_rejected(monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py', monkeypatch)
    manifest = adapter.load_manifest()
    harness = SimpleNamespace(**{manifest['case_table']:manifest['input_table']})
    harness.run_correctness = lambda case_index: (case_index != 3, 'negative control')
    harness.run_performance = lambda: []
    monkeypatch.setattr(adapter,'load_harness',lambda:harness)
    correctness = adapter.evaluate('candidate','correctness')
    assert correctness['status']=='FAIL'
    assert [r['status'] for r in correctness['cases']]==['PASS','PASS','PASS','FAIL','PASS']
    manifest_result = {'protocol':'arena-eval-v1','role':'task','action':'validate-task',
                       'status':'PASS','cases':manifest['cases']}
    CaseManifest.from_result(result_record(manifest_result)).validate(result_record(correctness))
    perf = adapter.evaluate('candidate','performance')
    assert perf['status']=='FAIL' and all(r['status']=='FAIL' for r in perf['cases'])
    assert perf['failure_kind'] != 'numerical_mismatch'


def test_vllm_candidate_stub_is_never_baseline_fallback(tmp_path,monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py',monkeypatch)
    data = adapter.load_manifest()
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    for source, targets in data['candidate_symbols'].items():
        p=tmp_path/source;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join('@triton.jit\ndef '+t['name']+'():\n    pass\n' for t in targets))
    assert adapter.inspect_candidate(data)=='unimplemented'
    with pytest.raises(ValueError,match='no baseline fallback'):
        adapter.inspect_candidate(data,require_implemented=True)



def test_staged_write_reference_known_answer_and_untouched_cells(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_apply_write'
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    initial = torch.full((3, 5), -7, dtype=torch.int32)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    starts = torch.tensor([1, 3], dtype=torch.int32)
    contents = torch.tensor([10, 11, 20], dtype=torch.int32)
    cumulative = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.tensor([[-7,-7,-7,20,-7],[-7,-7,-7,-7,-7],[-7,10,11,-7,-7]], dtype=torch.int32)
    torch.testing.assert_close(harness.reference_apply_write(initial,indices,starts,contents,cumulative),expected)
    checks.check_output(expected,initial,indices,starts,contents,cumulative,harness.reference_apply_write)
    invalid = expected.clone(); invalid[1,0] = 0
    with pytest.raises(AssertionError,match='untouched cells'):
        checks.check_output(invalid,initial,indices,starts,contents,cumulative,harness.reference_apply_write)
    with pytest.raises(AssertionError,match='dtype'):
        checks.check_output(expected.float(),initial,indices,starts,contents,cumulative,harness.reference_apply_write)


@pytest.mark.parametrize('mode', ['correct','incorrect_timed','stale','no_write','uniform_segments','zero_starts','identity_mapping'])
def test_staged_write_replay_checks_changed_mapping_offsets_and_lengths(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_apply_write'
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    output = torch.zeros(6, 12, dtype=torch.int32)
    output_work = output.clone()
    write_indices = torch.arange(4, dtype=torch.int32)
    write_starts = torch.zeros(4, dtype=torch.int32)
    write_contents = torch.arange(1, 13, dtype=torch.int32)
    cu_lens = torch.tensor([3,6,9,12], dtype=torch.int32)
    original = [v.clone() for v in (write_indices,write_starts,write_contents,cu_lens)]
    def fn():
        output_work.copy_(harness.reference_apply_write(output_work,write_indices,write_starts,write_contents,cu_lens))
    def prepare():
        output_work.copy_(output)
    observed = []
    def benchmark(measured, *, timed_run, **kwargs):
        observed.append({k:v for k,v in kwargs.items() if k != 'prepare_fn'})
        for actual,expected in zip((write_indices,write_starts,write_contents,cu_lens),original):
            torch.testing.assert_close(actual,expected)
        kwargs['prepare_fn'](); captured = measured(); cached = captured.clone()
        if mode == 'incorrect_timed': captured.zero_()
        def replay():
            kwargs['prepare_fn']()
            indices = original[0] if mode == 'identity_mapping' else write_indices
            starts = original[1] if mode == 'zero_starts' else write_starts
            cumulative = original[3] if mode == 'uniform_segments' else cu_lens
            if mode == 'stale': captured.copy_(cached)
            elif mode != 'no_write':
                captured.copy_(harness.reference_apply_write(output,indices,starts,write_contents,cumulative))
            return captured
        timed_run.outputs = captured; timed_run.rerun = replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    options = dict(warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)
    if mode == 'correct':
        ms,metadata = checks.checked_benchmark(harness,benchmark,fn,**options)
        assert ms == 0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError): checks.checked_benchmark(harness,benchmark,fn,**options)
    assert observed == [dict(warmup=10,repetition=100,target_ms=20.0)]


def test_staged_write_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_apply_write/_arena_eval.py',monkeypatch)
    assert adapter.load_harness().run_performance.__module__ == '_staged_write_checks'


@pytest.mark.parametrize('mode', ['correct','incorrect_timed','stale','no_write','omit_last','truncate_tail','wrong_source',
                                 'zero_source_and_dest_timed','zero_source_and_dest_replay',
                                 'mutate_source_only_timed','mutate_source_only_replay'])
def test_batch_memcpy_checks_every_timed_destination_and_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    checks = module_at(task/'_arena_checks.py',monkeypatch)
    harness = SimpleNamespace(_TimedRun=SimpleNamespace)
    sources = [torch.tensor([1,2,3],dtype=torch.uint8), torch.tensor([11,12,13,14,15],dtype=torch.uint8)]
    initial_sources = [source.clone() for source in sources]
    destinations = [torch.zeros_like(value) for value in sources]
    def fn():
        for dst,src in zip(destinations,sources): dst.copy_(src)
    observed = []
    def benchmark(measured, *, timed_run, **kwargs):
        observed.append(kwargs)
        for source, initial in zip(sources,initial_sources):
            assert torch.equal(source,initial)
        outputs = measured(); cached = [value.clone() for value in outputs]
        if mode == 'incorrect_timed': outputs[-1].zero_()
        if mode == 'zero_source_and_dest_timed':
            sources[-1].zero_(); outputs[-1].zero_()
        if mode == 'mutate_source_only_timed': sources[-1].zero_()
        def replay():
            for i,(out,src) in enumerate(zip(outputs,sources)):
                if mode == 'no_write' or mode == 'omit_last' and i == len(outputs)-1: continue
                if mode == 'stale': out.copy_(cached[i])
                elif mode == 'truncate_tail': out[:-1].copy_(src[:-1])
                elif mode == 'wrong_source': out.fill_(sources[1-i][0])
                else: out.copy_(src)
            if mode == 'zero_source_and_dest_replay':
                sources[-1].zero_(); outputs[-1].zero_()
            if mode == 'mutate_source_only_replay': sources[-1].zero_()
            return outputs
        timed_run.outputs = outputs; timed_run.rerun = replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    if mode == 'correct':
        ms,metadata=checks.checked_benchmark(harness,benchmark,fn,sources,destinations,warmup=10,repetition=100)
        assert ms==0.25 and metadata['perturbed_input_replay_checked'] and metadata['source_buffers_unchanged']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness,benchmark,fn,sources,destinations,warmup=10,repetition=100)
    assert observed == [dict(warmup=10,repetition=100)]


def test_batch_memcpy_retains_original_inputs_and_restores_hooks(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    checks = module_at(task/'_arena_checks.py',monkeypatch)
    sources=[torch.tensor([1,255,0],dtype=torch.uint8)]
    destinations=[torch.zeros_like(sources[0])]
    sentinel=(sources,destinations,object(),object(),object())
    calls=[]
    def make_inputs(*args,**kwargs): calls.append((args,kwargs));return sentinel
    benchmark=object()
    h=SimpleNamespace(make_inputs=make_inputs,_benchmark_cuda_graph_or_events=benchmark,
                      run_correctness=lambda **kwargs:(True,None))
    def run():
        assert h.make_inputs(4,8,device='cuda') is sentinel
        return h._benchmark_cuda_graph_or_events('protected-fn',warmup=10,repetition=100)
    def checked(actual_h,actual_b,fn,src,dst,**kwargs):
        assert actual_h is h and actual_b is benchmark and fn=='protected-fn'
        assert src is sources and dst is destinations
        assert kwargs==dict(warmup=10,repetition=100)
        raise AssertionError('simulated output mismatch')
    h.run_performance=run;monkeypatch.setattr(checks,'checked_benchmark',checked)
    checks.install(h)
    with pytest.raises(AssertionError,match='simulated output mismatch'): h.run_performance()
    assert h.make_inputs is make_inputs and h._benchmark_cuda_graph_or_events is benchmark
    assert calls==[((4,8),dict(device='cuda'))]


def test_batch_memcpy_adapter_installs_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_performance.__module__=='_batch_memcpy_checks'
    assert harness.run_correctness.__module__=='_batch_memcpy_checks'


@pytest.mark.parametrize('case_index', [None, 0, 4])
@pytest.mark.parametrize('mode', ['correct','zero_source_and_dest','mutate_source_only','incorrect_output'])
def test_batch_memcpy_original_correctness_uses_pristine_sources(monkeypatch, case_index, mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    original_inputs=harness.make_inputs
    current=[]
    seen=[]
    def make_inputs(batch,size,device):
        assert device=='cuda'
        inputs=original_inputs(batch,size,device='cpu')
        current[:]=inputs[:2]
        seen.append((batch,size))
        return inputs
    def candidate(src_ptrs,dst_ptrs,sizes):
        sources,destinations=current
        for source,destination in zip(sources,destinations): destination.copy_(source)
        if mode=='zero_source_and_dest':
            sources[-1].zero_(); destinations[-1].zero_()
        elif mode=='mutate_source_only': sources[-1].zero_()
        elif mode=='incorrect_output': destinations[-1].zero_()
    monkeypatch.setattr(harness,'make_inputs',make_inputs)
    monkeypatch.setattr(harness,'load_module',lambda:SimpleNamespace(batch_memcpy=candidate))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    checks.install(harness)
    ok,error=harness.run_correctness(case_index=case_index)
    assert ok is (mode=='correct')
    if mode=='zero_source_and_dest': assert 'caller-owned source' in error
    if mode=='correct' or mode=='zero_source_and_dest':
        assert seen==[value for i,value in enumerate(harness.TEST_SHAPES)
                      if case_index is None or case_index==i]
    assert harness.make_inputs is make_inputs

def bmm_known_inputs():
    a=torch.tensor([[[1.,2.],[3.,4.]],[[-1.,2.],[0.,3.]]],dtype=torch.float16)
    b=torch.tensor([[[5.,6.],[7.,8.]],[[4.,2.],[6.,1.]]],dtype=torch.float16)
    expected=torch.tensor([[[19.,22.],[43.,50.]],[[8.,0.],[18.,3.]]],dtype=torch.float16)
    return a,b,expected


@pytest.mark.parametrize('mode',['correct','zero_source_and_output','wrong_batch','shape','dtype','device','nonfinite'])
def test_bmm_correctness_preserves_inputs_and_full_output_contract(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_checks.py',monkeypatch)
    a,b,expected=bmm_known_inputs()
    torch.testing.assert_close(checks.reference(a,b),expected,atol=0,rtol=0)
    def candidate(a,b):
        output=expected.clone()
        if mode=='zero_source_and_output':a.zero_();output.zero_()
        elif mode=='wrong_batch':output[1].copy_(output[0])
        elif mode=='shape':output=output[:1]
        elif mode=='dtype':output=output.float()
        elif mode=='device':output=torch.empty_like(output,device='meta')
        elif mode=='nonfinite':output.fill_(float('inf'))
        return output
    module=SimpleNamespace(bmm_triton=candidate)
    load=lambda:module
    harness=SimpleNamespace(load_module=load)
    with checks.checked_modules(harness):
        if mode=='correct':torch.testing.assert_close(harness.load_module().bmm_triton(a,b),expected)
        else:
            with pytest.raises(AssertionError):harness.load_module().bmm_triton(a,b)
    assert harness.load_module is load and module.bmm_triton is candidate


@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong',
                                'zero_source_and_output_timed','zero_source_and_output_replay'])
def test_bmm_exact_timed_replay_and_pristine_input_oracles(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_checks.py',monkeypatch)
    a,b,expected=bmm_known_inputs()
    mod=SimpleNamespace(bmm_triton=checks.reference)
    original=mod.bmm_triton
    initial=(a.clone(),b.clone())
    harness=SimpleNamespace(_TimedRun=SimpleNamespace)
    observed=[]
    def fn():mod.bmm_triton(a,b)
    def benchmark(measured,*,timed_run,**kwargs):
        observed.append(kwargs)
        for value,saved in zip((a,b),initial):assert torch.equal(value,saved)
        output=measured();cached=output.clone()
        torch.testing.assert_close(output,expected)
        if mode=='incorrect_timed':output.zero_()
        if mode=='zero_source_and_output_timed':a.zero_();output.zero_()
        def replay():
            if mode=='correct':output.copy_(checks.reference(a,b))
            elif mode=='stale':output.copy_(cached)
            elif mode=='changing_wrong':output.fill_(a.flatten()[0])
            elif mode=='zero_source_and_output_replay':a.zero_();output.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    if mode=='correct':
        ms,metadata=checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
        assert ms==0.25 and metadata['perturbed_input_replay_checked'] and metadata['source_buffers_unchanged']
    else:
        with pytest.raises(AssertionError):checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    assert observed==[dict(warmup=10,repetition=100)] and mod.bmm_triton is original


def test_bmm_adapter_installs_correctness_and_timed_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_bmm_checks'


def test_rms_reference_has_independent_known_answers(monkeypatch):
    runner = ROOT/'tasks/triton2triton/vllm/triton_rms_norm/scripts/task_runner.py'
    harness = module_at(runner,monkeypatch)
    x = torch.tensor([[3.,4.],[0.,0.]])
    w = torch.tensor([2.,0.5])
    expected = torch.tensor([[6./(12.5+1e-6)**0.5, 2./(12.5+1e-6)**0.5],[0.,0.]])
    result = harness.reference_rms_norm(x,w)
    torch.testing.assert_close(result,expected)
    assert not torch.allclose(result,torch.ones_like(result),atol=1e-2,rtol=1e-2)


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'device', 'nonfinite', 'values'])
def test_rms_output_contract_rejects_invalid_outputs(monkeypatch, invalid):
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    expected = harness.reference_rms_norm(x, weight)
    checks.check_output(expected, x, weight, 1e-6, harness.reference_rms_norm)
    output = {'shape': expected[:1], 'dtype': expected.double(),
              'device': torch.empty_like(expected, device='meta'),
              'nonfinite': torch.full_like(expected, float('inf')),
              'values': torch.zeros_like(expected)}[invalid]
    with pytest.raises(AssertionError):
        checks.check_output(output, x, weight, 1e-6, harness.reference_rms_norm)


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_rms_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    """CPU simulation of captured buffers; this does not claim GPU execution."""
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    eps = 1e-6
    mod = SimpleNamespace(rms_norm=harness.reference_rms_norm)
    original = mod.rms_norm
    seen = []

    def fn():
        # Match the original harness, including its discarded return value.
        mod.rms_norm(x, weight, eps=eps)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(harness.reference_rms_norm(x, weight, eps))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.rms_norm is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


def test_rms_adapter_installs_output_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_rms_norm/_arena_eval.py', monkeypatch)
    harness = adapter.load_harness()
    assert harness.run_correctness.__module__ == '_rms_output_checks'
    assert harness.run_performance.__module__ == '_rms_output_checks'


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_geak_gemm_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    x = torch.tensor([[1., 2.], [3., 4.]])
    w = torch.tensor([[2., 1.], [4., 3.]])
    bias = torch.tensor([1., 2.])
    expected = torch.tensor([[5., 12.], [11., 26.]])
    checks.check_output(expected, x, w, bias)
    metadata = {'benchmark_method': 'cuda_graph'}
    seen = []

    def fn():
        return torch.nn.functional.linear(x, w, bias)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(torch.nn.functional.linear(x, w, bias))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.5, metadata

    if mode == 'correct':
        ms, returned = checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and returned is metadata
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert 'perturbed_input_replay_checked' not in metadata
    assert seen == [{'warmup': 50, 'repetition': 200}]


def feed_forward_reference():
    path=ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/test_kernel_harness.py'
    tree=ast.parse(path.read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference_ff')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['reference_ff']


def test_feed_forward_reference_independent_known_answer():
    import math
    reference=feed_forward_reference()
    x=torch.tensor([[[3.,4.]]]);rms_w=torch.tensor([2.,0.5]);w=torch.diag(torch.tensor([1.,-4.]))
    a=6./math.sqrt(12.5+1e-6);b=-8./math.sqrt(12.5+1e-6)
    expected=torch.tensor([[[a*a/(1+math.exp(-a)),b*b/(1+math.exp(-b))]]])
    torch.testing.assert_close(reference(x,w,w,rms_w),expected)


@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong'])
def test_feed_forward_timed_replay_restores_diagnostic_peer_inputs(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/_arena_checks.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(TimedRun=SimpleNamespace))
    h=SimpleNamespace(reference_ff=feed_forward_reference())
    x=torch.tensor([[[3.,4.]]]);rms_w=torch.tensor([2.,0.5])
    w1=torch.diag(torch.tensor([1.,-4.]));w3=w1.clone()
    original_x,original_rms=x.clone(),rms_w.clone()
    def fn(): return h.reference_ff(x,w1,w3,rms_w)
    seen=[]
    def benchmark(measured,*,timed_run,**kwargs):
        seen.append(kwargs);output=measured();cached=output.clone()
        if mode=='incorrect_timed':output.zero_()
        def replay():
            if mode=='correct':output.copy_(h.reference_ff(x,w1,w3,rms_w))
            elif mode=='stale':output.copy_(cached)
            elif mode=='changing_wrong':output.fill_(x.flatten()[0])
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    if mode=='correct':
        ms,metadata=checks.checked_benchmark(h,benchmark,fn,warmup=50,repetition=200)
        assert ms==0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError): checks.checked_benchmark(h,benchmark,fn,warmup=50,repetition=200)
    torch.testing.assert_close(x,original_x,atol=0,rtol=0)
    torch.testing.assert_close(rms_w,original_rms,atol=0,rtol=0)
    assert seen==[dict(warmup=50,repetition=200)]


def test_feed_forward_diagnostic_reference_remains_original_call(monkeypatch):
    checks=module_at(ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/_arena_checks.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(TimedRun=SimpleNamespace))
    ref_fn=lambda: 'diagnostic output'
    fn=lambda:ref_fn()
    seen=[]
    def benchmark(actual,**kwargs):
        assert actual is fn
        seen.append(kwargs)
        return 0.5,dict(benchmark_method='cuda_graph')
    assert checks.checked_benchmark(None,benchmark,fn,warmup=50,repetition=200)[0]==0.5
    assert seen==[dict(warmup=50,repetition=200)]


@pytest.mark.parametrize('value', [1., float('inf'), float('nan')])
def test_geak_gemm_correctness_rejects_nonfinite_output(monkeypatch, value):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    original = lambda: torch.tensor([[value]])
    harness = SimpleNamespace(gemm_a16w16=original)
    visited = []

    def run(indices):
        visited.extend(indices)
        # Even a harness comparing matching infinities must not bypass the gate.
        torch.testing.assert_close(harness.gemm_a16w16(), original(), equal_nan=True)

    harness.run_correctness = run
    if value == 1.:
        assert checks.checked_correctness(harness, [0, 1, 2]) is None
    else:
        with pytest.raises(AssertionError, match='finite tensor'):
            checks.checked_correctness(harness, [0, 1, 2])
    assert visited == [0, 1, 2] and harness.gemm_a16w16 is original


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_awq_packed_known_answers_and_exact_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_awq_dequantize'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    qweight = torch.tensor([[0x76543210], [-19088744]], dtype=torch.int32)  # second word: 0xfedcba98
    scales = torch.arange(1, 9, dtype=torch.float16).repeat(2, 1)
    zeros = torch.tensor([[0x11111111], [-2004318072]], dtype=torch.int32)  # 0x88888888
    expected = torch.tensor([[-1, 6, 0, 16, 5, 30, 14, 48],
                             [0, 8, 3, 20, 10, 36, 21, 56]], dtype=torch.float16)
    checks.check_output(expected, qweight, scales, zeros, harness.reference_awq_dequantize)
    with pytest.raises(AssertionError):
        checks.check_output(expected.flip(-1), qweight, scales, zeros, harness.reference_awq_dequantize)
    original = lambda q, s, z: harness.reference_awq_dequantize(q, s, z, 1)
    mod = SimpleNamespace(awq_dequantize_triton=original)
    seen = []

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed': output.zero_()

        def replay():
            if mode == 'correct': output.copy_(original(qweight, scales, zeros))
            elif mode == 'stale': output.copy_(cached)
            elif mode == 'changing_wrong': output.fill_(int(qweight[0, 0]) & 15)
            return output

        timed_run.outputs, timed_run.rerun = output, replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.awq_dequantize_triton is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'wrong_ids', 'wrong_weights'])
def test_shared_expert_append_known_answers_and_both_timed_outputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/geak_eval/L1/fused_append_shared_experts'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    node = next(n for n in ast.parse((task/'test_kernel_harness.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'reference_fused_append')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'protected-reference', 'exec'), namespace)
    reference = namespace['reference_fused_append']
    topk_ids = torch.tensor([[3, 7], [1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.75]])
    cfg = {'S': 1, 'N': 8, 'scale_factor': 1.0}
    expected = (torch.tensor([[3, 7, 8], [1, 0, 8]], dtype=torch.int32),
                torch.tensor([[0.5, 0.25, 1.], [0.125, 0.75, 1.]]))
    checks.check_output(expected, topk_ids, topk_weights, cfg, reference)
    original = lambda ids, weights, s, factor, N: reference(ids, weights, s, factor, N)
    harness = SimpleNamespace(reference_fused_append=reference, fused_append_shared_experts=original)
    seen = []

    def fn():
        harness.fused_append_shared_experts(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], N=cfg['N'])

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        outputs = measured()
        cached = tuple(out.clone() for out in outputs)
        if mode == 'incorrect_timed': outputs[0].zero_()

        def replay():
            if mode in ('correct', 'wrong_ids', 'wrong_weights'):
                expected = reference(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], cfg['N'])
                for out, ref in zip(outputs, expected): out.copy_(ref)
                if mode == 'wrong_ids': outputs[0][:, -1].zero_()
                if mode == 'wrong_weights': outputs[1][:, -1].zero_()
            elif mode == 'stale':
                for out, ref in zip(outputs, cached): out.copy_(ref)
            return outputs

        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.5, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
    assert harness.fused_append_shared_experts is original
    assert seen == [{'warmup': 50, 'repetition': 200}]


ROCM = sorted([*(ROOT/'tasks/triton2triton/rocmbench').rglob('config.yaml'),
               *(ROOT/'tasks/instruction2triton').rglob('config.yaml')])


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/gemm',
                                    'tasks/triton2triton/rocmbench/hard/gemm'])
def test_rocm_gemm_scope_retains_original_scored_cases_and_numerical_gate(relative,monkeypatch):
    task=ROOT/relative
    source=task/'gemm.py'
    original=subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    assert source.read_bytes()==original
    shapes=pure_functions(source,['get_x_vals']).get_x_vals()
    assert shapes==[(1024*v,1024*v,1024*v) for v in range(1,9)]+[
        (4864,4096,8192),(9728,8192,65536),(4864,8192,4160)]
    rows=json.loads((task/'workloads.json').read_text())['cases']
    assert len(rows)==22 and sum('performance' in r['checks'] for r in rows)==11
    for function,suffix in [('test_correctness',''),('test_performance','_str')]:
        selected=[r['params']['arguments'] for r in rows if r['params']['function']==function]
        assert {(r['M'],r['N'],r['K']) for r in selected}==set(shapes)
        assert len(selected)==11
        for args in selected:
            assert all(args[name+suffix]=='fp16' for name in ('in_dtype_a','in_dtype_b','out_dtype'))
            assert args['col_a'] is False and args['col_b'] is False
    # Independent small known answer, then a negative control, through both
    # original protected correctness and the added performance-input oracle.
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16)
    b=torch.tensor([[5.,6.],[7.,8.]],dtype=torch.float16)
    expected=torch.tensor([[19.,22.],[43.,50.]],dtype=torch.float16)
    context={'current_scale_a8_b8':None,'a':a,'b':b,'c':expected.clone()}
    reference=module_at(task/'_arena_reference.py',monkeypatch)
    check=reference.prepare(context,None)
    check(None)
    context['c'].zero_()
    with pytest.raises(reference.NumericalMismatch):check(None)
    def generator(m,n,dtype,col,seed,device):
        assert dtype==torch.float16 and col is False and device=='cuda'
        value=a if seed==1 else b
        return value,value.float(),None
    result=expected.clone()
    def candidate(a,b,c,*args,**kwargs):
        assert kwargs==dict(a_scale=None,b_scale=None,scale_a8_b8=None,activation='')
        c.copy_(result)
    protected=pure_functions(source,['get_x_vals','test_correctness'],dict(
        pytest=pytest,set_seed=lambda:None,name_to_torch_types={'fp16':torch.float16},gen_input=generator,
        dtype_is_8_bit=lambda dtype:False,matmul=candidate,result_gold={}))
    args=(2,2,2,False,False,'fp16','fp16','fp16',SimpleNamespace(node=SimpleNamespace(name='known-answer')))
    protected.test_correctness(*args)
    result.zero_()
    with pytest.raises(AssertionError):protected.test_correctness(*args)


@pytest.mark.parametrize('path', ROCM, ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_v2_preserves_original_source_and_complete_parameter_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=task/data['source']
    original=subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    expected_source = original
    if task.name == 'test_kernel_sub':
        historical_skip = b'    pytest.skip("Skipping ASTSource compile-in-subprocess check on Triton 3.3 due to known API/compiler instability; numerical correctness tests cover kernel behavior.")\n'
        assert original.count(historical_skip) == 1
        expected_source = original.replace(historical_skip, b'')
    if task.name == 'rmsnorm_fwd':
        skip = b'    # Ensure in_dtype and out_dtype are compatible for RMSNorm (usually they are the same for x and y)\n    # For benchmarking, let\'s assume in_dtype is the primary type for x and g, and y.\n    if in_dtype_str != out_dtype_str:\n         pytest.skip(f"Skipping perf test where in_dtype {in_dtype_str} != out_dtype {out_dtype_str} for simplicity.")\n\n'
        assert expected_source.count(skip) == 1
        expected_source = expected_source.replace(skip,b'').replace(
            b'y_buffer = torch.empty_like(x) # Output buffer for forward',
            b'y_buffer = torch.empty_like(x, dtype=arg_to_torch_dtype[out_dtype_str]) # Declared output dtype').replace(
            b'baseline_callable = lambda: torch_rmsnorm_fwd(x, g, ZERO_CENTERED_GAMMA, current_dtype, eps)',
            b'baseline_callable = lambda: torch_rmsnorm_fwd(x, g, ZERO_CENTERED_GAMMA, arg_to_torch_dtype[out_dtype_str], eps)')
    assert source.read_bytes() == expected_source
    assert hashlib.sha256(original).hexdigest()==data['migration']['original_source_sha256']
    rows=data['cases']
    assert len(rows)==len({row['test_case_id'] for row in rows})
    assert all('correctness' in row['checks'] for row in rows)
    assert any(row['checks']==['correctness','performance'] for row in rows)
    assert {entry.symbol for entry in spec.candidate.entrypoints}==set(data['kernel_symbols'])
    assert 'task_type' not in spec.to_mapping()
    ast.parse((task/'_arena_eval.py').read_text())
    ast.parse((task/'_arena_reference.py').read_text())



@pytest.mark.parametrize('exitcode,timed_out', [(0,False),(1,False),(None,True)])
@pytest.mark.parametrize('task', ['tasks/triton2triton/rocmbench/easy/test_kernel_sub',
                                'tasks/instruction2triton/rocmbench/test_kernel_sub'])
def test_kernel_sub_declared_compilation_case_executes_and_rejects_child_failure(task,exitcode,timed_out):
    # Run the protected test body against a controlled process, verifying that
    # a skipped/nonzero/timed-out compilation cannot become successful evidence.
    tree=ast.parse((ROOT/task/'test_kernel_sub.py').read_text())
    definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                    and node.name=='test_compile_kernel_sub_in_subproc')
    definition.decorator_list=[]
    calls=[]
    def compile_target(): pass
    class Process:
        def __init__(self, target):
            assert target is compile_target
            self.exitcode=exitcode
        def start(self): calls.append('start')
        def join(self, timeout=None): calls.append(('join',timeout))
        def is_alive(self): return timed_out
        def terminate(self): calls.append('terminate')
    namespace=dict(multiprocessing=SimpleNamespace(set_start_method=lambda *a,**k:None,Process=Process),
                   set_seed=lambda:None,pytest=pytest,torch=torch,result_gold={},
                   compile_kernel_sub_for_test=compile_target)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(ROOT/task/'test_kernel_sub.py'),'exec'),namespace)
    run=lambda: namespace[definition.name]('fresh-cache',SimpleNamespace(node=SimpleNamespace(name='compile-case')))
    if timed_out:
        with pytest.raises(pytest.fail.Exception,match='Process timed out'): run()
        assert 'terminate' in calls
    elif exitcode != 0:
        with pytest.raises(AssertionError): run()
    else:
        run()
        assert namespace['result_gold']['compile_case'].item()==1.0
    assert calls[:2]==['start',('join',60)]


@pytest.mark.parametrize('task',['tasks/instruction2triton/rocmbench/rmsnorm_fwd','tasks/triton2triton/rocmbench/medium/rmsnorm_fwd'])
@pytest.mark.parametrize('in_name',['fp16','bf16','fp32'])
@pytest.mark.parametrize('out_name',['fp16','bf16','fp32'])
@pytest.mark.parametrize('zero_centered',[False,True])
def test_rms_forward_declared_dtype_pairs_execute_original_performance_body(monkeypatch,task,in_name,out_name,zero_centered):
    import inspect,math
    oracle_module=module_at(ROOT/task/'_arena_reference.py',monkeypatch)
    mapping={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}
    calls=[]
    class CpuTorch:
        def __getattr__(self,name):
            value=getattr(torch,name)
            if name in ('randn','rand','empty','empty_like'):
                def cpu(*args,**kwargs):
                    kwargs['device']='cpu'
                    return value(*args,**kwargs)
                return cpu
            return value
    def run_kernel(x,g,y,rsigma,*args):
        assert x.dtype==mapping[in_name] and y.dtype==mapping[out_name]
        zero,eps=args[5],args[-1]
        for row in range(x.shape[0]):
            inv=1/math.sqrt(sum(float(v)**2 for v in x[row])/x.shape[1]+eps)
            rsigma[row]=inv
            for col in range(x.shape[1]):
                y[row,col]=float(x[row,col])*inv*(float(g[col])+int(zero))
        calls.append('kernel')
        return y
    class Benchmark:
        def __init__(self,op_callable,config,**kwargs):
            assert config==(10,100)
            self.context=dict(inspect.currentframe().f_back.f_locals)
            self.op=op_callable
        def run_benchmark(self,**kwargs):
            output=self.op()
            oracle_module.prepare(self.context,None)(output)
            invalid={**self.context,'y_buffer':output.double()}
            with pytest.raises(ValueError,match='dtype'): oracle_module.prepare(invalid,None)(output)
            output.zero_()
            with pytest.raises(oracle_module.NumericalMismatch): oracle_module.prepare(self.context,None)(output)
    tree=ast.parse((ROOT/task/'rmsnorm_fwd.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_performance');fn.decorator_list=[]
    namespace=dict(torch=CpuTorch(),pytest=pytest,arg_to_torch_dtype=mapping,set_seed=lambda:torch.manual_seed(42),
                   get_num_sms=lambda:2,triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),
                   rmsnorm=run_kernel,PytestBenchmarker=Benchmark,do_bench_config=lambda warm_up,repetition:(warm_up,repetition),
                   OP_NAME_FOR_BENCHMARK='fixture',calculate_rmsnorm_fwd_gbps=None,calculate_rmsnorm_fwd_tflops=None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/task/'rmsnorm_fwd.py'),'exec'),namespace)
    namespace['test_performance'](2,8,zero_centered,in_name,out_name,None)
    assert calls==['kernel']

def oracle(name,monkeypatch):
    return module_at(ROOT/'tasks/instruction2triton/rocmbench'/name/'_arena_reference.py',monkeypatch)


def test_philox_known_counter_and_uniform_negative_control(monkeypatch):
    reference=oracle('test_randn',monkeypatch)
    # Philox4x32-10 zero counter/key known answer (first lane).
    assert reference.philox32(0,1).tolist()==[0x6627e8d5]
    output=torch.zeros(16)
    check=reference.prepare({'seed_val':0,'N_elements':16,'x_output_buffer':output},None)
    with pytest.raises(reference.NumericalMismatch):check(output)


def test_swizzle_matches_original_explicit_known_order(monkeypatch):
    reference=oracle('test_triton_swizzle2d',monkeypatch)
    expected=reference.swizzle_reference(5,7,3,dtype=torch.int32,device='cpu')
    task=ROOT/'tasks/instruction2triton/rocmbench/test_triton_swizzle2d/test_triton_swizzle2d.py'
    tree=ast.parse(task.read_text());test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_swizzle2d')
    assignment=next(n for n in ast.walk(test) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_order' for t in n.targets))
    original_order=ast.literal_eval(assignment.value.func.value.args[0])
    assert expected.tolist()==original_order
    with pytest.raises(reference.NumericalMismatch):reference.compare(expected,torch.arange(35).reshape(5,7),exact=True,check_dtype=False)


def test_rms_backward_reference_matches_autograd_and_rejects_wrong_gradients(monkeypatch):
    reference=oracle('rmsnorm_bwd',monkeypatch)
    x=torch.tensor([[1.,2.,3.],[2.,-1.,4.]],requires_grad=True)
    g=torch.tensor([0.2,0.4,0.8],requires_grad=True)
    go=torch.tensor([[1.,0.,2.],[-1.,2.,0.5]])
    y=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*(g+1)
    y.backward(go)
    dx=x.grad.clone();dg=g.grad.clone()
    context={'x':x.detach(),'g':g.detach(),'grad_output':go,'ZERO_CENTERED_GAMMA':True,'eps':1e-5,
             'dx_bench':dx,'dg_tmp_bench':torch.stack([dg,torch.zeros_like(dg)])}
    check=reference.prepare(context,None);check(None)
    dx.zero_()
    with pytest.raises(reference.NumericalMismatch):check(None)


def test_oracle_distinguishes_output_contract_failure_from_numerical_error(monkeypatch):
    reference=oracle('test_add_kernel',monkeypatch)
    with pytest.raises(ValueError,match='shape/device'):
        reference.compare(torch.zeros(2),torch.zeros(3),atol=1e-3,rtol=1e-2)
    with pytest.raises(ValueError,match='nonfinite'):
        reference.compare(torch.tensor([float('nan')]),torch.zeros(1),atol=1e-3,rtol=1e-2)


@pytest.mark.parametrize('relative',['tasks/instruction2triton/rocmbench/test_add_kernel','tasks/triton2triton/rocmbench/easy/test_add_kernel'])
@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong','event_fallback','replay_raises',
                                'zero_inputs_and_output_timed','zero_inputs_and_output_replay'])
def test_add_canonical_samples_observe_exact_replay_and_reject_wrong_output(monkeypatch,relative,mode):
    # Use the actual PytestBenchmarker configuration and statistics helper,
    # with a CPU graph simulator only at the canonical device-timing boundary.
    task=ROOT/relative
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    reference=module_at(task/'_arena_reference.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_arena_reference',reference)
    class Timed:
        outputs=None
        def _bind(self,rerun,outputs):self.outputs=outputs;self.rerun=rerun
    calls=[]
    def timer(fn,*,timed_run=None,**kwargs):
        assert timed_run is not None
        assert kwargs['warmup']==10 and kwargs['repetition']==100
        calls.append(kwargs)
        if mode=='event_fallback':raise RuntimeError('timed_run requires an observable CUDA-graph replay')
        output=fn();saved=output.clone()
        if mode=='incorrect_timed':output.zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='correct' or mode.startswith('zero_inputs'):fn()
            elif mode=='stale':output.copy_(saved)
            elif mode=='changing_wrong':output.fill_(123)
            return output
        timed_run._bind(replay,output)
        return [0.25]*100,dict(benchmark_method='cuda_graph',benchmark_warmup=10,benchmark_samples=100)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(
        TimedRun=Timed,benchmark_cuda_graph_or_events_samples=timer))
    helper=module_at(ROOT/'src/tools/perf/performance_utils_pytest.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'performance_utils_pytest',helper)
    plugin=SimpleNamespace(action='performance',current_row={'test_case_id':'cpu-replay'},exercised=set())
    Checked=adapter.benchmark_type(helper.PytestBenchmarker,plugin,None)
    def make():
        x=torch.tensor([2.,4.,6.]);y=torch.tensor([1.,3.,5.]);output=torch.empty_like(x)
        launches=[]
        def launch():
            launches.append(None)
            output.copy_(x+y)
            if (mode=='zero_inputs_and_output_timed' and len(launches)>=2 or
                    mode=='zero_inputs_and_output_replay' and len(launches)>=3):
                x.zero_();y.zero_();output.zero_()
            return 'compiled-kernel-handle'
        return Checked(op_callable=launch,op_name='add',config=helper.do_bench_config(warm_up=10,repetition=100))
    benchmark=make();original=benchmark.op_callable
    inputs=tuple(benchmark.context[name] for name in ('x','y'))
    pristine=tuple(value.clone() for value in inputs)
    if mode=='correct':
        record=benchmark.run_benchmark(current_params_dict={})
        assert record['timing_ms']['mean']==0.25
        assert plugin.current_row['metadata']['perturbed_input_replay_checked']
        assert plugin.current_row['metadata']['device_timing']['benchmark_samples']==100
        assert plugin.exercised=={'cpu-replay'}
        # Compare effective public-timer arguments with the actual old helper
        # path, including defaults hidden behind the task's direct sample call.
        import inspect
        prior=[]
        monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',
                            lambda fn,**options:(prior.append(options) or [0.25],{}))
        helper._measure_times(original,benchmark.config,prepare_fn=benchmark.prepare_fn,
                              use_cuda_graph=benchmark.use_cuda_graph,fallback_reason=benchmark.fallback_reason)
        canonical=module_at(ROOT/'src/tools/perf/aka_benchmark.py',monkeypatch)
        signature=inspect.signature(canonical.benchmark_cuda_graph_or_events_samples)
        def effective(options):
            bound=signature.bind_partial(None,**options);bound.apply_defaults()
            return {k:v for k,v in bound.arguments.items() if k not in ('fn','timed_run')}
        assert effective(calls[0])==effective(prior[0])
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):
            benchmark.run_benchmark(current_params_dict={})
    elif mode=='event_fallback':
        with pytest.raises(RuntimeError,match='observable'):benchmark.run_benchmark(current_params_dict={})
    else:
        with pytest.raises((reference.NumericalMismatch,ValueError)):
            benchmark.run_benchmark(current_params_dict={})
    assert all(torch.equal(value,expected) for value,expected in zip(inputs,pristine))
    assert all(benchmark.context[name] is value for name,value in zip(('x','y'),inputs))
    if mode!='correct':assert not plugin.exercised
    assert benchmark.op_callable is original and len(calls)==1


@pytest.mark.parametrize('path', [p for p in ROCM if p.parent.name!='test_add_kernel'], ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_timing_evidence_retains_canonical_fallback_reason(monkeypatch, path):
    adapter = module_at(path.parent/'_arena_eval.py', monkeypatch)
    expected = torch.tensor([2.])
    def prepare(context, module):
        return lambda value: torch.testing.assert_close(value, expected)
    monkeypatch.setitem(__import__('sys').modules, '_arena_reference', SimpleNamespace(prepare=prepare))
    plugin = SimpleNamespace(action='performance', current_row={'test_case_id':'cpu-fixture'}, exercised=set())
    record = {'timing_ms': {'mean': 0.25}, 'benchmark_method':'cuda_event_fallback',
              'benchmark_fallback_reason':'protected invocation uses a GPU scalar on the host',
              'benchmark_samples':100, 'benchmark_warmup':10, 'benchmark_effective_repeats':1,
              'params':{'not_timing_metadata':True}}
    class Base:
        def __init__(self, op_callable, **kwargs):
            self.op_callable = op_callable
            self.prepare_fn = None
        def run_benchmark(self, **kwargs):
            self.op_callable()
            return record
    benchmark = adapter.benchmark_type(Base, plugin, None)(op_callable=lambda: expected.clone())
    benchmark.run_benchmark()
    assert plugin.current_row['metadata']['device_timing'] == {
        k:v for k,v in record.items() if k.startswith('benchmark_')}
    assert plugin.current_row['metadata']['timed_output_checked']
    assert plugin.current_row['execution_time_ms'] == 0.25


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/moe_gemm',
                                     'tasks/triton2triton/rocmbench/hard/moe_gemm'])
def test_moe_launcher_selects_events_before_attempting_unsupported_capture(monkeypatch, relative):
    adapter = module_at(ROOT/relative/'_arena_eval.py', monkeypatch)
    expected = torch.tensor([2.])
    monkeypatch.setitem(__import__('sys').modules, '_arena_reference', SimpleNamespace(
        prepare=lambda context,module: lambda value: torch.testing.assert_close(value, expected)))
    plugin = SimpleNamespace(action='performance', current_row={'test_case_id':'cpu-fixture'}, exercised=set())
    options = {}
    class Base:
        def __init__(self, op_callable, use_cuda_graph=True, fallback_reason=None):
            options.update(use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason)
            self.op_callable = op_callable
            self.prepare_fn = None
        def run_benchmark(self, **kwargs):
            if options['use_cuda_graph']:
                raise RuntimeError('The protected .item() launcher cannot be captured')
            self.op_callable()
            return {'timing_ms':{'mean':0.25}, 'benchmark_method':'cuda_event_fallback',
                    'benchmark_fallback_reason':options['fallback_reason']}
    benchmark = adapter.benchmark_type(Base,plugin,None)(op_callable=lambda:expected.clone())
    benchmark.run_benchmark()
    assert options['use_cuda_graph'] is False and '.item()' in options['fallback_reason']
    assert plugin.current_row['metadata']['device_timing']['benchmark_fallback_reason'] == options['fallback_reason']
    assert plugin.exercised == {'cpu-fixture'}


def test_skip_is_not_passing_protocol_evidence(monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    plugin=adapter.ReportPlugin(data,'correctness')
    key=next(iter(plugin.rows));plugin.node_rows['item']=key
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='setup',failed=False,skipped=True,longrepr='unsupported device'))
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='teardown',failed=False,skipped=False))
    assert plugin.rows[key]['status']=='FAIL' and plugin.rows[key]['failure_kind']=='not_executed'


def test_rocm_real_pytest_lifecycle_and_manifest_drift(tmp_path,monkeypatch):
    """CPU functions exercise real pytest setup/call/teardown and collection hooks."""
    import sys
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1')
    source='''import pytest
from types import SimpleNamespace
triton=SimpleNamespace(jit=lambda f:f)
@triton.jit
def kernel():
    return 1
@pytest.mark.parametrize('x',[1,2])
def test_value(x):
    assert x > 0
'''
    (tmp_path/'test_cpu_migration.py').write_text(source)
    rows=[{'test_case_id':adapter.identity('test_value',{'x':x}),
           'params':{'function':'test_value','arguments':{'x':x}},
           'checks':['correctness'],'status':'PASS'} for x in (1,2)]
    data={'source':'test_cpu_migration.py','kernel_symbols':['kernel'],'cases':rows}
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='PASS',result
    assert all(r['status']=='PASS' and r['metrics']['original_pytest_passed'] for r in result['cases'])
    # A manifest cannot silently lose one original case.
    data['cases'].pop();(tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']
    monkeypatch.delitem(sys.modules,'test_cpu_migration',raising=False)


def test_rocm_missing_candidate_fails_every_applicable_case(tmp_path,monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','performance')
    assert result['status']=='FAIL'
    assert result['cases'] and all(r['status']=='FAIL' and 'performance' in r['checks'] for r in result['cases'])
    result_record(result)


GEAK = sorted((ROOT/'tasks/triton2triton/geak_eval').rglob('config.yaml'))


@pytest.mark.parametrize('path', GEAK, ids=lambda p:p.parent.name)
def test_geak_v2_preserves_original_functions_and_freezes_the_complete_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert spec.candidate.language=='triton'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=(task/data['source']).relative_to(ROOT).as_posix()
    original=subprocess.check_output(['git','show',f'{BASE}:{source}'],cwd=ROOT)
    assert (ROOT/source).read_bytes()==original
    before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/test_kernel_harness.py'],cwd=ROOT,text=True)
    after=(task/'test_kernel_harness.py').read_text()
    assert data['migration']['original_harness_sha256']==hashlib.sha256(before.encode()).hexdigest()
    assert data['migration']['original_source_sha256']==hashlib.sha256(original).hexdigest()
    olddefs={n.name:n for n in ast.parse(before).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newdefs={n.name:n for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    bootstrap={'_find_baseline_kernel_dir','_load_baseline_triton','_resolve_geak_kernel_dir','_register_geak_aliases'}
    for name in olddefs.keys()-bootstrap-({'e8m0_to_f32'} if data['migration'].get('reference_fixes') else set()):
        assert ast.get_source_segment(before,olddefs[name])==ast.get_source_segment(after,newdefs[name]),name
    assert 'os.environ.get("GEAK_WORK_DIR"' not in after
    assert 'os.environ.get("GEAK_REPO_ROOT"' not in after
    perfrows=[r for r in data['cases'] if 'performance' in r['checks']]
    assert [r['params']['configuration'] for r in perfrows]==data['input_tables']['performance']
    assert [r['params']['case_index'] for r in perfrows]==list(range(len(perfrows)))
    configs=[r['params']['configuration'] for r in data['cases']]
    assert all(v in configs for v in data['input_tables']['original_correctness'])
    assert all('correctness' in r['checks'] for r in data['cases'])
    result_record({'protocol':'arena-eval-v1','role':'task','action':'validate-task','status':'PASS','cases':data['cases']})
    for role in ('baseline','candidate'):
        # Real command smoke: AST compilation only, no JIT/GPU execution.
        run=subprocess.run([__import__('sys').executable,'_arena_eval.py',role,'compile'],cwd=task,text=True,capture_output=True)
        parsed=parse_command_result(run.stdout,role=role,action='compile',returncode=run.returncode)
        assert parsed.status=='PASS',run.stdout+run.stderr


@pytest.mark.parametrize('stream_kind',['captured','text','reconfigure_error'])
def test_quant_sort_harness_bootstrap_supports_captured_stdout(monkeypatch,stream_kind):
    import io,sys
    from contextlib import redirect_stdout
    path=ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort/test_kernel_harness.py'
    tree=ast.parse(path.read_text())
    # Execute the actual entire import/bootstrap prefix up to GPU dependencies.
    end=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Import)
             and any(alias.name=='torch' for alias in node.names))
    prefix=compile(ast.Module(body=tree.body[:end],type_ignores=[]),str(path),'exec')
    monkeypatch.setitem(sys.modules,'_aka_benchmark',SimpleNamespace(benchmark_cuda_graph_or_events_samples=None))
    class FaultyStream(io.StringIO):
        def reconfigure(self,**kwargs):raise RuntimeError('broken text stream')
    stream=(io.TextIOWrapper(io.BytesIO()) if stream_kind=='text' else
            FaultyStream() if stream_kind=='reconfigure_error' else io.StringIO())
    try:
        with redirect_stdout(stream):
            if stream_kind=='reconfigure_error':
                with pytest.raises(RuntimeError,match='broken text stream'):exec(prefix,{})
            else:exec(prefix,{})
        if stream_kind=='text':assert stream.line_buffering
        if stream_kind=='captured':
            before=subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT,text=True)
            old=next(node for node in ast.parse(before).body if isinstance(node,ast.Expr)
                     and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                     and node.value.func.attr=='reconfigure')
            with pytest.raises(AttributeError,match='reconfigure'):
                exec(compile(ast.Module(body=[old],type_ignores=[]),str(path),'exec'),
                     {'sys':SimpleNamespace(stdout=stream)})
    finally:stream.close()


def test_geak_boolean_integer_and_skipped_result_contracts(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    adapter.require_success(None,'none',1)
    adapter.require_success(0,'zero',1)
    adapter.require_success(True,'bool',1)
    for value,kind in [(None,'bool'),(False,'zero'),(True,'zero'),(0,'bool'),
                       ({'correct':True,'num_correct':0,'num_failed':0,'skipped':True},'dict'),
                       ({'correct':True,'num_correct':1,'num_failed':0},'dict')]:
        with pytest.raises(RuntimeError):adapter.require_success(value,kind,2)


def test_geak_performance_uses_current_candidate_timing_and_rejects_missing_calls(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    measurements=iter([17.,3.,19.,5.])
    h=SimpleNamespace(benchmark_cuda_graph_or_events=lambda *a,**k:(next(measurements),{'benchmark_method':'cuda_graph'}))
    actions=SimpleNamespace(h=h,performance=lambda:[h.benchmark_cuda_graph_or_events(None) for _ in range(4)])
    data={'cases':[{'checks':['correctness','performance']},{'checks':['correctness','performance']}],
          'timing_calls_per_case':['reference','candidate']}
    assert [ms for ms,meta in adapter.capture_performance(actions,data)]==[3.,5.]
    # A stale saved report has no role in supplying the missing current call.
    h.benchmark_cuda_graph_or_events=lambda *a,**k:(2.,{'benchmark_method':'cuda_graph'})
    actions.performance=lambda:h.benchmark_cuda_graph_or_events(None)
    with pytest.raises(RuntimeError,match='Missing/extra'):adapter.capture_performance(actions,data)


def test_geak_failure_envelope_and_manifest_drift(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((GEAK[0].parent/'workloads.json').read_text())
    actions=SimpleNamespace(inputs=lambda:data['input_tables'],validate=lambda:None,
                            correctness=lambda require:require(False,'bool',len(data['cases'])))
    monkeypatch.setattr(adapter,'load_actions',lambda:actions)
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='FAIL' and all(row['status']=='FAIL' for row in result['cases'])
    assert result['failure_kind']!='numerical_mismatch'
    result_record(result)
    actions.inputs=lambda:{'performance':[],'original_correctness':[]}
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']


def pure_functions(path, names, namespace=None):
    """Load protected CPU reference functions without importing GPU dependencies."""
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    scope={'torch':torch,**(namespace or {})}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return SimpleNamespace(**scope)


def test_geak_rope_references_have_independent_known_answers():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/fused_qk_rope_cache_mla/test_kernel_harness.py'
    from enum import IntEnum
    ref=pure_functions(path,['RotateStyle','rotate_half_neox','rotate_half_gptj','ref_rope_sbhd_fwd'],{'IntEnum':IntEnum})
    x=torch.tensor([[1.,2.,3.,4.]])
    torch.testing.assert_close(ref.rotate_half_neox(x),torch.tensor([[-3.,-4.,1.,2.]]))
    torch.testing.assert_close(ref.rotate_half_gptj(x),torch.tensor([[-2.,1.,-4.,3.]]))
    got=ref.ref_rope_sbhd_fwd(x,torch.full((1,4),torch.pi/2),ref.RotateStyle.GPTJ,False,False)
    expected=torch.tensor([[-2.,1.,-4.,3.]])
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,x)


def test_geak_fp4_reference_decodes_independent_known_values():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/test_kernel_harness.py'
    # Only the lookup-table allocation is redirected to CPU; arithmetic is unchanged.
    cpu_torch=SimpleNamespace(float32=torch.float32,
        tensor=lambda values,**kwargs:torch.tensor(values,**{**kwargs,'device':'cpu'}))
    ref=pure_functions(path,['mxfp4_to_f32'],{'torch':cpu_torch})
    # Low/high nibbles enumerate the E2M1 positive and negative values.
    packed=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]],dtype=torch.uint8)
    expected=torch.tensor([[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]])
    got=ref.mxfp4_to_f32(packed)
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,torch.zeros_like(expected))


@pytest.mark.parametrize('name',['gemm_a16wfp4','fused_mxfp4_quant_moe_sort'])
def test_e8m0_reference_all_encodings_and_negative_control(name):
    path=ROOT/'tasks/triton2triton/geak_eval/L3'/name/'test_kernel_harness.py'
    ref=pure_functions(path,['e8m0_to_f32'])
    codes=torch.arange(256,dtype=torch.uint8)
    got=ref.e8m0_to_f32(codes)
    expected=codes.view(torch.float8_e8m0fnu).float()
    torch.testing.assert_close(got,expected,atol=0,rtol=0,equal_nan=True)
    assert got[126]==0.5 and got[134]==128 and torch.isnan(got[255])
    before=subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name=='e8m0_to_f32')
    scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(scope['e8m0_to_f32'](codes),expected,atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize('task',[VLLM[0].parent,ROCM[0].parent,GEAK[0].parent],ids=['vllm','rocm','geak'])
def test_bad_diagnostic_evidence_still_emits_failure(task,monkeypatch,capsys):
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    code=adapter.emit_result({'protocol':'arena-eval-v1','role':'candidate','action':'performance',
                             'status':'PASS','cases':[],'metadata':{'bad':float('nan')}})
    parsed=parse_command_result(capsys.readouterr().out,role='candidate',action='performance',returncode=code)
    assert parsed.status=='FAIL' and parsed.failure_kind=='invalid_evidence'


def test_vllm_additional_original_correctness_cases_are_manifested():
    for name in ['triton_bad_words','triton_logit_bias']:
        task=ROOT/'tasks/triton2triton/vllm'/name
        data=json.loads((task/'workloads.json').read_text())
        extra=[r for r in data['cases'] if r['checks']==['correctness']]
        before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/scripts/task_runner.py'],cwd=ROOT,text=True)
        tree=ast.parse(before)
        if name=='triton_bad_words':
            assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='multi_cases' for t in n.targets))
            assert [r['shape'] for r in extra]==[list(v) for v in ast.literal_eval(assignment.value)]
            assert [r['params']['seed'] for r in extra]==[1234,1235,1236]
        else:
            assert len(extra)==1 and extra[0]['shape']==[2,256,16] and extra[0]['params']['seed']==777
        assert all(r['params']['case_index']==-1 for r in extra)


def test_all_rocm_manifests_match_original_cpu_collected_pytest_parameters():
    """Import decorators with inert GPU stubs; execute no kernels or test bodies."""
    import sys
    fixture=ROOT/'tests/fixtures/triton_migration/collect_rocm_cpu.py'
    run=subprocess.run([sys.executable,str(fixture)],cwd=ROOT,text=True,capture_output=True,check=True)
    collected=json.loads(run.stdout)
    assert len(collected)==61
    for task,functions in collected.items():
        expected={}
        for fn in functions:
            for params in fn['cases']:
                content=json.dumps(params,sort_keys=True,separators=(',',':'))
                case=fn['function']+'/'+hashlib.sha256(content.encode()).hexdigest()[:20]
                expected[case]={'function':fn['function'],'arguments':params}
        data=json.loads((ROOT/task/'workloads.json').read_text())
        assert {r['test_case_id']:r['params'] for r in data['cases']}==expected,task
        assert 'built-in method' not in json.dumps(data)


DECORATOR_HELPERS={
    'gemm':{'leaky_relu'},
    'layernorm':{'get_autotune_config'},
    'rmsnorm_fwd':{'get_autotune_config'},
    'softmax':{'get_autotune_config'},
    'multreduce_matmul_dot_kernel':{'get_triton_dot_autotune_configs','get_triton_autotune_key','get_triton_heuristics','triton_matmul_kernel'},
    'triton_multreduce_matmul_kernel':{'get_triton_multreduce_autotune_configs','get_triton_autotune_key','get_triton_heuristics','triton_matmul_kernel'},
}


@pytest.mark.parametrize('path',[p for p in ROCM if p.parent.name in DECORATOR_HELPERS],ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_implementation_helpers_editable_without_exposing_tests_or_references(path,tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
    workspace=tmp_path/'task'
    shutil.copytree(path.parent,workspace,ignore=shutil.ignore_patterns('__pycache__'))
    spec=load_task_spec(workspace/'config.yaml',task_id=path.parent.name)
    helper_names=DECORATOR_HELPERS[path.parent.name]
    entry_names={e.symbol for e in spec.candidate.entrypoints}
    assert not (helper_names & entry_names), 'Implementation helpers must not become required entrypoints'
    assert helper_names <= set(spec.candidate.editable[0].symbols)
    assert set(spec.candidate.editable[0].symbols)==entry_names|helper_names
    snapshot=snapshot_workspace_harness(workspace)
    source=workspace/spec.candidate.editable[0].path
    text=source.read_text();tree=ast.parse(text);lines=text.splitlines(keepends=True)
    insertions=[n.body[0].lineno-1 for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helper_names]
    for index in sorted(insertions,reverse=True):lines.insert(index,'    _arena_helper_boundary_probe = 1\n')
    source.write_text(''.join(lines))
    verify_workspace_harness(snapshot)
    # Ordinary test/reference code remains protected in that same file.
    tree=ast.parse(source.read_text())
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    lines=source.read_text().splitlines(keepends=True)
    lines.insert(test.body[0].lineno-1,'    _arena_forbidden_test_change = 1\n')
    source.write_text(''.join(lines))
    with pytest.raises(RuntimeError):verify_workspace_harness(snapshot)


def conv_fwd_inputs():
    x=torch.tensor([[1.,2.,3.,4.,5.,6.],[7.,8.,9.,10.,11.,12.]])
    weight=torch.tensor([[1.,2.,3.],[1.,-1.,2.]])
    bias=torch.tensor([1.,-1.])
    state=torch.tensor([[[10.,20.],[30.,40.]],[[50.,60.],[70.,80.]]])
    starts=torch.tensor([0,3,6],dtype=torch.int32)
    indices=torch.tensor([0,1],dtype=torch.int32)
    has_init=torch.ones(2,dtype=torch.int32)
    return x,weight,bias,state,starts,indices,has_init


def independent_conv_fwd(x,weight,bias,state,starts,indices,has_init,activation=None):
    # An independent grouped-convolution implementation for the CPU controls.
    result=torch.empty_like(x)
    for i,slot in enumerate(indices.tolist()):
        start,end=starts[i:i+2].tolist()
        prefix=state[slot].clone() if has_init[i] else torch.zeros_like(state[slot])
        sequence=torch.cat((prefix,x[:,start:end]),dim=1)
        values=torch.nn.functional.conv1d(sequence.unsqueeze(0),weight.unsqueeze(1),bias,groups=x.shape[0])[0]
        if activation in ('silu','swish'):values=torch.nn.functional.silu(values)
        result[:,start:end]=values
        state[slot].copy_(sequence[:,-state.shape[-1]:])
    return result


def test_conv_fwd_original_reference_and_state_have_independent_known_answers(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=conv_fwd_inputs()
    expected=torch.tensor([[54.,29.,15.,183.,84.,33.],[3.,48.,16.,9.,91.,22.]])
    state_expected=torch.tensor([[[2.,3.],[8.,9.]],[[5.,6.],[11.,12.]]])
    actual=checks.references(harness,*args,None)
    assert torch.equal(actual[0],expected) and torch.equal(actual[1],state_expected)
    output=independent_conv_fwd(*args)
    assert torch.equal(output,expected) and torch.equal(args[3],state_expected)


@pytest.mark.parametrize('mode',['correct','wrong_state','no_state_write','wrong_output','dtype','shape','nonfinite','mutate_input'])
def test_conv_fwd_correctness_checks_output_state_and_pristine_input_oracles(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=conv_fwd_inputs()
    def candidate(x,w,b,state,starts,indices,has_init,activation=None):
        saved=state.clone()
        out=independent_conv_fwd(x,w,b,state,starts,indices,has_init,activation)
        if mode=='wrong_state':state[1,0,0]=0
        elif mode=='no_state_write':state.copy_(saved)
        elif mode=='wrong_output':out.zero_()
        elif mode=='dtype':out=out.double()
        elif mode=='shape':out=out[:1]
        elif mode=='nonfinite':state[0,0,0]=float('nan')
        elif mode=='mutate_input':x.zero_();out.zero_();state.zero_()
        return out
    module=SimpleNamespace(causal_conv1d_fwd=candidate)
    original_load=lambda:module
    harness.load_module=original_load
    with checks.checked_modules(harness):
        if mode=='correct':harness.load_module().causal_conv1d_fwd(*args)
        else:
            with pytest.raises(AssertionError):harness.load_module().causal_conv1d_fwd(*args)
    assert harness.load_module is original_load and module.causal_conv1d_fwd is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed_state','stale','no_write','wrong_replay_state','wrong_replay_output','mutate_input','replay_raises'])
def test_conv_fwd_actual_timed_output_state_and_replay_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    harness._TimedRun=SimpleNamespace
    x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init=conv_fwd_inputs()
    out=torch.empty_like(x);activation='silu'
    initial_conv_states=conv_states.clone()
    batch_ptr=torch.tensor([0,1],dtype=torch.int32)
    token_chunk_offset_ptr=torch.zeros(2,dtype=torch.int32)
    source=(x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init,initial_conv_states,batch_ptr,token_chunk_offset_ptr)
    pristine=tuple(value.clone() for value in source)
    def fn():
        assert batch_ptr.numel()==token_chunk_offset_ptr.numel()==2
        out.copy_(independent_conv_fwd(x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init,activation))
    def prepare():conv_states.copy_(initial_conv_states)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs)
        kwargs['prepare_fn']();outputs=measured();saved=tuple(value.clone() for value in outputs)
        if mode=='wrong_timed_state':conv_states.zero_()
        def replay():
            kwargs['prepare_fn']()
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':
                for value,cached in zip(outputs,saved):value.copy_(cached)
            elif mode!='no_write':
                measured()
                if mode=='wrong_replay_state':conv_states.zero_()
                elif mode=='wrong_replay_output':out.zero_()
                elif mode=='mutate_input':x.zero_();out.zero_();conv_states.zero_()
            return outputs
        timed_run.outputs=outputs;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100,prepare_fn=prepare)
    if mode=='correct':
        ms,metadata=call()
        assert ms==0.25 and metadata['cached_state_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100,prepare_fn=prepare)]
    assert all(torch.equal(value,saved) for value,saved in zip(source,pristine))


def test_conv_fwd_adapter_installs_output_state_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_conv_fwd_checks'
