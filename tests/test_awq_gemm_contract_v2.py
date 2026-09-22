"""CPU AWQ oracle and actual task-wrapper controls; GPU validation is separate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_awq_gemm'


def load(path):
    spec = importlib.util.spec_from_file_location('_awq_contract_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.chdir(TASK)
    harness = load(TASK/'scripts/task_runner.py')
    checks = load(TASK/'_arena_checks.py')
    harness._TimedRun = load(ROOT/'src/tools/perf/aka_benchmark.py').TimedRun
    return harness, checks


def vector_oracle(a, packed, scales, zeros, split=1, **kwargs):
    # Vectorized unpack/broadcast differs from the protected scalar CPU loops.
    shifts = torch.tensor([0,16,4,20,8,24,12,28])
    weights = ((packed.long()[...,None] >> shifts) & 15).reshape(packed.shape[0],-1)
    zero = ((zeros.long()[...,None] >> shifts) & 15).reshape(zeros.shape[0],-1)
    groups = torch.arange(packed.shape[0]) // (packed.shape[0]//zeros.shape[0])
    dequant = (weights.double()-zero[groups].double())*scales[groups].double()
    return (a.double() @ dequant).to(scales.dtype)


def inputs():
    return (torch.arange(1,65).reshape(2,32).half()/32,
            torch.full((32,1),0x76543210,dtype=torch.int32),
            torch.full((1,8),.0625,dtype=torch.float16),
            torch.full((1,1),0x11111111,dtype=torch.int32))


def test_reference_unpack_order_and_original_gate(modules):
    harness, checks = modules
    values = (torch.tensor([[2.]],dtype=torch.float16),
              torch.tensor([[0x76543210]],dtype=torch.int32),
              torch.full((1,8),.5,dtype=torch.float16),
              torch.tensor([[0x11111111]],dtype=torch.int32))
    expected = torch.tensor([[-1,3,0,4,1,5,2,6]],dtype=torch.float16)
    assert torch.equal(checks.reference(harness,values),expected)
    assert torch.equal(vector_oracle(*values),expected)
    value = torch.tensor([1.],dtype=torch.float16)
    checks.check_output(value+.15,value)
    with pytest.raises(AssertionError): checks.check_output(value+.25,value)


@pytest.mark.parametrize('mode', ['shape','dtype','device','nan','infinity','values','none'])
def test_complete_output_contract(modules, mode):
    _, checks = modules
    expected = vector_oracle(*inputs())
    output = {'shape':expected[:1], 'dtype':expected.float(),
              'device':torch.empty_like(expected,device='meta'),
              'nan':torch.full_like(expected,float('nan')),
              'infinity':torch.full_like(expected,float('inf')),
              'values':torch.zeros_like(expected), 'none':None}[mode]
    with pytest.raises(AssertionError): checks.check_output(output,expected)


@pytest.mark.parametrize('mode', ['correct','ignore_split','omit_tail','mutate_a',
                                 'mutate_weight','mutate_scales','mutate_zeros','raise'])
def test_public_correctness_adds_supported_tails_and_split_paths(modules, mode):
    harness, checks = modules
    calls, observed = [], []
    def public(a, weight, scales, zeros, split, **kwargs):
        values = (a, weight, scales, zeros)
        observed.append((values, checks.snapshot(values)))
        calls.append((a.shape,weight.shape,split))
        if mode.startswith('mutate_'):
            values[('a','weight','scales','zeros').index(mode.removeprefix('mutate_'))].zero_()
        if mode == 'raise': raise RuntimeError('controlled failure')
        output = vector_oracle(*values)
        if mode == 'ignore_split' and split>1: output.zero_()
        if mode == 'omit_tail' and a.shape[0]%32: output[-1].zero_()
        return output
    module = SimpleNamespace(awq_gemm_triton=public)
    harness.load_module = lambda:module
    with checks.checked_modules(harness):
        if mode=='correct':
            result = harness.load_module().awq_gemm_triton(*inputs(),1)
            checks.check_output(result,vector_oracle(*inputs()))
            assert calls[1:]==[(torch.Size([35,64]),torch.Size([64,3]),s) for s in (1,2,4)]
        else:
            with pytest.raises((AssertionError,RuntimeError)):
                harness.load_module().awq_gemm_triton(*inputs(),1)
    for values,saved in observed: checks.unchanged(values,saved)
    assert module.awq_gemm_triton is public


@pytest.mark.parametrize('mode', ['correct','events','wrong_timed','stale','no_write',
    'changing_wrong','raise_replay','mutate_timed_a','mutate_timed_weight',
    'mutate_timed_scales','mutate_timed_zeros','mutate_replay_a',
    'mutate_replay_weight','mutate_replay_scales','mutate_replay_zeros',
    'zero_input_and_output','unobservable'])
def test_exact_timed_result_replay_and_input_restoration(modules, mode):
    harness, checks = modules
    input_tensor,qweight,scales,qzeros = inputs()
    values = (input_tensor,qweight,scales,qzeros)
    pristine = checks.snapshot(values)
    mod = SimpleNamespace(awq_gemm_triton=vector_oracle)
    def fn():
        mod.awq_gemm_triton(input_tensor,qweight,scales,qzeros,1)
    observed = []
    def benchmark(measured, *, timed_run, **kwargs):
        observed.append(kwargs)
        checks.unchanged(values,pristine)
        output = measured(); cached = output.clone()
        if mode=='wrong_timed': output.zero_()
        if mode=='zero_input_and_output': input_tensor.zero_(); output.zero_()
        if mode.startswith('mutate_timed_'):
            values[('a','weight','scales','zeros').index(mode.removeprefix('mutate_timed_'))].zero_()
        def replay():
            assert torch.isnan(output).all()
            assert all(not torch.equal(v,s) for v,s in zip(values,pristine))
            if mode=='raise_replay': raise RuntimeError('controlled replay failure')
            if mode=='stale': output.copy_(cached)
            elif mode=='changing_wrong': output.fill_(input_tensor[0,0])
            elif mode!='no_write': output.copy_(measured())
            if mode.startswith('mutate_replay_'):
                values[('a','weight','scales','zeros').index(mode.removeprefix('mutate_replay_'))].zero_()
            return output
        if mode!='unobservable': timed_run._bind(replay,output)
        return .125, {'benchmark_method':'cuda_event_fallback' if mode=='events' else 'cuda_graph',
                      'benchmark_fallback_reason':'explicit observable event' if mode=='events' else None}
    if mode in ('correct','events'):
        ms,metadata = checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
        assert ms==.125 and metadata['timed_output_checked']
        assert metadata['perturbed_input_replay_checked'] and metadata['source_buffers_unchanged']
        if mode=='events': assert metadata['benchmark_fallback_reason']=='explicit observable event'
    else:
        with pytest.raises((AssertionError,RuntimeError)):
            checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    checks.unchanged(values,pristine)
    assert mod.awq_gemm_triton is vector_oracle
    assert observed==[dict(warmup=10,repetition=100)]


def test_original_five_case_correctness_and_performance_are_wrapped(modules, monkeypatch):
    harness, checks = modules
    for name in ('randn','randint'):
        factory = getattr(torch,name)
        monkeypatch.setattr(torch,name,lambda *args,_factory=factory,**kwargs:
                            _factory(*args,**{**kwargs,'device':'cpu'}))
    real_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*args,**kwargs:
                        real_to(self,*('cpu' if a=='cuda' else a for a in args),**kwargs))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    mod = SimpleNamespace(awq_gemm_triton=vector_oracle)
    harness.load_module = lambda:mod
    timings = []
    def benchmark(measured,*,timed_run,**options):
        timings.append(options)
        output = measured()
        def replay(): output.copy_(measured()); return output
        timed_run._bind(replay,output)
        return .125, {'benchmark_method':'cuda_graph'}
    harness._benchmark_cuda_graph_or_events = benchmark
    checks.install(harness)
    assert harness.run_correctness()==(True,None)
    rows = harness.run_performance()
    assert len(rows)==5 and all(r['execution_time_ms']==.125 for r in rows)
    assert timings==[dict(warmup=10,repetition=100)]*5
    assert [tuple(r['params'].values()) for r in rows]==harness.TEST_SHAPES
    assert mod.awq_gemm_triton is vector_oracle
    adapter = load(TASK/'_arena_eval.py')
    assert adapter.load_harness().run_performance.__module__=='_awq_gemm_checks'
