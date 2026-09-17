"""CPU numerical/adapter regressions; full GPU qualification is recorded separately."""
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASKS=[f'{s}/{n}' for s in ['instruction2triton/rocmbench','triton2triton/rocmbench/medium']
       for n in ['softmax','naive_softmax']]


def load(path):
    spec=importlib.util.spec_from_file_location('_softmax_test_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture(scope='module',autouse=True)
def cpu_budget():
    previous=torch.get_num_threads();rng=torch.random.get_rng_state();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous);torch.random.set_rng_state(rng)


@pytest.fixture(params=TASKS)
def task(request,monkeypatch):
    path=ROOT/'tasks'/request.param;ref=load(path/'_arena_reference.py')
    monkeypatch.setitem(sys.modules,'_arena_reference',ref)
    return path,ref


def scalar_reference(x):
    rows=[]
    for row in x.tolist():
        maximum=max(row);exps=[math.exp(v-maximum) for v in row];total=math.fsum(exps)
        rows.append([v/total for v in exps])
    return torch.tensor(rows,dtype=torch.float32)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_independent_known_answers_and_full_output_negative(task,dtype):
    _,ref=task
    x=torch.tensor([[0,0,0,0],[-1000,-1000,-1000,-1000],[0,1000,-1000,0],[1,2,3,4]],dtype=dtype)
    check=ref.SoftmaxCheck(x);expected=scalar_reference(x)
    assert expected[0].tolist()==[.25]*4 and expected[1].tolist()==[.25]*4
    assert expected[2].tolist()==[0,1,0,0]
    assert torch.allclose(check.expected,expected,atol=1e-8,rtol=1e-5)
    good=expected.to(dtype);check(good)
    for bad in [torch.zeros_like(good),good.roll(1,1),torch.full_like(good,1/x.shape[1])]:
        with pytest.raises(ref.NumericalMismatch):check(bad)
    bad=good.clone();bad[-1,-1]+=0.1
    with pytest.raises(ref.NumericalMismatch):check(bad)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_quantized_valid_fp32_interval_and_invalid_next_value(task,dtype):
    _,ref=task
    v=torch.linspace(-3,3,4001).to(dtype)
    x=torch.stack((torch.zeros_like(v),v),dim=1)
    check=ref.SoftmaxCheck(x);r=torch.softmax(x.float(),dim=1)
    assert (ref.ATOL,ref.RTOL)==(1e-8,1e-5)
    # Construct a value independently proven acceptable BEFORE output storage.
    # The gate is fixed; test data does not determine its tolerance.
    allowed=r+(1e-8+1e-5*r.abs())*.5
    assert torch.allclose(allowed,r,atol=1e-8,rtol=1e-5)
    stored=allowed.to(dtype)
    check(stored)
    # Adjacent rounded values can differ more than the FP32 gate applied after
    # storage, even though their pre-storage values satisfy that exact gate.
    assert not torch.allclose(stored,torch.softmax(x,dim=1),atol=1e-8,rtol=1e-5)
    # One representable step beyond the admissible endpoint must still fail.
    upper=(r+(1e-8+1e-5*r.abs())).to(dtype)
    bad=stored.clone();bad[0,0]=torch.nextafter(upper[0,0],torch.tensor(float('inf'),dtype=dtype))
    with pytest.raises(ref.NumericalMismatch):check(bad)


def test_fp32_gate_is_unchanged_and_reference_does_not_follow_input_mutation(task):
    _,ref=task;x=torch.tensor([[0.,0.,0.,0.],[1.,2.,3.,4.]])
    check=ref.SoftmaxCheck(x);reference=check.expected.clone();radius=1e-8+1e-5*reference.abs()
    check(reference+radius*.5)
    with pytest.raises(ref.NumericalMismatch):check(reference+radius*2)
    # A constant row shift leaves softmax unchanged but still violates read-only.
    x.add_(8)
    assert torch.allclose(torch.softmax(x,dim=1),reference)
    assert torch.equal(check.expected,reference)
    with pytest.raises(ValueError,match='Read-only'):check(reference)
    check.restore();x[0,0]=-0.
    with pytest.raises(ValueError,match='Read-only'):check(reference)
    check.restore()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_perturbation_changes_softmax_not_just_row_shift_and_restores(task,dtype):
    _,ref=task;x=torch.tensor([[0,1,2,3],[-1,-2,-3,-4]],dtype=dtype);pristine=x.clone()
    check=ref.SoftmaxCheck(x);old=check.expected.to(dtype);saved=old.clone()
    check.fresh(old)
    assert bool(torch.isnan(old).all())
    assert not torch.allclose(torch.softmax(pristine.float(),dim=1),check.expected,atol=1e-8,rtol=1e-5)
    with pytest.raises(ref.NumericalMismatch):check(saved)
    check(check.expected.to(dtype))
    check.restore()
    assert torch.equal(x,pristine) and torch.equal(old,saved)


@pytest.mark.parametrize('shape',[(1,1),(128,1),(1,128),(359,1),(1,359)])
@pytest.mark.parametrize('mutate',[False,True])
def test_original_correctness_hook_pristine_reference_and_single_column(task,shape,mutate):
    path,ref=task;captures=[]
    class Torch:
        def __getattr__(self,name):return getattr(torch,name)
        def randn(self,*a,**kw):kw['device']='cpu';return torch.randn(*a,**kw)
    def candidate(x):
        captures.append((x,x.clone()))
        result=torch.softmax(x,dim=1)
        if mutate:x.add_(8)
        return result
    source=(path/(path.name+'.py')).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='test_softmax');node.decorator_list=[]
    ns=dict(torch=Torch(),set_seed=lambda:torch.manual_seed(42),softmax=candidate,result_gold={})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path/(path.name+'.py')),'exec'),ns)
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu_original',user_properties=[]))
    if mutate:
        with pytest.raises(ValueError,match='Read-only'):ns['test_softmax'](*shape,request)
        assert not request.node.user_properties
    else:
        ns['test_softmax'](*shape,request)
        assert dict(request.node.user_properties)['softmax_contract']['original_fp32_gate']
    assert torch.equal(*captures[0])


@pytest.mark.parametrize('mode',['good_graph','good_events','cached','no_replay_write','wrong_tail',
                                 'wrong_timed','modify_initial','modify_timed','modify_replay',
                                 'crash','replay_crash','unobservable','bad_timing','alias_input'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_actual_allocating_adapter_timed_output_replay_state_and_timing(task,monkeypatch,mode,dtype):
    path,ref=task;adapter=load(path/'_arena_eval.py')
    x=torch.tensor([[0,1,2,3],[-1,-2,-3,-4]],dtype=dtype);pristine=x.clone()
    cached=torch.softmax(x.float(),dim=1).to(dtype);buffer=torch.empty_like(x)
    stage=['initial'];timer_calls=[]
    plugin=types.SimpleNamespace(action='performance',current_row={'test_case_id':'cpu'},exercised=set())
    class Base:
        def __init__(self,op_callable):
            self.op_callable=op_callable;self.prepare_fn=None;self.use_cuda_graph=mode!='good_events'
            self.fallback_reason='explicit observable events' if mode=='good_events' else None
            self.config=types.SimpleNamespace(warm_up=10,repetition=100)
    class Timed:
        def rerun(self):
            stage[0]='replay'
            if mode=='replay_crash':raise RuntimeError(mode)
            return self.fn()
    def benchmark(fn,**kwargs):
        timer_calls.append(kwargs)
        if mode in ['crash','unobservable']:raise RuntimeError(mode)
        stage[0]='timed';t=kwargs['timed_run'];t.fn=fn;t.outputs=fn()
        metadata={'benchmark_method':'cuda_event_fallback' if mode=='good_events' else 'cuda_graph',
                  'benchmark_samples':100,'benchmark_warmup':10}
        if mode=='good_events':metadata['benchmark_fallback_reason']='explicit observable events'
        return ([float('nan')] if mode=='bad_timing' else [1.,2.]),metadata
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=Timed,benchmark_cuda_graph_or_events_samples=benchmark))
    monkeypatch.setitem(sys.modules,'performance_utils_pytest',types.SimpleNamespace(_compute_timing_stats=lambda times,cfg:{'mean':sum(times)/len(times)}))
    def op():
        out=torch.empty_like(x) if stage[0]=='initial' or mode=='good_events' else buffer
        if not (stage[0]=='replay' and mode=='no_replay_write'):
            out.copy_(cached if mode=='cached' else torch.softmax(x.float(),dim=1).to(dtype))
        if mode=='modify_'+stage[0]:x.add_(8)
        if mode=='wrong_timed' and stage[0]=='timed' or mode=='wrong_tail' and stage[0]=='replay':out[-1,-1]+=0.1
        return x if mode=='alias_input' else out
    wrapped=adapter.benchmark_type(Base,plugin,None)(op);wrapped.context={'x':x}
    if mode in ['good_graph','good_events']:
        wrapped.run_benchmark(baseline_callable=lambda:pytest.fail('peer timing forbidden'))
        assert plugin.exercised=={'cpu'} and plugin.current_row['execution_time_ms']==1.5
        metadata=plugin.current_row['metadata']
        assert metadata['fresh_input_replay_checked'] and metadata['timed_output_checked'] and metadata['readonly_input_checked']
        assert metadata['input_state_restored'] and metadata['poisoned_output_restored']
        if mode=='good_events':assert metadata['device_timing']['benchmark_fallback_reason']=='explicit observable events'
        helper=load(ROOT/'src/tools/perf/performance_utils_pytest.py');prior=[]
        monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',lambda fn,**kw:(prior.append(kw) or [1.],{}))
        helper._measure_times(op,wrapped.config,prepare_fn=wrapped.prepare_fn,use_cuda_graph=wrapped.use_cuda_graph,fallback_reason=wrapped.fallback_reason)
        canonical=load(ROOT/'src/tools/perf/aka_benchmark.py');sig=inspect.signature(canonical.benchmark_cuda_graph_or_events_samples)
        def effective(kw):
            bound=sig.bind_partial(None,**kw);bound.apply_defaults()
            return {k:v for k,v in bound.arguments.items() if k not in ['fn','timed_run']}
        assert effective(timer_calls[0])==effective(prior[0])
    else:
        with pytest.raises((ref.NumericalMismatch,ValueError,RuntimeError)):wrapped.run_benchmark()
        assert not plugin.exercised and 'execution_time_ms' not in plugin.current_row
    assert torch.equal(x,pristine)
    if mode in ['good_graph','cached','no_replay_write','wrong_tail','modify_replay','replay_crash']:
        assert torch.equal(buffer,cached)


@pytest.mark.parametrize('bad',['dtype','shape','nan','non_tensor'])
def test_output_contract(task,bad):
    _,ref=task;x=torch.zeros(2,4);check=ref.SoftmaxCheck(x);out=torch.softmax(x,dim=1)
    if bad=='dtype':out=out.half()
    if bad=='shape':out=out.reshape(4,2)
    if bad=='nan':out[-1,-1]=float('nan')
    if bad=='non_tensor':out=None
    with pytest.raises((ValueError,TypeError)):check(out)


# Original immutable hashes; tests require no transient git history.
ORIGINAL = {'instruction2triton/rocmbench/naive_softmax': {'config.yaml': '029522ae12fc59a4153f52f731091ac3cc3ed51657e7ec8e8936b48e05253413',
                                                'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                'softmax': '385a722dee67c1b5c26600de6d27d68a52e5651d6c5a74c9c17e66356aeb1081',
                                                'softmax_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                'softmax_kernel_naive': '0eef73deaa488269b7622da7b881c0b0e327513eac4b8b3a97ef631b39ae5ec3',
                                                'softmax_kernel_naive_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                'test_performance': '9c25ada07ba95a021787641d8ea0c135aaad14c3293fdb0b2b32f450bdbae8c1',
                                                'test_performance_decorators': '7745185e667ca62d684f5fbdaffca20e02354c43f5d73684d4e603c178965a9d',
                                                'test_softmax_decorators': '74d22fcbd91c2c85aa15e4a700453c5d2fb812662fc643563c73e78240c405a5',
                                                'workloads.json': '573ff277e86b6af6aacd73ac4a0fd4460df979f410d0d08a9aa0e102c143759f'},
 'instruction2triton/rocmbench/softmax': {'config.yaml': 'f26b3115bd19e55bbfbea4e94da645efb040bb617b23be9ad5d0d938e55b8e29',
                                          'get_autotune_config': '6244982d0e7695fb733182da152884bb0b46fcdfcfdac2eae73743112cbbb1bd',
                                          'get_autotune_config_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                          'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                          'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                          'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                          'softmax': 'fc439017574b4a60dce5a57dd642ea52754d2c0441f92abb4bedbb3eee6b9fa0',
                                          'softmax_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                          'softmax_kernel_online': '7682d5f9402972e803255f366f40fce0c88a7cd956bd16cdac5ddfd152745776',
                                          'softmax_kernel_online_decorators': '66970bc765542eb3bf9c2eb2774adead9b9757bd364b54a3efcafe160a1cc17c',
                                          'test_performance': '9c25ada07ba95a021787641d8ea0c135aaad14c3293fdb0b2b32f450bdbae8c1',
                                          'test_performance_decorators': '7745185e667ca62d684f5fbdaffca20e02354c43f5d73684d4e603c178965a9d',
                                          'test_softmax_decorators': '74d22fcbd91c2c85aa15e4a700453c5d2fb812662fc643563c73e78240c405a5',
                                          'workloads.json': '876bff095c2126dc6ddd90beccd64a66e997f6559ff038b558ba854c39755684'},
 'triton2triton/rocmbench/medium/naive_softmax': {'config.yaml': '029522ae12fc59a4153f52f731091ac3cc3ed51657e7ec8e8936b48e05253413',
                                                  'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                  'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                  'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                  'softmax': '385a722dee67c1b5c26600de6d27d68a52e5651d6c5a74c9c17e66356aeb1081',
                                                  'softmax_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                  'softmax_kernel_naive': '0eef73deaa488269b7622da7b881c0b0e327513eac4b8b3a97ef631b39ae5ec3',
                                                  'softmax_kernel_naive_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                  'test_performance': '04eaed48b32f8bc89c15eddbd78a127244f673061d4db664a8497a0d496dde2f',
                                                  'test_performance_decorators': '7745185e667ca62d684f5fbdaffca20e02354c43f5d73684d4e603c178965a9d',
                                                  'test_softmax_decorators': '74d22fcbd91c2c85aa15e4a700453c5d2fb812662fc643563c73e78240c405a5',
                                                  'workloads.json': '5fdcfcaf3c23b6f59bebb4c4ef22c9a67b7c0bc4af5c936d191e88b9ed27c961'},
 'triton2triton/rocmbench/medium/softmax': {'config.yaml': 'f26b3115bd19e55bbfbea4e94da645efb040bb617b23be9ad5d0d938e55b8e29',
                                            'get_autotune_config': '6244982d0e7695fb733182da152884bb0b46fcdfcfdac2eae73743112cbbb1bd',
                                            'get_autotune_config_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                            'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                            'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                            'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                            'softmax': 'fc439017574b4a60dce5a57dd642ea52754d2c0441f92abb4bedbb3eee6b9fa0',
                                            'softmax_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                            'softmax_kernel_online': '7682d5f9402972e803255f366f40fce0c88a7cd956bd16cdac5ddfd152745776',
                                            'softmax_kernel_online_decorators': '66970bc765542eb3bf9c2eb2774adead9b9757bd364b54a3efcafe160a1cc17c',
                                            'test_performance': '04eaed48b32f8bc89c15eddbd78a127244f673061d4db664a8497a0d496dde2f',
                                            'test_performance_decorators': '7745185e667ca62d684f5fbdaffca20e02354c43f5d73684d4e603c178965a9d',
                                            'test_softmax_decorators': '74d22fcbd91c2c85aa15e4a700453c5d2fb812662fc643563c73e78240c405a5',
                                            'workloads.json': '6eb8ffa67f661c05841796b81713eeb3b6dd8a52f9be564c3238d24ffc18cfa9'}}


def test_kernels_autotune_wrappers_manifest_and_timing_are_original(task):
    path,_=task;expected=ORIGINAL[path.relative_to(ROOT/'tasks').as_posix()]
    source=(path/(path.name+'.py')).read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,digest in expected.items():
        if name.endswith('.yaml') or name.endswith('.json') or name=='performance_utils_pytest.py':
            actual=(path/name).read_bytes()
        elif name.endswith('_decorators'):
            actual=ast.dump(ast.Module(body=nodes[name[:-11]].decorator_list,type_ignores=[])).encode()
        else:actual=ast.get_source_segment(source,nodes[name]).encode()
        assert hashlib.sha256(actual).hexdigest()==digest
    cases=json.loads((path/'workloads.json').read_text())['cases']
    assert len(cases)==31 and sum('performance' in r['checks'] for r in cases)==21
    perf=[r['params']['arguments'] for r in cases if 'performance' in r['checks']]
    assert {(r['M'],r['N'],r['dtype_str']) for r in perf}=={
        (m,n,d) for m,n in [(2048,2048),(4096,4096),(8192,8192),(1,32000),(1,131072),(1024,8192),(512,32000)]
        for d in ['fp16','bf16','fp32']}
    assert 'assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)' in source
    assert 'y_torch = check.expected' in source and "x = torch.randn(M, N, device='cuda')" in source
