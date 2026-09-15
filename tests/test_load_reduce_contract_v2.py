"""CPU contract/control simulations; actual GPU qualification is recorded separately."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['triton2triton/rocmbench/easy/test_load_reduce', 'instruction2triton/rocmbench/test_load_reduce']
SHAPES = [(128,64),(128,128),(128,256),(128,512),(128,1024),
          (256,64),(256,128),(256,256),(256,512),(512,64),(512,128),(512,256),
          (1024,64),(1024,128)]


def load(path):
    spec = importlib.util.spec_from_file_location('_reduce_test_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    rng = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.random.set_rng_state(rng)


@pytest.fixture(params=TASKS, ids=['triton','instruction'])
def task(request, monkeypatch):
    path = ROOT/'tasks'/request.param
    ref = load(path/'_arena_reference.py')
    monkeypatch.setitem(sys.modules, '_arena_reference', ref)
    return path, ref


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_known_rows_negative_ties_last_column_and_preserved_gate(task, dtype):
    _, ref = task
    x = torch.tensor([[-9,-2,-5,-2], [0,2,3,9], [7,7,-1,0], [-1,-8,-4,-0.5]], dtype=dtype)
    y = torch.empty(4, dtype=dtype)
    check = ref.ReductionCheck(x,y)
    assert check.expected.tolist() == [-2,9,7,-0.5]
    assert check.expected.data_ptr() != y.data_ptr()
    y.copy_(torch.tensor([-2,9,7,-0.5], dtype=dtype));check(y)
    y[-1] = 0
    with pytest.raises(ref.NumericalMismatch):check(y)
    # The original tolerance and check_dtype=False remain real behavior.
    ref.compare(torch.tensor([3.02],dtype=torch.float64),torch.tensor([3.]),
                atol=1e-3,rtol=1e-2,check_dtype=False)
    with pytest.raises(ref.NumericalMismatch):
        ref.compare(torch.tensor([3.1]),torch.tensor([3.]),atol=1e-3,rtol=1e-2,check_dtype=False)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape', SHAPES)
def test_all42_original_performance_shapes_fresh_oracle_and_restore(task, dtype, shape):
    _, ref = task
    x = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(42))
    y = torch.full((shape[0],),19.,dtype=dtype)
    original_x=x.clone();original_y=y.clone()
    check = ref.prepare({'x':x,'y_buffer':y},None)
    try:
        y.copy_(x.max(dim=1).values);check(y)
        cached=y.clone()
        check.fresh()
        assert x.shape==original_x.shape and x.stride()==original_x.stride() and x.dtype==dtype
        assert not torch.allclose(check.expected,cached,atol=1e-3,rtol=1e-2)
        assert bool(torch.all(check.expected[::2]<0)) and bool(torch.all(check.expected[1::2]>0))
        assert bool(torch.isnan(y).all())
        y.copy_(cached)
        with pytest.raises(ref.NumericalMismatch):check(y)
        y.copy_(x.max(dim=1).values);check(y)
        # Changing a non-maximal input is still forbidden, even if y is unchanged.
        minimum=int(torch.argmin(x[0]));x[0,minimum]-=1
        with pytest.raises(ValueError,match='Read-only'):check(y)
    finally:
        check.restore()
    assert torch.equal(x,original_x) and torch.equal(y,original_y)


def test_input_bytes_and_reference_independence(task):
    _, ref = task
    x=torch.tensor([[0.,3.],[-5.,-2.]])
    y=torch.empty(2)
    check=ref.ReductionCheck(x,y)
    expected=check.expected.clone()
    x.zero_();y.zero_()
    # This would pass the original post-invocation oracle; it must fail now.
    torch.testing.assert_close(y,x.max(dim=1).values,rtol=1e-2,atol=1e-3,check_dtype=False)
    assert torch.equal(check.expected,expected)
    with pytest.raises(ValueError,match='Read-only'):check(y)
    check.restore();y.copy_(expected);x[0,0]=-0.
    with pytest.raises(ValueError,match='Read-only'):check(y)
    check.restore()


@pytest.mark.parametrize('mode',['good','zero_input_and_output','modify_nonmaximum'])
def test_actual_original_correctness_hook_uses_snapshot_and_restores(task,mode):
    path,ref=task
    captures=[]
    class Torch:
        def __getattr__(self,name):return getattr(torch,name)
        def randn(self,*a,**kw):kw['device']='cpu';return torch.randn(*a,**kw)
        def empty(self,*a,**kw):kw['device']='cpu';return torch.empty(*a,**kw)
        def set_printoptions(self,**kw):pass
    class Kernel:
        def __getitem__(self,grid):
            def launch(x,y,sm,sn,sy,m,n):
                assert grid==(1,) and (sm,sn,sy,m,n)==(64,1,1,128,64)
                captures.append((x,y,x.clone()))
                y.copy_(x.max(dim=1).values)
                if mode=='zero_input_and_output':x.zero_();y.zero_()
                if mode=='modify_nonmaximum':x[0,torch.argmin(x[0])]-=1
            return launch
    source=(path/'test_load_reduce.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='test_load_reduce')
    node.decorator_list=[]
    ns=dict(torch=Torch(),set_seed=lambda:torch.manual_seed(42),dtype_mapping={'float16':torch.float16},
            load_reduce_kernel=Kernel(),result_gold={},assert_close=torch.testing.assert_close)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path/'test_load_reduce.py'),'exec'),ns)
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='original_case',user_properties=[]))
    if mode=='good':
        ns['test_load_reduce'](128,64,'float16',request)
        assert dict(request.node.user_properties)['load_reduce_contract']['independent_input_snapshot']
    else:
        with pytest.raises((AssertionError,ValueError)):
            ns['test_load_reduce'](128,64,'float16',request)
        assert not request.node.user_properties
    assert len(captures)==1 and torch.equal(captures[0][0],captures[0][2])


@pytest.mark.parametrize('action',['correctness','performance'])
@pytest.mark.parametrize('mode',['good','events','cached','no_replay_write','wrong_tail','wrong_timed',
                                 'modify_initial','modify_timed','modify_replay','crash','replay_crash',
                                 'unobservable','bad_timing','wrong_alias'])
def test_real_adapter_observed_output_fresh_replay_readonly_and_finally_restore(task,monkeypatch,action,mode):
    path,ref=task;adapter=load(path/'_arena_eval.py')
    x=torch.tensor([[-5.,-2.,-4.,-9.],[0.,7.,3.,2.],[-2.,-1.,-3.,-8.],[4.,2.,4.,1.]])
    y=torch.full((4,),19.);original_x=x.clone();original_y=y.clone();cached=x.max(dim=1).values
    stage=['initial'];calls=[]
    plugin=types.SimpleNamespace(action=action,current_row={'test_case_id':'cpu'},exercised=set())
    class Base:
        def __init__(self,op_callable):
            self.op_callable=op_callable;self.prepare_fn=None;self.use_cuda_graph=mode!='events'
            self.fallback_reason='explicit event path' if mode=='events' else None
            self.config=types.SimpleNamespace(warm_up=10,repetition=100)
    class Timed:
        def rerun(self):
            stage[0]='replay'
            if mode=='replay_crash':raise RuntimeError('injected replay failure')
            return self.fn()
    def benchmark(fn,**kwargs):
        calls.append(kwargs)
        if mode in ['crash','unobservable']:raise RuntimeError(mode)
        stage[0]='timed';t=kwargs['timed_run'];t.fn=fn;t.outputs=fn()
        metadata={'benchmark_method':'cuda_event_fallback' if mode=='events' else 'cuda_graph',
                  'benchmark_warmup':10,'benchmark_samples':100}
        if mode=='events':metadata['benchmark_fallback_reason']='explicit event path'
        return ([float('nan')] if mode=='bad_timing' else [1.,2.]),metadata
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(
        TimedRun=Timed,benchmark_cuda_graph_or_events_samples=benchmark))
    monkeypatch.setitem(sys.modules,'performance_utils_pytest',types.SimpleNamespace(
        _compute_timing_stats=lambda times,cfg:{'mean':sum(times)/len(times)}))
    def op():
        if not (stage[0]=='replay' and mode=='no_replay_write'):
            y.copy_(cached if mode=='cached' else x.max(dim=1).values)
        if mode=='modify_'+stage[0]:x[0,torch.argmin(x[0])]-=1
        if stage[0]=='replay' and mode=='wrong_tail':y[-1]+=5
        if stage[0]=='timed' and mode=='wrong_timed':y[-1]+=5
        return y.clone() if stage[0]=='timed' and mode=='wrong_alias' else y
    wrapper=adapter.benchmark_type(Base,plugin,None)(op)
    wrapper.context={'x':x,'y_buffer':y}
    succeeds=mode in ['good','events'] or (action=='correctness' and mode!='modify_initial')
    if succeeds:
        wrapper.run_benchmark(baseline_callable=lambda:pytest.fail('peer timing forbidden'))
        assert plugin.exercised=={'cpu'}
        if action=='performance':
            metadata=plugin.current_row['metadata']
            assert metadata['timed_output_checked'] and metadata['fresh_input_replay_checked']
            assert metadata['readonly_input_checked'] and metadata['input_state_restored']
            assert plugin.current_row['execution_time_ms']==1.5
            if mode=='events':assert metadata['device_timing']['benchmark_fallback_reason']=='explicit event path'
            helper=load(ROOT/'src/tools/perf/performance_utils_pytest.py');prior=[]
            monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',
                lambda fn,**kw:(prior.append(kw) or [1.],{}))
            helper._measure_times(op,wrapper.config,prepare_fn=wrapper.prepare_fn,
                                  use_cuda_graph=wrapper.use_cuda_graph,fallback_reason=wrapper.fallback_reason)
            canonical=load(ROOT/'src/tools/perf/aka_benchmark.py')
            sig=inspect.signature(canonical.benchmark_cuda_graph_or_events_samples)
            def effective(options):
                bound=sig.bind_partial(None,**options);bound.apply_defaults()
                return {k:v for k,v in bound.arguments.items() if k not in ['fn','timed_run']}
            assert effective(calls[0])==effective(prior[0])
    else:
        with pytest.raises((ref.NumericalMismatch,ValueError,RuntimeError)):
            wrapper.run_benchmark()
        assert not plugin.exercised and 'execution_time_ms' not in plugin.current_row
    assert torch.equal(x,original_x) and torch.equal(y,original_y)


def test_output_contract_nonfinite_shape_and_alias(task):
    _,ref=task;x=torch.tensor([[1.,2.],[3.,4.]]);y=torch.empty(2)
    check=ref.ReductionCheck(x,y);y.copy_(check.expected);check(y)
    with pytest.raises(ValueError,match='declared output'):check(y.clone())
    y[0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):check(y)
    with pytest.raises(ValueError,match='shape/device'):
        ref.compare(torch.ones(2,1),torch.ones(2),atol=1e-3,rtol=1e-2,check_dtype=False)


# Fixed original digests, independent of transient branch history/shallow clones.
ORIGINAL = {'instruction2triton/rocmbench/test_load_reduce': {'calculate_load_reduce_gbps': 'e32f29203ca54047e975524d020df5ac76d98c6c6418352f842da28c907f4b3c',
                                                   'calculate_load_reduce_gbps_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'calculate_load_reduce_tflops': '80d90eff19322092fb706767299d9ba04d6e4c33b42eabbbd941dcbc42c17f9c',
                                                   'calculate_load_reduce_tflops_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'config.yaml': '15fb6b4428ad91c448ea37d55a9d9157986df54f88e4453b1124d29d4bcd0174',
                                                   'load_reduce_kernel': 'b00a412ad8eba94ffd379773aea4bfdc58dd0d42636671502293f120e7867e8e',
                                                   'load_reduce_kernel_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                   'load_reduce_triton_wrapper': '6f1b329472d2ecb75a24a47182049a7fc616ae192a9c561604917a9fe673608c',
                                                   'load_reduce_triton_wrapper_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                   'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                   'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'test_load_reduce': '920d2447113354badd65589a1675e5bda58a3bfb6b3d91fe62c6c1028416a2bc',
                                                   'test_load_reduce_decorators': '83f4e91200252fdbe53b438f1160105741a1bce10c3d91bf5100b4613c7ab601',
                                                   'test_performance': 'd1eae0fca09932e0c07fc509f506734210adc5576c193a4f070fde279667ae39',
                                                   'test_performance_decorators': '274599a0a653d7020c6ebe33f9844ffc965e18f696421b7a225bc379e19d93f2',
                                                   'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                   'test_save_performance_results_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                   'test_save_results_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'workloads.json': '84c85de315d05598e05e2d1d57a0ea79c92dab070408d95b7afb2b51dd53b384'},
 'triton2triton/rocmbench/easy/test_load_reduce': {'calculate_load_reduce_gbps': 'e32f29203ca54047e975524d020df5ac76d98c6c6418352f842da28c907f4b3c',
                                                   'calculate_load_reduce_gbps_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'calculate_load_reduce_tflops': '80d90eff19322092fb706767299d9ba04d6e4c33b42eabbbd941dcbc42c17f9c',
                                                   'calculate_load_reduce_tflops_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'config.yaml': '15fb6b4428ad91c448ea37d55a9d9157986df54f88e4453b1124d29d4bcd0174',
                                                   'load_reduce_kernel': 'b00a412ad8eba94ffd379773aea4bfdc58dd0d42636671502293f120e7867e8e',
                                                   'load_reduce_kernel_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                   'load_reduce_triton_wrapper': '6f1b329472d2ecb75a24a47182049a7fc616ae192a9c561604917a9fe673608c',
                                                   'load_reduce_triton_wrapper_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                   'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                   'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'test_load_reduce': '920d2447113354badd65589a1675e5bda58a3bfb6b3d91fe62c6c1028416a2bc',
                                                   'test_load_reduce_decorators': '83f4e91200252fdbe53b438f1160105741a1bce10c3d91bf5100b4613c7ab601',
                                                   'test_performance': 'a54e3d5b51852c3be4554784587921a6f1bdfb2d6118c39337ff51526839c7b1',
                                                   'test_performance_decorators': '274599a0a653d7020c6ebe33f9844ffc965e18f696421b7a225bc379e19d93f2',
                                                   'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                   'test_save_performance_results_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                   'test_save_results_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                   'workloads.json': '6fb4c1bd444646f4d4350d0593f12abe1e1684ab96b595b45883e363b09fc885'}}


def test_original_kernel_workloads_timing_and_symbol_boundary_preserved(task):
    path,_=task;expected=ORIGINAL[path.relative_to(ROOT/'tasks').as_posix()]
    source=(path/'test_load_reduce.py').read_text();tree=ast.parse(source)
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        assert hashlib.sha256(ast.dump(ast.Module(body=node.decorator_list,type_ignores=[])).encode()).hexdigest()==expected[node.name+'_decorators']
        if node.name!='test_load_reduce':
            assert hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()==expected[node.name]
    for name in ['config.yaml','performance_utils_pytest.py','workloads.json']:
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==expected[name]
    manifest=json.loads((path/'workloads.json').read_text())
    assert len(manifest['cases'])==43
    perf=[r for r in manifest['cases'] if 'performance' in r['checks']]
    assert len(perf)==42
    combinations={(r['params']['arguments']['block_m_const'],r['params']['arguments']['block_n_const'],r['params']['arguments']['dtype_str']) for r in perf}
    assert combinations=={(m,n,d) for m,n in SHAPES for d in ['fp16','fp32','bf16']}
    assert 'assert_close(y, golden, rtol=1e-2, atol=1e-3, check_dtype=False)' in source
    assert 'golden = check.expected' in source and 'check = ReductionCheck(x, y)' in source
