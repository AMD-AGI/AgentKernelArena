"""RMS backward contract: full per-row outputs, private inputs and timed replay."""
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

ROOT=Path(__file__).resolve().parents[1]
TASKS=['instruction2triton/rocmbench/rmsnorm_bwd','triton2triton/rocmbench/hard/rmsnorm_bwd']


def load(path):
    spec=importlib.util.spec_from_file_location('_rms_'+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.fixture(scope='module',autouse=True)
def budget():
    threads=torch.get_num_threads();state=torch.random.get_rng_state();torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads);torch.random.set_rng_state(state)


@pytest.fixture(params=TASKS)
def task(request,monkeypatch):
    path=ROOT/'tasks'/request.param;ref=load(path/'_arena_reference.py');monkeypatch.setitem(sys.modules,'_arena_reference',ref);return path,ref


def context(dtype=torch.float32,centered=False):
    x=torch.tensor([[1.,2.,-3.],[2.,-1.,4.]],dtype=dtype)
    g=torch.tensor([.5,-1.,2.],dtype=dtype);go=torch.tensor([[2.,-3.,1.],[-1.,2.,3.]],dtype=dtype);eps=1e-5
    r=torch.rsqrt(x.float().square().mean(-1)+eps)
    return dict(x=x,g=g,grad_output=go,rsigma_buffer=r,dx_bench=torch.full_like(x,-7),dg_tmp_bench=torch.full(x.shape,-9,dtype=torch.float32),ZERO_CENTERED_GAMMA=centered,eps=eps)


def independent_autograd(c):
    x=c['x'].double().clone().requires_grad_();g=c['g'].double().clone().requires_grad_();go=c['grad_output'].double()
    r=torch.rsqrt(x.square().mean(-1,keepdim=True)+c['eps'])
    y=x*r*(g+(1 if c['ZERO_CENTERED_GAMMA'] else 0));(y*go).sum().backward()
    return x.grad.to(c['x'].dtype), (go*x.detach()*r.detach()).float(),g.grad


def execute(c):
    dx,dg,_=independent_autograd(c);c['dx_bench'].copy_(dx);c['dg_tmp_bench'].copy_(dg);return object()


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('centered',[False,True])
def test_independent_autograd_and_both_full_outputs(task,dtype,centered):
    _,ref=task;c=context(dtype,centered);check=ref.BackwardCheck(c);handle=execute(c);check(handle)
    _,_,gradient=independent_autograd(c);torch.testing.assert_close(check.expected[1].double().sum(0),gradient,atol=1e-5,rtol=1e-5)
    # Equal-and-opposite row errors preserve the summed gradient: they must fail.
    original_sum=c['dg_tmp_bench'].sum(0).clone();c['dg_tmp_bench'][0,1]+=5;c['dg_tmp_bench'][1,1]-=5
    torch.testing.assert_close(c['dg_tmp_bench'].sum(0),original_sum)
    with pytest.raises(ref.NumericalMismatch):check(handle)
    check.restore()


@pytest.mark.parametrize('name',['x','g','grad_output','rsigma_buffer'])
def test_all_four_inputs_are_immutable(task,name):
    _,ref=task;c=context();check=ref.BackwardCheck(c);execute(c);c[name].flatten()[0]+=1
    with pytest.raises(ValueError,match='Read-only'):check(object())
    check.restore();assert all(ref.equal_bytes(a,b) for a,b in zip(check.inputs,check.original))


def test_protected_forward_rsigma_is_independently_validated(task):
    _,ref=task;c=context();c['rsigma_buffer'].zero_()
    with pytest.raises(AssertionError):ref.BackwardCheck(c)


@pytest.mark.parametrize('mode',['valid','cache','no_write','wrong_timed','wrong_replay','dg_only_wrong','input_mutation','crash'])
def test_real_event_timer_full_outputs_and_fresh_restore(task,monkeypatch,mode):
    path,ref=task;adapter=load(path/'_arena_eval.py');c=context();keys=['x','g','grad_output','rsigma_buffer','dx_bench','dg_tmp_bench'];snapshots=[c[k].clone() for k in keys]
    phase=['initial'];cache=independent_autograd(c)[:2];seen=[]
    plugin=types.SimpleNamespace(action='performance',current_row={'test_case_id':'cpu'},exercised=set())
    class Base:
        def __init__(self,fn):
            self.op_callable=fn;self.prepare_fn=None;self.use_cuda_graph=False;self.fallback_reason='rmsnorm_bwd_graph_capture_native_segfault';self.config=types.SimpleNamespace(warm_up=10,repetition=100)
    class Timed:
        outputs=None
        def rerun(self):
            phase[0]='replay'
            if mode=='crash':raise RuntimeError('injected replay crash')
            return op()
    def timer(fn,**kwargs):
        phase[0]='timed';kwargs['timed_run'].outputs=fn();seen.append(kwargs);return [1.,2.],{'benchmark_method':'cuda_event_fallback'}
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=Timed,benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules,'performance_utils_pytest',types.SimpleNamespace(_compute_timing_stats=lambda times,cfg:{'mean':sum(times)/len(times)}))
    def op():
        if not(mode=='no_write' and phase[0]=='replay'):
            if mode=='cache':
                for name,value in zip(['dx_bench','dg_tmp_bench'],cache):c[name].copy_(value)
            else:execute(c)
        if mode=='wrong_'+phase[0]:c['dx_bench'][-1,-1]+=5
        if mode=='dg_only_wrong' and phase[0]=='replay':c['dg_tmp_bench'][-1,-1]+=5
        if mode=='input_mutation' and phase[0]=='replay':c['rsigma_buffer'][0]+=1
        return object()  # Real launch returns a kernel handle, not a tensor.
    bench=adapter.benchmark_type(Base,plugin,None)(op);bench.context=c
    if mode=='valid':
        bench.run_benchmark();assert plugin.current_row['metadata']['timed_output_checked']
        helper=load(ROOT/'src/tools/perf/performance_utils_pytest.py');before=[]
        monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',lambda fn,**kwargs:(before.append(kwargs) or [1.],{}))
        helper._measure_times(op,bench.config,prepare_fn=None,use_cuda_graph=False,fallback_reason=bench.fallback_reason)
        sig=inspect.signature(load(ROOT/'src/tools/perf/aka_benchmark.py').benchmark_cuda_graph_or_events_samples)
        def effective(kwargs):
            b=sig.bind_partial(None,**kwargs);b.apply_defaults();return {k:v for k,v in b.arguments.items() if k not in ['fn','timed_run']}
        assert effective(before[0])==effective(seen[0])
    else:
        with pytest.raises((ValueError,AssertionError,RuntimeError)):bench.run_benchmark()
        assert not plugin.exercised
    assert all(ref.equal_bytes(c[k],s) for k,s in zip(keys,snapshots))


def test_frozen_autograd_inputs_restore_without_leaf_inplace_errors(task):
    _,ref=task;x=torch.tensor([1.,2.],requires_grad=True);frozen=ref.FrozenInputs([x])
    with torch.no_grad():x[0]+=1
    with pytest.raises(ValueError):frozen.check()
    frozen.restore();frozen.check();assert x.requires_grad


def test_actual_scored_callable_is_backward_in_both_suites(task):
    path,_=task;source=(path/'rmsnorm_bwd.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    perf=nodes['test_performance'];op=next(n for n in ast.walk(perf) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='op_lambda')
    assert isinstance(op.value,ast.Lambda) and isinstance(op.value.body.func,ast.Subscript)
    assert ast.unparse(op.value.body.func.value)=='rms_bwd_kernel'
    assert [ast.unparse(arg) for arg in op.value.body.args[:6]]==['grad_output','x','g','rsigma_buffer','dx_bench','dg_tmp_bench']
    assert 'use_cuda_graph=False' in ast.get_source_segment(source,perf)
    assert hashlib.sha256(ast.get_source_segment(source,perf).encode()).hexdigest()==ORIGINAL_INSTRUCTION_PERFORMANCE


def test_original_kernels_autograd_classes_case_manifest_and_gates(task,monkeypatch):
    path,_=task;old=ORIGINAL[path.relative_to(ROOT).as_posix()];source=(path/'rmsnorm_bwd.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name,sha in {**old['functions'],**old['classes']}.items():assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in old['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    rows=json.loads((path/'workloads.json').read_text())['cases'];assert len(rows)==82 and sum('performance' in r['checks'] for r in rows)==42
    for expression in ['torch.allclose(y_triton, y_torch, atol=atol, rtol=rtol)','torch.allclose(rsigma, rsigma_torch, atol=atol, rtol=rtol)','torch.allclose(grad_x_triton, grad_x_ref, atol=atol, rtol=rtol)','torch.allclose(grad_g_triton, grad_g_ref, atol=atol, rtol=rtol)']:assert expression in source
    adapter=load(path/'_arena_eval.py');monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task');assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'

# Original task-byte expectations; no runtime Git dependency.
ORIGINAL = {'tasks/instruction2triton/rocmbench/rmsnorm_bwd': {'classes': {'RMSNorm': '598d8df1821cb5e9e7df07c185eed764ec41cfe32288a8189eb6ebdb41f0d6c4'},
                                                    'files': {'config.yaml': 'adbc3e5beca4c42d059c29eed4a2e380ef3893305cb8013df0df7025ebac04ac',
                                                              'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                              'workloads.json': 'a78e744d6d49f216e2800d25dae41805849bd83892ad52d21d65269b4fd5b2cb'},
                                                    'functions': {'_rmsnorm_bwd_dg_reduce': '87aed0f11db92ada44fd4a161eb05336ff3b5952f157849e8299b38f450ce108',
                                                                  'calculate_rmsnorm_bwd_gbps': 'a72db096973f29220e3793002d11daa7bdbf855be605224f56a44894934eac77',
                                                                  'calculate_rmsnorm_bwd_tflops': 'ebdb5620cc5f91ef56734f9af13da223aa96e0faac5ebb0c90cc00e009012010',
                                                                  'get_autotune_config': '6244982d0e7695fb733182da152884bb0b46fcdfcfdac2eae73743112cbbb1bd',
                                                                  'get_available_models': 'c9886e07603ca34a4ec2457d7b8499d18c91a5310645ee41156d8823fd6e5199',
                                                                  'get_cuda_autotune_config': '2c7b0e90a4f86912e3b7f206a7b5af2ff37a37b55952ddf870f3d257556e4356',
                                                                  'get_hip_autotune_config': '3a9b1d278dc1789294d406510bb0a4cc33a61868b495fbaf68c666bfff50c5e1',
                                                                  'get_model_configs': '24a2fe80e5ea5bf96a3b082158d51c4cd74b63a040faf4aae50e5fc9b442f9a3',
                                                                  'get_num_sms': '99d56b1c10718a3e08fe382fb864d22bf4720707c376739a8ed99bf623f8301d',
                                                                  'is_cuda': '344ddb015296f2ac8a4ec12629daceefbddc48714bc9071e21aa54eec0864218',
                                                                  'is_hip': '18a1484321fc6773db7fa5328ed8ad9af0261147caf1ed9d9ef3be4d42408c57',
                                                                  'main': '9abbc8849ff72de3a55131cb28c11d78ea5890fc31e4e7679409460786cd7a9d',
                                                                  'model_benchmark_configs': '14c5d82570da8c6f0a033896c5ee8e4db6052d1405a93281839a7854da0ae697',
                                                                  'parse_args': 'd02bcd72ba9eb30217c93f1221c51d47f6a0273b16955ae6ae30fac3761e573a',
                                                                  'rms_bwd_kernel': '11be699f801b66376fd93f0239ff070c909ecccb13396dc8ff5355cdf85f4b4f',
                                                                  'rms_kernel': '9cf645dc326bfde55777c7fd888ee33b33a5bb25ff1d018dc5037cf69cb19d94',
                                                                  'run_benchmark': '4d76de183623c5c05c9432f9237b4ac541531cb52b8041da32cf28e9a5debe8b',
                                                                  'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                  'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                  'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                                  'torch_rmsnorm_fwd': '63a44b586eba61b05faea142d56c6383ba3b5ecfb0eeba84ff071f636d71e65c'}},
 'tasks/triton2triton/rocmbench/hard/rmsnorm_bwd': {'classes': {'RMSNorm': '598d8df1821cb5e9e7df07c185eed764ec41cfe32288a8189eb6ebdb41f0d6c4'},
                                                    'files': {'config.yaml': 'adbc3e5beca4c42d059c29eed4a2e380ef3893305cb8013df0df7025ebac04ac',
                                                              'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                              'workloads.json': '8ff98739d9c4eeaaf64309e1bd3d2c1444e390c47a3db8f87320eb00dcb3ab88'},
                                                    'functions': {'_rmsnorm_bwd_dg_reduce': '87aed0f11db92ada44fd4a161eb05336ff3b5952f157849e8299b38f450ce108',
                                                                  'calculate_rmsnorm_fwd_gbps': 'e05d3d26c3ce03e078872fd0d8cb93b86c0f91f06cc2ed8f2b5fec020e0bc8f8',
                                                                  'calculate_rmsnorm_fwd_tflops': '272514842ca9416aa01612dc1cbc92ef15f7d299ee2372dea226f289b546b015',
                                                                  'get_autotune_config': '6244982d0e7695fb733182da152884bb0b46fcdfcfdac2eae73743112cbbb1bd',
                                                                  'get_available_models': 'c9886e07603ca34a4ec2457d7b8499d18c91a5310645ee41156d8823fd6e5199',
                                                                  'get_cuda_autotune_config': '2c7b0e90a4f86912e3b7f206a7b5af2ff37a37b55952ddf870f3d257556e4356',
                                                                  'get_hip_autotune_config': '3a9b1d278dc1789294d406510bb0a4cc33a61868b495fbaf68c666bfff50c5e1',
                                                                  'get_model_configs': '24a2fe80e5ea5bf96a3b082158d51c4cd74b63a040faf4aae50e5fc9b442f9a3',
                                                                  'get_num_sms': '99d56b1c10718a3e08fe382fb864d22bf4720707c376739a8ed99bf623f8301d',
                                                                  'is_cuda': '344ddb015296f2ac8a4ec12629daceefbddc48714bc9071e21aa54eec0864218',
                                                                  'is_hip': '18a1484321fc6773db7fa5328ed8ad9af0261147caf1ed9d9ef3be4d42408c57',
                                                                  'main': '9abbc8849ff72de3a55131cb28c11d78ea5890fc31e4e7679409460786cd7a9d',
                                                                  'model_benchmark_configs': '14c5d82570da8c6f0a033896c5ee8e4db6052d1405a93281839a7854da0ae697',
                                                                  'parse_args': 'd02bcd72ba9eb30217c93f1221c51d47f6a0273b16955ae6ae30fac3761e573a',
                                                                  'rms_bwd_kernel': '11be699f801b66376fd93f0239ff070c909ecccb13396dc8ff5355cdf85f4b4f',
                                                                  'rms_kernel': '9cf645dc326bfde55777c7fd888ee33b33a5bb25ff1d018dc5037cf69cb19d94',
                                                                  'run_benchmark': '4d76de183623c5c05c9432f9237b4ac541531cb52b8041da32cf28e9a5debe8b',
                                                                  'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                  'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                  'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                                  'torch_rmsnorm_fwd': '63a44b586eba61b05faea142d56c6383ba3b5ecfb0eeba84ff071f636d71e65c'}}}
ORIGINAL_INSTRUCTION_PERFORMANCE = '98e2c027a61ecfa120d97db3678c7c4d5afbb685c84f5948bfcba6d9bf11ead5'


@pytest.mark.parametrize('dtype_str',['fp16','bf16'])
@pytest.mark.parametrize('centered',[False,True])
@pytest.mark.parametrize('mode',['good','mutate_go','wrong_dx','crash'])
def test_actual_original_autograd_body_protects_backward_inputs(task,dtype_str,centered,mode):
    path,ref=task;tree=ast.parse((path/'rmsnorm_bwd.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['RMSNorm','torch_rmsnorm_fwd','test_rmsnorm']]
    for node in nodes:node.decorator_list=[]
    records=[]
    class Kernel:
        def __init__(self,fn):self.fn=fn
        def __getitem__(self,grid):return self.fn
    def forward(y,x,g,r,*args,**kwargs):
        epsilon=args[4];is_centered=args[5]
        inv=torch.rsqrt(x.float().square().mean(-1)+epsilon);r.copy_(inv)
        y.copy_(x.float()*inv[:,None]*(g.float()+(1 if is_centered else 0)))
    def backward(go,x,g,r,dx,dg,*args,**kwargs):
        records.extend([(t,t.clone()) for t in (go,x,g,r)])
        gamma=g.float()+(1 if args[4] else 0);z=x.float();dy=go.float();inv=r[:,None]
        dx.copy_(dy*inv*gamma-inv**3*z*(dy*z*gamma).mean(-1,keepdim=True))
        dg.copy_(dy*z*inv)
        if mode in ['mutate_go','crash']:go[0,0]+=1
        if mode=='wrong_dx':dx[-1,-1]+=10
        if mode=='crash':raise RuntimeError('injected kernel failure')
    def reduce(dg,out,*args,**kwargs):out.copy_(dg.sum(0).reshape_as(out))
    class CPUTorch:
        def __getattr__(self,name):
            fn=getattr(torch,name)
            if name not in ['randn','ones','zeros','empty','empty_like','zeros_like']:return fn
            def call(*args,**kwargs):kwargs.pop('device',None);return fn(*args,**kwargs)
            return call
    ns={'torch':CPUTorch(),'triton':types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()),'get_num_sms':lambda:1,'set_seed':lambda:torch.manual_seed(42),'result_gold':{},
        'rms_kernel':Kernel(forward),'rms_bwd_kernel':Kernel(backward),'_rmsnorm_bwd_dg_reduce':Kernel(reduce),'arg_to_torch_dtype':{'fp16':torch.float16,'bf16':torch.bfloat16}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-rms-autograd>','exec'),ns);ns['rmsnorm']=ns['RMSNorm'].apply
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu',user_properties=[]))
    run=lambda:ns['test_rmsnorm'](2,10,centered,dtype_str,dtype_str,request)
    if mode=='good':run();assert request.node.user_properties
    else:
        with pytest.raises((ValueError,AssertionError,RuntimeError)):run()
        assert not request.node.user_properties
    assert records and all(ref.equal_bytes(x,y) for x,y in records)
