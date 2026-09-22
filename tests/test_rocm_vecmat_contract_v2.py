"""Batched VecMat actual kernel-body CPU execution, FP16 products and FP32 sums."""
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
TASKS=['instruction2triton/rocmbench/test_batched_vecmat','triton2triton/rocmbench/easy/test_batched_vecmat']


def load(path):
    spec=importlib.util.spec_from_file_location('_bias_'+path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture(scope='module',autouse=True)
def budget():
    threads=torch.get_num_threads();state=torch.random.get_rng_state();torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads);torch.random.set_rng_state(state)


@pytest.fixture(params=TASKS)
def task(request,monkeypatch):
    path=ROOT/'tasks'/request.param;ref=load(path/'_arena_reference.py');monkeypatch.setitem(sys.modules,'_arena_reference',ref);return path,ref


def inputs(dtype):
    a=torch.tensor([[2,-3,4],[1,5,-2]],dtype=dtype)
    b=torch.tensor([[[1,2,-2],[3,-1,4]],[[1,2,-2],[3,-1,4]]],dtype=dtype)
    return a,b


def answer(a,b):
    return (a[:,None,:]*b).sum(2,dtype=torch.float32).to(a.dtype)


class Pointer:
    def __init__(self,tensor,index=0):self.tensor=tensor;self.index=index
    def __add__(self,index):return Pointer(self.tensor,self.index+index)


class TensorLanguage:
    float32=torch.float32
    constexpr=object
    def __init__(self,program):self.program=program;self.loads=0;self.stores=0
    def program_id(self,axis):return self.program[axis]
    def arange(self,start,stop):return torch.arange(start,stop)
    def zeros(self,shape,dtype):return torch.zeros(shape,dtype=dtype)
    def cdiv(self,a,b):return (a+b-1)//b
    def broadcast(self,a,b):return torch.broadcast_tensors(a,b)
    def trans(self,a):return a.T
    def sum(self,a,axis,dtype=None):return a.sum(axis,dtype=dtype)
    def load(self,pointer,mask=None,other=0):
        index=torch.as_tensor(pointer.index);mask=torch.ones_like(index,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,index.shape)
        active=index[mask];assert bool(((active>=0)&(active<pointer.tensor.numel())).all()),'out-of-bounds load'
        result=torch.full(index.shape,other,dtype=pointer.tensor.dtype);result[mask]=pointer.tensor.flatten()[active];self.loads+=1;return result
    def store(self,pointer,value,mask=None):
        index=torch.as_tensor(pointer.index);mask=torch.ones_like(index,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,index.shape)
        active=index[mask];assert bool(((active>=0)&(active<pointer.tensor.numel())).all()),'out-of-bounds store'
        pointer.tensor.flatten()[active]=value[mask].to(pointer.tensor.dtype);self.stores+=1


def run_actual_kernel(path,a,b,blocks,*,remove_masks=False):
    tree=ast.parse((path/'test_batched_vecmat.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batched_vecmat');node.decorator_list=[]
    if remove_masks:
        for child in ast.walk(node):
            if isinstance(child,ast.Call) and ast.unparse(child.func) in ['tl.load','tl.store']:child.keywords=[kw for kw in child.keywords if kw.arg!='mask']
    m,k=a.shape;n=b.shape[1];out=torch.full((m,n),float('nan'),dtype=a.dtype)
    for mi in range((m+blocks[0]-1)//blocks[0]):
        for ni in range((n+blocks[1]-1)//blocks[1]):
            ns={'tl':TensorLanguage((mi,ni))};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-vecmat-kernel>','exec'),ns)
            ns[node.name](Pointer(a),Pointer(b),m,n,k,Pointer(out),*blocks)
    return out


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('shape,blocks',[((2,3,5),(4,4,8)),((17,19,35),(16,32,64)),((16,32,128),(16,64,64)),((3,7,128),(4,8,32))])
def test_actual_kernel_full_outputs_partial_tiles_and_pristine_inputs(task,dtype,shape,blocks):
    path,ref=task;m,n,k=shape;a=((torch.arange(m*k)%17-8).reshape(m,k)/7).to(dtype);b=((torch.arange(m*n*k)%13-6).reshape(m,n,k)/3).to(dtype)
    snapshot=[a.clone(),b.clone()];out=run_actual_kernel(path,a,b,blocks);ref.VecmatCheck(a,b)(out)
    # Higher precision independent summation of input-dtype products.
    expected=(a[:,None,:]*b).double().sum(2).to(dtype);torch.testing.assert_close(out,expected,atol=1e-3,rtol=1e-2)
    assert all(torch.equal(x,y) for x,y in zip([a,b],snapshot))


def test_memory_negative_control_detects_missing_masks(task):
    path,_=task;a=torch.ones((2,5));b=torch.ones((2,3,5))
    with pytest.raises(AssertionError,match='out-of-bounds'):run_actual_kernel(path,a,b,(4,4,8),remove_masks=True)


def test_fp16_product_rounding_and_fp32_reduction_are_both_required(task):
    path,ref=task
    a=torch.tensor([[1.0009765625,1.0009765625]],dtype=torch.float16)
    b=torch.tensor([[[1.0009765625,-1.0]]],dtype=torch.float16)
    check=ref.VecmatCheck(a,b);expected=(a[:,None,:]*b).double().sum(2).half()
    assert torch.equal(check.expected,expected)
    # Large cancellation after a long sum: repeated half block rounding loses the unit tail.
    a=torch.ones((1,256),dtype=torch.float16);b=torch.ones((1,1,256),dtype=torch.float16);b[:,:,0]=2048;b[:,:,128]=-2048;b[:,:,-1]=-252.5
    actual=run_actual_kernel(path,a,b,(1,1,64));assert actual.item()==.5
    ref.VecmatCheck(a,b)(actual)
    wrong=torch.zeros((1,1),dtype=torch.float16)
    for start in range(0,256,64):wrong+=b[:,:,start:start+64].sum(2).half()
    assert wrong.item()!=actual.item()
    with pytest.raises(ref.NumericalMismatch):ref.VecmatCheck(a,b)(wrong)


def test_original_manifest_performance_and_numerical_gate(task,monkeypatch):
    path,_=task;old=ORIGINAL[path.relative_to(ROOT).as_posix()];source=(path/'test_batched_vecmat.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,sha in old['functions'].items():assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in old['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256(ast.dump(nodes['test_performance']).encode()).hexdigest()==old['performance_without_skip']
    assert 'np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)' in source
    rows=json.loads((path/'workloads.json').read_text())['cases'];assert len(rows)==33 and sum('performance' in r['checks'] for r in rows)==30
    assert hashlib.sha256(json.dumps(rows[:31],sort_keys=True,separators=(',',':')).encode()).hexdigest()==old['rows']
    assert all(r['checks']==['correctness'] for r in rows[31:])
    adapter=load(path/'_arena_eval.py');monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task');assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'


def test_only_reviewed_tail_and_accumulator_changes_to_kernel(task):
    path,_=task;old=ORIGINAL[path.relative_to(ROOT).as_posix()];node=next(n for n in ast.parse((path/'test_batched_vecmat.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='batched_vecmat');node.decorator_list=[]
    # Actual execution above verifies the new masks and accumulator. Normalize
    # exactly those reviewed changes to preserve every other kernel operation.
    for child in ast.walk(node):
        if isinstance(child,ast.Assign) and isinstance(child.targets[0],ast.Name):
            if child.targets[0].id=='vecmat':
                kw=next(k for k in child.value.keywords if k.arg=='dtype');assert ast.unparse(kw.value)=='tl.float32';kw.value=ast.parse('A.dtype.element_ty',mode='eval').body
            if child.targets[0].id=='k_blocks':
                assert ast.unparse(child.value)=='tl.cdiv(dim_k, block_k)';child.value=ast.parse('dim_k // block_k',mode='eval').body
        if isinstance(child,ast.Call) and ast.unparse(child.func) in ['tl.load','tl.store']:
            assert any(k.arg=='mask' for k in child.keywords);child.keywords=[]
        if isinstance(child,ast.Call) and ast.unparse(child.func)=='tl.sum':
            kw=next(k for k in child.keywords if k.arg=='dtype');assert ast.unparse(kw.value)=='tl.float32';child.keywords.remove(kw)
    assert hashlib.sha256(ast.dump(node).encode()).hexdigest()==old['kernel_ast']
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('mode',['graph','events','cache','no_write','wrong_timed','wrong_replay','mutation','crash'])
def test_allocating_wrapper_real_timer_replay_and_restore(task,monkeypatch,dtype,mode):
    path,ref=task;adapter=load(path/'_arena_eval.py');a,b=inputs(dtype);original=[x.clone() for x in (a,b)]
    phase=['initial'];timed_original=[];cached=answer(a,b);observed=[]
    plugin=types.SimpleNamespace(action='performance',current_row={'test_case_id':'cpu'},exercised=set())
    class Base:
        def __init__(self,fn):
            self.op_callable=fn;self.prepare_fn=None;self.use_cuda_graph=mode!='events';self.fallback_reason=None
            self.config=types.SimpleNamespace(warm_up=10,repetition=100)
    class Timed:
        outputs=None
        def rerun(self):
            phase[0]='replay'
            if mode=='crash':raise RuntimeError('injected replay crash')
            return self.outputs if mode=='no_write' else op()
    def timer(fn,**kwargs):
        phase[0]='timed';out=fn();kwargs['timed_run'].outputs=out;timed_original.append((out,out.clone()));observed.append(kwargs)
        return [1.,2.],{'benchmark_method':'cuda_event_fallback' if mode=='events' else 'cuda_graph'}
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=Timed,benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules,'performance_utils_pytest',types.SimpleNamespace(_compute_timing_stats=lambda times,cfg:{'mean':sum(times)/len(times)}))
    def op():
        out=cached.clone() if mode=='cache' else answer(a,b)
        if mode=='wrong_'+phase[0]:out[-1,-1]+=10
        if mode=='mutation' and phase[0]=='replay':a[0,0]+=1
        return out
    bench=adapter.benchmark_type(Base,plugin,None)(op);bench.context=dict(A_tri=a,B_tri=b)
    if mode in ['graph','events']:
        bench.run_benchmark();assert plugin.current_row['metadata']['fresh_input_replay_checked']
        helper=load(ROOT/'src/tools/perf/performance_utils_pytest.py');previous=[]
        monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',lambda fn,**kw:(previous.append(kw) or [1.],{}))
        helper._measure_times(op,bench.config,prepare_fn=None,use_cuda_graph=bench.use_cuda_graph,fallback_reason=bench.fallback_reason)
        sig=inspect.signature(load(ROOT/'src/tools/perf/aka_benchmark.py').benchmark_cuda_graph_or_events_samples)
        def effective(kwargs):
            bound=sig.bind_partial(None,**kwargs);bound.apply_defaults();return {k:v for k,v in bound.arguments.items() if k not in ['fn','timed_run']}
        assert effective(observed[0])==effective(previous[0])
    else:
        with pytest.raises((ValueError,AssertionError,RuntimeError)):bench.run_benchmark()
        assert not plugin.exercised
    assert all(ref.equal_bytes(x,y) for x,y in zip((a,b),original))
    assert all(ref.equal_bytes(x,y) for x,y in timed_original)

# Stable originals; no CI Git-history dependency.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_batched_vecmat': {'files': {'config.yaml': 'c7d0c82734863968d20ed2523462ceda06f06d319a062a066945853d9b56de99',
                                                                      'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                            'functions': {'batched_vecmat_triton_wrapper': '5296dd80ea1db6284199f671c84e7ae191bb90f46d91f293f34a9f0d82b28ca9',
                                                                          'calculate_batched_vecmat_gbps': '558ba8e0d85140a049e7671fca345e0c9045f3ddf5b895cbb94c23eb110e3d82',
                                                                          'calculate_batched_vecmat_tflops': '92b6d979d6a76eb876ce1f3a05e84cc8b56424ecb2b6fa82c8e9a2ffcadf5b93',
                                                                          'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                          'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                          'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                            'kernel_ast': '3fbfcbae473ffd0e577db74836c08ae01d91a5829ace5b6e92c48707dd151974',
                                                            'performance_without_skip': '9a859de9a7a59bd3c283ba2fb32db6bb1aedb7352da7c5fe5b0d66322ba9e4f1',
                                                            'rows': '544eb99ca853d9d2681c3bff5ecc927c84bdadd88ba03bedc3f8f1010c96b4a9'},
 'tasks/triton2triton/rocmbench/easy/test_batched_vecmat': {'files': {'config.yaml': 'c7d0c82734863968d20ed2523462ceda06f06d319a062a066945853d9b56de99',
                                                                      'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                            'functions': {'batched_vecmat_triton_wrapper': '5296dd80ea1db6284199f671c84e7ae191bb90f46d91f293f34a9f0d82b28ca9',
                                                                          'calculate_batched_vecmat_gbps': '558ba8e0d85140a049e7671fca345e0c9045f3ddf5b895cbb94c23eb110e3d82',
                                                                          'calculate_batched_vecmat_tflops': '92b6d979d6a76eb876ce1f3a05e84cc8b56424ecb2b6fa82c8e9a2ffcadf5b93',
                                                                          'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                          'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                          'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                            'kernel_ast': '3fbfcbae473ffd0e577db74836c08ae01d91a5829ace5b6e92c48707dd151974',
                                                            'performance_without_skip': '0664259ecf55cf4a86bb49b6138c512e5b985eabbef8ce614e8b91215cb22425',
                                                            'rows': '544eb99ca853d9d2681c3bff5ecc927c84bdadd88ba03bedc3f8f1010c96b4a9'}}


@pytest.mark.parametrize('mode',['good','tail','input_mutation','no_op'])
def test_actual_original_numpy_case_stays_independent(task,mode):
    import numpy as np
    path,ref=task;node=[n for n in ast.parse((path/'test_batched_vecmat.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_vecmat'][-1]
    node.decorator_list=[];observed=[]
    class Kernel:
        def __getitem__(self,grid):
            def launch(a,b,m,n,k,out,**kwargs):
                observed.extend([(x,x.clone()) for x in (a,b)])
                if mode!='no_op':out.copy_(answer(a,b))
                if mode=='tail':out[-1,-1]+=100
                if mode=='input_mutation':a[0,0]+=1
            return launch
    ns={'torch':torch,'np':np,'RandomState':np.random.RandomState,'set_seed':lambda:None,'batched_vecmat':Kernel(),'result_gold':{}}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-vecmat>','exec'),ns)
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu',user_properties=[]))
    if mode=='good':ns['test_vecmat'](request,device='cpu');assert request.node.user_properties
    else:
        with pytest.raises((AssertionError,ValueError)):ns['test_vecmat'](request,device='cpu')
        assert not request.node.user_properties
    assert all(ref.equal_bytes(x,y) for x,y in observed)
