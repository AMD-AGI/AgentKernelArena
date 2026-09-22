"""CPU contract simulations; actual GPU qualification is recorded separately."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASKS=[ROOT/'tasks/triton2triton/rocmbench/easy/test_block_copy',
       ROOT/'tasks/instruction2triton/rocmbench/test_block_copy']


def load(path):
    spec=importlib.util.spec_from_file_location('_block_test_'+path.parent.parent.name+'_'+path.stem,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class CompilationError(Exception):pass


def error(message):
    exc=CompilationError('compiler location')
    exc.__cause__=ValueError(message)
    return exc


@pytest.fixture(params=TASKS,ids=['triton','instruction'])
def task(request,monkeypatch):
    ref=load(request.param/'_arena_reference.py')
    monkeypatch.setitem(sys.modules,'_arena_reference',ref)
    monkeypatch.setitem(sys.modules,'triton.compiler.errors',types.SimpleNamespace(CompilationError=CompilationError))
    return request.param,ref


# Checked-in immutable regression digests; no branch history needed at runtime.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_block_copy': {'block_copy_kernel': 'da69d81c6f0af511078235aaafda91eb73c7f3240fd97bb3191b0bb1b102215d',
                                                        'block_copy_kernel_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                        'config': '9bd9f8fb9b1a474224fd057feab504586695c819f762b616d6e54b5da9a626a8',
                                                        'generated_stub': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                        'rows': 'f68924ae7b7b4038600ce5e2cf90c545d4e3ecea9206a3c91e3f3ad8343f4ba7',
                                                        'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                        'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                        'test_performance': '44b96cc31b7939015bb67395647fce914c605029c8914f1afc61db1764b8b992',
                                                        'test_performance_decorators': 'cbb7c4dc16d294bfff2a3c0763da2721a50ebadd9cff052eb1f94d960a9c7566'},
 'tasks/triton2triton/rocmbench/easy/test_block_copy': {'block_copy_kernel': 'da69d81c6f0af511078235aaafda91eb73c7f3240fd97bb3191b0bb1b102215d',
                                                        'block_copy_kernel_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                        'config': '9bd9f8fb9b1a474224fd057feab504586695c819f762b616d6e54b5da9a626a8',
                                                        'generated_stub': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                        'rows': 'f68924ae7b7b4038600ce5e2cf90c545d4e3ecea9206a3c91e3f3ad8343f4ba7',
                                                        'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                        'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                        'test_performance': '4e69a75ae3cb814d3e4969e8fb71aeda7bb1f2b5fc395f4763e51f037cf668d2',
                                                        'test_performance_decorators': 'cbb7c4dc16d294bfff2a3c0763da2721a50ebadd9cff052eb1f94d960a9c7566'}}


def digest(data):return hashlib.sha256(data).hexdigest()


def test_original_manifest_kernel_timing_and_guard_are_preserved(task):
    path,_=task;key=str(path.relative_to(ROOT));expected=ORIGINAL[key]
    manifest=json.loads((path/'workloads.json').read_text())
    extra=manifest['cases'][130:]
    if key.startswith('tasks/triton2triton/'):
        assert [(r['params']['arguments']['dtypes_str'][0], r['params']['arguments']['n'],
                 r['params']['arguments']['padding_option']) for r in extra] == TAIL_CONTROLS
        assert all(r['checks']==['correctness'] for r in extra)
    else:
        assert extra == []
    rows=[{k:v for k,v in r.items() if k!='metadata'} for r in manifest['cases'][:130]]
    assert digest(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())==expected['rows']
    assert len(rows)==130
    assert sum(r['params']['function']=='test_block_copy' for r in rows)==90
    assert sum('performance' in r['checks'] for r in rows)==40
    source=(path/'test_block_copy.py').read_text();tree=ast.parse(source)
    for name in ['block_copy_kernel','test_performance','set_seed']:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert digest(ast.get_source_segment(source,node).encode())==expected[name]
        assert digest(ast.dump(ast.Module(body=node.decorator_list,type_ignores=[])).encode())==expected[name+'_decorators']
    assert digest((path/'config.yaml').read_bytes())==expected['config']
    assert digest((path/'performance_utils_pytest.py').read_bytes())==expected['generated_stub']
    invalid=[r for r in manifest['cases'] if r['metadata']['contract']['expected_outcome']=='compile_time_rejection']
    assert len(invalid)==15
    assert all(r['metadata']['contract']['buffers_unchanged'] for r in invalid)


@pytest.mark.parametrize('dtype',[torch.bool,torch.int16,torch.int32])
def test_only_specific_rejection_is_accepted_and_buffers_unchanged(task,dtype):
    _,ref=task;a=(torch.arange(8)%2).to(dtype);b=torch.zeros_like(a)
    def launch():raise error(ref.INTEGER_NAN_ERROR)
    assert ref.expect_integer_nan_rejection(launch,a,b)['expected_rejection_checked']
    for exc in [RuntimeError(ref.INTEGER_NAN_ERROR),ValueError(ref.INTEGER_NAN_ERROR),
                CompilationError(ref.INTEGER_NAN_ERROR),error('unrelated compilation failure')]:
        def broken():raise exc
        with pytest.raises(type(exc)):
            ref.expect_integer_nan_rejection(broken,a,b)
    with pytest.raises(AssertionError,match='unexpectedly succeeded'):
        ref.expect_integer_nan_rejection(lambda:None,a,b)
    def mutation():
        b[0]=1
        raise error(ref.INTEGER_NAN_ERROR)
    with pytest.raises(ref.NumericalMismatch):ref.expect_integer_nan_rejection(mutation,a,b)


TAIL_CONTROLS=[('float32',63,None),('int32',65,'zero'),('float16',127,'nan')]
DTYPES=['bool','int16','int32','float16','float32','bfloat16']
@pytest.mark.parametrize('dtype,n,padding',[(d,n,p) for d in DTYPES for n in [64,128,256,512,1024] for p in [None,'zero','nan']]+TAIL_CONTROLS)
def test_original_correctness_function_executes_all90_rows_on_cpu(task,dtype,n,padding):
    path,ref=task
    class Kernel:
        def __getitem__(self,grid):
            def launch(a_ptr,b_ptr,N,BLOCK_SIZE,padding_option):
                assert grid({'BLOCK_SIZE':BLOCK_SIZE})==((N+63)//64,)
                if not a_ptr.is_floating_point() and padding_option=='nan':raise error(ref.INTEGER_NAN_ERROR)
                b_ptr[:N//2].copy_(a_ptr[:N//2])
                b_ptr[N//2:].fill_(0 if padding_option=='zero' else float('nan') if padding_option=='nan' else 1)
            return launch
    source=(path/'test_block_copy.py').read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_block_copy');fn.decorator_list=[]
    ns={'torch':torch,'pytest':pytest,'triton':types.SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b),
        'check_type_supported':lambda *a:None,'block_copy_kernel':Kernel(),'result_gold':{}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path/'test_block_copy.py'),'exec'),ns)
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu_domain_case',user_properties=[]))
    state=torch.random.get_rng_state().clone()
    ns['test_block_copy']((dtype,dtype),n,padding,request,device='cpu')
    evidence=dict(request.node.user_properties)['block_copy_contract']
    if dtype in DTYPES[:3] and padding=='nan':
        assert evidence['expected_rejection_checked']
        assert torch.equal(torch.random.get_rng_state(),state)
    else:assert evidence['readonly_input_checked'] and evidence['defined_output_checked']


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.int32])
@pytest.mark.parametrize('padding',[None,'zero'])
def test_full_defined_output_and_readonly_input(task,dtype,padding):
    _,ref=task;a=torch.arange(16).to(dtype);before=a.clone();b=torch.full_like(a,-7)
    b[:8]=a[:8]
    if padding=='zero':b[8:]=0
    ref.check_output(a,b,before,padding,untouched_tail=torch.full_like(a[8:],-7) if padding is None else None)
    b[0]+=1
    with pytest.raises(ref.NumericalMismatch):ref.check_output(a,b,before,padding)
    b[0]=a[0];a[0]+=1
    with pytest.raises(ref.NumericalMismatch):ref.check_output(a,b,before,padding)
    a.copy_(before);b[8:]=3
    with pytest.raises(ref.NumericalMismatch):ref.check_output(a,b,before,padding,untouched_tail=torch.full_like(a[8:],-7))


def test_undefined_suffix_nan_padding_metadata_and_subnormal(task):
    _,ref=task;a=torch.arange(8,dtype=torch.float32);b=a.clone();b[4:]=float('nan')
    ref.check_output(a,b,a.clone(),None) # Full grid None suffix really is undefined.
    ref.check_output(a,b,a.clone(),'nan')
    b[-1]=float('inf')
    with pytest.raises(ref.NumericalMismatch):ref.check_output(a,b,a.clone(),'nan')
    b[4:]=0;b.view(torch.int32)[-1]=1
    with pytest.raises(ref.NumericalMismatch):ref.check_output(a,b,a.clone(),'zero')
    with pytest.raises(ValueError):ref.check_output(a,b.double(),a.clone(),None)


@pytest.mark.parametrize('mode',['correct','cached','readonly_mutation','wrong_padding','crash',
                                 'observable_events','unobservable_fallback'])
@pytest.mark.parametrize('padding',[None,'zero'])
def test_real_adapter_replay_restore_and_unchanged_timing_parameters(task,monkeypatch,mode,padding):
    path,ref=task;adapter=load(path/'_arena_eval.py');a=torch.arange(16,dtype=torch.float32);b=torch.full_like(a,19)
    initial_a=a.clone();initial_b=b.clone();c={'a':a,'b':b,'n':16,'padding_option':padding}
    plugin=types.SimpleNamespace(action='performance',current_row={'test_case_id':'cpu_case'},exercised=set())
    class Base:
        def __init__(self,op_callable,**kwargs):
            self.op_callable=op_callable;self.prepare_fn=None;self.use_cuda_graph=mode!='observable_events'
            self.fallback_reason='explicit observable event path' if mode=='observable_events' else None
            self.config=types.SimpleNamespace(warm_up=10,repetition=100)
    class Timed:
        def rerun(self):return self.fn()
    def benchmark(fn,**kwargs):
        assert {k:v for k,v in kwargs.items() if k!='timed_run'}==dict(
            warmup=10,repetition=100,prepare_fn=None,use_cuda_graph=mode!='observable_events',
            fallback_reason='explicit observable event path' if mode=='observable_events' else None)
        if mode=='crash':raise RuntimeError('injected timing failure')
        if mode=='unobservable_fallback':raise RuntimeError('timed_run requires observable replay')
        t=kwargs['timed_run'];t.fn=fn;t.outputs=fn()
        return [1.,2.],({'benchmark_method':'cuda_event_fallback',
                         'benchmark_fallback_reason':'explicit observable event path'}
                        if mode=='observable_events' else {'benchmark_method':'cuda_graph'})
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=Timed,benchmark_cuda_graph_or_events_samples=benchmark))
    monkeypatch.setitem(sys.modules,'performance_utils_pytest',types.SimpleNamespace(_compute_timing_stats=lambda ts,config:{'mean':sum(ts)/len(ts)}))
    cached=a[:8].clone()
    def op():
        b[:8].copy_(cached if mode=='cached' else a[:8])
        if padding=='zero':b[8:].fill_(1 if mode=='wrong_padding' else 0)
        elif mode=='wrong_padding':b[8:]=0
        if mode=='readonly_mutation':a[0]+=1
    wrapped=adapter.benchmark_type(Base,plugin,None)(op_callable=op)
    wrapped.context=c
    if mode in ('correct','observable_events'):
        wrapped.run_benchmark()
        assert plugin.current_row['metadata']['fresh_input_replay_checked']
        assert plugin.current_row['execution_time_ms']==1.5 and plugin.exercised=={'cpu_case'}
        if mode=='observable_events':
            assert plugin.current_row['metadata']['device_timing']['benchmark_fallback_reason']=='explicit observable event path'
    else:
        with pytest.raises((ref.NumericalMismatch,RuntimeError)):wrapped.run_benchmark()
        assert not plugin.exercised and 'execution_time_ms' not in plugin.current_row
    assert torch.equal(a,initial_a) and torch.equal(b,initial_b)
