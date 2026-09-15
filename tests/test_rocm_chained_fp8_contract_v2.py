"""Chained FP8: scaled known answers, true dtype, full batches and replay."""
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
TASKS=['instruction2triton/rocmbench/test_chained_dot_fp8','triton2triton/rocmbench/hard/test_chained_dot_fp8']


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
    q=torch.tensor([[[1.,2.],[3.,-1.]],[[2.,1.],[-1.,2.]]],dtype=dtype)
    k=torch.tensor([[[1.,-1.],[2.,1.]],[[1.,2.],[-1.,1.]]],dtype=dtype)
    v=torch.tensor([[[1.,2.],[-2.,1.]],[[2.,1.],[1.,-1.]]],dtype=dtype)
    return q,k,v


def answer(q,k,v,scales):
    qd,kd,vd,ss,sd,os=scales
    if q.dtype==torch.float8_e4m3fnuz:
        middle=((q.float()*qd)@(k.float().transpose(-1,-2)*kd)*ss).to(q.dtype)
        return ((middle.float()*sd)@(v.float().transpose(-1,-2)*vd)*os).to(q.dtype)
    return ((q.float()@k.float().transpose(-1,-2)).half().float()@v.float().transpose(-1,-2)).half()


def test_scaled_fp8_known_answer_and_ignoring_scales_fails(task):
    _,ref=task;q=k=v=torch.ones((2,2,2),dtype=torch.float8_e4m3fnuz);scales=(.5,.5,2.,2.,.5,.5)
    check=ref.DotCheck(q.clone(),k.clone(),v.clone(),scales)
    expected=torch.ones((2,2,2),dtype=q.dtype);check(expected)
    with pytest.raises(ref.NumericalMismatch):check(torch.full((2,2,2),4.,dtype=q.dtype))
    with pytest.raises(ValueError,match='metadata'):check(expected.float())


def test_mandatory_fp8_intermediate_rounding_has_independent_counterexample(task):
    _,ref=task;dtype=torch.float8_e4m3fnuz
    q=torch.tensor([[[1.125,0.]]],dtype=dtype);k=torch.tensor([[[1.125,0.],[1.25,0.]]],dtype=dtype);v=torch.tensor([[[128.,-128.],[128.,-128.]]],dtype=dtype)
    check=ref.DotCheck(q,k,v);expected=torch.full((1,1,2),-16.,dtype=dtype);check(expected)
    wrong=(q.float()@k.float().transpose(-1,-2)@v.float().transpose(-1,-2)).to(dtype)
    assert torch.equal(wrong.float(),torch.full((1,1,2),-18.))
    with pytest.raises(ref.NumericalMismatch):check(wrong)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float8_e4m3fnuz])
def test_full_second_batch_and_readonly_fields(task,dtype):
    _,ref=task;q,k,v=inputs(dtype);check=ref.DotCheck(q,k,v);out=answer(q,k,v,(1.,)*6);check(out)
    corrupted=out.float();corrupted[1,-1,-1]+=16
    with pytest.raises(ref.NumericalMismatch):check(corrupted.to(dtype))
    for x in [q,k,v]:
        original=x.clone();x.view(torch.uint8).flatten()[0]^=1
        with pytest.raises(ValueError,match='Read-only'):check(out)
        x.copy_(original)


class CPUTorch:
    def __getattr__(self,name):
        fn=getattr(torch,name)
        if name not in ['empty','arange']:return fn
        def call(*args,**kwargs):kwargs.pop('device',None);return fn(*args,**kwargs)
        return call


def protected_functions(path,names,ns):
    nodes=[n for n in ast.parse((path/'test_chained_dot_fp8.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in nodes:n.decorator_list=[]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-fp8>','exec'),ns)


@pytest.mark.parametrize('which',['original','tail_fp8','tail_fp16'])
@pytest.mark.parametrize('mode',['good','cache','wrong_dtype','tail','mutation'])
def test_actual_original_and_tail_bodies_reject_invalid_candidates(task,which,mode):
    import math
    path,ref=task;ns={'torch':CPUTorch(),'math':math,'float8':torch.float8_e4m3fnuz,'set_seed':lambda:torch.manual_seed(42),'result_gold':{}}
    protected_functions(path,['to_float8','test_chained_dot','test_batched_tail_and_scale_control'],ns)
    calls=[];snapshots=[]
    def launch(q,k,v,msize=32,*scales):
        if not snapshots:snapshots.extend([(x,x.clone()) for x in (q,k,v)])
        out=answer(q,k,v,scales or (1.,)*6);calls.append(out.clone())
        if mode=='cache':out=calls[0].clone()
        if mode=='wrong_dtype':out=out.float()
        if mode=='tail':
            changed=out.float();changed[-1,-1,-1]+=16;out=changed.to(out.dtype)
        if mode=='mutation':q.view(torch.uint8).flatten()[0]^=1
        return out
    ns['chained_dot']=launch;request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu',user_properties=[]))
    run=lambda:ns['test_chained_dot'](128,64,32,'fp8',16,request) if which=='original' else ns['test_batched_tail_and_scale_control']('fp8' if which=='tail_fp8' else 'fp16',request)
    if mode=='good':
        run();assert len(calls)==2 and request.node.user_properties
        assert not torch.equal(calls[0].float(),calls[1].float())
    else:
        with pytest.raises((ValueError,AssertionError)):run()
        assert not request.node.user_properties
    assert all(ref.equal_bytes(a,b) for a,b in snapshots)


def test_original_classes_scored_cases_gates_and_reviewed_masks(task,monkeypatch):
    path,_=task;old=ORIGINAL[path.relative_to(ROOT).as_posix()];source=(path/'test_chained_dot_fp8.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name,sha in {**old['functions'],**old['classes']}.items():assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in old['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    kernel=nodes['_chained_dot'];kernel.decorator_list=[]
    expected={'Q_block_ptr':(0,),'K_block_ptr':(1,),'V_block_ptr':(0,),'O_block_ptr':(0,)};seen=set()
    for node in ast.walk(kernel):
        if isinstance(node,ast.Call) and ast.unparse(node.func) in ['tl.load','tl.store']:
            pointer=ast.unparse(node.args[0]);kw={k.arg:ast.literal_eval(k.value) for k in node.keywords}
            assert kw['boundary_check']==expected[pointer]
            if ast.unparse(node.func)=='tl.load':assert kw['padding_option']=='zero'
            seen.add(pointer);node.keywords=[]
    assert seen==set(expected) and hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==old['kernel_ast']
    rows=json.loads((path/'workloads.json').read_text())['cases'];assert len(rows)==26 and sum('performance' in r['checks'] for r in rows)==20
    assert hashlib.sha256(json.dumps(rows[:24],sort_keys=True,separators=(',',':')).encode()).hexdigest()==old['rows']
    assert all(r['checks']==['correctness'] for r in rows[24:])
    assert 'assert_close(tri_out[0].float(), ref_f8.float(), atol=1e-2, rtol=0)' in source
    assert 'assert_close(tri_out, ref, atol=1e-2, rtol=0)' in source
    adapter=load(path/'_arena_eval.py');monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task');assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'
@pytest.mark.parametrize('dtype',[torch.float16,torch.float8_e4m3fnuz])
@pytest.mark.parametrize('mode',['graph','events','cache','no_write','wrong_timed','wrong_replay','mutation','crash'])
def test_allocating_wrapper_real_timer_replay_and_restore(task,monkeypatch,dtype,mode):
    path,ref=task;adapter=load(path/'_arena_eval.py');q,k,v=inputs(dtype);original=[x.clone() for x in (q,k,v)]
    phase=['initial'];timed_original=[];cached=answer(q,k,v,(1.,)*6);observed=[]
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
        out=cached.clone() if mode=='cache' else answer(q,k,v,(1.,)*6)
        if mode=='wrong_'+phase[0]:
            changed=out.float();changed[-1,-1]+=10;out=changed.to(dtype)
        if mode=='mutation' and phase[0]=='replay':q.view(torch.uint8).flatten()[0]^=1
        return out
    bench=adapter.benchmark_type(Base,plugin,None)(op);bench.context=dict(q_for_kernel=q,k_for_kernel=k,v_for_kernel_call=v,q_desc_py=1.,k_desc_py=1.,v_desc_py=1.,s_sc_py=1.,s_desc_py=1.,o_sc_py=1.)
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
    assert all(ref.equal_bytes(x,y) for x,y in zip((q,k,v),original))
    assert all(ref.equal_bytes(x,y) for x,y in timed_original)

# Original expectations; no runtime branch/history dependency.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_chained_dot_fp8': {'classes': {'chained_dot_fn': 'ddcf0dfdac9a7f17cb76517cc9dccf8b1f26a63d5660041da94f27bf61722062'},
                                                             'files': {'config.yaml': 'f49b517b0e13bf0e8a82036562a271f01c541d6b70caef60fc4613c05d515b40',
                                                                       'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                             'functions': {'calculate_chained_dot_gbps': 'ae486787e5ee14cf0c0628c4f2d1d51776a26a51e253dadd63d2b42295af9508',
                                                                           'calculate_chained_dot_tflops': 'c4c5a2d8c7f4a10735707de7517ea0f3c629695853bf22947750382b7fbf1c2b',
                                                                           'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                           'test_performance': '075366593111da2882a98349989fbb85106be914347ebee561bb1279fdcf4870',
                                                                           'test_save_performance_results': '450ff29b0859d464447fbe214ebf21542bb77256c164329607aee2e48a703ec1',
                                                                           'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                                           'to_float8': 'a34b908ef8251af63976f13730cd53555bd8b6a1a7b993ea62d719db263a56a1'},
                                                             'kernel_ast': '617e279d290073c1216a5a1fe96a252ca144c63ac1134c4175eb22d8c496f791',
                                                             'rows': '55983a50e5b62794c270f6b35ac45082d76d77b2684bb8adf7567e1ec9b28f8d'},
 'tasks/triton2triton/rocmbench/hard/test_chained_dot_fp8': {'classes': {'chained_dot_fn': 'ddcf0dfdac9a7f17cb76517cc9dccf8b1f26a63d5660041da94f27bf61722062'},
                                                             'files': {'config.yaml': 'f49b517b0e13bf0e8a82036562a271f01c541d6b70caef60fc4613c05d515b40',
                                                                       'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                             'functions': {'calculate_chained_dot_gbps': 'ae486787e5ee14cf0c0628c4f2d1d51776a26a51e253dadd63d2b42295af9508',
                                                                           'calculate_chained_dot_tflops': 'c4c5a2d8c7f4a10735707de7517ea0f3c629695853bf22947750382b7fbf1c2b',
                                                                           'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                           'test_performance': 'e1bcb7a49c5593c0b5102e3001e81950c5d281e02d7063eb45fb3231cb033cb6',
                                                                           'test_save_performance_results': '450ff29b0859d464447fbe214ebf21542bb77256c164329607aee2e48a703ec1',
                                                                           'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8',
                                                                           'to_float8': 'a34b908ef8251af63976f13730cd53555bd8b6a1a7b993ea62d719db263a56a1'},
                                                             'kernel_ast': '617e279d290073c1216a5a1fe96a252ca144c63ac1134c4175eb22d8c496f791',
                                                             'rows': '55983a50e5b62794c270f6b35ac45082d76d77b2684bb8adf7567e1ec9b28f8d'}}
