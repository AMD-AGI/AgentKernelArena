"""GEMM row-bias rounding, operand tail safety, immutable inputs and real replay."""
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
TASKS=['instruction2triton/rocmbench/multreduce_matmul_dot_kernel','triton2triton/rocmbench/hard/multreduce_matmul_dot_kernel']


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
    return (torch.tensor([[2,-3,4],[1,5,-2]],dtype=dtype),torch.tensor([[1,3],[2,-1],[-2,4]],dtype=dtype),torch.tensor([.5,-1],dtype=dtype))


def answer(a,b,bias):
    result=(a.float()@b.float()).to(a.dtype)
    return result+bias[:,None] if bias is not None else result


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_known_answer_full_output_and_readonly(task,dtype):
    _,ref=task;a,b,bias=inputs(dtype);check=ref.BiasCheck(a,b,bias)
    out=torch.tensor([[-11.5,25.5],[14,-11]],dtype=dtype);check(out)
    out[-1,-1]+=2
    with pytest.raises(ref.NumericalMismatch):check(out)
    with pytest.raises(ValueError,match='aliases'):check(a[:,:2])
    bias[0]+=1
    with pytest.raises(ValueError,match='Read-only'):check(out)
    check.restore()


def test_bias_is_added_after_required_bf16_rounding(task):
    _,ref=task;a=torch.tensor([[1.0078125]],dtype=torch.bfloat16);b=a.clone();bias=torch.tensor([-1.015625],dtype=torch.bfloat16)
    check=ref.BiasCheck(a,b,bias);check(torch.zeros((1,1),dtype=torch.bfloat16))
    wrong=(a.double()@b.double()+bias.double()[:,None]).bfloat16()
    assert wrong.item()==2**-14
    with pytest.raises(ref.NumericalMismatch):check(wrong)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mode',['graph','events','cache','no_write','wrong_timed','wrong_replay','mutation','crash'])
def test_allocating_wrapper_real_timer_replay_and_restore(task,monkeypatch,dtype,mode):
    path,ref=task;adapter=load(path/'_arena_eval.py');a,b,bias=inputs(dtype);original=[x.clone() for x in (a,b,bias)]
    phase=['initial'];timed_original=[];cached=answer(a,b,bias);observed=[]
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
        out=cached.clone() if mode=='cache' else answer(a,b,bias)
        if mode=='wrong_'+phase[0]:out[-1,-1]+=10
        if mode=='mutation' and phase[0]=='replay':bias[0]+=1
        return out
    bench=adapter.benchmark_type(Base,plugin,None)(op);bench.context=dict(a=a,b=b,bias=bias)
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
    assert all(ref.equal_bytes(x,y) for x,y in zip((a,b,bias),original))
    assert all(ref.equal_bytes(x,y) for x,y in timed_original)


class CPUTorch:
    def __getattr__(self,name):
        fn=getattr(torch,name)
        if name not in ['arange','tensor']:return fn
        def call(*args,**kwargs):kwargs.pop('device',None);return fn(*args,**kwargs)
        return call


@pytest.mark.parametrize('dtype',['fp16','bf16'])
@pytest.mark.parametrize('mode',['good','omit_bias','tail','mutate_bias','cache'])
def test_actual_nonzero_bias_controls(task,dtype,mode):
    path,ref=task;source=ast.parse((path/'multreduce_matmul_dot_kernel.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='test_nonzero_row_bias_control');node.decorator_list=[]
    ns={'torch':CPUTorch()};observed=[];calls=[]
    def launch(provider,a,b,bias):
        if not observed:observed.extend([(x,x.clone()) for x in (a,b,bias)])
        out=answer(a,b,None if mode=='omit_bias' else bias);calls.append(out.clone())
        if mode=='cache':out=calls[0].clone()
        if mode=='tail':out[-1,-1]+=10
        if mode=='mutate_bias':bias[0]+=1
        return out
    ns['triton_matmul']=launch;exec(compile(ast.Module(body=[node],type_ignores=[]),'<bias-control>','exec'),ns)
    request=types.SimpleNamespace(node=types.SimpleNamespace(user_properties=[]))
    if mode=='good':ns[node.name](dtype,request);assert request.node.user_properties and len(calls)==3
    else:
        with pytest.raises((ValueError,AssertionError)):ns[node.name](dtype,request)
        assert not request.node.user_properties
    assert all(ref.equal_bytes(x,snapshot) for x,snapshot in observed)


@pytest.mark.parametrize('shape',[(1,23,31),(1,23,128),(2,16384,16384),(16,32,128)])
def test_actual_operand_load_masks_cover_only_valid_rows_columns_and_k(task,shape):
    path,_=task;source=ast.parse((path/'multreduce_matmul_dot_kernel.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='triton_matmul_kernel')
    loads=[n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load' and ast.unparse(n.args[0]) in ['a_ptrs','b_ptrs']]
    assert len(loads)==4
    M,N,K=shape
    for load_node in loads:
        kwargs={kw.arg:kw.value for kw in load_node.keywords};assert ast.literal_eval(kwargs['other'])==0
        for BK in [128,256,512]:
            BLOCK_SIZE_K=BK
            offs_am=torch.arange(16)+((M-1)//16)*16;offs_bn=torch.arange(32)+((N-1)//32)*32;offs_k=torch.arange(BK);k=(K-1)//BK
            mask=eval(compile(ast.Expression(body=kwargs['mask']),'<load-mask>','eval'),locals())
            # Every active lane must have a valid logical M or N; masked-K forms also bound the final K tile.
            expected=(offs_am[:,None]<M) if ast.unparse(load_node.args[0])=='a_ptrs' else (offs_bn[None,:]<N)
            if isinstance(kwargs['mask'],ast.BinOp):expected=expected&((offs_k[None,:]<K-k*BK) if ast.unparse(load_node.args[0])=='a_ptrs' else (offs_k[:,None]<K-k*BK))
            assert torch.equal(mask,expected)


def test_only_reviewed_load_masks_changed_kernel_and_original_cases_retained(task,monkeypatch):
    path,_=task;old=ORIGINAL[path.relative_to(ROOT).as_posix()];source=(path/'multreduce_matmul_dot_kernel.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,sha in old['functions'].items():assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in old['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    kernel=nodes['triton_matmul_kernel'];kernel.decorator_list=[]
    for n in ast.walk(kernel):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load' and ast.unparse(n.args[0]) in ['a_ptrs','b_ptrs']:
            mask=next(kw for kw in n.keywords if kw.arg=='mask')
            if isinstance(mask.value,ast.BinOp):mask.value=mask.value.right
            else:n.keywords=[]
    assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==old['kernel_ast']
    rows=json.loads((path/'workloads.json').read_text())['cases'];assert len(rows)==40 and sum('performance' in r['checks'] for r in rows)==24
    assert hashlib.sha256(json.dumps(rows[:38],sort_keys=True,separators=(',',':')).encode()).hexdigest()==old['rows']
    assert all(r['checks']==['correctness'] and not r['coverage']['scored'] for r in rows[38:])
    adapter=load(path/'_arena_eval.py');monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task')
    assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'

# Stable source/case expectations, independent of Git history.
ORIGINAL = {'tasks/instruction2triton/rocmbench/multreduce_matmul_dot_kernel': {'files': {'config.yaml': 'ebc28a5fab2dd87b7ec4aeca4ca6aa75079bbeca6949ad16339f1d67289b7177',
                                                                               'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                                     'functions': {'allclose': 'ea6ddc8a99b520e26dfac2ef1524d3a25ce4c4fe2add5c7197c8f12e992939b1',
                                                                                   'calculate_gemm_gbps': '7fe8b4fe7178d7076751611e942359f1e47654813b8d387de1854fdf0d85afb2',
                                                                                   'calculate_gemm_tflops': 'a91dc4b494aa2aaae56f121022891bcd0f7e044752a458b4101c83240049de7f',
                                                                                   'gen_input': 'b812ff15f66c1ecf2e0c069959667914e4240e725a76f111c4a1c520b492a213',
                                                                                   'gen_input_benchmark': 'fa8d338f61c4352e800706ff04a0e05dae4864e5e975caf6b4c3842f7d6c5bc9',
                                                                                   'get_target_shapes': '467c4c0b4367b8ec289b73b72c2f366845dd552a4b52aa2c0e57313bc339d77f',
                                                                                   'get_target_shapes_for_perf': 'b549441d2fc8d2518467208ff2469fd1976f7b9d4f6ac4d6c61f9878445a45f2',
                                                                                   'get_triton_autotune_key': '67ddfbc4712ede5b0ac0b88c54ec37b1d1a8899d6585586b2a46adbb572834b3',
                                                                                   'get_triton_dot_autotune_configs': '3718af43458b8f8e63b7ab84d3306f8ec53b2ed8f2bb93418173a849575fc1a5',
                                                                                   'get_triton_heuristics': 'b16d9f06bbfd7c6ae5a42668a59524e1cead9b2cbdfe13ad5afe1117d3b987e2',
                                                                                   'main': '7a7baa59c6b7c882ed7ebf7a209cc20a4ec004a9aa7ca0d02eb08ebfee24e7a9',
                                                                                   'matmul': '7b19151b80c5225f8c5c16bff201aafd74091345068152547ec1c3c58080d17e',
                                                                                   'ms_to_gibps': 'cc24cf65990c046072e0cf055bd87fc577528ad5631ac79b77ec33bd9714e92d',
                                                                                   'parse_args': '051ca1908494081b087379865e852c8cce673444d43459de1efb10f652894cd7',
                                                                                   'positive_int': '1f8569bfc1225c7a7b58f8a4a8392d479a16e0584e8fdda483381dc4734d6f99',
                                                                                   'run_benchmark': '869ece821d0a2eacf64ed8e54aaa13d556fb4aaa0264ed6be0c18b507211adc6',
                                                                                   'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                                   'test_performance': '590e572282c1170151c7eef2399c2f2386a443a03551201c21dea92ac18cf719',
                                                                                   'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                                   'test_save_results': 'e65c749df181c1c11b002cda5185ba56e40dab445af669fa59337de2e2509d3a',
                                                                                   'torch_matmul': 'f1155e0c41510d3dd52f5dd0d489fff5da5475c6f616c11b9a0e9daaeeb5b573',
                                                                                   'triton_dot_matmul_kernel': '55eeb6fad1b00f73a2e5201175f6f7dbc9a3a7f73c298a6059bb166753dcefda',
                                                                                   'triton_matmul': '14ab68bcc12baf55b05dc2ee4e3bdbf72b3f6de2ef5729fa67d69f97f3dc191e'},
                                                                     'kernel_ast': '2d6cfd78c6b97bb40d06e08dfcb081db9f6d8a7f4de24ef832fb5c1937f67636',
                                                                     'rows': 'ac381686e9f8affb1c6cb59b1116d385e93f82154d78c9465be0c1c01f4c08b2'},
 'tasks/triton2triton/rocmbench/hard/multreduce_matmul_dot_kernel': {'files': {'config.yaml': 'ebc28a5fab2dd87b7ec4aeca4ca6aa75079bbeca6949ad16339f1d67289b7177',
                                                                               'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                                     'functions': {'allclose': 'ea6ddc8a99b520e26dfac2ef1524d3a25ce4c4fe2add5c7197c8f12e992939b1',
                                                                                   'calculate_gemm_gbps': '7fe8b4fe7178d7076751611e942359f1e47654813b8d387de1854fdf0d85afb2',
                                                                                   'calculate_gemm_tflops': 'a91dc4b494aa2aaae56f121022891bcd0f7e044752a458b4101c83240049de7f',
                                                                                   'gen_input': 'b812ff15f66c1ecf2e0c069959667914e4240e725a76f111c4a1c520b492a213',
                                                                                   'gen_input_benchmark': 'fa8d338f61c4352e800706ff04a0e05dae4864e5e975caf6b4c3842f7d6c5bc9',
                                                                                   'get_target_shapes': '467c4c0b4367b8ec289b73b72c2f366845dd552a4b52aa2c0e57313bc339d77f',
                                                                                   'get_target_shapes_for_perf': 'b549441d2fc8d2518467208ff2469fd1976f7b9d4f6ac4d6c61f9878445a45f2',
                                                                                   'get_triton_autotune_key': '67ddfbc4712ede5b0ac0b88c54ec37b1d1a8899d6585586b2a46adbb572834b3',
                                                                                   'get_triton_dot_autotune_configs': '3718af43458b8f8e63b7ab84d3306f8ec53b2ed8f2bb93418173a849575fc1a5',
                                                                                   'get_triton_heuristics': 'b16d9f06bbfd7c6ae5a42668a59524e1cead9b2cbdfe13ad5afe1117d3b987e2',
                                                                                   'main': '7a7baa59c6b7c882ed7ebf7a209cc20a4ec004a9aa7ca0d02eb08ebfee24e7a9',
                                                                                   'matmul': '7b19151b80c5225f8c5c16bff201aafd74091345068152547ec1c3c58080d17e',
                                                                                   'ms_to_gibps': 'cc24cf65990c046072e0cf055bd87fc577528ad5631ac79b77ec33bd9714e92d',
                                                                                   'parse_args': '051ca1908494081b087379865e852c8cce673444d43459de1efb10f652894cd7',
                                                                                   'positive_int': '1f8569bfc1225c7a7b58f8a4a8392d479a16e0584e8fdda483381dc4734d6f99',
                                                                                   'run_benchmark': '869ece821d0a2eacf64ed8e54aaa13d556fb4aaa0264ed6be0c18b507211adc6',
                                                                                   'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                                   'test_performance': '539519403e1feea2f1e8a7180f0fb26bf58f4b04f9dd6d92663c3351dc776a31',
                                                                                   'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                                   'test_save_results': 'e65c749df181c1c11b002cda5185ba56e40dab445af669fa59337de2e2509d3a',
                                                                                   'torch_matmul': 'f1155e0c41510d3dd52f5dd0d489fff5da5475c6f616c11b9a0e9daaeeb5b573',
                                                                                   'triton_dot_matmul_kernel': '55eeb6fad1b00f73a2e5201175f6f7dbc9a3a7f73c298a6059bb166753dcefda',
                                                                                   'triton_matmul': '14ab68bcc12baf55b05dc2ee4e3bdbf72b3f6de2ef5729fa67d69f97f3dc191e'},
                                                                     'kernel_ast': '2d6cfd78c6b97bb40d06e08dfcb081db9f6d8a7f4de24ef832fb5c1937f67636',
                                                                     'rows': 'ac381686e9f8affb1c6cb59b1116d385e93f82154d78c9465be0c1c01f4c08b2'}}
