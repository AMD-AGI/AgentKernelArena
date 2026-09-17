"""Two-stage matmul: original binary exact gate and independent rounding enclosure."""
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
TASKS=['instruction2triton/rocmbench/test_chained_matmul','triton2triton/rocmbench/medium/test_chained_matmul']

def load(path):
    spec = importlib.util.spec_from_file_location('_mxfp_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    threads = torch.get_num_threads(); state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads); torch.random.set_rng_state(state)


@pytest.fixture(params=TASKS)
def task(request, monkeypatch):
    path=ROOT/'tasks'/request.param
    ref=load(path/'_arena_reference.py')
    monkeypatch.setitem(sys.modules,'_arena_reference',ref)
    return path,ref


def protected_functions(path,names,namespace):
    tree=ast.parse((path/'test_chained_matmul.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in nodes:n.decorator_list=[]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<protected-chain>','exec'),namespace)
    return namespace


def inputs():
    a=torch.tensor([[2,-3,4],[1,5,-2]],dtype=torch.float16)
    b=torch.tensor([[1,2,-2],[3,-1,4]],dtype=torch.float16)
    c=torch.tensor([[1,2,3],[-2,3,-1]],dtype=torch.float16)
    return a,b,c,torch.full((2,3),-7,dtype=torch.float16)


def test_hand_computed_two_stage_answer_and_exact_gate(task):
    _,ref=task;a,b,c,out=inputs();check=ref.ChainCheck(a,b,c,out,exact=True)
    expected=torch.tensor([[-62,51,-61],[35,0,55]],dtype=torch.float16)
    assert torch.equal(check.expected,expected)
    out.copy_(expected);check(out)
    out[-1,-1]+=1
    with pytest.raises(ref.NumericalMismatch):check(out)
    out.copy_(expected);a[0,0]+=1
    with pytest.raises(ValueError,match='Read-only'):check(out)
    check.restore()


@pytest.mark.parametrize('shape',[(3,17,32),(17,48,32),(4,64,128),(2,16,512)])
@pytest.mark.parametrize('block',[8,16,32])
def test_rounding_interval_encloses_different_fp32_reductions(task,shape,block):
    _,ref=task;m,n,k=shape
    a=(torch.arange(m*k).reshape(m,k)%23-11).half()/7
    b=(torch.arange(n*k).reshape(n,k)%17-8).half()/5
    c=(torch.arange(n*k).reshape(n,k)%13-6).half()/3
    lower,upper=ref.chain_bounds(a,b,c)
    out=torch.zeros((m,k),dtype=torch.float32)
    for start in range(0,n,block):
        mid=(a.float()@b[start:start+block].float().T).half()
        out+=mid.float()@c[start:start+block].float()
    out=out.half();check=ref.ChainCheck(a,b,c,out);check(out)
    assert bool((lower<=upper).all())
    out[-1,-1]=upper[-1,-1]+max(1.,abs(float(upper[-1,-1]))*.05)
    with pytest.raises(ref.NumericalMismatch):check(out)


def test_omitting_required_intermediate_half_rounding_is_rejected(task):
    _,ref=task
    a=torch.tensor([[1.,.0004]],dtype=torch.float16)
    b=torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float16)
    c=torch.tensor([[100.,100.],[-100.,-100.]],dtype=torch.float16)
    out=torch.zeros((1,2),dtype=torch.float16);check=ref.ChainCheck(a,b,c,out)
    out.copy_((a.double()@b.double().T@c.double()).half())
    assert bool((out.abs()>.03).all())
    with pytest.raises(ref.NumericalMismatch):check(out)
    out.zero_();check(out)


@pytest.mark.parametrize('original',[False,True])
@pytest.mark.parametrize('mode',['good','cache','tail','input_mutation','no_write'])
def test_actual_functional_bodies_and_restore(task,original,mode):
    path,ref=task
    ns={'torch':CPUTorch(),'triton':types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),'set_seed':lambda:None,'result_gold':{}}
    protected_functions(path,['chained_matmul_reference','test_chained_matmul','test_signed_partial_m_control'],ns)
    calls=[];before=[]
    def launch(a,b,c,out,*args,**kwargs):
        if not before:before.extend([(x,x.clone()) for x in (a,b,c,out)])
        value=(a.float()@b.float().T).half().float()@c.float();calls.append(value.clone())
        if not(mode=='no_write' and len(calls)>1):out.copy_(calls[0] if mode=='cache' else value)
        if mode=='tail':out[-1,-1]+=10
        if mode=='input_mutation':a[0,0]+=1
        return out
    class Kernel:
        def __getitem__(self,grid):return launch
    ns['chained_matmul_kernel']=Kernel();ns['chained_matmul_triton_wrapper']=launch
    request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu',user_properties=[]))
    run=lambda:ns['test_chained_matmul'](request,device='cpu') if original else ns['test_signed_partial_m_control'](17,48,32,16,16,request)
    # The original case runs once; cache/no-write replay negatives apply to new controls.
    if mode=='good' or (original and mode in ['cache','no_write']):
        run();assert request.node.user_properties
    else:
        with pytest.raises((ValueError,AssertionError)):run()
        assert not request.node.user_properties
    assert all(ref.equal_bytes(x,v) for x,v in before)


def test_original_source_scoring_rows_and_exact_assertion(task,monkeypatch):
    path,ref=task;expected=ORIGINAL[path.relative_to(ROOT).as_posix()]
    source=(path/'test_chained_matmul.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,sha in expected['functions'].items():assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in expected['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    rows=json.loads((path/'workloads.json').read_text())['cases']
    assert len(rows)==152 and sum('performance' in r['checks'] for r in rows)==150
    assert hashlib.sha256(json.dumps(rows[:151],sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected['rows']
    assert rows[-1]['checks']==['correctness'] and not rows[-1]['coverage']['scored']
    assert 'assert (torch_result == triton_result).all()' in source
    adapter=load(path/'_arena_eval.py');monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task')
    assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'
@pytest.mark.parametrize('mode', ['graph', 'events', 'cached', 'no_write', 'wrong_timed', 'wrong_replay', 'mutation', 'crash', 'bad_timing'])
def test_real_timer_replay_and_finally_restore(task, monkeypatch, mode):
    path,ref=task
    adapter = load(path/'_arena_eval.py'); a, b, c, out = inputs()
    original = [x.clone() for x in (a, b, c, out)]; phase = ['initial']; cached = (a.float() @ b.float().T).half().float() @ c.float(); timer_args = []
    plugin = types.SimpleNamespace(action='performance', current_row={'test_case_id': 'cpu'}, exercised=set())
    class Base:
        def __init__(self, fn):
            self.op_callable = fn; self.prepare_fn = None; self.use_cuda_graph = mode != 'events'
            self.fallback_reason = 'explicit events' if mode == 'events' else None
            self.config = types.SimpleNamespace(warm_up=10, repetition=100)
    class Timed:
        outputs = None
        def rerun(self):
            phase[0] = 'replay'
            if mode == 'crash': raise RuntimeError('injected replay crash')
            return op()
    def timer(fn, **kw):
        timer_args.append(kw); phase[0] = 'timed'; kw['timed_run'].outputs = fn()
        return ([float('nan')] if mode == 'bad_timing' else [1., 2.]), {'benchmark_method': 'cuda_event_fallback' if mode == 'events' else 'cuda_graph'}
    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=Timed, benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules, 'performance_utils_pytest', types.SimpleNamespace(_compute_timing_stats=lambda ts, cfg: {'mean': sum(ts)/len(ts)}))
    def op():
        if not (mode == 'no_write' and phase[0] == 'replay'): out.copy_(cached if mode == 'cached' else (a.float() @ b.float().T).half().float() @ c.float())
        if mode == 'wrong_' + phase[0]: out[-1, -1] += 10
        if mode == 'mutation' and phase[0] == 'replay': a[0, 0] += 1
        return out
    bench = adapter.benchmark_type(Base, plugin, None)(op); bench.context = dict(a=a,b=b,c_mat=c,triton_result_buffer=out)
    if mode in ['graph', 'events']:
        bench.run_benchmark(baseline_callable=lambda: pytest.fail('peer baseline'))
        assert all(bench_value for key, bench_value in plugin.current_row['metadata'].items() if key.endswith('_checked') or key.endswith('_restored'))
        helper = load(ROOT/'src/tools/perf/performance_utils_pytest.py'); previous = []
        monkeypatch.setattr(helper, 'benchmark_cuda_graph_or_events_samples', lambda fn, **kw: (previous.append(kw) or [1.], {}))
        helper._measure_times(op, bench.config, prepare_fn=None, use_cuda_graph=bench.use_cuda_graph, fallback_reason=bench.fallback_reason)
        sig = inspect.signature(load(ROOT/'src/tools/perf/aka_benchmark.py').benchmark_cuda_graph_or_events_samples)
        def effective(kwargs):
            bound = sig.bind_partial(None, **kwargs); bound.apply_defaults()
            return {k: v for k, v in bound.arguments.items() if k not in ['fn', 'timed_run']}
        assert effective(previous[0]) == effective(timer_args[0])
    else:
        with pytest.raises((ValueError, RuntimeError, AssertionError)): bench.run_benchmark()
        assert not plugin.exercised
    assert all(ref.equal_bytes(x, y) for x, y in zip((a, b, c, out), original))
class CPUTorch:
    def __getattr__(self, name):
        original = getattr(torch, name)
        if name not in ['empty', 'randn', 'randint', 'arange', 'tensor', 'full']: return original
        def call(*args, **kwargs):
            kwargs.pop('device', None)
            return original(*args, **kwargs)
        return call



# Stable expected hashes; no runtime Git history dependency.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_chained_matmul': {'files': {'config.yaml': 'c17101058eb2756c6b9cf9bb49bfd653c0ba1ff8a9f3978ec36de6780d06504f',
                                                                      'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                            'functions': {'calculate_chained_matmul_gbps': 'ebb646f1543b3a056231cc2633ed020e56cd1162c6ad66a21790e1f28afd61db',
                                                                          'calculate_chained_matmul_tflops': '7195211f778aaca797117a706e5f695cfc4a49c9ba66a8018679760470e80205',
                                                                          'chained_matmul_kernel': '9610429036ea6574ceef3a8de69c47f17949fe7c8031264ebe78d65542a330a8',
                                                                          'chained_matmul_reference': '7189496a5d90520841b3e74edac8c51535b125b6dab0ac718e3eba1446d219ea',
                                                                          'chained_matmul_triton_wrapper': '69082f027e636bac7d710babb83a4e4fa14443f77b06c9ad44a4a1243a4c482e',
                                                                          'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                          'test_performance': '3b40bb6ea2582e7e1f3958dc965375e23cb5b0664f5da8ec38b9d9726ed6f381',
                                                                          'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                          'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                            'rows': '990f2ec29f6020f504f13a97295593bf923d680c797fffc5c02d7cc05cb2dfa1'},
 'tasks/triton2triton/rocmbench/medium/test_chained_matmul': {'files': {'config.yaml': 'c17101058eb2756c6b9cf9bb49bfd653c0ba1ff8a9f3978ec36de6780d06504f',
                                                                        'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                              'functions': {'calculate_chained_matmul_gbps': 'ebb646f1543b3a056231cc2633ed020e56cd1162c6ad66a21790e1f28afd61db',
                                                                            'calculate_chained_matmul_tflops': '7195211f778aaca797117a706e5f695cfc4a49c9ba66a8018679760470e80205',
                                                                            'chained_matmul_kernel': '9610429036ea6574ceef3a8de69c47f17949fe7c8031264ebe78d65542a330a8',
                                                                            'chained_matmul_reference': '7189496a5d90520841b3e74edac8c51535b125b6dab0ac718e3eba1446d219ea',
                                                                            'chained_matmul_triton_wrapper': '69082f027e636bac7d710babb83a4e4fa14443f77b06c9ad44a4a1243a4c482e',
                                                                            'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                            'test_performance': 'ba6854f55a4ae2f2dc39c9fd76c4a95473b54722b344cbefb21286eed2e6c615',
                                                                            'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                            'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                              'rows': '990f2ec29f6020f504f13a97295593bf923d680c797fffc5c02d7cc05cb2dfa1'}}
