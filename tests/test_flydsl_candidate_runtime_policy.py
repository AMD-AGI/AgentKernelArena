"""Task-owned candidate policy regressions; CPU fixtures do not qualify GPU code."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASKS=['torch2flydsl/fused_add_rmsnorm_kernel', 'torch2flydsl/layernorm2d_kernel', 'torch2flydsl/layernorm2d_with_add_kernel', 'torch2flydsl/per_token_i8_quant_kernel', 'torch2flydsl/rmsnorm2d_kernel', 'torch2flydsl/rope_fwd_kernel', 'torch2flydsl/silu_and_mul_kernel', 'torch2flydsl/swiglu_and_mul_kernel', 'triton2flydsl/sglang/fused_dual_residual_rmsnorm', 'triton2flydsl/sglang/gdn_l2norm_fwd', 'triton2flydsl/sglang/fused_gdn_gating', 'triton2flydsl/sglang/fused_norm_gate', 'torch2flydsl/smoothquant_kernel', 'torch2flydsl/rmsnorm2d_smoothquant_kernel', 'torch2flydsl/moe_topk_sigmoid_kernel', 'triton2flydsl/sglang/merge_state', 'triton2flydsl/sglang/experts_combine', 'triton2flydsl/aiter/fused_clamp_act_mul', 'torch2flydsl/moe_topk_softplus_kernel', 'torch2flydsl/moe_biased_grouped_topk_kernel', 'torch2flydsl/rope_thd_fwd_kernel', 'torch2flydsl/per_1x128_fp8_quant_kernel', 'torch2flydsl/per_token_fp8_quant_kernel', 'torch2flydsl/rmsnorm2d_dynamicquant_kernel', 'torch2flydsl/silu_and_mul_quant_kernel', 'triton2flydsl/sglang/gdn_fused_recurrent_decode', 'triton2flydsl/sglang/chunk_local_cumsum', 'triton2flydsl/generative_recommenders/jagged_dense_broadcast_add', 'torch2flydsl/gelu_and_mul_kernel', 'triton2flydsl/sglang/dsv4_fp4_indexer', 'triton2flydsl/sglang/lightning_attn', 'triton2flydsl/sglang/triton_mrope_fused', 'torch2flydsl/per_tensor_fp8_quant_kernel']

def checks(task):
    return load(ROOT/'tasks'/task/'scripts/candidate_checks.py','_policy_'+task.replace('/','_'))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def fake_fly_kernel():
    # CPU-only profile-event fixture. It is deliberately not native FlyDSL/GPU.
    namespace={'__name__':'flydsl.compiler._cpu_profile_fixture','torch':torch}
    exec('class FakeCompiledKernel:\n def __call__(self,x):\n  return x.clone()\n',namespace)
    return namespace['FakeCompiledKernel']()

def harness(task,target,benchmark,*,baseline=False):
    h=SimpleNamespace(ARENA_FINAL_CANDIDATE=not baseline,ARENA_PROVIDED_BASELINE=baseline,ENTRY='op',ENTRIES=('op',),KERNEL_FILE='kernel.py',benchmark_cuda_graph_or_events=benchmark)
    # The real protected loaders execute a fresh candidate module per load.
    source=lambda:SimpleNamespace(op=target,flydsl_op=target)
    h.load_module=lambda: source()
    h._load_source=lambda: source()
    h._load_module=lambda kernel_dir,filename,alias: source()
    return h

def get_candidate(h,task):
    if task.startswith('torch2flydsl/'):
        return h._load_module('.',h.KERNEL_FILE,'candidate').flydsl_op
    return (h._load_source() if '/aiter/' in task else h.load_module()).op

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('switch_after',[1,7,31])
def test_late_PyTorch_math_on_actual_benchmark_callback_rejected(task,switch_after):
    c=checks(task); fly=fake_fly_kernel(); state={'calls':0,'inside':False}
    x=torch.arange(8,dtype=torch.float32)
    def target(x):
        state['calls']+=1
        if state['calls']>switch_after:
            return x+0  # Numerically correct, but target-language delegation.
        return fly(x)
    def benchmark(fn,*args,**kwargs):
        state['inside']=True
        try:
            for _ in range(40):fn()
        finally:state['inside']=False
    h=harness(task,target,benchmark)
    original=h.benchmark_cuda_graph_or_events
    with pytest.raises(RuntimeError,match='non-preparation PyTorch operation aten::add'):
        with c.audit_candidate_calls(h,performance=True):
            fn=get_candidate(h,task)
            h.benchmark_cuda_graph_or_events(lambda:fn(x),warmup=10,repetition=100)
    assert state['calls']==switch_after+1
    assert h.benchmark_cuda_graph_or_events is original

@pytest.mark.parametrize('task',TASKS)
def test_eager_replay_late_delegate_is_still_guarded(task):
    c=checks(task);fly=fake_fly_kernel();state={'calls':0};x=torch.ones(8)
    def target(x):
        state['calls']+=1
        return fly(x) if state['calls']<=4 else x.sum().expand_as(x)
    def benchmark(fn,**kwargs):
        for _ in range(3):fn()
        return fn
    h=harness(task,target,benchmark)
    with pytest.raises(RuntimeError,match='aten::sum'):
        with c.audit_candidate_calls(h,performance=True):
            fn=get_candidate(h,task)
            replay=h.benchmark_cuda_graph_or_events(lambda:fn(x))
            assert state['calls']==4
            try:
                replay()
            except RuntimeError:
                pass  # A surrounding harness cannot hide the sticky violation.

@pytest.mark.parametrize('task',TASKS)
def test_legal_preparation_and_Fly_fixture_no_profile_inside_timer(task,monkeypatch):
    c=checks(task);fly=fake_fly_kernel();x=torch.ones(2,4)
    inside={'value':False};original_profile=sys.setprofile;outer=sys.getprofile();calls=[];prepared=[]
    def profile(fn):
        assert not inside['value'],'profiling must stay outside the canonical clock/capture'
        original_profile(fn)
    monkeypatch.setattr(sys,'setprofile',profile)
    def target(x):
        y=torch.empty_like(x).view(-1).view_as(x)
        y.copy_(x)
        return fly(y)
    def benchmark(fn,*args,**kwargs):
        calls.append((args,kwargs))
        inside['value']=True
        try:
            assert sys.getprofile() is outer
            for _ in range(6):assert torch.equal(fn(),x)
            return (0.125,{'benchmark_samples':100,'benchmark_warmup':10})
        finally:inside['value']=False
    h=harness(task,target,benchmark)
    original=h.benchmark_cuda_graph_or_events
    with c.audit_candidate_calls(h,performance=True) as observed:
        fn=get_candidate(h,task)
        sentinel=object();prepare=lambda:prepared.append('restored')
        result=h.benchmark_cuda_graph_or_events(lambda:fn(x),warmup=10,repetition=100,prepare_fn=prepare,timed_run=sentinel)
        assert observed
    assert result==(0.125,{'benchmark_samples':100,'benchmark_warmup':10})
    assert calls[0][1]=={'warmup':10,'repetition':100,'prepare_fn':prepare,'timed_run':sentinel}
    assert prepared==['restored','restored']
    assert h.benchmark_cuda_graph_or_events is original

@pytest.mark.parametrize('task',TASKS)
def test_baseline_policy_loader_timer_and_Torch_math_untouched(task):
    c=checks(task);x=torch.ones(8);h=harness(task,lambda x:x+1,lambda fn,**kwargs:fn(),baseline=True)
    timer=h.benchmark_cuda_graph_or_events
    with c.audit_candidate_calls(h,performance=True) as observed:
        assert h.benchmark_cuda_graph_or_events is timer
        assert torch.equal(get_candidate(h,task)(x),x+1)
    assert not observed

REGISTRY_BAD=[
    'import sys\ndef op(): return sys.modules["arena_harness"]',
    'import sys as registry\ndef op(): return registry.modules',
    'from sys import modules as registry\ndef op(): return registry["arena_harness"]',
    'import sys\na=sys\nb=a\ndef op(): return b.modules',
    'import sys\ndef op(): return getattr(sys,"mod"+"ules")',
    'import sys\ndef op(): return vars(sys)["modules"]',
    'import sys\ndef op(): return sys.__dict__["modules"]',
    'import sys\ndef op(): return sys._getframe()',
    'import sys\ndef op(): return sys.setprofile(None)',
    'import sys\nfrom builtins import getattr as fetch\ndef op(): return fetch(sys,\"modules\")',
    'import sys\ndef op(): return (lambda:sys)().modules',
    'import sys\nregistry=(sys,)\ndef op(): return registry[0].modules',
]

@pytest.mark.parametrize('task',TASKS)
def test_sys_safe_metadata_and_installed_library_internal_sys_allowed(task,tmp_path):
    entry=tmp_path/'candidate.py';entry.write_text('import flydsl\nimport sys\nVERSION=sys.version_info\ndef op(): return sys.platform\n')
    after=load(ROOT/'tasks'/task/'task_runtime.py','_safe_runtime');after.ROOT=tmp_path
    assert after.check_dependencies([entry],True)==[entry]

@pytest.mark.parametrize('task',TASKS)
def test_local_helper_cannot_hide_registry_access(task,tmp_path):
    entry=tmp_path/'candidate.py';entry.write_text('import flydsl\nfrom helper import op\n')
    (tmp_path/'helper.py').write_text('import sys as registry\ndef op(): return registry.modules\n')
    after=load(ROOT/'tasks'/task/'task_runtime.py','_closure_runtime');after.ROOT=tmp_path
    with pytest.raises(ValueError,match='module registry'):
        after.check_dependencies([entry],True)

def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

@pytest.mark.parametrize('shape',[(1,128),(3,384),(5,1152),(2,4224)])
def test_FP8_legal_small_group_model_comparator_tail_and_wrong_scale_controls(shape):
    b=ROOT/'tasks/torch2flydsl/per_1x128_fp8_quant_kernel'
    model=load(b/'model.py','_model_cpu');reference=model.Model()
    funcs=functions(b/'test_kernel_harness.py')
    namespace={'SCALE_RTOL':1e-3,'CODE_TOL':1}
    exec(compile(ast.Module([funcs['_compare']],type_ignores=[]),'<actual protected comparator>','exec'),namespace)
    gen=torch.Generator().manual_seed(20260401)
    x=torch.randn(*shape,dtype=torch.bfloat16,generator=gen);original=x.clone()
    y,s=reference(x)
    assert y.shape==x.shape and s.shape==(shape[0],shape[1]//128)
    assert namespace['_compare']((y,s),(y,s))[0]
    # A vectorized implementation that ignores the last complete 128-group
    # returns finite, correctly shaped values but fails the original comparator.
    bad_y=y.clone();bad_y.view(torch.uint8)[:,-128:]=0
    assert not namespace['_compare']((bad_y,s),(y,s))[0]
    bad_s=s.clone();bad_s[:,-1]*=2
    assert not namespace['_compare']((y,bad_s),(y,s))[0]
    assert torch.equal(x,original)
    zero=torch.zeros_like(x);zy,zs=reference(zero)
    assert bool((zy.float()==0).all()) and bool((zs==1).all())

@pytest.mark.parametrize('task',TASKS)
def test_actual_task_actions_performance_entry_enables_late_callback_policy(task,monkeypatch):
    cp=checks(task);x=torch.ones(8);fly=fake_fly_kernel();state={'calls':0}
    def target(x):
        state['calls']+=1
        return fly(x) if state['calls']==1 else x+0
    h=harness(task,target,lambda fn,**kwargs:fn())
    def performance(*args,**kwargs):
        fn=get_candidate(h,task)
        return h.benchmark_cuda_graph_or_events(lambda:fn(x))
    h.run_performance=performance;h.arena_benchmark=performance;h.run_benchmark=performance
    package=ModuleType('scripts');package.__path__=[str(ROOT/'tasks'/task/'scripts')]
    monkeypatch.setitem(sys.modules,'scripts',package)
    monkeypatch.setitem(sys.modules,'scripts.candidate_checks',cp)
    actions=load(ROOT/'tasks'/task/'scripts/task_actions.py','_actual_actions')
    with pytest.raises(RuntimeError,match='aten::add'):
        actions.performance(h)
    assert state['calls']==2

@pytest.mark.parametrize('task',TASKS)
def test_capture_only_delegate_sticky_even_if_timer_suppresses_and_event_fallback_succeeds(task):
    c=checks(task);fly=fake_fly_kernel();x=torch.ones(8);capture={'active':False,'caught':False,'fallback':False}
    def target(x):
        out=fly(x)
        return out+0 if capture['active'] else out
    def benchmark(fn,**kwargs):
        capture['active']=True
        try:fn()
        except RuntimeError:capture['caught']=True
        finally:capture['active']=False
        fallback=fn();capture['fallback']=True
        assert torch.equal(fallback,x)
        return .1,{'benchmark_method':'cuda_event_fallback','benchmark_samples':100}
    h=harness(task,target,benchmark)
    with pytest.raises(RuntimeError,match='aten::add'):
        with c.audit_candidate_calls(h,performance=True):
            fn=get_candidate(h,task)
            h.benchmark_cuda_graph_or_events(lambda:fn(x))
    assert capture=={'active':False,'caught':True,'fallback':True}

@pytest.mark.parametrize('task',TASKS)
def test_candidate_cannot_swallow_operation_violation_and_return_legal_output(task):
    c=checks(task);fly=fake_fly_kernel();x=torch.ones(8)
    def target(x):
        try:x+0
        except RuntimeError:pass
        return fly(x)
    h=harness(task,target,lambda fn,**kwargs:fn())
    with pytest.raises(RuntimeError,match='aten::add'):
        with c.audit_candidate_calls(h,performance=True):
            fn=get_candidate(h,task)
            h.benchmark_cuda_graph_or_events(lambda:fn(x))

@pytest.mark.parametrize('task',TASKS)
def test_legitimate_capture_fallback_policy_metadata_unchanged(task):
    c=checks(task);fly=fake_fly_kernel();x=torch.ones(8)
    def benchmark(fn,**kwargs):
        try:raise RuntimeError('legitimate unsupported graph capture')
        except RuntimeError:pass
        assert torch.equal(fn(),x)
        return .1,{'benchmark_method':'cuda_event_fallback','benchmark_samples':100,'benchmark_warmup':10}
    h=harness(task,lambda x:fly(x),benchmark)
    with c.audit_candidate_calls(h,performance=True):
        fn=get_candidate(h,task)
        result=h.benchmark_cuda_graph_or_events(lambda:fn(x))
    assert result==(.1,{'benchmark_method':'cuda_event_fallback','benchmark_samples':100,'benchmark_warmup':10})

@pytest.mark.parametrize('bad_candidate',[False,True])
def test_real_FP8_correctness_loop_executes_new_groups_and_rejects_last_group_omission(bad_candidate,monkeypatch):
    root=ROOT/'tasks/torch2flydsl/per_1x128_fp8_quant_kernel'
    rt=load(root/'task_runtime.py','_fp8_runtime_loop')
    package=ModuleType('scripts');package.__path__=[str(root/'scripts')]
    benchmark=ModuleType('_aka_benchmark');benchmark.TimedRun=object;benchmark.benchmark_cuda_graph_or_events=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,'task_runtime',rt)
    monkeypatch.setitem(sys.modules,'scripts',package)
    monkeypatch.setitem(sys.modules,'_aka_benchmark',benchmark)
    monkeypatch.delitem(sys.modules,'scripts.replay_checks',raising=False)
    h=load(root/'test_kernel_harness.py','_fp8_harness_loop')
    m=load(root/'model.py','_fp8_model_loop');model=m.Model()
    # Numeric harness-loop unit check only: CPU tensors, no AITER/GPU claims.
    controls=h.CORRECTNESS_CONTROLS
    h.CORRECTNESS_SHAPES=controls
    make=h._make_inputs;draws=[]
    def make_cpu(shape):
        draws.append(shape['name']);return make(shape,device='cpu')
    h._make_inputs=make_cpu;h._aiter_op=lambda x:model(x)
    def candidate(x):
        y,s=model(x)
        if bad_candidate and x.shape[1]//128 not in (32,64):
            y.view(torch.uint8)[:,-128:]=0
        return y,s
    h._load_module=lambda kernel_dir,filename,alias:m if filename==h.MODEL_FILE else SimpleNamespace(flydsl_per_1x128_fp8_quant=candidate)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.nn.Module,'to',lambda self,*args,**kwargs:self)
    if bad_candidate:
        with pytest.raises(AssertionError,match='correctness FAILED'):
            h.run_correctness(verbose=False)
    else:
        assert h.run_correctness(verbose=False) is True
    assert draws==[c['name'] for c in controls]

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('source',REGISTRY_BAD)
def test_candidate_registry_and_control_frame_routes_rejected_before_import(task,source,tmp_path):
    entry=tmp_path/'candidate.py';entry.write_text('import flydsl\n'+source+'\n')
    runtime=load(ROOT/'tasks'/task/'task_runtime.py','_registry_policy');runtime.ROOT=tmp_path
    with pytest.raises(ValueError,match='module registry|control frames|Dynamic sys'):
        runtime.check_dependencies([entry],True)
    runtime.check_dependencies([entry],False)


def test_group_geometry_controls_preserve_original_six_scored_cases():
    import json
    path=ROOT/'tasks/torch2flydsl/per_1x128_fp8_quant_kernel'
    rows=json.loads((path/'cases.json').read_text())['cases']
    scored=[r for r in rows if 'performance' in r['checks']]
    assert [(r['params']['m'],r['params']['n']) for r in scored]==[(1,4096),(16,8192),(128,4096),(256,8192),(1024,4096),(4096,4096)]
    controls=[r for r in rows if r['checks']==['correctness']]
    assert {r['params']['n']//128 for r in controls}=={1,3,9,33}
    assert all(r['params']['n']%128==0 for r in controls)


def test_real_edit_boundary_allows_kernel_edit_but_registry_policy_rejects_before_import(tmp_path):
    import shutil
    from src.task_spec import load_task_spec
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    task='triton2flydsl/sglang/experts_combine'
    work=tmp_path/'task';shutil.copytree(ROOT/'tasks'/task,work)
    spec=load_task_spec(work/'config.yaml',task_id=task)
    guard=snapshot_workspace_harness(work,task_spec=spec)
    source=work/'experts_combine.py'
    source.write_text('import flydsl\nimport sys as registry\ndef experts_combine_triton(*args): return registry.modules["arena_harness"]\n')
    verify_workspace_harness(guard,discard_added=False)
    runtime=load(work/'task_runtime.py','_real_guard_policy')
    with pytest.raises(ValueError,match='module registry'):
        runtime.check_dependencies([source],True)
    protected=work/'cases.json';protected.write_text(protected.read_text()+'\n')
    with pytest.raises(Exception):verify_workspace_harness(guard,discard_added=False)
