"""CPU contract regressions; real Triton/GPU qualification is separate evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import hashlib
import sys
import types

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {'triton_fla_fused_recurrent': {'allclose': ["Call(func=Attribute(value=Name(id='torch', "
                                             "ctx=Load()), attr='allclose', ctx=Load()), "
                                             "args=[Name(id='r_cpu', ctx=Load()), Name(id='ref_f', "
                                             "ctx=Load())], keywords=[keyword(arg='atol', "
                                             "value=Constant(value=0.05)), keyword(arg='rtol', "
                                             'value=Constant(value=0.05))])'],
                                'files': {'source/triton_fla_fused_recurrent.py': '931a0358c40f99eb3359afc054db4c89978d641f7252b123111e69f341db60e8',
                                          'workloads.json': 'cc6d49941de90d729fe34c69f6de250f6f76c9b2ad893235ca1f8ae14b562601'},
                                'generated': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                                'protected': [('reference',
                                               'a418b973e5313aa5f609337eab4ecc453a939ff342f56697b3d088e8039ca305'),
                                              ('gen_inputs',
                                               'f8912a9692e61ed111e640a27b6a07f724bae9b0d5f55047cc26612ef12e5a7b')],
                                'timing': [('_benchmark_cuda_graph_or_events',
                                            [('warmup', 'WARMUP_ITERATIONS'),
                                             ('repetition', 'BENCHMARK_ITERATIONS')])]},
 'triton_linear_attn_decode': {'allclose': ["Call(func=Attribute(value=Name(id='torch', "
                                            "ctx=Load()), attr='allclose', ctx=Load()), "
                                            "args=[Call(func=Attribute(value=Name(id='triton_out', "
                                            "ctx=Load()), attr='float', ctx=Load()), args=[], "
                                            'keywords=[]), '
                                            "Call(func=Attribute(value=Name(id='ref_out', "
                                            "ctx=Load()), attr='float', ctx=Load()), args=[], "
                                            "keywords=[])], keywords=[keyword(arg='atol', "
                                            "value=Constant(value=0.01)), keyword(arg='rtol', "
                                            'value=Constant(value=0.01))])',
                                            "Call(func=Attribute(value=Name(id='torch', "
                                            "ctx=Load()), attr='allclose', ctx=Load()), "
                                            "args=[Call(func=Attribute(value=Name(id='kv_caches_triton', "
                                            "ctx=Load()), attr='float', ctx=Load()), args=[], "
                                            'keywords=[]), '
                                            "Call(func=Attribute(value=Name(id='kv_caches_ref', "
                                            "ctx=Load()), attr='float', ctx=Load()), args=[], "
                                            "keywords=[])], keywords=[keyword(arg='atol', "
                                            "value=Constant(value=0.01)), keyword(arg='rtol', "
                                            'value=Constant(value=0.01))])'],
                               'files': {'source/triton_linear_attn_decode.py': 'e8429e9326718bff4af57ea980b1c2b415af0bb4e24b6aa65eb9754dba3b5af5',
                                         'workloads.json': '6438b5703598d840b564609318bb794e95b5f1184f42aaf14f7b49455aaf099e'},
                               'generated': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                               'protected': [('reference_linear_attn_decode',
                                              '65b9a17df6b52f94770248615a953957b8f6a5b1cc1379791ab950ab2f4a511d')],
                               'timing': [('_benchmark_cuda_graph_or_events',
                                           [('warmup', 'WARMUP_ITERATIONS'),
                                            ('repetition', 'BENCHMARK_ITERATIONS'),
                                            ('prepare_fn', 'lambda: kv_work.copy_(kv_caches)')])]},
 'triton_selective_scan_update': {'allclose': ["Call(func=Attribute(value=Name(id='torch', "
                                               "ctx=Load()), attr='allclose', ctx=Load()), "
                                               "args=[Call(func=Attribute(value=Name(id='out', "
                                               "ctx=Load()), attr='cpu', ctx=Load()), args=[], "
                                               "keywords=[]), Name(id='ref', ctx=Load())], "
                                               "keywords=[keyword(arg='atol', "
                                               "value=Constant(value=0.01)), keyword(arg='rtol', "
                                               'value=Constant(value=0.01))])'],
                                  'files': {'source/triton_selective_scan_update.py': '0f51b765808afe31d980bcada0bb406279231b58a9cdae2a28af61f47fde7320',
                                            'workloads.json': 'dca4c9d570eb82a27b18dc46a0d3a9a0c8c8d97846b7e82779db35d74dc54ad0'},
                                  'generated': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                                  'protected': [('reference',
                                                 '9bed052ab5855154ac2ce76ea77e3ed2795df43441f1dfe9ccc198e18d6a31f4')],
                                  'timing': [('_benchmark_cuda_graph_or_events',
                                              [('warmup', 'WARMUP_ITERATIONS'),
                                               ('repetition', 'BENCHMARK_ITERATIONS'),
                                               ('target_ms', '20.0'),
                                               ('prepare_fn',
                                                'lambda: state_work.copy_(state)')])]}}
NAMES = ('triton_fla_fused_recurrent', 'triton_linear_attn_decode', 'triton_selective_scan_update')

@pytest.fixture(params=NAMES)
def task(request, monkeypatch):
    root = ROOT / 'tasks/triton2triton/vllm' / request.param
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    spec = importlib.util.spec_from_file_location('_fla_test_harness', root / 'scripts/task_runner.py')
    h = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, h)
    spec.loader.exec_module(h)
    return root, h, h.semantic_controls, sys.modules['scripts.contract_checks']


def test_independent_reference_and_comparator_controls(task):
    _, h, _, _ = task
    rows = h.run_reference_controls()
    assert rows and all(row['status'] == 'PASS' and row['negative_control'] == 'rejected' for row in rows)


@pytest.mark.parametrize('fault',['dtype','shape','nonfinite','missing_tuple','alias'])
def test_output_contract_checks_full_tensors(task,fault):
    _,_,_,checks=task
    ref=torch.arange(1.,9.).reshape(2,4)
    actual=ref.clone();expected=ref;inputs=()
    if fault=='dtype':actual=actual.half()
    if fault=='shape':actual=actual.reshape(4,2)
    if fault=='nonfinite':actual[0,0]=float('nan')
    if fault=='missing_tuple':expected=(ref,ref);actual=(actual,)
    if fault=='alias':inputs=(actual,)
    with pytest.raises(checks.ContractFailure):checks.check_outputs(actual,expected,atol=.01,rtol=.01,inputs=inputs)


@pytest.mark.parametrize('fault',['none','wrong_measured','cached_replay','input_mutation','replay_mutation','no_binding'])
def test_actual_timed_output_replay_and_readonly_guards(task,fault):
    _,_,_,checks=task
    from src.tools.perf.aka_benchmark import TimedRun
    x=torch.tensor([2.,3.]);original=x.clone();readonly=checks.InputSnapshot({'x':x})
    output=x*3
    def replay():
        if fault=='replay_mutation':x.add_(7)
        output.copy_(original*3 if fault=='cached_replay' else x*3)
        return output
    timed=TimedRun()
    if fault!='no_binding':timed._bind(replay,output)
    if fault=='wrong_measured':output.add_(10)
    if fault=='input_mutation':x.add_(1)
    def run():return checks.validate_timed(timed,readonly,lambda:x*3,lambda:x.neg_(),atol=.01,rtol=.01)
    if fault=='none':
        assert run()['replay_correctness']=='PASS'
        assert torch.equal(x,original)
    else:
        with pytest.raises(checks.ContractFailure):run()


def test_original_state_contract_preserved(task):
    root,h,_,_=task;saved=BASELINE[root.name]
    digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
    for f,expected in saved['files'].items():assert digest((root/f).read_text())==expected
    text=(root/'scripts/task_runner.py').read_text();tree=ast.parse(text)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name,expected in saved['protected']:assert digest(ast.dump(functions[name],include_attributes=False))==expected
    for text_ast in saved['allclose']:
        assert text_ast in [ast.dump(n,include_attributes=False) for n in ast.walk(tree)]
    assert digest(text.split('# >>> AKA-GENERATED:')[1].split('# <<< AKA-GENERATED <<<')[0])==saved['generated']
    timing=[(ast.unparse(n.func),[(k.arg,ast.unparse(k.value)) for k in n.keywords if k.arg!='timed_run'])
            for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_benchmark_cuda_graph_or_events']
    assert timing==saved['timing']
    assert (h.WARMUP_ITERATIONS,h.BENCHMARK_ITERATIONS)==(10,100)


@pytest.mark.parametrize('fault',['none','output','state','input','missing'])
def test_public_controls_check_outputs_and_entire_state(task,monkeypatch,fault):
    root,h,controls,checks=task
    for case in list(controls.control_cases('cpu')):
        monkeypatch.setattr(controls,'control_cases',lambda device,c=case:iter([c]))
        def operation(**kw):
            expected=case['expected'];out=expected[0].clone();state=expected[1].clone()
            if fault=='output':out.add_(100)
            if fault=='state':state.add_(100)
            if fault=='input':
                target=next(v for k,v in kw.items() if k in ('q','x') and isinstance(v,torch.Tensor));target.add_(1)
            if root.name=='triton_fla_fused_recurrent':return (out,) if fault=='missing' else (out,state)
            if root.name=='triton_linear_attn_decode':
                kw['kv_caches'].copy_(state);return None if fault=='missing' else out.to(kw['q'].dtype)
            kw['state'].copy_(state);kw['out'].copy_(out)
            return None if fault=='missing' else kw['out']
        mod=types.SimpleNamespace(**{case['entrypoint']:operation})
        if fault=='none':assert controls.run_controls(mod,'cpu')
        else:
            with pytest.raises(checks.ContractFailure):controls.run_controls(mod,'cpu')


@pytest.mark.parametrize('fault',['none','output','state','cached','input'])
def test_actual_stateful_performance_runner(task,monkeypatch,fault):
    root,h,controls,_=task
    from src.tools.perf.aka_benchmark import TimedRun
    for name in ('randn','rand','arange','zeros','empty'):
        original=getattr(torch,name)
        def factory(*args,_original=original,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return _original(*args,**kwargs)
        monkeypatch.setattr(torch,name,factory)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    if root.name=='triton_fla_fused_recurrent':
        monkeypatch.setattr(h,'SEEDS',[42]);op=controls.reference_outputs;name='fused_recurrent_gated_delta_rule_fwd'
    elif root.name=='triton_linear_attn_decode':
        monkeypatch.setattr(h,'TEST_SHAPES',[(2,2,4,32)])
        def op(q,k,v,cache,slope,slots):
            temp=cache.float().clone();out=h.reference_linear_attn_decode(q,k,v,temp,slope,slots)
            cache.copy_(temp);return out.to(q.dtype)
        name='linear_attn_decode_forward'
    else:
        monkeypatch.setattr(h,'TEST_SHAPES',[(2,4,5,3,2,True,True)])
        def op(state,x,dt,A,B,C,D=None,z=None,out=None):
            expected,updated=controls.reference_outputs(state,x,dt,A,B,C,D,z,h.reference)
            state.copy_(updated);out.copy_(expected);return out
        name='selective_state_update'
    monkeypatch.setattr(h,'load_module',lambda:types.SimpleNamespace(**{name:op}))
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=TimedRun))
    def benchmark(fn,*,warmup,repetition,timed_run,prepare_fn=None,**kwargs):
        assert (warmup,repetition)==(10,100)
        if prepare_fn:prepare_fn()
        result=fn();assert len(result)==2
        saved=tuple(x.clone() for x in result)
        def replay():
            if prepare_fn:prepare_fn()
            if fault=='cached':
                for x,old in zip(result,saved):x.copy_(old)
                return result
            return fn()
        timed_run._bind(replay,result)
        if fault=='output':result[0].add_(100)
        if fault=='state':result[1].add_(100)
        if fault=='input':
            items=[c.cell_contents for c in fn.__closure__]
            items=[v for item in items for v in (item if isinstance(item,tuple) else (item,))]
            target=next(x for x in items if isinstance(x,torch.Tensor) and x.is_floating_point() and all(x is not y for y in result));target.add_(1)
        return 1.,dict(benchmark_method='cuda_graph',benchmark_timed_run_kind='captured_graph',cpu_fixture=True)
    monkeypatch.setattr(h,'_benchmark_cuda_graph_or_events',benchmark)
    rows=h.run_performance();assert len(rows)==1
    if fault=='none':assert rows[0]['execution_time_ms']==1. and rows[0]['replay_correctness']=='PASS', rows[0]
    else:assert rows[0]['execution_time_ms']<0 and rows[0]['error']


def test_actual_framework_guard_allows_kernel_edits_but_rejects_wrapper_bypass(task,tmp_path):
    import shutil
    from src.task_spec import load_task_spec
    from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
    root,_,_,_=task
    workspace=tmp_path/'workspace';shutil.copytree(root,workspace)
    spec=load_task_spec(workspace/'config.yaml',task_id='triton2triton/vllm/'+root.name)
    snapshot=snapshot_workspace_harness(workspace,task_spec=spec)
    source=workspace/spec.candidate.editable[0].path
    tree=ast.parse(source.read_text())
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==spec.candidate.entrypoints[0].symbol)
    kernel.body.append(ast.Pass())
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    verify_workspace_harness(snapshot)
    wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
                 n.name not in spec.candidate.editable[0].symbols)
    wrapper.body=[ast.Return(ast.Constant(None))]
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(RuntimeError,match='[Hh]arness|protected'):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize('task',['triton_linear_attn_decode'],indirect=True)
def test_half_output_uses_original_fp32_gate_without_rounding_reference(task):
    _,_,_,checks=task
    actual=torch.tensor([1.],dtype=torch.float16)
    # Rounding the oracle to half would change this acceptance conclusion.
    expected=torch.tensor([1.0197],dtype=torch.float32)
    assert torch.allclose(actual.float(),expected.half().float(),atol=.0196,rtol=0)
    with pytest.raises(checks.NumericalMismatch):
        checks.check_outputs(actual,expected,atol=.0196,rtol=0,output_dtypes=torch.float16)


@pytest.mark.parametrize('task',['triton_fla_fused_recurrent'],indirect=True)
def test_recurrent_state_oracle_independently_rejects_ignored_initial_state(task):
    _,h,controls,checks=task
    c=list(controls.control_cases('cpu'))[-1];kw=dict(c['kwargs']);kw['initial_state']=None
    with pytest.raises(checks.NumericalMismatch):
        checks.check_outputs(controls.reference_outputs(**kw),c['expected'],atol=c['atol'],rtol=c['rtol'])
