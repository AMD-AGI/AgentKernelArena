"""CPU contract regressions; real Triton/GPU qualification is separate evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = '66b187244d9d08d3fa46a24e9245a02367e14d06'
NAMES = ('triton_ssd_chunk_cumsum', 'triton_ssd_chunk_scan', 'triton_ssd_chunk_state',
         'triton_ssd_chunk_state_varlen', 'triton_ssd_state_passing')


@pytest.fixture(params=NAMES)
def task(request, monkeypatch):
    root = ROOT / 'tasks/triton2triton/vllm' / request.param
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    spec = importlib.util.spec_from_file_location('_ssd_test_harness', root / 'scripts/task_runner.py')
    h = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, h)
    spec.loader.exec_module(h)
    return root, h, h.semantic_controls, sys.modules['scripts.ssd_checks']


def test_independent_reference_and_comparator_controls(task):
    _, h, _, _ = task
    rows = h.run_reference_controls()
    assert rows and all(row['status'] == 'PASS' and row['negative_control'] == 'rejected' for row in rows)


def test_control_rejects_wrong_full_outputs_and_input_mutation(task, monkeypatch):
    _, _, controls, checks = task
    original = list(controls.control_cases('cpu'))
    for case in original:
        monkeypatch.setattr(controls, 'control_cases', lambda device, c=case: iter([c]))
        expected = case['expected']
        def bad_output(**kwargs):
            if case.get('output_args'):
                out = kwargs[case['output_args'][0]]
                out.copy_(expected + 100)
                return out
            return tuple(x+100 for x in expected) if isinstance(expected, tuple) else expected+100
        mod = types.SimpleNamespace(**{case['entrypoint']: bad_output})
        with pytest.raises(checks.NumericalMismatch):
            controls.run_controls(mod, device='cpu')
        def mutates_input(**kwargs):
            key = next(k for k,v in kwargs.items() if isinstance(v, torch.Tensor) and
                       v.is_floating_point() and k not in case.get('output_args', ()))
            kwargs[key].add_(1)
            if case.get('output_args'):
                out=kwargs[case['output_args'][0]];out.copy_(expected);return out
            return tuple(x.clone() for x in expected) if isinstance(expected, tuple) else expected.clone()
        mod = types.SimpleNamespace(**{case['entrypoint']: mutates_input})
        with pytest.raises(checks.ContractFailure, match='Read-only input'):
            controls.run_controls(mod, device='cpu')


@pytest.mark.parametrize("task", ["triton_ssd_state_passing"], indirect=True)
def test_state_passing_controls_reject_ignoring_initial_state(task):
    root,h,controls,checks=task
    def ignores_initial(states,dA_cumsum,cu_chunk_seqlens,seq_idx,initial_states=None,out_dtype=None):
        return h.reference(states,dA_cumsum,seq_idx).to(out_dtype)
    with pytest.raises(checks.NumericalMismatch):
        controls.run_controls(types.SimpleNamespace(state_passing_fwd=ignores_initial),'cpu')


@pytest.mark.parametrize("task", ["triton_ssd_chunk_state_varlen"], indirect=True)
def test_varlen_controls_reject_ignoring_initial_state(task):
    root,h,controls,checks=task
    def ignores_initial(B,x,dt,dA_cumsum,cu_seqlens,chunk_states,initial_states=None):
        return h.reference_chunk_state_varlen(B,x,dt,dA_cumsum,cu_seqlens,chunk_states,dt.shape[-1])
    with pytest.raises(checks.NumericalMismatch):
        controls.run_controls(types.SimpleNamespace(chunk_state_varlen=ignores_initial),'cpu')


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


def _base(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT)}'],cwd=ROOT,text=True)


def test_original_workloads_gates_kernels_and_timing_preserved(task):
    root,h,_,_=task
    for path in [root/'workloads.json',*list((root/'source').glob('*.py'))]:
        assert path.read_text()==_base(path)
    original_config=yaml.safe_load(_base(root/'config.yaml'))
    current_config=yaml.safe_load((root/'config.yaml').read_text())
    edit=current_config['candidate']['editable'][0]
    assert edit['scope']=='symbols' and edit['allow_new_helpers'] is True
    assert edit['path']==original_config['candidate']['editable'][0]
    symbols=[current_config['candidate']['entrypoints'][0]['symbol']]
    if root.name=='triton_ssd_chunk_cumsum':symbols.append('softplus')
    assert edit['symbols']==symbols
    current_config['candidate']['editable']=original_config['candidate']['editable']
    assert current_config==original_config
    path=root/'scripts/task_runner.py';before=_base(path);after=path.read_text()
    trees=[ast.parse(s) for s in (before,after)]
    def protected(tree):
        return [ast.dump(n,include_attributes=False) for n in tree.body if
                isinstance(n,ast.FunctionDef) and (n.name.startswith('reference') or n.name=='ref_softplus') or
                isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                ('TEST_SHAPES','WARMUP_ITERATIONS','BENCHMARK_ITERATIONS') for t in n.targets)]
    assert protected(trees[0])==protected(trees[1])
    marker='# >>> AKA-GENERATED:';end='# <<< AKA-GENERATED <<<'
    assert before.split(marker)[1].split(end)[0]==after.split(marker)[1].split(end)[0]
    def generation(tree):
        found=[]
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in ('run_correctness','run_performance'):
                for n in ast.walk(fn):
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('manual_seed','randn','rand','arange','cumsum'):
                        found.append((fn.name,ast.dump(n,include_attributes=False)))
        return sorted(found)
    assert generation(trees[0])==generation(trees[1])
    calls=[n for n in ast.walk(trees[1]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_benchmark_cuda_graph_or_events']
    assert len(calls)==1
    assert {k.arg:ast.unparse(k.value) for k in calls[0].keywords}=={
        'warmup':'WARMUP_ITERATIONS','repetition':'BENCHMARK_ITERATIONS','timed_run':'timed'}


def test_protocol_manifest_controls_and_failed_performance(task,monkeypatch):
    root,h,_,_=task
    spec=importlib.util.spec_from_file_location('_ssd_eval_test',root/'_arena_eval.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'load_harness',lambda:h)
    real_find=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name,*a: object() if name=='triton' else real_find(name,*a))
    report=m.evaluate('task','validate-task')
    assert report['status']=='PASS' and report['metadata']['reference_controls']
    assert len(report['cases'])==5
    from src.task_spec import load_task_spec
    from src.task_protocol import parse_command_result,CaseManifest
    spec=load_task_spec(root/'config.yaml',task_id='triton2triton/vllm/'+root.name)
    assert spec.candidate.initial_state=='implemented'
    result=parse_command_result('ARENA_EVAL_RESULT='+json.dumps(report),role='task',action='validate-task',returncode=0)
    manifest=CaseManifest.from_result(result)
    monkeypatch.setattr(h,'run_performance',lambda:[{'test_case_id':r['test_case_id'],'execution_time_ms':-1.,'error':'replay rejected'} for r in report['cases']])
    failed=m.evaluate('candidate','performance')
    assert failed['status']=='FAIL'
    assert all(r['reason']=='replay rejected' for r in failed['cases'])
    parsed=parse_command_result('ARENA_EVAL_RESULT='+json.dumps(failed),role='candidate',action='performance',returncode=1)
    manifest.validate(parsed)


@pytest.mark.parametrize('fault', ['none', 'wrong_measured', 'cached_replay', 'timed_input_write'])
def test_real_performance_runner_wires_collector_and_rejects_bad_paths(task, monkeypatch, fault):
    """Execute the actual runner on CPU tensors with explicitly fake device timing."""
    root,h,_,_=task
    from src.tools.perf.aka_benchmark import TimedRun
    for name in ('randn','rand','arange','zeros'):
        original=getattr(torch,name)
        def factory(*args,_original=original,_name=name,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            value = _original(*args,**kwargs)
            # Large CPU fixture signals distinguish stale results even at the
            # retained chunk-state atol=.5; this is not a scored workload.
            return value * 10 if _name == "randn" else value
        monkeypatch.setattr(torch,name,factory)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    if root.name=='triton_ssd_chunk_cumsum':
        shape=(4,2,2,True,False)
        def op(dt,A,chunk_size,cu,dt_bias=None,dt_softplus=False):
            return h.reference(dt,A,chunk_size,cu,dt_bias,dt_softplus)
        name='chunk_cumsum_fwd'
    elif root.name=='triton_ssd_chunk_scan':
        shape=(4,2,2,1,2,2)
        def op(cb,x,dt,dA,C,states,cu,out,seq):
            out.copy_(h.reference_chunk_scan(cb,x,dt,dA,C,states,seq,dt.shape[-1]))
        name='chunk_scan_fwd'
    elif root.name=='triton_ssd_chunk_state':
        shape=(4,2,2,1,2,2)
        def op(B,x,dt,dA,cu):return h.reference(B,x,dt,dA,cu)
        name='chunk_state_fwd'
    elif root.name=='triton_ssd_chunk_state_varlen':
        shape=(8,2,2,2,1,2,2)
        def op(B,x,dt,dA,cu,states):return h.reference_chunk_state_varlen(B,x,dt,dA,cu,states,dt.shape[-1])
        name='chunk_state_varlen'
    else:
        shape=(2,2,2,2)
        def op(states,dA,cu,seq):return h.reference(states,dA,seq)
        name='state_passing_fwd'
    monkeypatch.setattr(h,'TEST_SHAPES',[shape])
    monkeypatch.setattr(h,'load_module',lambda:types.SimpleNamespace(**{name:op}))
    monkeypatch.setitem(sys.modules,'_aka_benchmark',types.SimpleNamespace(TimedRun=TimedRun))
    calls=[]
    def fake_benchmark(fn,*,warmup,repetition,timed_run):
        assert (warmup,repetition)==(10,100)
        output=fn();assert output is not None
        values=output if isinstance(output,tuple) else (output,)
        saved=tuple(v.clone() for v in values)
        def replay():
            if fault=='cached_replay':
                for out,old in zip(values,saved):out.copy_(old)
                return output
            return fn()
        timed_run._bind(replay,output)
        if fault=='wrong_measured':
            for out in values:out.add_(100)
        if fault=='timed_input_write':
            target=next(c.cell_contents for c in fn.__closure__ if isinstance(c.cell_contents,torch.Tensor) and
                        c.cell_contents.is_floating_point() and all(c.cell_contents is not o for o in values))
            target.add_(1)
        calls.append(True)
        return 1.0,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph','cpu_fixture':True}
    monkeypatch.setattr(h,'_benchmark_cuda_graph_or_events',fake_benchmark)
    records=h.run_performance()
    assert len(records)==1 and calls
    if fault=='none':
        assert records[0]['execution_time_ms']==1.
        assert records[0]['timed_output_correctness']=='PASS' and records[0]['replay_correctness']=='PASS'
    else:
        assert records[0]['execution_time_ms']<0 and records[0].get('error')


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
