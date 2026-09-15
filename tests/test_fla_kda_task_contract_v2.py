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
BASELINE = {'triton_fla_chunk_fwd_o': {'files': {'source/triton_fla_chunk_fwd_o.py': '3b47b94d96369bd0494c3a7996c47f62325a9371840382590c898ee65d417123',
                                      'workloads.json': 'db237f9668446ac0851f5bfee765b2e7ac223f5a218764749c842e42dcd2962c'},
                            'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                            'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                            'reference_and_constants_ast': 'b7466441ef03672e93f32a1288f52017c769bf8a269eb9bea29dcd84c461159b'},
 'triton_fla_scaled_dot_kkt': {'files': {'source/triton_fla_scaled_dot_kkt.py': '51fb28e231b5ab1575e84b7cfc4e14a5285f3e15d8aa4dd439b4e95236008f0e',
                                         'workloads.json': '8fd0010abf74c83b7629f8faa79f3417cc04626784bc79fb1719769bbfcecf47'},
                               'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                               'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                               'reference_and_constants_ast': '32fef019dc490e01648b8a59305b79bd333114054f880840a87a22bc394f3371'},
 'triton_kda_dot_kkt_inter': {'files': {'source/triton_kda_dot_kkt_inter.py': '98f2bc1a3d57be04b6f9587a1d4a64d99bca9f993e8b062bc6912c3abcd40820',
                                        'workloads.json': 'd71d254ac8d75eff6aaa21542e306d00df06c2a64d1abc326d5e4288aa7d7c79'},
                              'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                              'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                              'reference_and_constants_ast': 'f2a1ecc5b5d02b243528ea7d57d1282fb4e02e691040ac3148afe1a6d550566e'},
 'triton_kda_dot_kkt_intra': {'files': {'source/triton_kda_dot_kkt_intra.py': 'd4aec65d7a8577f40268b98e6ba168f1b0489ab49500a99870e2ff37a249277a',
                                        'workloads.json': '6a034ed061c7ed8cf2dcd057fb31597a3ed7a5d557e21a7b9e97c129524111de'},
                              'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                              'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                              'reference_and_constants_ast': 'e2ce91d5af6a3cbdf2452c5f30ee115b8f85876aa9c2b9ad5c4e2bac980b1860'},
 'triton_kda_gate': {'files': {'source/triton_kda_gate.py': '59be7ebf3b2762cb028a97821abf0aefc15d23a97c96c66ee72996e0340aab78',
                               'workloads.json': 'eb5799050238e47fd15cacdc92c24be626eab987bf9bc0e05cc37bffc0cd9e15'},
                     'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                     'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                     'reference_and_constants_ast': 'a22716f61770f9e5e43dc623934d52928e6548e8fb3308383b133f54962aa932'},
 'triton_kda_gla_fwd_o': {'files': {'source/triton_kda_gla_fwd_o.py': 'c6643a478127fccbebd11f30e51915111043a72d5331123c858b283231afa9a8',
                                    'workloads.json': 'd2a697ffc63411c5df5d61ad03d539faf1220fd468e25e74a7c19c8be2cc95cb'},
                          'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                          'generation_ast': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                          'reference_and_constants_ast': 'e590adef6e8d05c5b02038a0de23e69486605f1de5bd959d7a810f0f9be9a71a'},
 'triton_ssd_bmm': {'files': {'source/triton_ssd_bmm.py': '3c34add862bf6abb13c4ed3f53fef6ddf618d9b520d648213eb21d1432509554',
                              'workloads.json': '5a81adb73e7f1562ebdf96652ea026d0f46417d0e0d952cfcee92644041e9e09'},
                    'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a',
                    'generation_ast': '960935b3bbc3fe4cd6bd3eca31036f5f73518b3d0c48daa84704caac8d13688b',
                    'reference_and_constants_ast': 'a5438528a71db68adb74c76e310f56f87c0a674a123f04a19ec7b4a80a059df3'}}
NAMES = ('triton_ssd_bmm', 'triton_kda_gate', 'triton_fla_scaled_dot_kkt', 'triton_kda_dot_kkt_inter', 'triton_kda_dot_kkt_intra', 'triton_fla_chunk_fwd_o', 'triton_kda_gla_fwd_o')

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


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_protocol_manifest_controls_and_failed_performance(task,monkeypatch):
    root,h,_,_=task
    spec=importlib.util.spec_from_file_location('_fla_eval_test',root/'_arena_eval.py')
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


def test_original_contract_preserved(task):
    root,h,_,_=task
    saved=BASELINE[root.name]
    for relative,digest in saved['files'].items():
        assert _digest((root/relative).read_text())==digest
    text=(root/'scripts/task_runner.py').read_text();tree=ast.parse(text)
    def protected(tree):
        return [ast.dump(n,include_attributes=False) for n in tree.body if
            isinstance(n,ast.FunctionDef) and (n.name.startswith('reference') or n.name=='gen_inputs') or
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            ('TEST_SHAPES','SEEDS','WARMUP_ITERATIONS','BENCHMARK_ITERATIONS','PERF_SEED_IDX') for t in n.targets)]
    assert _digest(json.dumps(protected(tree)))==saved['reference_and_constants_ast']
    def generation(tree):
        return sorted((f.name,ast.dump(n,include_attributes=False)) for f in tree.body
            if isinstance(f,ast.FunctionDef) and f.name in ('run_correctness','run_performance')
            for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ('manual_seed','randn','rand','arange','cumsum'))
    assert _digest(json.dumps(generation(tree)))==saved['generation_ast']
    assert _digest(text.split('# >>> AKA-GENERATED:')[1].split('# <<< AKA-GENERATED <<<')[0])==saved['generated_region']
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_benchmark_cuda_graph_or_events')
    assert {k.arg:ast.unparse(k.value) for k in call.keywords}=={
        'warmup':'WARMUP_ITERATIONS','repetition':'BENCHMARK_ITERATIONS','timed_run':'timed'}


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
            return value
        monkeypatch.setattr(torch,name,factory)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    if root.name=='triton_ssd_bmm':
        monkeypatch.setattr(h,'TEST_SHAPES',[(8,2,16,4,False)])
        op=h.reference_bmm;name='bmm_chunk_fwd'
    else:
        monkeypatch.setattr(h,'SEEDS',[42])
        op=h.reference
        name={'triton_kda_gate':'fused_kda_gate','triton_fla_scaled_dot_kkt':'chunk_scaled_dot_kkt_fwd',
              'triton_kda_dot_kkt_inter':'kda_dot_kkt_inter','triton_kda_dot_kkt_intra':'kda_dot_kkt_intra',
              'triton_fla_chunk_fwd_o':'chunk_fwd_o','triton_kda_gla_fwd_o':'kda_gla_fwd_o'}[root.name]
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
            items=[c.cell_contents for c in fn.__closure__]
            items=[v for item in items for v in (item if isinstance(item,tuple) else (item,))]
            target=next(v for v in items if isinstance(v,torch.Tensor) and v.is_floating_point() and all(v is not o for o in values))
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

@pytest.mark.parametrize('name,missing', [('triton_kda_gate','g_bias'),('triton_fla_scaled_dot_kkt','g')])
def test_optional_gate_controls_reject_omitted_dependency(name,missing,monkeypatch):
    root=ROOT/'tasks/triton2triton/vllm'/name
    for module in list(sys.modules):
        if module=='scripts' or module.startswith('scripts.'):monkeypatch.delitem(sys.modules,module)
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('_negative_control',root/'scripts/task_runner.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    controls=h.semantic_controls
    c=list(controls.control_cases('cpu'))[-1]
    kwargs=dict(c['kwargs']);kwargs[missing]=None
    wrong=h.reference(**kwargs)
    from scripts.contract_checks import check_outputs,NumericalMismatch
    with pytest.raises(NumericalMismatch):
        check_outputs(wrong,c['expected'],atol=c['atol'],rtol=c['rtol'])

@pytest.mark.parametrize('task',['triton_kda_dot_kkt_inter','triton_kda_dot_kkt_intra'],indirect=True)
def test_real_correctness_rejects_missing_kkt_output(task,monkeypatch):
    _,h,_,_=task
    args,kwargs=h.gen_inputs(42,'cpu')
    monkeypatch.setattr(h,'gen_inputs',lambda *a:(args,kwargs))
    name='kda_dot_kkt_inter' if 'inter' in h.SOURCE_FILE else 'kda_dot_kkt_intra'
    monkeypatch.setattr(h,'load_module',lambda:types.SimpleNamespace(**{name:lambda *a,**kw:()}))
    # Missing outputs must fail before any device conversion is necessary.
    ok,error=h.run_correctness(case_index=0)
    assert not ok and error


@pytest.mark.parametrize('task',['triton_kda_dot_kkt_inter','triton_kda_dot_kkt_intra',
                                 'triton_fla_chunk_fwd_o','triton_kda_gla_fwd_o'],indirect=True)
def test_structured_control_rejects_omitted_gate_or_state(task):
    _,h,controls,checks=task
    c=list(controls.control_cases('cpu'))[-1];kw=dict(c['kwargs'])
    if 'gk' in kw:kw['gk']=torch.zeros_like(kw['gk'])
    else:kw['h']=torch.zeros_like(kw['h'])
    wrong=h.reference(**kw)
    with pytest.raises(checks.NumericalMismatch):
        checks.check_outputs(wrong,c['expected'],atol=c['atol'],rtol=c['rtol'])

@pytest.mark.parametrize('task',['triton_ssd_bmm'],indirect=True)
def test_bmm_causal_flag_preserves_original_full_matrix_gate(task):
    _,h,controls,checks=task
    c=next(c for c in controls.control_cases('cpu') if c['kwargs']['causal'])
    wrong=torch.tril(c['expected'])
    assert torch.count_nonzero(c['expected']-wrong)>0
    with pytest.raises(checks.NumericalMismatch):
        checks.check_outputs(wrong,c['expected'],atol=c['atol'],rtol=c['rtol'])
