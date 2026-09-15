"""CPU contract regressions; real Triton/GPU qualification is separate evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import hashlib
from pr107_integration_helpers import original_manifest
import sys
import types

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Preserved contract fingerprints from parent 66b18724; no Git history is needed.
BASELINE = {'triton_ssd_chunk_cumsum': {'files': {'workloads.json': 'ad32e8ad77a4063b2140b9d723b3cae37b40a3d8c30785d96314b51010cce418',
                                       'source/triton_ssd_chunk_cumsum.py': '61231def1708392a863c28963f8fc6fd0e192d7d3c5b8e7398d6e537626ad741'},
                             'config_except_editable': '4da3c58d2cff25ab5ca596aa294e3c4f13fba5ab7fb735c880bf4c7c3e2c5b8c',
                             'reference_and_constants_ast': '991f24124d7455334ce6800c918f2e2f7e38d14e06228acd57dc06cc34a8ad35',
                             'generation_ast': '785e10e6c5e63929ae56b935e2f3446161b752341db503c0fbc32147f8910570',
                             'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'triton_ssd_chunk_scan': {'files': {'workloads.json': 'a364d5de4f5a21ea976f55d89ba61bee5b22cd6f754af7c62b2306249f2f4fb5',
                                     'source/triton_ssd_chunk_scan.py': '6530336992c466bf4e76810d159a6efd51c05751fa3b577644b20febace5885a'},
                           'config_except_editable': 'fd83c7d977b63066857bc3245ff49e243549d1ecd89df04977d94b6eeea8bfad',
                           'reference_and_constants_ast': '4dba5db02a4cee5b57da72e944186995eda20ea4fec7f8face1135b59ae0f812',
                           'generation_ast': 'de3391ed91d33da5a8055c539d9906c6c25c31e80cee95845ae5c5e4525d7296',
                           'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'triton_ssd_chunk_state': {'files': {'workloads.json': '92785807450f9dc163b88ff7d4b57e7daeeafc7bfec321cd84d6223ebce154f9',
                                      'source/triton_ssd_chunk_state.py': '1653dd642efba27293c3315e201584ad62a5fc253cbdb282cc414376f14d6ffc'},
                            'config_except_editable': '9dd95eab05545c104659dce85d7a92c5520cb3adfd5d98766fb50b07d6de437d',
                            'reference_and_constants_ast': '9d5a0a2b2e5fd91f425058beadf088bf4d6f9e65a1a59b6a8812014bf62ca440',
                            'generation_ast': 'c5f7d8e8644b32cb97757af790146cd46b4d965a7d76c0ec8431161d9ba477c4',
                            'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'triton_ssd_chunk_state_varlen': {'files': {'workloads.json': 'bc0ea9ad9d41022c64e02e790f6ee928d91481b2f708e6cc0e50b2aa3e717320',
                                             'source/triton_ssd_chunk_state_varlen.py': '2550d9117263494144050f74b200cea07568f9564daec753022c29ca60d6f954'},
                                   'config_except_editable': '7826598c69b4d125c2137f49de88b25f684007565cf250303c21dd45e7aea1e1',
                                   'reference_and_constants_ast': '302fb4c4f3680aec791c1d38627a76b3140f22dd667cc8a7a3297a35fc5a2fd7',
                                   'generation_ast': 'a54eb7af435b8ca776874c74e40713aed6abf814f114f394625711cf920cf861',
                                   'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'triton_ssd_state_passing': {'files': {'workloads.json': '93788e8fe4150acddbbd6014a348568aa24a212afd62b90b6677416885769460',
                                        'source/triton_ssd_state_passing.py': 'af5eaac0616a9b0d7ec9a7bd1aed9ec728f27c24b9d75ba498c4a727a2611257'},
                              'config_except_editable': '3417d7c375e4a6f380a2aa8f37cd7aac81ebe071494a054b3cf1cf948c48df0f',
                              'reference_and_constants_ast': '45261791b4930659d5dd471bf87438070183d8706ceb86dd89c9c37d30c7c299',
                              'generation_ast': '9a557362d418b071f9ab74c9ff96b2834486b28e76eb24d890def7a8362791fb',
                              'generated_region': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'}}
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


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_original_workloads_gates_kernels_and_timing_preserved(task):
    root,h,_,_=task
    saved=BASELINE[root.name]
    for relative,digest in saved['files'].items():
        text=(root/relative).read_text()
        if relative=='workloads.json':
            text=json.dumps(original_manifest(json.loads(text)),indent=2)+'\n'
        assert _digest(text)==digest
    config=yaml.safe_load((root/'config.yaml').read_text())
    edit=config['candidate'].pop('editable')[0]
    assert edit['scope']=='symbols' and edit['allow_new_helpers'] is True
    assert edit['path']=='source/'+root.name+'.py'
    symbols=[config['candidate']['entrypoints'][0]['symbol']]
    if root.name=='triton_ssd_chunk_cumsum':symbols.append('softplus')
    assert edit['symbols']==symbols
    assert _digest(json.dumps(config,sort_keys=True))==saved['config_except_editable']
    after=(root/'scripts/task_runner.py').read_text()
    tree=ast.parse(after)
    def protected(tree):
        return [ast.dump(n,include_attributes=False) for n in tree.body if
                isinstance(n,ast.FunctionDef) and (n.name.startswith('reference') or n.name=='ref_softplus') or
                isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                ('TEST_SHAPES','WARMUP_ITERATIONS','BENCHMARK_ITERATIONS') for t in n.targets)]
    assert _digest(json.dumps(protected(tree)))==saved['reference_and_constants_ast']
    marker='# >>> AKA-GENERATED:';end='# <<< AKA-GENERATED <<<'
    assert _digest(after.split(marker)[1].split(end)[0])==saved['generated_region']
    def generation(tree):
        found=[]
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in ('run_correctness','run_performance'):
                for n in ast.walk(fn):
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('manual_seed','randn','rand','arange','cumsum'):
                        found.append((fn.name,ast.dump(n,include_attributes=False)))
        return sorted(found)
    assert _digest(json.dumps(generation(tree)))==saved['generation_ast']
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_benchmark_cuda_graph_or_events']
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
    assert len(original_manifest(m.load_manifest())['cases'])==5
    assert len(report['cases'])==len(m.load_manifest()['cases'])
    from src.task_spec import load_task_spec
    from src.task_protocol import parse_command_result,CaseManifest
    spec=load_task_spec(root/'config.yaml',task_id='triton2triton/vllm/'+root.name)
    assert spec.candidate.initial_state=='implemented'
    result=parse_command_result('ARENA_EVAL_RESULT='+json.dumps(report),role='task',action='validate-task',returncode=0)
    manifest=CaseManifest.from_result(result)
    monkeypatch.setattr(h,'run_performance',lambda:[{'test_case_id':r['test_case_id'],'execution_time_ms':-1.,'error':'replay rejected'} for r in report['cases'] if 'performance' in r['checks']])
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
