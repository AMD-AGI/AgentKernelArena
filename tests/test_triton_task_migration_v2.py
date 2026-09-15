"""CPU contract regressions; synthetic measurements below are not GPU validation."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
VLLM = sorted((ROOT/'tasks/triton2triton/vllm').glob('*/config.yaml'))
BASE = '5c9f8ef2'


def module_at(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    spec = importlib.util.spec_from_file_location('_isolated_task_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_record(result):
    return parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result), role=result['role'],
                                action=result['action'], returncode=0 if result['status']=='PASS' else 1)


@pytest.mark.parametrize('path', VLLM, ids=lambda p:p.parent.name)
def test_vllm_v2_preserves_all_original_cases_checks_sources_and_helpers(path):
    task = path.parent
    spec = load_task_spec(path, task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state == 'implemented'
    assert spec.candidate.language == 'triton' and spec.baseline.kind == 'initial_candidate'
    relative = (task/'scripts/task_runner.py').relative_to(ROOT).as_posix()
    before = subprocess.check_output(['git','show',f'{BASE}:{relative}'], cwd=ROOT, text=True)
    after = (ROOT/relative).read_text()
    bt, at = ast.parse(before), ast.parse(after)
    bf = {n.name:n for n in bt.body if isinstance(n,ast.FunctionDef)}
    af = {n.name:n for n in at.body if isinstance(n,ast.FunctionDef)}
    correction = af['run_correctness']
    correction.args = deepcopy(bf['run_correctness'].args)
    loop = next(n for n in ast.walk(correction) if isinstance(n,ast.For))
    assert ast.unparse(loop.body[0].test).startswith('case_index is not None')
    loop.body.pop(0)
    assert ast.dump(correction, include_attributes=False) == ast.dump(bf['run_correctness'], include_attributes=False)
    for name in bf.keys()-{'run_correctness'}:
        assert ast.get_source_segment(before,bf[name]) == ast.get_source_segment(after,af[name])
    manifest = json.loads((task/'workloads.json').read_text())
    assert len(manifest['cases']) == 5
    assert all(row['checks']==['correctness','performance'] for row in manifest['cases'])
    assert manifest['migration']['original_harness_sha256'] == hashlib.sha256(before.encode()).hexdigest()
    for edit in spec.candidate.editable:
        source = task/edit.path
        original = subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
        assert source.read_bytes() == original
    assert 'Evaluation contract' in (task/'README.md').read_text()


def test_vllm_per_case_failure_and_incomplete_measurement_rejected(monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py', monkeypatch)
    manifest = adapter.load_manifest()
    harness = SimpleNamespace(**{manifest['case_table']:manifest['input_table']})
    harness.run_correctness = lambda case_index: (case_index != 3, 'negative control')
    harness.run_performance = lambda: []
    monkeypatch.setattr(adapter,'load_harness',lambda:harness)
    correctness = adapter.evaluate('candidate','correctness')
    assert correctness['status']=='FAIL'
    assert [r['status'] for r in correctness['cases']]==['PASS','PASS','PASS','FAIL','PASS']
    manifest_result = {'protocol':'arena-eval-v1','role':'task','action':'validate-task',
                       'status':'PASS','cases':manifest['cases']}
    CaseManifest.from_result(result_record(manifest_result)).validate(result_record(correctness))
    perf = adapter.evaluate('candidate','performance')
    assert perf['status']=='FAIL' and all(r['status']=='FAIL' for r in perf['cases'])
    assert perf['failure_kind'] != 'numerical_mismatch'


def test_vllm_candidate_stub_is_never_baseline_fallback(tmp_path,monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py',monkeypatch)
    data = adapter.load_manifest()
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    for source, targets in data['candidate_symbols'].items():
        p=tmp_path/source;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join('@triton.jit\ndef '+t['name']+'():\n    pass\n' for t in targets))
    assert adapter.inspect_candidate(data)=='unimplemented'
    with pytest.raises(ValueError,match='no baseline fallback'):
        adapter.inspect_candidate(data,require_implemented=True)


def test_rms_reference_has_independent_known_answers(monkeypatch):
    runner = ROOT/'tasks/triton2triton/vllm/triton_rms_norm/scripts/task_runner.py'
    harness = module_at(runner,monkeypatch)
    x = torch.tensor([[3.,4.],[0.,0.]])
    w = torch.tensor([2.,0.5])
    expected = torch.tensor([[6./(12.5+1e-6)**0.5, 2./(12.5+1e-6)**0.5],[0.,0.]])
    result = harness.reference_rms_norm(x,w)
    torch.testing.assert_close(result,expected)
    assert not torch.allclose(result,torch.ones_like(result),atol=1e-2,rtol=1e-2)


ROCM = sorted([*(ROOT/'tasks/triton2triton/rocmbench').rglob('config.yaml'),
               *(ROOT/'tasks/instruction2triton').rglob('config.yaml')])


@pytest.mark.parametrize('path', ROCM, ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_v2_preserves_original_source_and_complete_parameter_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=task/data['source']
    original=subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    assert source.read_bytes()==original
    assert hashlib.sha256(original).hexdigest()==data['migration']['original_source_sha256']
    rows=data['cases']
    assert len(rows)==len({row['test_case_id'] for row in rows})
    assert all('correctness' in row['checks'] for row in rows)
    assert any(row['checks']==['correctness','performance'] for row in rows)
    assert {entry.symbol for entry in spec.candidate.entrypoints}==set(data['kernel_symbols'])
    assert 'task_type' not in spec.to_mapping()
    ast.parse((task/'_arena_eval.py').read_text())
    ast.parse((task/'_arena_reference.py').read_text())


def oracle(name,monkeypatch):
    return module_at(ROOT/'tasks/instruction2triton/rocmbench'/name/'_arena_reference.py',monkeypatch)


def test_philox_known_counter_and_uniform_negative_control(monkeypatch):
    reference=oracle('test_randn',monkeypatch)
    # Philox4x32-10 zero counter/key known answer (first lane).
    assert reference.philox32(0,1).tolist()==[0x6627e8d5]
    output=torch.zeros(16)
    check=reference.prepare({'seed_val':0,'N_elements':16,'x_output_buffer':output},None)
    with pytest.raises(reference.NumericalMismatch):check(output)


def test_swizzle_matches_original_explicit_known_order(monkeypatch):
    reference=oracle('test_triton_swizzle2d',monkeypatch)
    expected=reference.swizzle_reference(5,7,3,dtype=torch.int32,device='cpu')
    task=ROOT/'tasks/instruction2triton/rocmbench/test_triton_swizzle2d/test_triton_swizzle2d.py'
    tree=ast.parse(task.read_text());test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_swizzle2d')
    assignment=next(n for n in ast.walk(test) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_order' for t in n.targets))
    original_order=ast.literal_eval(assignment.value.func.value.args[0])
    assert expected.tolist()==original_order
    with pytest.raises(reference.NumericalMismatch):reference.compare(expected,torch.arange(35).reshape(5,7),exact=True,check_dtype=False)


def test_rms_backward_reference_matches_autograd_and_rejects_wrong_gradients(monkeypatch):
    reference=oracle('rmsnorm_bwd',monkeypatch)
    x=torch.tensor([[1.,2.,3.],[2.,-1.,4.]],requires_grad=True)
    g=torch.tensor([0.2,0.4,0.8],requires_grad=True)
    go=torch.tensor([[1.,0.,2.],[-1.,2.,0.5]])
    y=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*(g+1)
    y.backward(go)
    dx=x.grad.clone();dg=g.grad.clone()
    context={'x':x.detach(),'g':g.detach(),'grad_output':go,'ZERO_CENTERED_GAMMA':True,'eps':1e-5,
             'dx_bench':dx,'dg_tmp_bench':torch.stack([dg,torch.zeros_like(dg)])}
    check=reference.prepare(context,None);check(None)
    dx.zero_()
    with pytest.raises(reference.NumericalMismatch):check(None)


def test_oracle_distinguishes_output_contract_failure_from_numerical_error(monkeypatch):
    reference=oracle('test_add_kernel',monkeypatch)
    with pytest.raises(ValueError,match='shape/device'):
        reference.compare(torch.zeros(2),torch.zeros(3),atol=1e-3,rtol=1e-2)
    with pytest.raises(ValueError,match='nonfinite'):
        reference.compare(torch.tensor([float('nan')]),torch.zeros(1),atol=1e-3,rtol=1e-2)


def test_timed_wrong_path_rejected_by_rocm_adapter(monkeypatch):
    task=ROOT/'tasks/instruction2triton/rocmbench/test_add_kernel'
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    reference=module_at(task/'_arena_reference.py',monkeypatch)
    import sys
    monkeypatch.setitem(sys.modules,'_arena_reference',reference)
    plugin=SimpleNamespace(action='performance',current_row={'test_case_id':'synthetic-cpu'},exercised=set())
    class Base:
        def __init__(self,op_callable,**kwargs):self.op_callable=op_callable;self.prepare_fn=None
        def run_benchmark(self,*args,**kwargs):
            self.op_callable()
            return {'timing_ms':{'mean':1.0},'benchmark_method':'cuda_graph'}
    Checked=adapter.benchmark_type(Base,plugin,None)
    def scenario():
        x=torch.ones(4);y=torch.ones(4);output=torch.empty_like(x)
        calls=[]
        def op():
            output.copy_(x+y if not calls else torch.zeros_like(x))
            calls.append(1)
        return Checked(op_callable=op)
    benchmark=scenario()
    with pytest.raises(reference.NumericalMismatch):benchmark.run_benchmark()
    assert not plugin.exercised


def test_skip_is_not_passing_protocol_evidence(monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    plugin=adapter.ReportPlugin(data,'correctness')
    key=next(iter(plugin.rows));plugin.node_rows['item']=key
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='setup',failed=False,skipped=True,longrepr='unsupported device'))
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='teardown',failed=False,skipped=False))
    assert plugin.rows[key]['status']=='FAIL' and plugin.rows[key]['failure_kind']=='not_executed'


def test_rocm_real_pytest_lifecycle_and_manifest_drift(tmp_path,monkeypatch):
    """CPU functions exercise real pytest setup/call/teardown and collection hooks."""
    import sys
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1')
    source='''import pytest
from types import SimpleNamespace
triton=SimpleNamespace(jit=lambda f:f)
@triton.jit
def kernel():
    return 1
@pytest.mark.parametrize('x',[1,2])
def test_value(x):
    assert x > 0
'''
    (tmp_path/'test_cpu_migration.py').write_text(source)
    rows=[{'test_case_id':adapter.identity('test_value',{'x':x}),
           'params':{'function':'test_value','arguments':{'x':x}},
           'checks':['correctness'],'status':'PASS'} for x in (1,2)]
    data={'source':'test_cpu_migration.py','kernel_symbols':['kernel'],'cases':rows}
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='PASS',result
    assert all(r['status']=='PASS' and r['metrics']['original_pytest_passed'] for r in result['cases'])
    # A manifest cannot silently lose one original case.
    data['cases'].pop();(tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']
    monkeypatch.delitem(sys.modules,'test_cpu_migration',raising=False)


def test_rocm_missing_candidate_fails_every_applicable_case(tmp_path,monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','performance')
    assert result['status']=='FAIL'
    assert result['cases'] and all(r['status']=='FAIL' and 'performance' in r['checks'] for r in result['cases'])
    result_record(result)
