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
    assert sum('performance' in row['checks'] for row in manifest['cases']) == 5
    assert all('correctness' in row['checks'] for row in manifest['cases'])
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


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'device', 'nonfinite', 'values'])
def test_rms_output_contract_rejects_invalid_outputs(monkeypatch, invalid):
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    expected = harness.reference_rms_norm(x, weight)
    checks.check_output(expected, x, weight, 1e-6, harness.reference_rms_norm)
    output = {'shape': expected[:1], 'dtype': expected.double(),
              'device': torch.empty_like(expected, device='meta'),
              'nonfinite': torch.full_like(expected, float('inf')),
              'values': torch.zeros_like(expected)}[invalid]
    with pytest.raises(AssertionError):
        checks.check_output(output, x, weight, 1e-6, harness.reference_rms_norm)


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_rms_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    """CPU simulation of captured buffers; this does not claim GPU execution."""
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    eps = 1e-6
    mod = SimpleNamespace(rms_norm=harness.reference_rms_norm)
    original = mod.rms_norm
    seen = []

    def fn():
        # Match the original harness, including its discarded return value.
        mod.rms_norm(x, weight, eps=eps)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(harness.reference_rms_norm(x, weight, eps))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.rms_norm is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


def test_rms_adapter_installs_output_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_rms_norm/_arena_eval.py', monkeypatch)
    harness = adapter.load_harness()
    assert harness.run_correctness.__module__ == '_rms_output_checks'
    assert harness.run_performance.__module__ == '_rms_output_checks'


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_geak_gemm_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    x = torch.tensor([[1., 2.], [3., 4.]])
    w = torch.tensor([[2., 1.], [4., 3.]])
    bias = torch.tensor([1., 2.])
    expected = torch.tensor([[5., 12.], [11., 26.]])
    checks.check_output(expected, x, w, bias)
    metadata = {'benchmark_method': 'cuda_graph'}
    seen = []

    def fn():
        return torch.nn.functional.linear(x, w, bias)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(torch.nn.functional.linear(x, w, bias))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.5, metadata

    if mode == 'correct':
        ms, returned = checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and returned is metadata
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert 'perturbed_input_replay_checked' not in metadata
    assert seen == [{'warmup': 50, 'repetition': 200}]


@pytest.mark.parametrize('value', [1., float('inf'), float('nan')])
def test_geak_gemm_correctness_rejects_nonfinite_output(monkeypatch, value):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    original = lambda: torch.tensor([[value]])
    harness = SimpleNamespace(gemm_a16w16=original)
    visited = []

    def run(indices):
        visited.extend(indices)
        # Even a harness comparing matching infinities must not bypass the gate.
        torch.testing.assert_close(harness.gemm_a16w16(), original(), equal_nan=True)

    harness.run_correctness = run
    if value == 1.:
        assert checks.checked_correctness(harness, [0, 1, 2]) is None
    else:
        with pytest.raises(AssertionError, match='finite tensor'):
            checks.checked_correctness(harness, [0, 1, 2])
    assert visited == [0, 1, 2] and harness.gemm_a16w16 is original


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_awq_packed_known_answers_and_exact_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_awq_dequantize'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    qweight = torch.tensor([[0x76543210], [-19088744]], dtype=torch.int32)  # second word: 0xfedcba98
    scales = torch.arange(1, 9, dtype=torch.float16).repeat(2, 1)
    zeros = torch.tensor([[0x11111111], [-2004318072]], dtype=torch.int32)  # 0x88888888
    expected = torch.tensor([[-1, 6, 0, 16, 5, 30, 14, 48],
                             [0, 8, 3, 20, 10, 36, 21, 56]], dtype=torch.float16)
    checks.check_output(expected, qweight, scales, zeros, harness.reference_awq_dequantize)
    with pytest.raises(AssertionError):
        checks.check_output(expected.flip(-1), qweight, scales, zeros, harness.reference_awq_dequantize)
    original = lambda q, s, z: harness.reference_awq_dequantize(q, s, z, 1)
    mod = SimpleNamespace(awq_dequantize_triton=original)
    seen = []

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed': output.zero_()

        def replay():
            if mode == 'correct': output.copy_(original(qweight, scales, zeros))
            elif mode == 'stale': output.copy_(cached)
            elif mode == 'changing_wrong': output.fill_(int(qweight[0, 0]) & 15)
            return output

        timed_run.outputs, timed_run.rerun = output, replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.awq_dequantize_triton is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'wrong_ids', 'wrong_weights'])
def test_shared_expert_append_known_answers_and_both_timed_outputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/geak_eval/L1/fused_append_shared_experts'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    node = next(n for n in ast.parse((task/'test_kernel_harness.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'reference_fused_append')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'protected-reference', 'exec'), namespace)
    reference = namespace['reference_fused_append']
    topk_ids = torch.tensor([[3, 7], [1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.75]])
    cfg = {'S': 1, 'N': 8, 'scale_factor': 1.0}
    expected = (torch.tensor([[3, 7, 8], [1, 0, 8]], dtype=torch.int32),
                torch.tensor([[0.5, 0.25, 1.], [0.125, 0.75, 1.]]))
    checks.check_output(expected, topk_ids, topk_weights, cfg, reference)
    original = lambda ids, weights, s, factor, N: reference(ids, weights, s, factor, N)
    harness = SimpleNamespace(reference_fused_append=reference, fused_append_shared_experts=original)
    seen = []

    def fn():
        harness.fused_append_shared_experts(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], N=cfg['N'])

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        outputs = measured()
        cached = tuple(out.clone() for out in outputs)
        if mode == 'incorrect_timed': outputs[0].zero_()

        def replay():
            if mode in ('correct', 'wrong_ids', 'wrong_weights'):
                expected = reference(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], cfg['N'])
                for out, ref in zip(outputs, expected): out.copy_(ref)
                if mode == 'wrong_ids': outputs[0][:, -1].zero_()
                if mode == 'wrong_weights': outputs[1][:, -1].zero_()
            elif mode == 'stale':
                for out, ref in zip(outputs, cached): out.copy_(ref)
            return outputs

        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.5, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
    assert harness.fused_append_shared_experts is original
    assert seen == [{'warmup': 50, 'repetition': 200}]


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


GEAK = sorted((ROOT/'tasks/triton2triton/geak_eval').rglob('config.yaml'))


@pytest.mark.parametrize('path', GEAK, ids=lambda p:p.parent.name)
def test_geak_v2_preserves_original_functions_and_freezes_the_complete_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert spec.candidate.language=='triton'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=(task/data['source']).relative_to(ROOT).as_posix()
    original=subprocess.check_output(['git','show',f'{BASE}:{source}'],cwd=ROOT)
    assert (ROOT/source).read_bytes()==original
    before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/test_kernel_harness.py'],cwd=ROOT,text=True)
    after=(task/'test_kernel_harness.py').read_text()
    assert data['migration']['original_harness_sha256']==hashlib.sha256(before.encode()).hexdigest()
    assert data['migration']['original_source_sha256']==hashlib.sha256(original).hexdigest()
    olddefs={n.name:n for n in ast.parse(before).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newdefs={n.name:n for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    bootstrap={'_find_baseline_kernel_dir','_load_baseline_triton','_resolve_geak_kernel_dir','_register_geak_aliases'}
    for name in olddefs.keys()-bootstrap-({'e8m0_to_f32'} if data['migration'].get('reference_fixes') else set()):
        assert ast.get_source_segment(before,olddefs[name])==ast.get_source_segment(after,newdefs[name]),name
    assert 'os.environ.get("GEAK_WORK_DIR"' not in after
    assert 'os.environ.get("GEAK_REPO_ROOT"' not in after
    perfrows=[r for r in data['cases'] if 'performance' in r['checks']]
    assert [r['params']['configuration'] for r in perfrows]==data['input_tables']['performance']
    assert [r['params']['case_index'] for r in perfrows]==list(range(len(perfrows)))
    configs=[r['params']['configuration'] for r in data['cases']]
    assert all(v in configs for v in data['input_tables']['original_correctness'])
    assert all('correctness' in r['checks'] for r in data['cases'])
    result_record({'protocol':'arena-eval-v1','role':'task','action':'validate-task','status':'PASS','cases':data['cases']})
    for role in ('baseline','candidate'):
        # Real command smoke: AST compilation only, no JIT/GPU execution.
        run=subprocess.run([__import__('sys').executable,'_arena_eval.py',role,'compile'],cwd=task,text=True,capture_output=True)
        parsed=parse_command_result(run.stdout,role=role,action='compile',returncode=run.returncode)
        assert parsed.status=='PASS',run.stdout+run.stderr


def test_geak_boolean_integer_and_skipped_result_contracts(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    adapter.require_success(None,'none',1)
    adapter.require_success(0,'zero',1)
    adapter.require_success(True,'bool',1)
    for value,kind in [(None,'bool'),(False,'zero'),(True,'zero'),(0,'bool'),
                       ({'correct':True,'num_correct':0,'num_failed':0,'skipped':True},'dict'),
                       ({'correct':True,'num_correct':1,'num_failed':0},'dict')]:
        with pytest.raises(RuntimeError):adapter.require_success(value,kind,2)


def test_geak_performance_uses_current_candidate_timing_and_rejects_missing_calls(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    measurements=iter([17.,3.,19.,5.])
    h=SimpleNamespace(benchmark_cuda_graph_or_events=lambda *a,**k:(next(measurements),{'benchmark_method':'cuda_graph'}))
    actions=SimpleNamespace(h=h,performance=lambda:[h.benchmark_cuda_graph_or_events(None) for _ in range(4)])
    data={'cases':[{'checks':['correctness','performance']},{'checks':['correctness','performance']}],
          'timing_calls_per_case':['reference','candidate']}
    assert [ms for ms,meta in adapter.capture_performance(actions,data)]==[3.,5.]
    # A stale saved report has no role in supplying the missing current call.
    h.benchmark_cuda_graph_or_events=lambda *a,**k:(2.,{'benchmark_method':'cuda_graph'})
    actions.performance=lambda:h.benchmark_cuda_graph_or_events(None)
    with pytest.raises(RuntimeError,match='Missing/extra'):adapter.capture_performance(actions,data)


def test_geak_failure_envelope_and_manifest_drift(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((GEAK[0].parent/'workloads.json').read_text())
    actions=SimpleNamespace(inputs=lambda:data['input_tables'],validate=lambda:None,
                            correctness=lambda require:require(False,'bool',len(data['cases'])))
    monkeypatch.setattr(adapter,'load_actions',lambda:actions)
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='FAIL' and all(row['status']=='FAIL' for row in result['cases'])
    assert result['failure_kind']!='numerical_mismatch'
    result_record(result)
    actions.inputs=lambda:{'performance':[],'original_correctness':[]}
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']


def pure_functions(path, names, namespace=None):
    """Load protected CPU reference functions without importing GPU dependencies."""
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    scope={'torch':torch,**(namespace or {})}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return SimpleNamespace(**scope)


def test_geak_rope_references_have_independent_known_answers():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/fused_qk_rope_cache_mla/test_kernel_harness.py'
    from enum import IntEnum
    ref=pure_functions(path,['RotateStyle','rotate_half_neox','rotate_half_gptj','ref_rope_sbhd_fwd'],{'IntEnum':IntEnum})
    x=torch.tensor([[1.,2.,3.,4.]])
    torch.testing.assert_close(ref.rotate_half_neox(x),torch.tensor([[-3.,-4.,1.,2.]]))
    torch.testing.assert_close(ref.rotate_half_gptj(x),torch.tensor([[-2.,1.,-4.,3.]]))
    got=ref.ref_rope_sbhd_fwd(x,torch.full((1,4),torch.pi/2),ref.RotateStyle.GPTJ,False,False)
    expected=torch.tensor([[-2.,1.,-4.,3.]])
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,x)


def test_geak_fp4_reference_decodes_independent_known_values():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/test_kernel_harness.py'
    # Only the lookup-table allocation is redirected to CPU; arithmetic is unchanged.
    cpu_torch=SimpleNamespace(float32=torch.float32,
        tensor=lambda values,**kwargs:torch.tensor(values,**{**kwargs,'device':'cpu'}))
    ref=pure_functions(path,['mxfp4_to_f32'],{'torch':cpu_torch})
    # Low/high nibbles enumerate the E2M1 positive and negative values.
    packed=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]],dtype=torch.uint8)
    expected=torch.tensor([[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]])
    got=ref.mxfp4_to_f32(packed)
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,torch.zeros_like(expected))


@pytest.mark.parametrize('name',['gemm_a16wfp4','fused_mxfp4_quant_moe_sort'])
def test_e8m0_reference_all_encodings_and_negative_control(name):
    path=ROOT/'tasks/triton2triton/geak_eval/L3'/name/'test_kernel_harness.py'
    ref=pure_functions(path,['e8m0_to_f32'])
    codes=torch.arange(256,dtype=torch.uint8)
    got=ref.e8m0_to_f32(codes)
    expected=codes.view(torch.float8_e8m0fnu).float()
    torch.testing.assert_close(got,expected,atol=0,rtol=0,equal_nan=True)
    assert got[126]==0.5 and got[134]==128 and torch.isnan(got[255])
    before=subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name=='e8m0_to_f32')
    scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(scope['e8m0_to_f32'](codes),expected,atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize('task',[VLLM[0].parent,ROCM[0].parent,GEAK[0].parent],ids=['vllm','rocm','geak'])
def test_bad_diagnostic_evidence_still_emits_failure(task,monkeypatch,capsys):
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    code=adapter.emit_result({'protocol':'arena-eval-v1','role':'candidate','action':'performance',
                             'status':'PASS','cases':[],'metadata':{'bad':float('nan')}})
    parsed=parse_command_result(capsys.readouterr().out,role='candidate',action='performance',returncode=code)
    assert parsed.status=='FAIL' and parsed.failure_kind=='invalid_evidence'


def test_vllm_additional_original_correctness_cases_are_manifested():
    for name in ['triton_bad_words','triton_logit_bias']:
        task=ROOT/'tasks/triton2triton/vllm'/name
        data=json.loads((task/'workloads.json').read_text())
        extra=[r for r in data['cases'] if r['checks']==['correctness']]
        before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/scripts/task_runner.py'],cwd=ROOT,text=True)
        tree=ast.parse(before)
        if name=='triton_bad_words':
            assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='multi_cases' for t in n.targets))
            assert [r['shape'] for r in extra]==[list(v) for v in ast.literal_eval(assignment.value)]
            assert [r['params']['seed'] for r in extra]==[1234,1235,1236]
        else:
            assert len(extra)==1 and extra[0]['shape']==[2,256,16] and extra[0]['params']['seed']==777
        assert all(r['params']['case_index']==-1 for r in extra)


def test_all_rocm_manifests_match_original_cpu_collected_pytest_parameters():
    """Import decorators with inert GPU stubs; execute no kernels or test bodies."""
    import sys
    fixture=ROOT/'tests/fixtures/triton_migration/collect_rocm_cpu.py'
    run=subprocess.run([sys.executable,str(fixture)],cwd=ROOT,text=True,capture_output=True,check=True)
    collected=json.loads(run.stdout)
    assert len(collected)==61
    for task,functions in collected.items():
        expected={}
        for fn in functions:
            for params in fn['cases']:
                content=json.dumps(params,sort_keys=True,separators=(',',':'))
                case=fn['function']+'/'+hashlib.sha256(content.encode()).hexdigest()[:20]
                expected[case]={'function':fn['function'],'arguments':params}
        data=json.loads((ROOT/task/'workloads.json').read_text())
        assert {r['test_case_id']:r['params'] for r in data['cases']}==expected,task
        assert 'built-in method' not in json.dumps(data)


DECORATOR_HELPERS={
    'gemm':{'leaky_relu'},
    'layernorm':{'get_autotune_config'},
    'rmsnorm_fwd':{'get_autotune_config'},
    'softmax':{'get_autotune_config'},
    'multreduce_matmul_dot_kernel':{'get_triton_dot_autotune_configs','get_triton_autotune_key','get_triton_heuristics','triton_matmul_kernel'},
    'triton_multreduce_matmul_kernel':{'get_triton_multreduce_autotune_configs','get_triton_autotune_key','get_triton_heuristics','triton_matmul_kernel'},
}


@pytest.mark.parametrize('path',[p for p in ROCM if p.parent.name in DECORATOR_HELPERS],ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_implementation_helpers_editable_without_exposing_tests_or_references(path,tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
    workspace=tmp_path/'task'
    shutil.copytree(path.parent,workspace,ignore=shutil.ignore_patterns('__pycache__'))
    spec=load_task_spec(workspace/'config.yaml',task_id=path.parent.name)
    helper_names=DECORATOR_HELPERS[path.parent.name]
    entry_names={e.symbol for e in spec.candidate.entrypoints}
    assert not (helper_names & entry_names), 'Implementation helpers must not become required entrypoints'
    assert helper_names <= set(spec.candidate.editable[0].symbols)
    assert set(spec.candidate.editable[0].symbols)==entry_names|helper_names
    snapshot=snapshot_workspace_harness(workspace)
    source=workspace/spec.candidate.editable[0].path
    text=source.read_text();tree=ast.parse(text);lines=text.splitlines(keepends=True)
    insertions=[n.body[0].lineno-1 for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helper_names]
    for index in sorted(insertions,reverse=True):lines.insert(index,'    _arena_helper_boundary_probe = 1\n')
    source.write_text(''.join(lines))
    verify_workspace_harness(snapshot)
    # Ordinary test/reference code remains protected in that same file.
    tree=ast.parse(source.read_text())
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    lines=source.read_text().splitlines(keepends=True)
    lines.insert(test.body[0].lineno-1,'    _arena_forbidden_test_change = 1\n')
    source.write_text(''.join(lines))
    with pytest.raises(RuntimeError):verify_workspace_harness(snapshot)
