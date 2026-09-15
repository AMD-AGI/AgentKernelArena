"""Real CPU negative controls for the four public vLLM MoE task contracts."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function(path, name, **namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'torch': torch, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


CONTRACT = load(TASKS/'triton_batched_moe/_contract_checks.py', '_moe_contract_test')
TIMER = load(ROOT/'src/tools/perf/aka_benchmark.py', '_moe_timer_test')


@pytest.mark.parametrize('failure', [None, 'original', 'stale', 'mutate', 'replay_mutate', 'dtype', 'shape', 'nonfinite'])
def test_actual_measured_output_and_replay_fail_closed(failure):
    x = torch.tensor([2., -3.], dtype=torch.float16)
    original = x.clone()
    calls = []
    def reference(saved):
        calls.append('reference')
        return saved['A'] * 4
    def fn():
        if failure == 'mutate':
            x.zero_()
        return x * 4
    def benchmark(call, *, timed_run, **options):
        assert options == {'warmup': 10, 'repetition': 100}
        calls.append('timer')
        actual = call()
        if failure == 'original': actual.zero_()
        if failure == 'dtype': actual = actual.float()
        if failure == 'shape': actual = actual[:1]
        if failure == 'nonfinite': actual.fill_(float('inf'))
        def replay():
            calls.append('replay')
            if failure == 'stale': return original * 4
            if failure == 'replay_mutate': x.zero_()
            actual.copy_(call())
            return actual
        timed_run._bind(replay, actual)
        return 0.25, {'benchmark_method': 'cuda_graph'}
    def run():
        with patch.dict(sys.modules, {'_aka_benchmark': TIMER}):
            return CONTRACT.checked_benchmark(benchmark, fn, inputs={'A': x}, reference=reference,
                check=CONTRACT.compare_output, perturb=CONTRACT.perturb_activation,
                warmup=10, repetition=100)
    if failure is None:
        ms, metadata = run()
        assert ms == 0.25
        assert metadata['benchmark_original_output_checked'] and metadata['benchmark_replay_checked']
        assert calls == ['reference','timer','reference','replay']
    else:
        with pytest.raises(AssertionError): run()
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert calls[0] == 'reference'


def test_eager_input_mutation_cannot_change_oracle():
    x = torch.tensor([4.], dtype=torch.float16)
    def bad():
        x.zero_()
        return x.clone()
    with pytest.raises(AssertionError, match='Read-only input changed'):
        CONTRACT.checked_call(bad, inputs={'A':x}, reference=lambda s:s['A']*2, check=CONTRACT.compare_output)
    assert x.item()==4


def test_batched_known_answer_and_inactive_rows():
    ref = function(TASKS/'triton_batched_moe/scripts/task_runner.py', 'reference')
    inputs = {'A':torch.tensor([[[1.,2.],[3.,4.]], [[7.,8.],[9.,10.]]], dtype=torch.float16),
              'B':torch.tensor([[[5.,6.],[7.,8.]], [[1.,1.],[2.,2.]]], dtype=torch.float16),
              'counts':torch.tensor([1,0], dtype=torch.int32)}
    expected = torch.tensor([[[17.,23.],[0.,0.]],[[0.,0.],[0.,0.]]], dtype=torch.float16)
    torch.testing.assert_close(ref(inputs),expected,atol=0,rtol=0)
    for bad in [torch.zeros_like(expected),expected.flip(0),expected+1]:
        with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(bad,expected)


def test_mmk_independent_known_answer():
    ref = function(TASKS/'triton_moe_mmk/scripts/task_runner.py', 'reference')
    inputs={'A':torch.tensor([[1.,2.],[-1.,3.]], dtype=torch.float16),
            'B':torch.tensor([[5.,-2.,1.],[7.,4.,3.]], dtype=torch.float16)}
    expected=torch.tensor([[19.,6.,7.],[16.,14.,8.]],dtype=torch.float16)
    torch.testing.assert_close(ref(inputs),expected,atol=0,rtol=0)
    with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(torch.zeros_like(expected),expected)


@pytest.mark.parametrize('short,wrapper', [('batched_moe','batched_moe_gemm'),('moe_mmk','moe_matmul')])
def test_wrapper_cannot_bypass_declared_kernel(tmp_path, short, wrapper):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    task=TASKS/('triton_'+short)
    workspace=tmp_path/'task';shutil.copytree(task,workspace)
    snapshot=snapshot_workspace_harness(workspace)
    path=workspace/'source'/('triton_'+short+'.py');text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==wrapper)
    lines=text.splitlines(keepends=True)
    lines[node.body[-1].lineno-1:node.body[-1].end_lineno]=['    return torch.zeros_like(A)\n']
    path.write_text(''.join(lines))
    with pytest.raises(RuntimeError): verify_workspace_harness(snapshot)


@pytest.mark.parametrize('short', ['batched_moe','moe_mmk'])
def test_original_scored_cases_and_timing_intact(short):
    task=TASKS/('triton_'+short)
    manifest=json.loads((task/'workloads.json').read_text())
    measured=[r for r in manifest['cases'] if 'performance' in r['checks']]
    assert [r['test_case_id'] for r in measured]==[f'perf{i}' for i in range(1,6)]
    assert [r['params']['configuration'] for r in measured]==manifest['input_table']
    tree=ast.parse((task/'scripts/task_runner.py').read_text())
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['WARMUP_ITERATIONS','BENCHMARK_ITERATIONS']}
    assert assignments=={'WARMUP_ITERATIONS':10,'BENCHMARK_ITERATIONS':100}


def test_mmk_partial_tile_pointer_indices_stay_within_matrix():
    from types import SimpleNamespace
    path=TASKS/'triton_moe_mmk/source/triton_moe_mmk.py'
    kernel=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='moe_mmk_kernel')
    expr=next(n.value for n in kernel.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='offs_n')
    actual=eval(compile(ast.Expression(expr),str(path),'eval'),{'pid_n':1,'BLOCK_N':64,'N':70,'tl':SimpleNamespace(arange=torch.arange)})
    assert actual.min()>=0 and actual.max()<70
    assert torch.equal(actual[:6],torch.arange(64,70))
    # The original operator precedence used local modulo, yielding indices 70..127.
    legacy=64+torch.arange(64)%70
    assert (legacy>=70).sum()==58


def test_batched_weight_load_masks_both_partial_dimensions():
    path=TASKS/'triton_batched_moe/source/triton_batched_moe.py'
    tree=ast.parse(path.read_text())
    load=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='b')
    mask=next(k.value for k in load.value.keywords if k.arg=='mask')
    actual=eval(compile(ast.Expression(mask),str(path),'eval'),{'offs_k':torch.arange(32),'K':35,'k':1,'BLOCK_K':32,'offs_n':torch.arange(64),'cta_n_size':6})
    assert actual.shape==(32,64) and actual.sum()==18
    assert not actual[:,6:].any() and not actual[3:].any()
