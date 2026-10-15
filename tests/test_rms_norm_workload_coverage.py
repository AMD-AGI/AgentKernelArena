"""Finite RMS workload and wrapper ABI checks; CPU simulations are not GPU passes."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_rms_norm'
ORIGINAL_SHAPES = [(32, 128), (64, 512), (128, 1024), (256, 2048), (512, 4096)]
ADDED_SHAPES = [(1, 4096), (32, 576), (32, 1536)]


def module_at(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    spec = importlib.util.spec_from_file_location('_rms_coverage_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_task(monkeypatch):
    """Use the real protected wrapper and an explicit CPU simulated kernel."""
    triton = ModuleType('triton')
    language = ModuleType('triton.language')
    language.constexpr = object()
    triton.language = language
    triton.jit = lambda fn: fn
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', language)
    adapter = module_at(TASK / '_arena_eval.py', monkeypatch)
    harness = adapter.load_harness()
    source = module_at(TASK / 'source/triton_rms_norm.py', monkeypatch)
    calls = []
    mode = {'value': 'correct'}

    class Kernel:
        def __getitem__(self, grid):
            def launch(x, weight, output, input_stride, output_stride, hidden, eps, *, BLOCK_SIZE):
                calls.append({'x': x.clone(), 'weight': weight.clone(), 'eps': eps,
                              'grid': grid, 'block': BLOCK_SIZE, 'hidden': hidden,
                              'strides': (input_stride, output_stride),
                              'contiguous': (x.is_contiguous(), weight.is_contiguous())})
                active_eps = 1e-6 if mode['value'] == 'default_eps' else eps
                result = harness.reference_rms_norm(x, weight, active_eps)
                if mode['value'] == 'decode_noop' and x.shape[0] == 1:
                    result.zero_()
                if mode['value'] == 'omit_tail' and hidden in (576, 1536):
                    result[:, (hidden // 1024) * 1024:] = 0
                output.copy_(result)
            return launch

    source._rms_norm_kernel = Kernel()
    harness.load_module = lambda: source
    adapter.load_harness = lambda: harness
    original_randn = torch.randn

    def cpu_randn(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, 'randn', cpu_randn)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    harness._TimedRun = SimpleNamespace
    timings = []

    def benchmark(fn, *, timed_run, **kwargs):
        timings.append(kwargs)
        output = fn()
        timed_run.outputs = output
        timed_run.rerun = lambda: output.copy_(fn())
        return 0.25, {'benchmark_method': 'cuda_graph'}

    harness._benchmark_cuda_graph_or_events = benchmark
    return SimpleNamespace(adapter=adapter, harness=harness, source=source,
                           calls=calls, mode=mode, timings=timings)


def test_original_kernel_reference_timing_and_first_five_cases_are_preserved(monkeypatch):
    runner_path = TASK / 'scripts/task_runner.py'
    old = subprocess.check_output(
        ['git', 'show', f'5c9f8ef2:{runner_path.relative_to(ROOT)}'], cwd=ROOT, text=True)
    current = runner_path.read_text()
    old_tree, current_tree = ast.parse(old), ast.parse(current)
    functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    original_functions, current_functions = functions(old_tree), functions(current_tree)
    for name in ('reference_rms_norm', 'run_performance', 'load_module', 'run_compile',
                 '_benchmark_cuda_graph_or_events', '_measure_cuda_event_fallback'):
        assert ast.dump(current_functions[name]) == ast.dump(original_functions[name])
    correction = deepcopy(current_functions['run_correctness'])
    correction.body.pop(0)  # Only the explicit additional-control dispatch.
    correction.args = deepcopy(original_functions['run_correctness'].args)
    loop = next(node for node in ast.walk(correction) if isinstance(node, ast.For))
    loop.body.pop(0)  # The existing schema-v2 per-case selection guard.
    assert ast.dump(correction) == ast.dump(original_functions['run_correctness'])
    source_path = TASK / 'source/triton_rms_norm.py'
    original_source = subprocess.check_output(
        ['git', 'show', f'5c9f8ef2:{source_path.relative_to(ROOT)}'], cwd=ROOT)
    assert source_path.read_bytes() == original_source
    runner = module_at(runner_path, monkeypatch)
    assert runner.TEST_SHAPES == ORIGINAL_SHAPES + ADDED_SHAPES
    assert runner.WARMUP_ITERATIONS == 10 and runner.BENCHMARK_ITERATIONS == 100


def test_manifest_declares_all_and_only_the_finite_correctness_and_scored_cases(cpu_task):
    data = cpu_task.adapter.load_manifest()
    rows = data['cases']
    scored = [row for row in rows if 'performance' in row['checks']]
    assert [row['test_case_id'] for row in scored] == [f'perf{i}' for i in range(1, 9)]
    assert [row['params']['configuration'] for row in scored] == data['input_table']
    assert data['input_table'] == [list(shape) for shape in ORIGINAL_SHAPES + ADDED_SHAPES]
    assert all(row['dtype'] == 'float16' and row['params']['case_index'] == index
               and row['checks'] == ['correctness', 'performance']
               for index, row in enumerate(scored))
    assert data['correctness_controls'] == cpu_task.harness.CORRECTNESS_CONTROLS
    for index, (row, control) in enumerate(zip(rows[8:], data['correctness_controls'], strict=True)):
        assert row['test_case_id'] == control['test_case_id']
        assert row['dtype'] == control['dtype'] and row['checks'] == ['correctness']
        assert row['params'] == {'case_index': 10000 + index,
                                 **{key: control[key] for key in ('shape', 'eps', 'input_scale', 'seed')}}


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_adapter_runs_all_twelve_cases_through_the_actual_wrapper_on_cpu(cpu_task, role):
    result = cpu_task.adapter.evaluate(role, 'correctness')
    assert result['status'] == 'PASS' and len(result['cases']) == 12
    assert all(row['status'] == 'PASS' for row in result['cases'])
    assert len(cpu_task.calls) == 12
    for index, call in enumerate(cpu_task.calls):
        if index < 8:
            shape, dtype, eps, scale, seed = (
                (ORIGINAL_SHAPES + ADDED_SHAPES)[index], torch.float16, 1e-6, 1, 42 + index)
        else:
            control = cpu_task.harness.CORRECTNESS_CONTROLS[index - 8]
            shape, dtype, eps, scale, seed = (
                control['shape'], getattr(torch, control['dtype']), control['eps'],
                control['input_scale'], control['seed'])
        torch.manual_seed(seed)
        expected_x = torch.randn(*shape, dtype=dtype) * scale
        expected_weight = torch.randn(shape[-1], dtype=dtype)
        assert torch.equal(call['x'], expected_x.reshape(-1, shape[-1]))
        assert torch.equal(call['weight'], expected_weight)
        assert call['eps'] == eps and call['grid'] == (expected_x.numel() // shape[-1],)
        assert call['block'] == 1024 and call['contiguous'] == (True, True)


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_same_eight_performance_cases_keep_original_draws_options_and_replay_checks(cpu_task, role):
    result = cpu_task.adapter.evaluate(role, 'performance')
    assert result['status'] == 'PASS' and len(result['cases']) == 8
    assert cpu_task.timings == [{'warmup': 10, 'repetition': 100}] * 8
    assert len(cpu_task.calls) == 16  # Measured output and changed-input replay per case.
    for index, shape in enumerate(ORIGINAL_SHAPES + ADDED_SHAPES):
        torch.manual_seed(0)
        expected_x = torch.randn(*shape, dtype=torch.float16)
        expected_weight = torch.randn(shape[-1], dtype=torch.float16)
        measured, replay = cpu_task.calls[2 * index:2 * index + 2]
        assert torch.equal(measured['x'], expected_x)
        assert torch.equal(measured['weight'], expected_weight)
        assert torch.equal(replay['x'], -expected_x)
        assert torch.equal(replay['weight'], expected_weight * 0.5)
        metadata = result['cases'][index]['metadata']['harness_measurement']
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']


@pytest.mark.parametrize('mode,failed_ids', [
    ('default_eps', {'control_rank3_bf16_eps', 'control_eps'}),
    ('decode_noop', {'perf6'}),
    ('omit_tail', {'perf7', 'perf8', 'control_bf16', 'control_rank3',
                   'control_rank3_bf16_eps', 'control_eps'}),
])
def test_declared_controls_reject_epsilon_decode_and_tail_errors_on_cpu(cpu_task, mode, failed_ids):
    cpu_task.mode['value'] = mode
    result = cpu_task.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL'
    assert {row['test_case_id'] for row in result['cases'] if row['status'] == 'FAIL'} == failed_ids


def test_adapter_rejects_manifest_control_drift_and_unknown_control(cpu_task, monkeypatch):
    data = cpu_task.adapter.load_manifest()
    data['correctness_controls'][0]['eps'] = 1.0
    monkeypatch.setattr(cpu_task.adapter, 'load_manifest', lambda: data)
    result = cpu_task.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL'
    assert 'correctness controls disagree' in result['reason']
    ok, error = cpu_task.harness.run_correctness(case_index=10004)
    assert not ok and 'IndexError' in error
