"""Real AWQ wrapper/guard CPU regressions; simulated kernels are not GPU passes."""
import ast
import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_awq_dequantize'


def module_at(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    spec = importlib.util.spec_from_file_location('_awq_cpu_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_cpu(qweight, scales, zeros, group_size):
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    unpack = lambda x: ((x[:, :, None] >> shifts) & 15).reshape(x.shape[0], -1)
    return ((unpack(qweight).float() - unpack(zeros).repeat_interleave(group_size, 0).float())
            * scales.float().repeat_interleave(group_size, 0)).to(scales.dtype)


@pytest.fixture
def cpu_task(monkeypatch):
    triton, language = ModuleType('triton'), ModuleType('triton.language')
    triton.language, language.constexpr = language, object()
    triton.jit = lambda fn: fn
    triton.cdiv = lambda left, right: (left + right - 1) // right
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', language)
    adapter = module_at(TASK / '_arena_eval.py', monkeypatch)
    harness = adapter.load_harness()
    source = module_at(TASK / 'source/triton_awq_dequantize.py', monkeypatch)
    checks = module_at(TASK / '_arena_checks.py', monkeypatch)
    state = {'field': None, 'phase': 'measured', 'correct_before_write': False, 'raises': False}
    retained, calls = {}, {}

    class Kernel:
        def __getitem__(self, grid):
            def launch(q, s, z, group_size, output, *args, **kwargs):
                key = q.data_ptr()
                calls[key] = calls.get(key, 0) + 1
                retained.setdefault(key, ((q, s, z), tuple(t.clone() for t in (q, s, z))))
                pristine = reference_cpu(q, s, z, group_size)
                if state['field'] and (state['phase'] != 'replay' or calls[key] == 2):
                    {'qweight': q, 'scales': s, 'zeros': z}[state['field']].zero_()
                    if state['raises']:
                        raise RuntimeError('candidate failed after input write')
                output.copy_(pristine if state['correct_before_write'] else reference_cpu(q, s, z, group_size))
            return launch

    source.awq_dequantize_kernel = Kernel()
    harness.load_module = lambda: source
    original_to = torch.Tensor.to

    def to_cpu(self, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    for name in ('randn', 'randint'):
        original = getattr(torch, name)
        def cpu(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    harness._TimedRun = SimpleNamespace
    return SimpleNamespace(source=source, harness=harness, checks=checks,
                           state=state, retained=retained)


def inputs():
    return (torch.full((64, 8), 0x76543210, dtype=torch.int32),
            torch.arange(1, 65, dtype=torch.float16).repeat(2, 1) / 64,
            torch.full((2, 8), 0x11111111, dtype=torch.int32))


def assert_restored(setup):
    for values, saved in setup.retained.values():
        for value, initial in zip(values, saved):
            assert torch.equal(value, initial)


def test_actual_correct_wrapper_passes_original_scored_case(cpu_task):
    original = cpu_task.source.awq_dequantize_triton
    assert cpu_task.harness.run_correctness(case_index=0) == (True, None)
    assert_restored(cpu_task)
    assert cpu_task.source.awq_dequantize_triton is original


@pytest.mark.parametrize('field', ['qweight', 'scales', 'zeros'])
@pytest.mark.parametrize('correct_before_write', [False, True])
def test_actual_correctness_wrapper_rejects_operand_writes(cpu_task, field, correct_before_write):
    cpu_task.state.update(field=field, correct_before_write=correct_before_write)
    original = cpu_task.source.awq_dequantize_triton
    ok, reason = cpu_task.harness.run_correctness(case_index=0)
    assert not ok and f'AWQ {field} is read-only' in reason
    assert_restored(cpu_task)
    assert cpu_task.source.awq_dequantize_triton is original


@pytest.mark.parametrize('field', ['qweight', 'scales', 'zeros'])
@pytest.mark.parametrize('phase', ['measured', 'replay'])
def test_actual_wrapper_timed_and_replay_writes_are_rejected_and_restored(cpu_task, field, phase):
    cpu_task.state.update(field=field, phase=phase)
    qweight, scales, zeros = inputs()
    saved = tuple(t.clone() for t in (qweight, scales, zeros))
    mod = cpu_task.source
    original = mod.awq_dequantize_triton

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def benchmark(measured, *, timed_run, **kwargs):
        assert kwargs == {'warmup': 10, 'repetition': 100}
        output = measured()
        timed_run.outputs, timed_run.rerun = output, lambda: output.copy_(measured())
        return .25, {'benchmark_method': 'cuda_graph'}

    with pytest.raises(AssertionError, match=f'AWQ {field} is read-only'):
        cpu_task.checks.checked_benchmark(cpu_task.harness, benchmark, fn, warmup=10, repetition=100)
    assert all(torch.equal(value, initial) for value, initial in zip((qweight, scales, zeros), saved))
    assert mod.awq_dequantize_triton is original


@pytest.mark.parametrize('field', ['qweight', 'scales', 'zeros'])
def test_correctness_restores_after_candidate_exception(cpu_task, field):
    cpu_task.state.update(field=field, raises=True)
    original = cpu_task.source.awq_dequantize_triton
    ok, reason = cpu_task.harness.run_correctness(case_index=0)
    assert not ok and 'candidate failed after input write' in reason
    assert_restored(cpu_task)
    assert cpu_task.source.awq_dequantize_triton is original


@pytest.mark.parametrize('phase', ['benchmark', 'replay'])
def test_benchmark_and_replay_exceptions_restore_all_inputs(cpu_task, phase):
    qweight, scales, zeros = inputs()
    saved = tuple(t.clone() for t in (qweight, scales, zeros))
    mod = cpu_task.source
    original = mod.awq_dequantize_triton

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def fail():
        for value in (qweight, scales, zeros):
            value.zero_()
        raise RuntimeError('execution failed')

    def benchmark(measured, *, timed_run, **kwargs):
        if phase == 'benchmark':
            fail()
        timed_run.outputs, timed_run.rerun = measured(), fail
        return .25, {'benchmark_method': 'cuda_graph'}

    with pytest.raises(RuntimeError, match='execution failed'):
        cpu_task.checks.checked_benchmark(cpu_task.harness, benchmark, fn, warmup=10, repetition=100)
    assert all(torch.equal(value, initial) for value, initial in zip((qweight, scales, zeros), saved))
    assert mod.awq_dequantize_triton is original


def test_readonly_scales_are_exact_below_output_tolerance(cpu_task):
    qweight, scales, zeros = inputs()
    scales = scales.float()
    saved = cpu_task.checks.snapshot_inputs(qweight, scales, zeros)
    scales[0, 0] += 1e-6
    with pytest.raises(AssertionError, match='AWQ scales is read-only'):
        cpu_task.checks.check_inputs(qweight, scales, zeros, saved)


def test_snapshot_reference_and_validation_stay_outside_timing(cpu_task, monkeypatch):
    qweight, scales, zeros = inputs()
    mod = cpu_task.source
    original = mod.awq_dequantize_triton
    outside = {'value': True}
    events = []
    for name in ('snapshot_inputs', 'reference_output', 'check_inputs', 'check_output', 'restore_inputs'):
        operation = getattr(cpu_task.checks, name)
        def checked(*args, _name=name, _operation=operation, **kwargs):
            assert outside['value']
            events.append(_name)
            return _operation(*args, **kwargs)
        monkeypatch.setattr(cpu_task.checks, name, checked)

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def benchmark(measured, *, timed_run, **kwargs):
        assert events == ['snapshot_inputs', 'reference_output']
        def invoke():
            outside['value'] = False
            try:
                return measured()
            finally:
                outside['value'] = True
        output = invoke()
        timed_run.outputs, timed_run.rerun = output, lambda: output.copy_(invoke())
        return .25, {'benchmark_method': 'cuda_graph'}

    ms, metadata = cpu_task.checks.checked_benchmark(cpu_task.harness, benchmark, fn, warmup=10, repetition=100)
    assert ms == .25 and metadata['readonly_inputs_checked']
    assert events.count('reference_output') == 2 and events[-1] == 'restore_inputs'
    assert mod.awq_dequantize_triton is original


@pytest.fixture
def boundary(tmp_path):
    workspace = tmp_path / 'workspace'
    shutil.copytree(TASK, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_awq_dequantize')
    edit, = spec.candidate.editable
    assert edit.scope == 'symbols' and edit.symbols == ('awq_dequantize_kernel',)
    assert edit.allow_new_helpers
    return workspace, snapshot_workspace_harness(workspace, task_spec=spec)


def test_real_guard_accepts_kernel_body_and_new_helper(boundary):
    workspace, snapshot = boundary
    path = workspace / 'source/triton_awq_dequantize.py'
    tree = ast.parse(path.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'awq_dequantize_kernel')
    kernel.body.insert(0, ast.parse('candidate_value = _candidate_identity(0)\n').body[0])
    tree.body.extend(ast.parse('@triton.jit\ndef _candidate_identity(value):\n    return value\n').body)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    verify_workspace_harness(snapshot, discard_added=False)


@pytest.mark.parametrize('target', ['wrapper', 'imports', 'group_sizes', 'checks', 'runner', 'manifest'])
def test_real_guard_rejects_wrapper_contract_and_harness_changes(boundary, target):
    workspace, snapshot = boundary
    if target in ('checks', 'runner', 'manifest'):
        path = workspace / {'checks': '_arena_checks.py', 'runner': 'scripts/task_runner.py', 'manifest': 'workloads.json'}[target]
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path = workspace / 'source/triton_awq_dequantize.py'
        tree = ast.parse(path.read_text())
        if target == 'wrapper':
            wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'awq_dequantize_triton')
            wrapper.body = [ast.Return(value=ast.Constant(None))]
        elif target == 'imports':
            tree.body.extend(ast.parse('import os\n').body)
        else:
            node = next(n for n in tree.body if isinstance(n, ast.Assign))
            node.value = ast.List(elts=[ast.Constant(1)], ctx=ast.Load())
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(RuntimeError, match='[Hh]arness|protected|policy'):
        verify_workspace_harness(snapshot, discard_added=False)
