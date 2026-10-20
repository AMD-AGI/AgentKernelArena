"""CPU execution of the real RMS adapter/wrapper; no GPU qualification claims."""
from types import SimpleNamespace

import pytest
import torch

from test_rms_norm_workload_coverage import TASK, cpu_task, module_at


@pytest.mark.parametrize('field', ['input', 'weight'])
@pytest.mark.parametrize('phase', ['correctness', 'measured', 'replay'])
def test_actual_adapter_rejects_kernel_input_writes_and_restores_buffers(cpu_task, field, phase):
    """The old harness accepted zero input/weight plus zero output in every case."""
    retained = []
    calls = {}

    class InputWritingKernel:
        def __getitem__(self, grid):
            def launch(x, weight, output, *args, **kwargs):
                key = x.data_ptr()
                calls[key] = calls.get(key, 0) + 1
                if calls[key] == 1:
                    retained.append((x, weight, x.clone(), weight.clone()))
                if phase != 'replay' or calls[key] == 2:
                    (x if field == 'input' else weight).zero_()
                output.copy_(cpu_task.harness.reference_rms_norm(x, weight, args[-1]))
            return launch

    cpu_task.source._rms_norm_kernel = InputWritingKernel()
    original_wrapper = cpu_task.source.rms_norm
    action = 'correctness' if phase == 'correctness' else 'performance'
    result = cpu_task.adapter.evaluate('candidate', action)
    assert result['status'] == 'FAIL'
    assert len(result['cases']) == (12 if action == 'correctness' else 8)
    assert all(row['status'] == 'FAIL' for row in result['cases'])
    if action == 'correctness':
        assert all(f'RMSNorm {field} is read-only' in row['reason'] for row in result['cases'])
    else:
        assert cpu_task.timings == [{'warmup': 10, 'repetition': 100}] * 8
        assert all(row['failure_kind'] == 'measurement_failure' for row in result['cases'])
    for x, weight, original_x, original_weight in retained:
        assert torch.equal(x, original_x)
        assert torch.equal(weight, original_weight)
    assert cpu_task.source.rms_norm is original_wrapper


@pytest.mark.parametrize('field', ['input', 'weight'])
@pytest.mark.parametrize('raises', [False, True])
def test_correctness_snapshot_precedes_candidate_and_restores_on_exception(monkeypatch, field, raises):
    checks = module_at(TASK / '_arena_checks.py', monkeypatch)
    events = []
    x, weight = torch.tensor([[3., 4.]]), torch.tensor([2., 0.5])
    originals = (x.clone(), weight.clone())

    def reference(value, scale, eps):
        events.append('reference')
        assert torch.equal(value, originals[0]) and torch.equal(scale, originals[1])
        return value * (value.square().mean(-1, keepdim=True) + eps).rsqrt() * scale

    def candidate(value, scale, eps):
        events.append('candidate')
        output = value * (value.square().mean(-1, keepdim=True) + eps).rsqrt() * scale
        (value if field == 'input' else scale).zero_()
        if raises:
            raise RuntimeError('candidate raised after writing')
        return output  # Correct output must not excuse an input write.

    module = SimpleNamespace(rms_norm=candidate)
    loader = lambda: module
    harness = SimpleNamespace(load_module=loader, reference_rms_norm=reference)
    with checks.checked_modules(harness):
        checked = harness.load_module()
        error = RuntimeError if raises else AssertionError
        with pytest.raises(error, match='candidate raised' if raises else 'is read-only'):
            checked.rms_norm(x, weight, eps=1e-6)
    assert events == ['reference', 'candidate']
    assert torch.equal(x, originals[0]) and torch.equal(weight, originals[1])
    assert module.rms_norm is candidate and harness.load_module is loader


@pytest.mark.parametrize('phase', ['benchmark', 'replay'])
def test_performance_restores_original_inputs_and_wrapper_when_execution_raises(monkeypatch, phase):
    checks = module_at(TASK / '_arena_checks.py', monkeypatch)
    harness = module_at(TASK / 'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    x, weight, eps = torch.tensor([[3., 4.]]), torch.tensor([2., 0.5]), 1e-6
    originals = (x.clone(), weight.clone())
    mod = SimpleNamespace(rms_norm=harness.reference_rms_norm)
    original = mod.rms_norm

    def fn():
        mod.rms_norm(x, weight, eps=eps)

    def fail():
        x.zero_()
        weight.zero_()
        raise RuntimeError('device execution failed')

    def benchmark(measured, *, timed_run, **kwargs):
        if phase == 'benchmark':
            fail()
        timed_run.outputs = measured()
        timed_run.rerun = fail
        return 0.25, {'benchmark_method': 'cuda_graph'}

    with pytest.raises(RuntimeError, match='device execution failed'):
        checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert torch.equal(x, originals[0]) and torch.equal(weight, originals[1])
    assert mod.rms_norm is original


@pytest.mark.parametrize('field', ['input', 'weight'])
def test_readonly_check_requires_exact_values_even_below_output_tolerance(monkeypatch, field):
    checks = module_at(TASK / '_arena_checks.py', monkeypatch)
    x, weight = torch.tensor([[1., 2.]]), torch.ones(2)
    saved = checks.snapshot_inputs(x, weight)
    (x if field == 'input' else weight).view(-1)[0] += 1e-5
    with pytest.raises(AssertionError, match=f'RMSNorm {field} is read-only'):
        checks.check_inputs(x, weight, saved)
    checks.restore_inputs(x, weight, saved)
    checks.check_inputs(x, weight, saved)
