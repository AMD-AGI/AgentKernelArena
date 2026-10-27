"""CPU controls for state and numerical gates added to the HIP task adapters."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

TASKS = Path(__file__).resolve().parents[1] / 'tasks'


def load(path):
    spec = importlib.util.spec_from_file_location('hip_v5_control_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loss_rejects_live_smoothing_state_mutation():
    root = TASKS / 'hip2hip/CrossEntropyLossLabelSmoothing'
    controls = load(root / 'eval_tools/case_controls.py')
    model_module = load(root / 'pytorch_code_module/py_12501_CrossEntropyLossLabelSmoothing.py')
    model = model_module.CrossEntropyLossLabelSmoothing()
    torch.manual_seed(0)
    inputs = next(iter(model_module.get_inputs()))
    controls.configure_models((model,), inputs)
    controls.assert_declared_control(model, inputs)
    model.smooth_eps += .1
    with pytest.raises(ValueError, match='epsilon'):
        controls.assert_declared_control(model, inputs)
    controls.configure_models((model,), inputs)
    model.smooth_dist.mul_(.8)
    with pytest.raises(AssertionError):
        controls.assert_declared_control(model, inputs)


def test_loss_rejects_state_change_after_correct_reported_sample(monkeypatch):
    root = TASKS / 'hip2hip/CrossEntropyLossLabelSmoothing'
    controls = load(root / 'eval_tools/case_controls.py')
    model_module = load(root / 'pytorch_code_functional/py_12501_CrossEntropyLossLabelSmoothing_func.py')
    validator = load(root / 'eval_tools/replay_validation.py')
    runner = load(root / 'eval_tools/evaluate.py')
    model = model_module.CrossEntropyLossLabelSmoothing()
    torch.manual_seed(0)
    inputs = next(iter(model_module.get_inputs()))
    controls.configure_models((model,), inputs)

    class TimedRun:
        def __init__(self):
            self.after_sample = None
            self.outputs = None

    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=TimedRun))
    monkeypatch.setitem(sys.modules, 'case_controls', controls)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    def benchmark(invoke, *, timed_run, repetition, max_graph_repeats, **_kwargs):
        assert max_graph_repeats == 1
        for _ in range(repetition):
            timed_run.outputs = invoke()
            timed_run.after_sample(timed_run.outputs)
        # A candidate can mutate state after returning a numerically correct output.
        model.smooth_dist.mul_(.8)
        return .1, {'benchmark_method': 'cuda_event_fallback',
                    'benchmark_timed_run_kind': 'eager_callable',
                    'benchmark_samples': repetition}

    perf = types.SimpleNamespace(cal_kernel_perf=lambda rtol=1e-4, atol=1e-5: None,
        benchmark_cuda_graph_or_events=benchmark,
        _compare_results=lambda left, right, **kw: torch.allclose(left, right, **kw))
    validator.install(perf, runner.output_contract)
    with pytest.raises(AssertionError):
        perf.cal_hip_latency(model, inputs, n_iter=2)
    controls.assert_declared_control(model, inputs)


def test_roiaware_scored_max_rejects_systematic_fifteen_percent_error():
    path = TASKS / 'hip2hip/roiaware_pool3d/scripts/reference_checks.py'
    checks = load(path)
    expected = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 1, 1, 4)
    actual = expected * 1.15
    checks.full_output(actual, expected)
    with pytest.raises(AssertionError):
        checks.strict_max_output(actual, expected)
    with pytest.raises(AssertionError):
        checks.check_timed_output(actual, expected, 'max', gpu=False)


@pytest.mark.parametrize('fault', ('clamp_below_minus_three', 'erase_minus_four_to_minus_two'))
def test_l2n55_negative_replay_rejects_clamped_candidate(monkeypatch, fault):
    root = TASKS / 'torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale'
    validator = load(root / 'eval_tools/replay_validation.py')

    class TimedRun:
        def __init__(self):
            self.after_sample = None
            self.outputs = None
            self._invoke = None
        def rerun(self):
            self.outputs = self._invoke()
            return self.outputs

    class Model(torch.nn.Module):
        def forward(self, x, fn=None):
            return x.sum(dim=1).clone() if fn is None else fn(x)

    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=TimedRun))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    inputs = [torch.tensor([[.1, .2, .3, .4]])]
    original = inputs[0].clone()

    def benchmark(invoke, *, timed_run, repetition, max_graph_repeats, **_kwargs):
        assert max_graph_repeats == 1
        timed_run._invoke = invoke
        for _ in range(repetition):
            timed_run.outputs = invoke()
            timed_run.after_sample(timed_run.outputs)
        return .1, {'benchmark_method': 'cuda_event_fallback',
                    'benchmark_timed_run_kind': 'eager_callable',
                    'benchmark_samples': repetition}

    perf = types.SimpleNamespace(cal_kernel_perf=lambda rtol=1e-4, atol=1e-5: None,
        benchmark_cuda_graph_or_events=benchmark,
        _compare_results=lambda left, right, **kw: torch.allclose(left, right, **kw))
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    def incorrect(value):
        if fault == 'clamp_below_minus_three':
            value = value.clamp_min(-3)
        else:
            value = torch.where((value > -4) & (value < -2), 0, value)
        return value.sum(dim=1).clone()
    with pytest.raises(ValueError, match='Timed operator output disagrees'):
        perf.cal_hip_latency(Model(), inputs, incorrect, n_iter=2)
    torch.testing.assert_close(inputs[0], original, rtol=0, atol=0)
    elapsed, metadata = perf.cal_hip_latency(Model(), inputs,
        lambda value: value.sum(dim=1).clone(), n_iter=2)
    assert elapsed == .1 and metadata['negative_domain_replay_valid'] is True
