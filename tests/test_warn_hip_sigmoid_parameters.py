"""CPU negative controls for public Sigmoid parameters and measured replay."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/hip2hip/Sigmoid'


def load(path):
    spec = importlib.util.spec_from_file_location('sigmoid_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch):
    evaluate = load(TASK / 'eval_tools/evaluate.py')
    module_cls = load(TASK / 'pytorch_code_module/py_11184_Sigmoid.py').Sigmoid
    functional_cls = load(TASK / 'pytorch_code_functional/py_11184_Sigmoid_func.py').Sigmoid
    compare = lambda expected, actual, **_: torch.allclose(expected, actual, rtol=1e-4, atol=1e-5)
    monkeypatch.setitem(sys.modules, 'correctness_check', SimpleNamespace(_compare_results=compare))
    monkeypatch.syspath_prepend(str(TASK / 'eval_tools'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return evaluate, module_cls, functional_cls, compare


@pytest.mark.parametrize('kind,failed', [
    ('correct', None), ('constant_a', 'a_only'), ('constant_max', 'max_only')])
def test_public_parameter_controls_reject_hardcoding(harness, kind, failed):
    evaluate, module_cls, functional_cls, _ = harness
    def hip(v, a, maximum):
        return torch.sigmoid((1 if kind == 'constant_a' else a) * v) * (
            10 if kind == 'constant_max' else maximum)
    if failed:
        with pytest.raises(ValueError, match=failed):
            evaluate.check_sigmoid_controls(module_cls(), functional_cls(), hip,
                                            1e-4, 1e-5, device='cpu')
    else:
        assert evaluate.check_sigmoid_controls(module_cls(), functional_cls(), hip,
                                               1e-4, 1e-5, device='cpu') == [
            'scalar', 'empty', 'extreme', 'a_only', 'max_only', 'negative_a']


@pytest.mark.parametrize('kind,failed', [
    ('correct', False), ('constant_a', True), ('constant_max', True)])
def test_unscored_measured_parameter_replay_rejects_hardcoding(harness, monkeypatch, kind, failed):
    evaluate, module_cls, functional_cls, compare = harness
    class FakeTimedRun:
        def __init__(self):
            self.outputs = None
            self.after_sample = None
            self.rerun = None
        @property
        def bound(self):
            return callable(self.rerun)
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=FakeTimedRun))
    def hip(v, a, maximum):
        return torch.sigmoid((1 if kind == 'constant_a' else a) * v) * (
            10 if kind == 'constant_max' else maximum)
    monkeypatch.setattr(evaluate, 'compile_hip', lambda *args, **kwargs: hip)
    monkeypatch.setattr(evaluate, 'prepare_models', lambda *args, **kwargs: (
        module_cls().eval(), functional_cls().eval()))
    def benchmark(fn, *, warmup, repetition, use_cuda_graph, max_graph_repeats, timed_run):
        assert (warmup, repetition, use_cuda_graph, max_graph_repeats) == (10, 100, True, 1)
        for _ in range(warmup):
            fn()
        output = fn()
        timed_run.outputs = output
        def replay():
            timed_run.outputs.copy_(fn())
            return timed_run.outputs
        timed_run.rerun = replay
        for _ in range(repetition):
            replay()
            timed_run.after_sample(timed_run.outputs)
        return .01, {'benchmark_method': 'cuda_graph',
                      'benchmark_timed_run_kind': 'captured_graph',
                      'benchmark_effective_repeats': 1, 'benchmark_samples': repetition}
    perf = SimpleNamespace(_compare_results=compare, benchmark_cuda_graph_or_events=benchmark)
    args = SimpleNamespace(baseline_hip='reference.hip', candidate='candidate.hip')
    if failed:
        with pytest.raises(ValueError, match='scaled_parameters'):
            evaluate.check_unscored_layout_timing(args, 'candidate', perf, device='cpu')
    else:
        assert evaluate.check_unscored_layout_timing(args, 'candidate', perf, device='cpu') == [
            'strided_float32', 'strided_float64', 'scaled_parameters']
