"""CPU controls for live affine parameters; no GPU timing claim."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'tasks/torch2hip/gpumode/11754_layer_normalization'


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parts(monkeypatch):
    controls = load('eval_tools/case_controls.py', 'layernorm_controls')
    reference = load('pytorch_code_module/py_11754_layer_normalization.py', 'layernorm_reference')
    functional = load('pytorch_code_functional/py_11754_layer_normalization_func.py',
                      'layernorm_functional')
    monkeypatch.setitem(sys.modules, 'case_controls', controls)
    return controls, reference, functional


def compare(left, right, *, rtol, atol):
    return torch.allclose(left, right, rtol=rtol, atol=atol)


def output_contract(expected, actual):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    assert torch.isfinite(actual).all()


def states(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_states(model, expected):
    for name, value in expected.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(4, 4), (8, 4), (4, 8, 4), (8, 16, 4),
                                   (16, 32, 4), (32, 64, 4)])
def test_each_declared_geometry_rejects_stale_gamma_and_beta(parts, shape):
    controls, reference, functional = parts
    torch.manual_seed(1337 + shape[0])
    x = torch.randn(shape)
    baseline = reference.layer_normalization(4).eval()
    candidate = functional.layer_normalization(4).eval()
    controls.configure_models((baseline, candidate), [x])
    expected = baseline(x)
    original = (states(baseline), states(candidate))
    controls.check_live_affine_variants(
        baseline, candidate, [x], expected, lambda values: candidate(*values),
        compare, output_contract, 1e-4, 1e-5)
    assert_states(baseline, original[0])
    assert_states(candidate, original[1])
    saved_gamma = candidate.gamma.detach().clone()
    saved_beta = candidate.beta.detach().clone()
    for missing, fake in (
        ('gamma', lambda value, gamma, beta, eps: functional.layer_norm_fn(value, saved_gamma, beta, eps)),
        ('beta', lambda value, gamma, beta, eps: functional.layer_norm_fn(value, gamma, saved_beta, eps)),
    ):
        with pytest.raises(ValueError, match=f'ignored live {missing}'):
            controls.check_live_affine_variants(
                baseline, candidate, [x], expected,
                lambda values: candidate(*values, fn=fake),
                compare, output_contract, 1e-4, 1e-5)
        assert_states(baseline, original[0])
        assert_states(candidate, original[1])


def test_timed_replay_rejects_affine_constant_after_reported_samples(parts, monkeypatch):
    controls, _, functional = parts
    replay = load('eval_tools/replay_validation.py', 'layernorm_replay')
    torch.manual_seed(7)
    x = torch.randn(4, 4)
    model = functional.layer_normalization(4).eval()
    controls.configure_models((model,), [x])
    original_state = states(model)
    original_input = x.clone()
    saved_gamma = model.gamma.detach().clone()
    saved_beta = model.beta.detach().clone()

    class TimedRun:
        def __init__(self):
            self.after_sample = None
            self.outputs = None
            self.rerun = None

    fake_benchmark_module = ModuleType('_aka_benchmark')
    fake_benchmark_module.TimedRun = TimedRun
    monkeypatch.setitem(sys.modules, '_aka_benchmark', fake_benchmark_module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    def benchmark(fn, *, repetition, timed_run, **kwargs):
        for _ in range(repetition):
            timed_run.outputs = fn()
            timed_run.after_sample(timed_run.outputs)
        timed_run.rerun = fn
        return 1.0, {'benchmark_timed_run_kind': 'captured_graph',
                     'benchmark_method': 'cuda_graph', 'benchmark_samples': repetition}

    def cal_kernel_perf(rtol=1e-4, atol=1e-5):
        pass

    perf = SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
                           benchmark_cuda_graph_or_events=benchmark,
                           _compare_results=compare)
    replay.install(perf, output_contract)
    positive = lambda value, gamma, beta, eps: functional.layer_norm_fn(value, gamma, beta, eps)
    _, metadata = perf.cal_hip_latency(model, [x], hip_fn=positive, n_iter=3, n_warmup=0)
    assert metadata['validated_sample_count'] == 3
    assert metadata['changed_affine_replay_count'] == 2
    assert_states(model, original_state)
    torch.testing.assert_close(x, original_input, rtol=0, atol=0)
    for missing, fake in (
        ('gamma', lambda value, gamma, beta, eps: functional.layer_norm_fn(value, saved_gamma, beta, eps)),
        ('beta', lambda value, gamma, beta, eps: functional.layer_norm_fn(value, gamma, saved_beta, eps)),
    ):
        with pytest.raises(ValueError, match=f'Timed operator ignored live {missing}'):
            perf.cal_hip_latency(model, [x], hip_fn=fake, n_iter=3, n_warmup=0)
        assert_states(model, original_state)
        torch.testing.assert_close(x, original_input, rtol=0, atol=0)
