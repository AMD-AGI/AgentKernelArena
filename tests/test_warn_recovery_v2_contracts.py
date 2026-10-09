"""Controls for live HIP affine state and wide/long Triton operator domains."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(4, 4), (8, 4), (4, 8, 4), (8, 16, 4),
                                    (16, 32, 4), (32, 64, 4)])
@pytest.mark.parametrize("ignored", ["gamma", "beta"])
def test_layer_normalization_live_parameters_reject_fixed_state(shape, ignored):
    task = TASKS / "hip2hip/gpumode/layer_normalization"
    adapter = load(task / "eval_tools/evaluate.py", "hip_layer_adapter")
    controls = load(task / "eval_tools/case_controls.py", "hip_layer_controls")
    source = load(task / "pytorch_code_module/py_11754_layer_normalization.py", "hip_layer_module")
    functional_source = load(task / "pytorch_code_functional/py_11754_layer_normalization_func.py",
                             "hip_layer_functional")
    torch.manual_seed(1337)
    x = torch.randn(shape)
    module = source.layer_normalization(4).eval()
    functional = functional_source.layer_normalization(4).eval()
    controls.configure_models((module, functional), [x])
    expected = module(x)
    compare = lambda left, right, **bounds: torch.allclose(left, right, **bounds)
    before_module = {name: value.clone() for name, value in module.state_dict().items()}
    before_functional = {name: value.clone() for name, value in functional.state_dict().items()}
    adapter.check_affine_parameter_variants(module, functional, None, [x], expected,
                                            compare, 1e-4, 1e-5)
    frozen_gamma = functional.gamma.detach().clone()
    frozen_beta = functional.beta.detach().clone()

    def fixed_parameter(value, gamma, beta, epsilon):
        return functional_source.layer_norm_fn(
            value, frozen_gamma if ignored == "gamma" else gamma,
            frozen_beta if ignored == "beta" else beta, epsilon)

    with pytest.raises(ValueError, match=f"changed layer normalization {ignored}"):
        adapter.check_affine_parameter_variants(module, functional, fixed_parameter,
                                                [x], expected, compare, 1e-4, 1e-5)
    assert all(torch.equal(value, before_module[name])
               for name, value in module.state_dict().items())
    assert all(torch.equal(value, before_functional[name])
               for name, value in functional.state_dict().items())


class FakeTimedRun:
    def __init__(self):
        self.after_sample = None
        self.outputs = None
        self.invoke = None

    def rerun(self):
        self.outputs = self.invoke()
        return self.outputs


@pytest.mark.parametrize("ignored", ["gamma", "beta"])
def test_layer_normalization_measured_replay_rejects_fixed_parameter(monkeypatch, ignored):
    task = TASKS / "hip2hip/gpumode/layer_normalization"
    validator = load(task / "eval_tools/replay_validation.py", "hip_layer_replay")
    controls = load(task / "eval_tools/case_controls.py", "hip_layer_replay_controls")
    functional_source = load(task / "pytorch_code_functional/py_11754_layer_normalization_func.py",
                             "hip_layer_replay_functional")
    monkeypatch.setitem(sys.modules, "_aka_benchmark", types.SimpleNamespace(TimedRun=FakeTimedRun))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = torch.randn(4, 4)
    model = functional_source.layer_normalization(4).eval()
    controls.configure_models((model,), [x])
    original_state = {name: value.clone() for name, value in model.state_dict().items()}

    def benchmark(invoke, *, timed_run, repetition, max_graph_repeats, **kwargs):
        assert max_graph_repeats == 1
        timed_run.invoke = invoke
        for _ in range(repetition):
            timed_run.outputs = invoke()
            timed_run.after_sample(timed_run.outputs)
        return 0.1, {"benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph",
                     "benchmark_samples": repetition}

    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=1e-4, atol=1e-5: None,
        benchmark_cuda_graph_or_events=benchmark,
        _compare_results=lambda left, right, **bounds: torch.allclose(left, right, **bounds),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    _, metadata = perf.cal_hip_latency(model, [x], n_iter=3)
    assert metadata["changed_parameter_replay_count"] == 2
    frozen_gamma = model.gamma.detach().clone()
    frozen_beta = model.beta.detach().clone()

    def fixed_parameter(value, gamma, beta, epsilon):
        return functional_source.layer_norm_fn(
            value, frozen_gamma if ignored == "gamma" else gamma,
            frozen_beta if ignored == "beta" else beta, epsilon)

    with pytest.raises(ValueError, match="Timed operator output disagrees"):
        perf.cal_hip_latency(model, [x], hip_fn=fixed_parameter, n_iter=3)
    assert all(torch.equal(value, original_state[name])
               for name, value in model.state_dict().items())


def test_wide_l2_control_rejects_missing_upper_features():
    task = TASKS / "triton2triton/vllm/triton_fla_l2norm"
    checks = load(task / "_arena_checks.py", "wide_l2_checks")
    calls = []

    def reference(x, eps):
        values = x.float()
        return values / torch.sqrt((values * values).sum(-1, keepdim=True) + eps)

    def candidate(x, eps=1e-6):
        calls.append(x.shape[-1])
        expected = reference(x, eps)
        if x.shape[-1] > 16384:
            expected[:, 16384:] = 0
        return expected

    def valid_candidate(x, eps=1e-6):
        calls.append(x.shape[-1])
        return reference(x, eps)

    valid_module = types.SimpleNamespace(l2norm_fwd=valid_candidate)
    valid_harness = types.SimpleNamespace(load_module=lambda: valid_module, reference=reference)
    with checks.checked_modules(valid_harness):
        checked = valid_harness.load_module()
        checked.l2norm_fwd(torch.randn(512, 128))
        checked.l2norm_fwd(torch.randn(512, 128))
    assert calls.count(32768) == 1 and calls.count(32769) == 1
    calls.clear()
    module = types.SimpleNamespace(l2norm_fwd=candidate)
    harness = types.SimpleNamespace(load_module=lambda: module, reference=reference)
    with checks.checked_modules(harness):
        checked = harness.load_module()
        with pytest.raises(AssertionError):
            checked.l2norm_fwd(torch.randn(512, 128))
    assert 32768 in calls
    assert module.l2norm_fwd is candidate


def test_l2_wrapper_dispatches_wide_rows_without_changing_scored_path(monkeypatch):
    task = TASKS / "triton2triton/vllm/triton_fla_l2norm"
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_language.constexpr = object()
    fake_triton.Config = lambda *args, **kwargs: (args, kwargs)
    fake_triton.jit = lambda function=None, **kwargs: function if function is not None else (lambda value: value)
    fake_triton.autotune = lambda **kwargs: (lambda value: value)
    fake_triton.cdiv = lambda left, right: (left + right - 1) // right
    fake_triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    fake_triton.language = fake_language
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)
    source = load(task / "source/triton_fla_l2norm.py", "l2_wide_dispatch")
    calls = []

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def invoke(x, y, eps, **kwargs):
                calls.append((self.name, x.shape[-1], kwargs))
                values = x.float()
                y.copy_(values / torch.sqrt((values * values).sum(-1, keepdim=True) + eps))
            return invoke

    source.l2norm_fwd_kernel = Launch("original")
    source._l2norm_fwd_wide_kernel = Launch("wide")
    for width, expected_path in ((128, "original"), (32768, "wide"), (32769, "wide")):
        x = torch.randn(2, width)
        result = source.l2norm_fwd(x)
        expected = x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert calls[-1][0:2] == (expected_path, width)
    assert calls[-1][2]["BD"] == 1024


def test_lightning_control_rejects_two_main_block_only_kernel():
    task = TASKS / "triton2triton/vllm/triton_lightning_attn_diag"
    checks = load(task / "_arena_checks.py", "lightning_length_checks")
    runner = load(task / "scripts/task_runner.py", "lightning_length_runner")
    calls = []

    def candidate(q, k, v, s, BLOCK=256, CBLOCK=32):
        calls.append((q.shape[2], BLOCK, CBLOCK))
        result = runner.reference_diag_attention(q, k, v, s.reshape(-1), BLOCK, CBLOCK).to(q.dtype)
        if BLOCK == 256 and q.shape[2] > 512:
            result[:, :, 512:] = 0
        return result

    def valid_candidate(q, k, v, s, BLOCK=256, CBLOCK=32):
        calls.append((q.shape[2], BLOCK, CBLOCK))
        return runner.reference_diag_attention(q, k, v, s.reshape(-1), BLOCK, CBLOCK).to(q.dtype)

    q = torch.full((1, 2, 64, 32), 0.25, dtype=torch.float16)
    k = torch.full_like(q, 0.375)
    v = torch.full_like(q, 0.5)
    slopes = torch.tensor([0., 0.03], dtype=torch.float32)
    valid_module = types.SimpleNamespace(lightning_attn_diag_forward=valid_candidate)
    valid_harness = types.SimpleNamespace(load_module=lambda: valid_module,
                                          reference_diag_attention=runner.reference_diag_attention)
    with checks.checked_modules(valid_harness):
        checked = valid_harness.load_module()
        checked.lightning_attn_diag_forward(q, k, v, slopes)
        checked.lightning_attn_diag_forward(q, k, v, slopes)
    assert calls.count((545, 256, 32)) == 1
    calls.clear()
    module = types.SimpleNamespace(lightning_attn_diag_forward=candidate)
    harness = types.SimpleNamespace(load_module=lambda: module,
                                    reference_diag_attention=runner.reference_diag_attention)
    with checks.checked_modules(harness):
        checked = harness.load_module()
        with pytest.raises(AssertionError):
            checked.lightning_attn_diag_forward(q, k, v, slopes)
    assert (545, 256, 32) in calls
    assert module.lightning_attn_diag_forward is candidate
