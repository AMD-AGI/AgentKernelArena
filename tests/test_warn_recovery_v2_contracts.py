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




class FakeTimedRun:
    def __init__(self):
        self.after_sample = None
        self.outputs = None
        self.invoke = None

    def rerun(self):
        self.outputs = self.invoke()
        return self.outputs




def test_wide_l2_control_rejects_missing_upper_features(monkeypatch):
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

    # Model the named launch boundary while keeping this numerical tail test
    # CPU-only. The real dispatch observer and suppression control stay active.
    runtime = types.SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)

    class CompiledKernel:
        def __init__(self, fn):
            self.src = types.SimpleNamespace(fn=fn)

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def run(self, x, y, eps, *, warmup=False):
            kernel = CompiledKernel(self)
            if not warmup:
                if runtime.launch_enter_hook is not None:
                    runtime.launch_enter_hook(None)
                y.copy_(self.fn(x, eps))
                if runtime.launch_exit_hook is not None:
                    runtime.launch_exit_hook(None)
            return kernel

    compiler = types.SimpleNamespace(CompiledKernel=CompiledKernel)
    monkeypatch.setattr(checks, '_triton_runtime',
                        lambda: (runtime, compiler, Kernel, ()))

    def launched_module(fn):
        kernel = Kernel(fn)

        def wrapper(x, eps=1e-6):
            output = torch.full_like(x, float('nan'))
            kernel.run(x, output, eps)
            return output

        return types.SimpleNamespace(l2norm_fwd=wrapper, l2norm_fwd_kernel=kernel)

    valid_module = launched_module(valid_candidate)
    valid_harness = types.SimpleNamespace(load_module=lambda: valid_module, reference=reference)
    with checks.checked_modules(valid_harness):
        checked = valid_harness.load_module()
        checked.l2norm_fwd(torch.randn(512, 128))
        checked.l2norm_fwd(torch.randn(512, 128))
    assert calls.count(32768) == 1 and calls.count(32769) == 1
    calls.clear()
    module = launched_module(candidate)
    original_wrapper = module.l2norm_fwd
    harness = types.SimpleNamespace(load_module=lambda: module, reference=reference)
    with checks.checked_modules(harness):
        checked = harness.load_module()
        with pytest.raises(AssertionError):
            checked.l2norm_fwd(torch.randn(512, 128))
    assert 32768 in calls
    assert module.l2norm_fwd is original_wrapper


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
