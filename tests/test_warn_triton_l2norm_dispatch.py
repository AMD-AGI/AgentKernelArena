"""CPU controls for L2 kernel-dispatch enforcement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


TASK = (Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_fla_l2norm')
spec = importlib.util.spec_from_file_location('l2_dispatch_checks', TASK / '_arena_checks.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
runtime = SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)
compiler = ModuleType('l2_dispatch_compiler')


class CompiledKernel:
    def __init__(self, fn):
        self.src = SimpleNamespace(fn=fn)


compiler.CompiledKernel = CompiledKernel


def normalize(x, eps=1e-6):
    x = x.float()
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class Kernel:
    def __init__(self, *, inert=False):
        self.inert = inert
        self.launches = 0

    def run(self, x, y, eps, *, warmup=False, **_):
        kernel = CompiledKernel(self)
        if not warmup:
            if runtime.launch_enter_hook is not None:
                runtime.launch_enter_hook(None)
            self.launches += 1
            if not self.inert:
                y.copy_(normalize(x, eps))
            if runtime.launch_exit_hook is not None:
                runtime.launch_exit_hook(None)
        return kernel

    def __getitem__(self, _grid):
        return self.run


class Autotuner:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        return self.fn.run(*args, **kwargs)

    def __getitem__(self, _grid):
        return self.run


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    runtime.launch_enter_hook = runtime.launch_exit_hook = None
    monkeypatch.setattr(checks, '_triton_runtime',
                        lambda: (runtime, compiler, Kernel, (Autotuner,)))


def module_for(kind, *, autotuned=False):
    kernel = Kernel(inert=kind == 'inert')
    declared = Autotuner(kernel) if autotuned else kernel

    def wrapper(x, eps=1e-6):
        y = torch.full_like(x, float('nan'))
        if kind != 'bypass':
            declared[(1,)](x, y, eps, warmup=kind == 'compile_only')
        if kind in ('bypass', 'inert', 'compile_only'):
            return normalize(x, eps)
        return y

    return SimpleNamespace(l2norm_fwd=wrapper, l2norm_fwd_kernel=declared), kernel


def harness_for(module):
    return SimpleNamespace(load_module=lambda: module, reference=normalize)


@pytest.mark.parametrize('kind', ['bypass', 'inert'])
def test_correct_output_without_named_kernel_contribution_is_rejected(kind):
    module, kernel = module_for(kind)
    harness = harness_for(module)
    original = module.l2norm_fwd
    with checks.checked_modules(harness):
        wrapped = harness.load_module()
        with pytest.raises(AssertionError, match='declared kernel|independent'):
            wrapped.l2norm_fwd(torch.randn(512, 128))
    assert module.l2norm_fwd is original
    assert not ('run' in vars(kernel))


def test_correct_named_kernel_passes_and_restores_run_on_failure():
    module, kernel = module_for('valid')
    harness = harness_for(module)
    original = module.l2norm_fwd
    x = torch.randn(512, 128)
    with checks.checked_modules(harness):
        result = harness.load_module().l2norm_fwd(x)
        torch.testing.assert_close(result, normalize(x), atol=1e-4, rtol=1e-4)
    assert module.l2norm_fwd is original
    assert not ('run' in vars(kernel))
    assert kernel.launches >= 3


class TimedRun:
    def __init__(self):
        self.bound = False
        self.outputs = None
        self.after_sample = None
        self._fn = None

    def rerun(self):
        return self._fn()


def benchmark(fn, *, timed_run, warmup, repetition, **_):
    assert (warmup, repetition) == (10, 100)
    timed_run._fn = fn
    for _ in range(warmup):
        fn()
    for _ in range(repetition):
        output = fn()
        timed_run.outputs = output
        timed_run.bound = True
        timed_run.after_sample(output)
    return 1.0, {'benchmark_method': 'cuda_event_fallback'}


@pytest.mark.parametrize('kind', ['bypass', 'valid'])
def test_all_reported_samples_require_named_dispatch(kind):
    module, kernel = module_for(kind)
    harness = harness_for(module)
    harness._TimedRun = TimedRun
    args, kwargs = (torch.randn(512, 128),), {}
    mod = module

    def fn():
        mod.l2norm_fwd(*args, **kwargs)

    before = args[0].clone()
    if kind == 'bypass':
        with pytest.raises(AssertionError, match='Measured L2 wrapper bypassed'):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    else:
        _, metadata = checks.checked_benchmark(
            harness, benchmark, fn, warmup=10, repetition=100)
        assert metadata['measured_samples_checked'] == 100
        assert metadata['named_kernel_dispatches_checked'] == 100
    assert torch.equal(args[0], before)
    assert not ('run' in vars(kernel))


def test_one_correct_intermediate_sample_without_named_launch_is_rejected():
    module, kernel = module_for('valid')
    regular = module.l2norm_fwd
    invocation = [0]

    def one_bypass(x, eps=1e-6):
        invocation[0] += 1
        if invocation[0] == 10 + 51:
            return normalize(x, eps)
        return regular(x, eps)

    module.l2norm_fwd = one_bypass
    harness = harness_for(module)
    harness._TimedRun = TimedRun
    args, kwargs = (torch.randn(512, 128),), {}
    mod = module

    def fn():
        mod.l2norm_fwd(*args, **kwargs)

    with pytest.raises(AssertionError, match='Measured L2 wrapper bypassed'):
        checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert invocation[0] == 61
    assert not ('run' in vars(kernel))


def test_alias_of_named_kernel_is_observed_and_restored():
    module, kernel = module_for('valid', autotuned=True)
    alias = module.l2norm_fwd_kernel

    def alias_wrapper(x, eps=1e-6):
        y = torch.full_like(x, float('nan'))
        alias[(1,)](x, y, eps)
        return y

    module.l2norm_fwd = alias_wrapper
    harness = harness_for(module)
    x = torch.randn(512, 128)
    with checks.checked_modules(harness):
        checks.check_output(harness.load_module().l2norm_fwd(x), normalize(x))
    assert not ('run' in vars(kernel))
    assert not ('run' in vars(alias))


def test_same_named_fake_run_object_is_rejected():
    fake_type = type('JITFunction', (), {
        '__module__': 'triton.runtime.jit',
        'run': lambda self, x, y, eps: y.copy_(normalize(x, eps)),
        '__getitem__': lambda self, grid: self.run,
    })
    fake = fake_type()
    module = SimpleNamespace(l2norm_fwd=lambda x, eps=1e-6: normalize(x, eps),
                             l2norm_fwd_kernel=fake)
    harness = harness_for(module)
    with checks.checked_modules(harness):
        with pytest.raises(AssertionError, match='genuine Triton JITFunction'):
            harness.load_module().l2norm_fwd(torch.randn(512, 128))


def test_compile_only_named_run_is_not_counted_as_launch():
    module, kernel = module_for('compile_only', autotuned=True)
    harness = harness_for(module)
    with checks.checked_modules(harness):
        with pytest.raises(AssertionError, match='diagnostic bypassed'):
            harness.load_module().l2norm_fwd(torch.randn(512, 128))
    assert kernel.launches == 0
    assert runtime.launch_enter_hook is runtime.launch_exit_hook is None


def test_compile_only_correct_outputs_cannot_pass_reported_event_sample():
    module, kernel = module_for('compile_only', autotuned=True)
    harness = harness_for(module)
    harness._TimedRun = TimedRun
    args, kwargs = (torch.randn(512, 128),), {}
    mod = module

    def fn():
        mod.l2norm_fwd(*args, **kwargs)

    with pytest.raises(AssertionError, match='Measured L2 wrapper bypassed'):
        checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert kernel.launches == 0
    assert runtime.launch_enter_hook is runtime.launch_exit_hook is None
