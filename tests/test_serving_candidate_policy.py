"""CPU Torch dispatch checks, not Triton compilation or GPU qualification."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def checks():
    path = Path(__file__).resolve().parents[1] / 'tasks/e2e/qwen3_0_6b_sglang/scripts/checks.py'
    spec = importlib.util.spec_from_file_location('rmsnorm_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('compute', [
    lambda x: x.float(), lambda x: x.square(), lambda x: x.mean(-1),
    lambda x: x.rsqrt(), lambda x: x + x, lambda x: x * x,
    lambda x: x.copy_(x), lambda x: x.cpu().numpy(),
    lambda x: getattr(torch, 'square')(x),
])
def test_tensor_methods_arithmetic_and_aliased_ops_are_not_kernel_rewrites(checks, compute):
    x = torch.ones((2, 8), dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError)):
        with checks.candidate_operations():
            compute(x)


def test_allocation_metadata_views_and_scalar_grid_math_remain_allowed(checks):
    x = torch.ones((2, 8), dtype=torch.float16)
    with checks.candidate_operations():
        for allocate in (torch.empty_like, torch.zeros_like):
            output = allocate(x)
            assert output.shape == x.shape and output.dtype == x.dtype
        assert torch.empty(x.shape, dtype=x.dtype).numel() == x.numel()
        assert torch.zeros(x.shape).stride(0) == x.stride(0)
        assert torch.empty_strided(x.shape, x.stride()).numel() == 16
        assert x.view(-1).numel() == 16
        assert (x.numel() + 255) // 256 == 1


def test_source_policy_counterexample_is_rejected_by_actual_tensor_dispatch(checks, tmp_path):
    # The old AST-only gate permits methods, arithmetic and assignment aliases.
    # The runtime gate must reject their execution even with a declared Triton
    # kernel; a copy-only kernel launched afterwards cannot legitimize the math.
    source = '''import torch
import triton
@triton.jit
def copy_kernel(src, dst): pass
def rms_norm(x, weight, eps=1e-6):
    value = x.float()
    return value * (value.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()
'''
    (tmp_path/'source').mkdir()
    (tmp_path/'source/rmsnorm.py').write_text(source)
    checks.ROOT = tmp_path
    checks.source_policy(target_only=True)
    assert checks.target_kernels() == ['copy_kernel']
    scope = {'torch': torch}
    exec(source[source.index('def rms_norm'):], scope)
    x, weight = torch.ones((2, 8), dtype=torch.float16), torch.ones(8, dtype=torch.float16)
    with pytest.raises(ValueError, match='delegates computation'):
        with checks.candidate_operations():
            scope['rms_norm'](x, weight)


def test_final_compile_requires_guarded_operator_cases_before_gpu_profiling(checks, monkeypatch):
    monkeypatch.setattr(checks, 'target_kernels', lambda: ['kernel'])
    class GuardedCases(Exception): pass
    def operator_checks(lock, *, target_only=False):
        assert target_only is True
        raise GuardedCases
    monkeypatch.setattr(checks, 'operator_checks', operator_checks)
    with pytest.raises(GuardedCases):
        checks.compile_candidate({})


def test_operator_harness_rejects_tensor_math_before_numerical_acceptance(checks, tmp_path, monkeypatch):
    (tmp_path/'source').mkdir()
    (tmp_path/'source/rmsnorm.py').write_text('''import torch
def rms_norm(x, weight, eps):
    value = x.float()
    return (value * (value.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()).to(x.dtype)
''')
    checks.ROOT = tmp_path
    randn = torch.randn
    def cpu_randn(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return randn(*args, **kwargs)
    monkeypatch.setattr(torch, 'randn', cpu_randn)
    with pytest.raises(ValueError, match='delegates computation'):
        checks.operator_checks({'correctness': {'seed': 0, 'rtol': 1e-3, 'atol': 1e-3}}, target_only=True)
