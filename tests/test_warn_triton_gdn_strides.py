"""CPU contract checks for fused GDN gating's public strided inputs."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_fused_gdn_gating'


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    g = -A_log.float().exp() * F.softplus(x, beta=beta, threshold=threshold)
    return g.unsqueeze(0), torch.sigmoid(b.float()).to(b.dtype).unsqueeze(0)


@pytest.fixture
def modules(monkeypatch):
    launches = []

    class FakeKernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(g, beta_output, A_log, a, b, dt_bias, seq_len, heads,
                       beta, threshold, block_heads, **kwargs):
                assert seq_len == 1 and heads == a.shape[1] and block_heads == 8
                assert all(value.is_contiguous() for value in (A_log, a, b, dt_bias))
                launches.append((grid, (A_log, a, b, dt_bias)))
                expected = reference(A_log, a, b, dt_bias, beta, threshold)
                g.copy_(expected[0])
                beta_output.copy_(expected[1])
            return launch

    triton = types.ModuleType('triton')
    triton.jit = FakeKernel
    triton.cdiv = lambda n, d: (n + d - 1) // d
    tl = types.ModuleType('triton.language')
    tl.constexpr = object()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', tl)
    triton.language = tl
    source = load_file(TASK / 'source/triton_fused_gdn_gating.py', 'gdn_source_under_test')
    checks = load_file(TASK / '_arena_checks.py', 'gdn_checks_under_test')
    return source, checks, launches


def inputs():
    torch.manual_seed(31)
    return (torch.linspace(-.6, .5, 11),
            torch.randn(2, 11, dtype=torch.float16),
            torch.randn(2, 11, dtype=torch.float16),
            torch.linspace(-.2, .3, 11))


def flat_read(x):
    if x.ndim == 1:
        strides = (1,)
    else:
        strides = (x.shape[1], 1)
    return torch.as_strided(x, x.shape, strides, storage_offset=x.storage_offset())


def test_scored_contiguous_path_keeps_exact_input_pointers(modules):
    source, checks, launches = modules
    args = inputs()
    state = checks.readonly_state(args)
    actual = source.fused_gdn_gating(*args)
    checks.check_outputs(actual, reference(*args))
    checks.unchanged(args, checks.snapshots(args), state)
    assert len(launches) == 1
    assert all(got.data_ptr() == original.data_ptr()
               for got, original in zip(launches[0][1], args))


@pytest.mark.parametrize('operand', range(4))
def test_each_strided_public_operand_is_normalized_without_caller_mutation(modules, operand):
    source, checks, launches = modules
    args = list(inputs())
    args[operand] = checks.strided_view(args[operand])
    args = tuple(args)
    pristine = checks.snapshots(args)
    state = checks.readonly_state(args)
    actual = source.fused_gdn_gating(*args, beta=.5, threshold=2.0)
    checks.check_outputs(actual, reference(*args, beta=.5, threshold=2.0))
    checks.unchanged(args, pristine, state)
    assert all(value.is_contiguous() for value in launches[0][1])
    assert launches[0][1][operand].data_ptr() != args[operand].data_ptr()
    assert all(launches[0][1][i].data_ptr() == args[i].data_ptr()
               for i in range(4) if i != operand)


@pytest.mark.parametrize('operand', range(4))
def test_unscored_control_rejects_independent_flat_reader_fault(modules, operand):
    source, checks, _ = modules
    class Harness:
        reference = staticmethod(reference)
        load_module = staticmethod(lambda: source)
    args = inputs()
    original = source.fused_gdn_gating

    def wrong(*values, beta=1.0, threshold=20.0):
        values = list(values)
        if not values[operand].is_contiguous():
            values[operand] = flat_read(values[operand])
        return reference(*values, beta=beta, threshold=threshold)

    source.fused_gdn_gating = wrong
    try:
        # The old scored inputs and existing contiguous tail control admit it.
        checks.check_outputs(wrong(*args), reference(*args))
        harness = Harness()
        with checks.checked_modules(harness):
            checked = harness.load_module()
            with pytest.raises(AssertionError, match='GDN gating .* mismatch|Not equal|Tensor-likes are not close'):
                checked.fused_gdn_gating(*args)
    finally:
        source.fused_gdn_gating = original
    assert source.fused_gdn_gating is original


def test_checker_rejects_write_to_padding_outside_logical_view(modules):
    _, checks, _ = modules
    x = checks.strided_view(inputs()[1])
    pristine = checks.snapshots((x,))
    state = checks.readonly_state((x,))
    assert torch.equal(x, pristine[0])
    full = torch.as_strided(x, (x.untyped_storage().nbytes() // x.element_size(),),
                            (1,), storage_offset=0)
    full[1] = 42
    assert torch.equal(x, pristine[0])
    with pytest.raises(AssertionError, match='storage or layout'):
        checks.unchanged((x,), pristine, state)


def test_state_restoration_recovers_padding_and_layout_after_failure(modules):
    _, checks, _ = modules
    x = checks.strided_view(inputs()[1])
    pristine = checks.snapshots((x,))
    state = checks.readonly_state((x,))
    full = torch.as_strided(x, (x.untyped_storage().nbytes() // x.element_size(),),
                            (1,), storage_offset=0)
    full[1] = 42
    x.transpose_(0, 1)
    with pytest.raises(AssertionError):
        checks.unchanged((x,), pristine, state)
    checks.restore_storage((x,), state)
    checks.unchanged((x,), pristine, state)


def test_honest_wrapper_passes_scored_and_all_unscored_controls(modules):
    source, checks, launches = modules
    class Harness:
        reference = staticmethod(reference)
        load_module = staticmethod(lambda: source)
    harness = Harness()
    with checks.checked_modules(harness):
        checked = harness.load_module()
        actual = checked.fused_gdn_gating(*inputs())
        checks.check_outputs(actual, reference(*inputs()))
    # Original scored call, existing masked-tail diagnostic, and one case for
    # each independently strided caller operand all reached the fake launch.
    assert len(launches) == 6
