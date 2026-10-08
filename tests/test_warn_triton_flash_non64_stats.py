"""Exercise all declared FlashAttention outputs on legal non-64-width controls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REFERENCE = (Path(__file__).resolve().parents[1] /
             'tasks/instruction2triton/rocmbench/test_flashattention_fwd/_arena_reference.py')


def _reference():
    spec = importlib.util.spec_from_file_location('flash_non64_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(*, bad_width=None, bad_side=None, mutate_input=False):
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(q, k, v, scale, L, m, output):
                width = q.shape[-1]
                seen.append((width, q.stride(-1), k.stride(-1), v.stride(-1)))
                scores = (q @ k.transpose(-1, -2)) * scale
                future = torch.ones(scores.shape[-2:], dtype=torch.bool).triu_(1)
                scores.masked_fill_(future, float('-inf'))
                output.copy_(torch.softmax(scores.float(), -1).to(q.dtype) @ v)

                stats = (q.float() @ k.float().transpose(-1, -2)) * scale
                stats.masked_fill_(future, float('-inf'))
                row_max = stats.max(-1).values
                if bad_width != width or bad_side != 'omit_L':
                    L.copy_(torch.exp(stats - row_max[..., None]).sum(-1).flatten(0, 1))
                if bad_width != width or bad_side != 'omit_m':
                    m.copy_(row_max.flatten(0, 1))
                if bad_width == width and bad_side == 'wrong_L':
                    L.add_(1)
                if bad_width == width and bad_side == 'wrong_m':
                    m.add_(1)
                if bad_width == width and mutate_input:
                    q.add_(1)
            return launch

    def attention(q, k, v, scale):
        output = torch.empty_like(q)
        L = torch.full((q.shape[0] * q.shape[1], q.shape[2]), float('nan'))
        m = torch.full_like(L, float('nan'))
        module.flash_fwd_kernel[(1,)](q, k, v, scale, L, m, output)
        return output

    module = SimpleNamespace(flash_fwd_kernel=Kernel(), attention=attention)
    return module, seen


def test_all_non64_width_controls_check_output_and_side_buffers():
    reference = _reference()
    candidate, seen = _candidate()
    reference.check_width_tail_controls(candidate, 'cpu')
    assert [row[0] for row in seen] == [16, 32, 128]
    assert all(row[1:] == (2, 3, 4) for row in seen)


@pytest.mark.parametrize('bad_side', ['wrong_L', 'wrong_m', 'omit_L', 'omit_m'])
def test_nondefault_scale_stride_control_rejects_bad_stats(bad_side):
    reference = _reference()
    candidate, seen = _candidate(bad_width=64, bad_side=bad_side)
    with pytest.raises((reference.NumericalMismatch, ValueError)):
        reference.check_scale_stride_control(candidate, 'cpu')
    assert seen == [(64, 2, 2, 2)]


def test_control_side_observer_composes_with_correctness_observer():
    reference = _reference()
    candidate, seen = _candidate()
    original_kernel = candidate.flash_fwd_kernel
    with reference.observe_kernel_side_outputs(candidate) as outer:
        reference.check_scale_stride_control(candidate, 'cpu')
        assert outer.latest is not None
    assert candidate.flash_fwd_kernel is original_kernel
    assert seen == [(64, 2, 2, 2)]


@pytest.mark.parametrize('width', [16, 32, 128])
@pytest.mark.parametrize('bad_side', ['wrong_L', 'wrong_m', 'omit_L', 'omit_m'])
def test_non64_width_controls_reject_independent_bad_stats(width, bad_side):
    reference = _reference()
    candidate, seen = _candidate(bad_width=width, bad_side=bad_side)
    with pytest.raises((reference.NumericalMismatch, ValueError)):
        reference.check_width_tail_controls(candidate, 'cpu')
    assert any(row[0] == width for row in seen)


@pytest.mark.parametrize('width', [16, 32, 128])
def test_non64_width_controls_reject_readonly_input_mutation(width):
    reference = _reference()
    candidate, seen = _candidate(bad_width=width, mutate_input=True)
    with pytest.raises(AssertionError, match='modified a width/tail control input'):
        reference.check_width_tail_controls(candidate, 'cpu')
    assert any(row[0] == width for row in seen)
