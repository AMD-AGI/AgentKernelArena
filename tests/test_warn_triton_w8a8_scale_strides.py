"""CPU controls for the block-GEMM wrapper's accepted scale strides."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_w8a8_block_int8_matmul"
spec = importlib.util.spec_from_file_location("w8a8_scale_stride_checks", TASK / "_arena_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def cpu_reference(a, b, sa, sb, blocks, dtype):
    block_n, block_k = blocks
    k_groups = torch.arange(a.shape[1]) // block_k
    n_groups = torch.arange(b.shape[0]) // block_n
    a_dequantized = a.float() * sa.float()[:, k_groups]
    b_dequantized = b.float() * sb.float()[n_groups][:, k_groups]
    return (a_dequantized @ b_dequantized.T).to(dtype)


def assume_contiguous_storage(scale):
    strides = []
    for index in range(scale.ndim):
        stride = 1
        for extent in scale.shape[index + 1:]:
            stride *= extent
        strides.append(stride)
    return torch.as_strided(scale, scale.shape, tuple(strides))


def inputs():
    a = ((torch.arange(3 * 67).reshape(3, 67) % 7) - 3).to(torch.int8)
    b = ((torch.arange(65 * 67).reshape(65, 67) % 5) - 2).to(torch.int8)
    sa = torch.tensor([[.5, 1.25], [.75, .25], [1.5, .5]], dtype=torch.float32)
    sb = torch.tensor([[.75, 1.5], [.25, .5]], dtype=torch.float32)
    return a, b, sa, sb


def run_checked(mode):
    calls = []

    def candidate(a, b, sa, sb, blocks, output_dtype=None):
        sa_bad, sb_bad = not sa.is_contiguous(), not sb.is_contiguous()
        calls.append((sa_bad, sb_bad))
        if mode == "wrong_as" and sa_bad:
            sa = assume_contiguous_storage(sa)
        elif mode == "wrong_bs" and sb_bad:
            sb = assume_contiguous_storage(sb)
        elif mode == "wrong_both" and sa_bad and sb_bad:
            sa = assume_contiguous_storage(sa)
            sb = assume_contiguous_storage(sb)
        rows = a.reshape(-1, a.shape[-1])
        sa_rows = sa.reshape(-1, sa.shape[-1])
        output = cpu_reference(rows, b, sa_rows, sb, blocks, output_dtype or torch.float16)
        return output.reshape(*a.shape[:-1], b.shape[0])

    module = SimpleNamespace(w8a8_block_int8_matmul=candidate)
    harness = SimpleNamespace(load_module=lambda: module,
                              reference_w8a8_block_int8_matmul=cpu_reference)
    original_loader = harness.load_module
    failure = None
    try:
        with checks.checked_modules(harness):
            checked = harness.load_module()
            checked.w8a8_block_int8_matmul(*inputs(), [64, 64], output_dtype=torch.float16)
    except AssertionError as exc:
        failure = exc
    assert harness.load_module is original_loader
    assert module.w8a8_block_int8_matmul is candidate
    return calls, failure


def test_padded_scales_change_both_physical_strides_but_not_values():
    _, _, sa, sb = inputs()
    for original in (sa, sb):
        padded = checks.padded_scale_view(original)
        assert torch.equal(padded, original)
        assert padded.stride(-1) != original.stride(-1)
        assert padded.stride(-2) != original.stride(-2)
        assert not padded.is_contiguous()
        assert torch.all(padded._base[..., 1] == 3.5)


def test_positive_controls_cover_each_scale_layout_independently():
    calls, failure = run_checked("good")
    assert failure is None
    assert calls == [(False, False)] * 4 + [
        (True, False), (False, True), (True, True)
    ]


@pytest.mark.parametrize("mode,first_bad_call", [
    ("wrong_as", 5), ("wrong_bs", 6), ("wrong_both", 7),
])
def test_wrong_stride_candidate_fails_after_contiguous_controls(mode, first_bad_call):
    calls, failure = run_checked(mode)
    assert failure is not None
    assert len(calls) == first_bad_call
    assert "Mismatched elements:" in str(failure)
