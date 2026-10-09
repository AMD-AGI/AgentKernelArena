"""Correctness-only probes for public GEMM branches outside the scored workload.

The matrices use small, exactly representable values so the existing numerical
gate is meaningful for BF16, FP8 and scaled paths too. No probe is timed.
"""
from __future__ import annotations

import torch


CONTROL_NAMES = (
    'bf16_column_a',
    'fp32_column_b_strided_output',
    'fp16_both_inputs_strided',
    'fp8_tensor_scale',
    'int8_int32',
    'fp16_block_scale',
    'fp16_block_b_only',
    'leaky_relu',
)


def _values(rows, cols, dtype, device, *, offset=0, scale=1):
    indices = torch.arange(rows * cols, dtype=torch.int32, device=device).reshape(rows, cols)
    return (((indices * 3 + offset) % 7 - 3).float() * scale).to(dtype)


def _column_major(values):
    return values.T.contiguous().T


def _check(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError('GEMM output shape, dtype or device changed')
    if not torch.isfinite(actual).all():
        raise AssertionError('GEMM output contains nonfinite values')
    if actual.dtype in (torch.int8, torch.int32):
        if not torch.equal(actual, expected):
            raise AssertionError('Integer GEMM output differs from the independent oracle')
    else:
        atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (5e-3, 1e-2)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _block_oracle(a, b, a_scale, b_scale, *, group=128):
    rows, reduction = a.shape
    columns = b.shape[1]
    expected = torch.zeros((rows, columns), dtype=torch.float32, device=a.device)
    for k in range(0, reduction, group):
        product = a[:, k:k + group].float() @ b[k:k + group].float()
        if a_scale is not None:
            product *= a_scale[:, k // group, None]
        for n in range(0, columns, group):
            expected[:, n:n + group] += product[:, n:n + group] * b_scale[k // group, n // group]
    return expected


def run_control(module, name, *, device='cuda'):
    """Run the public wrapper and verify the full caller-owned output and inputs."""
    if name not in CONTROL_NAMES:
        raise ValueError(f'Unknown GEMM domain control {name}')
    m, n, k = (17, 145, 256) if 'block' in name else (17, 35, 33)
    dtype = torch.float16
    if name == 'bf16_column_a':
        dtype = torch.bfloat16
    elif name == 'fp32_column_b_strided_output':
        dtype = torch.float32
    elif name == 'int8_int32':
        dtype = torch.int8
    a = _values(m, k, dtype, device)
    b_dtype = module.e4m3_type if name == 'fp8_tensor_scale' else dtype
    b = _values(k, n, b_dtype, device, offset=2,
                scale=.25 if name == 'fp8_tensor_scale' else 1)
    if name == 'fp32_column_b_strided_output':
        a = a + .13
        b = b - .07
    if name in ('bf16_column_a', 'fp16_both_inputs_strided'):
        a = _column_major(a)
    if name in ('fp32_column_b_strided_output', 'fp16_both_inputs_strided'):
        b = _column_major(b)
    out_dtype = torch.int32 if name == 'int8_int32' else dtype
    c = torch.empty_strided((m, n), (1, m), dtype=out_dtype, device=device) if name == 'fp32_column_b_strided_output' else torch.empty((m, n), dtype=out_dtype, device=device)
    a_scale = b_scale = None
    mode = None
    activation = ''
    if name == 'fp8_tensor_scale':
        a_scale = torch.tensor(.5, dtype=torch.float32, device=device)
        b_scale = torch.tensor(.25, dtype=torch.float32, device=device)
        mode = 'tensor'
    elif name in ('fp16_block_scale', 'fp16_block_b_only'):
        mode = 'block'
        a_scale = None if name == 'fp16_block_b_only' else _column_major(
            torch.tensor([[.5, 2.] if row % 2 else [2., .5] for row in range(m)],
                         dtype=torch.float32, device=device))
        b_scale = _column_major(torch.tensor([[.25, 2.], [2., .5]], dtype=torch.float32, device=device))
    elif name == 'leaky_relu':
        activation = 'leaky_relu'
    originals = tuple((tensor, tensor.clone(), tuple(tensor.stride())) for tensor in (a, b, a_scale, b_scale) if tensor is not None)
    if out_dtype.is_floating_point:
        c.fill_(float('nan'))
    else:
        c.fill_(torch.iinfo(out_dtype).min)
    module.matmul(a, b, c, a_scale, b_scale, scale_a8_b8=mode, activation=activation)
    if mode == 'block':
        expected = _block_oracle(a, b, a_scale, b_scale)
    else:
        expected = a.float() @ b.float()
        if mode == 'tensor':
            expected *= a_scale * b_scale
    if activation == 'leaky_relu':
        shifted = expected + 1
        expected = torch.where(shifted >= 0, shifted, shifted * .01)
    _check(c, expected.to(out_dtype))
    for tensor, original, stride in originals:
        if tensor.shape != original.shape or tensor.dtype != original.dtype or tuple(tensor.stride()) != stride or not torch.equal(tensor, original):
            raise AssertionError('GEMM changed a caller-owned input or scale')
