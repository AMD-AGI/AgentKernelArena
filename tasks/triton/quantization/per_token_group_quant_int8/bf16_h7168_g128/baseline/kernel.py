# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token-group INT8 quantization (Triton), vendored from vLLM.

Source: vllm/model_executor/layers/quantization/utils/int8_utils.py,
`_per_token_group_quant_int8` and the Triton fallback launch of `per_token_group_quant_int8`.
Adapted from sglang's int8_kernel.py.

Adaptation: the scale and the quotient use IEEE correctly-rounded division (`tl.div_rn`) instead of `/`,
which the ROCm Triton build lowers to an approximate division; see README.md.
"""
import torch
import triton
import triton.language as tl

GROUP_SIZE = 128
EPS = 1e-10


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = tl.div_rn(_absmax, int8_max * 1.0)
    y_q = tl.clamp(tl.div_rn(y, y_s), int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def run(*, x, x_q, x_s) -> None:
    group_size = GROUP_SIZE
    iinfo = torch.iinfo(x_q.dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        EPS,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
