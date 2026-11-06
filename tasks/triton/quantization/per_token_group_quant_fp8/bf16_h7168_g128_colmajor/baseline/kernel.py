# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token-group FP8 quantization with column-major scales (Triton), vendored from vLLM.

Source: vllm/model_executor/layers/quantization/utils/fp8_utils.py,
`_per_token_group_quant_fp8_colmajor` and the column-major Triton fallback launch of
`per_token_group_quant_fp8`. Adapted from https://github.com/sgl-project/sglang/pull/2575

The scales are stored column-major: the scale of token `t`, group `g` lives at `x_s[g, t]` of the
contiguous `[num_groups, n_tokens]` output, which is the `[n_tokens, num_groups]` scale matrix with
strides `(1, n_tokens)` that vLLM's launcher passes (`x_s.stride(1)` there is `x_s.stride(0)` here).
"""
import torch
import triton
import triton.language as tl

GROUP_SIZE = 128
EPS = 1e-10


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    # Ensure offset calculation uses int64 for y_s_ptr
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def run(*, x, x_q, x_s) -> None:
    group_size = GROUP_SIZE
    finfo = torch.finfo(x_q.dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_fp8_colmajor[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        x_s.stride(0),
        EPS,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=False,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
