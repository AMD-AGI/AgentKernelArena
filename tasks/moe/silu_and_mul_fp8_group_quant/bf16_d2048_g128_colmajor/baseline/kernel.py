# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiLU-and-mul with per-token-group FP8 quantization and column-major scales, in Triton.

Vendored from vLLM vllm/model_executor/layers/quantization/utils/fp8_utils.py
(_silu_mul_per_token_group_quant_fp8_colmajor and its launcher): the kernel is unchanged; the launcher is
fixed to the float8_e4m3fn range, group size 128, silu (alpha=1, beta=0), no clamp, no ue8m0, and writes
into preallocated outputs.
"""
import triton
import triton.language as tl

GROUP_SIZE = 128
BLOCK_M = 8
EPS = 1e-10
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,  # [M, N]
    y_q_ptr,  # [M, N // 2]
    y_s_ptr,  # [M, (N // 2) // GROUP_SIZE]
    M,  # num tokens
    N,  # intermediate size
    # Stride
    y_s_col_stride: tl.int64,
    # Information for float8
    eps,
    clamp_limit,
    alpha,
    beta,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    # Meta-parameters
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # TODO(varun) : Add expert_ids so we may early-exit no-op thread blocks.
    """
    Each thread block (BLOCK_N) computes [BLOCK_M, GROUP_SIZE] act-mul outputs. Then
    the thread block quantizes the [BLOCK_M, GROUP_SIZE] block of values and fills
    the outputs tensors at the right positions.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m.to(tl.int64) * BLOCK_M
    n_offset = pid_n.to(tl.int64) * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset

    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    # silu & mul — match C++ silu_and_mul: clamp in fp32 then store back to the
    # input dtype, run silu in fp32 then narrow, and do the mul at input
    # precision so HAS_CLAMP True/False share the same multiplication path.
    if HAS_CLAMP:
        act_in = tl.minimum(act_in.to(tl.float32), clamp_limit).to(
            y_ptr.dtype.element_ty
        )
        mul_in = tl.clamp(mul_in.to(tl.float32), -clamp_limit, clamp_limit).to(
            y_ptr.dtype.element_ty
        )
    # Unified gated activation: silu == swigluoai with alpha=1, beta=0.
    #   glu = gate * sigmoid(alpha * gate); y = (up + beta) * glu
    # Keep glu/up at input precision (narrow before the mul) so the alpha=1,
    # beta=0 defaults match the C++ silu_and_mul path bit-for-bit.
    act_in = act_in.to(tl.float32)
    glu = (act_in / (1.0 + tl.exp(-act_in * alpha))).to(y_ptr.dtype.element_ty)
    up = (mul_in.to(tl.float32) + beta).to(y_ptr.dtype.element_ty)
    y = (glu * up).to(tl.float32)

    # quant
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    # store y_q
    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    # store y_s
    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


def run(*, x, out_q, out_scales):
    M, N = x.shape
    N_2 = N // 2
    assert M % GROUP_SIZE == 0 and M % BLOCK_M == 0
    assert N_2 % GROUP_SIZE == 0
    grid = (M // BLOCK_M, N_2 // GROUP_SIZE)
    # out_scales is [n_groups, M]: the column-major storage of the logical [M, n_groups] scales.
    _silu_mul_per_token_group_quant_fp8_colmajor[grid](
        x,
        out_q,
        out_scales,
        M,
        N,
        out_scales.stride(0),
        EPS,
        0.0,
        1.0,
        0.0,
        -FP8_MAX,
        FP8_MAX,
        False,
        False,
        GROUP_SIZE,
        BLOCK_M,
        GROUP_SIZE,
    )
