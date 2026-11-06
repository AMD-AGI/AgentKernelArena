# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SiLU * mul + per-group FP8 quantization for the DeepGEMM masked MoE layout.

Adapted from vLLM ``_silu_mul_fp8_quant_deep_gemm`` (batched_deep_gemm_moe.py): one program per
(expert, quantization group) walks the valid tokens of the expert. The rows at or beyond the
expert's token count are zero-filled so the whole output is defined.
"""
import triton
import triton.language as tl

GROUP_SIZE = 128
FP8_MAX = 448.0  # float8_e4m3fn
EPS = 1e-10


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    input_ptr,   # bf16 activations (E, T, 2*H): gate then up
    y_q_ptr,     # fp8 quantized activations (E, T, H)
    y_s_ptr,     # float32 scales, stored (E, G, T) (strides: stride_ys_e, stride_ys_g, stride_ys_t; the kernel's stride_ys_t is 1 and stride_ys_g is T)
    counts_ptr,  # int32 valid tokens per expert (E)
    T,
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    stride_counts_e,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Multiply-by-reciprocal, as in the vLLM kernel, to match PyTorch's tensor/scalar division.
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)

    zero_q = tl.zeros([BLOCK], dtype=tl.float32).to(y_q_ptr.dtype.element_ty)
    zero_s = tl.zeros([1], dtype=tl.float32)
    for t in tl.range(n_tokens, T):
        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, zero_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t + tl.arange(0, 1), zero_s)


def run(*, y, tokens_per_expert, y_q, y_s):
    E, T, H2 = y.shape
    H = H2 // 2
    G = H // GROUP_SIZE
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
    # y_s is (E, G, T) contiguous: the DeepGEMM column-major layout of a logical (E, T, G) scale tensor.
    _silu_mul_fp8_quant_deep_gemm[(E * G,)](
        y, y_q, y_s, tokens_per_expert,
        T,
        H, GROUP_SIZE,
        stride_i_e, stride_i_t, stride_i_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        G * T, 1, T,
        tokens_per_expert.stride(0),
        EPS, -FP8_MAX, FP8_MAX,
        ceil_ue8m0=False,
        BLOCK=GROUP_SIZE,
        NUM_STAGES=4,
        num_warps=1,
    )
