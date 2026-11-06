# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MXFP4 MoE (a4w4) from AITER's Triton MXFP4 MoE kernels.

The sequence is the two-stage Triton fused MoE that AITER's MXFP4 kernels are written for (AITER
tests each stage on its own in op_tests/triton_tests/moe/test_moe_mx.py): align the (token, slot)
pairs to the expert tiles, quantize the bfloat16 activations to MXFP4, stage 1 (gate/up GEMM with fused SiLU(gate) * up,
bfloat16 output), quantize that intermediate to MXFP4, stage 2 (down GEMM scaled by the routing
weight, bfloat16 per (token, slot)), then add the top_k slots of each token.

Launchers: fused_moe_mxfp4_silu (aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py), fused_moe_mxfp4
(aiter/ops/triton/moe/moe_op_mxfp4.py) and mxfp4_quant (aiter/ops/triton/moe/moe_op_gemm_a4w4.py),
reduced to these fixed formats and writing into preallocated buffers (mxfp4_quant reads the bfloat16
input and converts it in the kernel instead of on the host). Tile configuration: AITER's gfx950 MX_FP4
MoE table (aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json) selected by token count as
get_optimal_moe_config does (aiter/ops/triton/utils/moe_config_utils.py). Numerics follow AITER's
production fused_moe a4w4 path: MXFP4 scales use AITER's default RoundUp rule (baseline/quant.py) and
stage 1 applies SiLU to its float32 accumulator (baseline/moe_op_mxfp4_silu_fused.py).
"""
import torch
import triton
import triton.language as tl

from .moe_op_mxfp4 import _fused_moe_kernel_mxfp4
from .moe_op_mxfp4_silu_fused import _fused_moe_kernel_mxfp4_silu
from .quant import _mxfp4_quant_kernel
from .routing import moe_align_block_size, moe_sum

# aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json ("kpack": 1 is the compiler default and omitted).
_MOE_MX_FP4_CONFIGS = {
    "small_M": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=2, num_warps=4,
                    num_stages=4, waves_per_eu=0, matrix_instr_nonkdim=16),
    "medium_M": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=8,
                     num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),
    "large_M": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=8,
                    num_stages=4, waves_per_eu=0, matrix_instr_nonkdim=16),
}
# aiter/ops/triton/utils/moe_config_utils.py
M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024
# This is fixed by spec for MXFP4. Do not tune this.
MXFP4_QUANT_BLOCK_SIZE = 32
NUM_XCDS = 8  # aiter/ops/triton/utils/device_info.py get_num_xcds() for gfx942/gfx950


def get_optimal_moe_config(M):
    if M < M_THRESHOLD_SMALL:
        return dict(_MOE_MX_FP4_CONFIGS["small_M"])
    if M < M_THRESHOLD_MEDIUM:
        return dict(_MOE_MX_FP4_CONFIGS["medium_M"])
    return dict(_MOE_MX_FP4_CONFIGS["large_M"])


def mxfp4_quant(x, x_fp4, x_scale, block_size_m=16, block_size_n=256):
    """MXFP4 (E2M1) quantization of x [M, N] along N into x_fp4 [M, N // 2] and x_scale [>= M, N // 32]."""
    M, N = x.shape
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    _mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        x_scale,
        x.stride(0),
        x.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        M,
        N,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        EVEN_M_N=(M % block_size_m == 0) and (N % block_size_n == 0),
    )


def _effective_rows(A, sorted_token_ids, top_k, config):
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    return EM


def fused_moe_mxfp4_silu(A, B, C, A_mx_scale, B_mx_scale, topk_weights, sorted_token_ids, expert_ids,
                         num_tokens_post_padded, num_valid_tokens, top_k, config):
    """Stage 1: C [tokens * top_k, N // 2] = SiLU(gate) * up of A [tokens, K // 2] x B [E, N, K // 2]."""
    EM = _effective_rows(A, sorted_token_ids, top_k, config)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_silu[grid](
        A,
        B,
        C,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        A_mx_scale.stride(0),
        A_mx_scale.stride(1),
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT="e2m1",
        B_DTYPE_FORMAT="e2m1",
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        compute_type=tl.bfloat16,
        SWIZZLE_MX_A=False,
        SWIZZLE_MX_B=False,
        **config,
    )


def fused_moe_mxfp4(A, B, C, A_mx_scale, B_mx_scale, topk_weights, sorted_token_ids, expert_ids,
                    num_tokens_post_padded, num_valid_tokens, top_k, config):
    """Stage 2: C [tokens, top_k, N] = topk_weight * (A [tokens * top_k, K // 2] x B [E, N, K // 2])."""
    EM = _effective_rows(A, sorted_token_ids, top_k, config)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4[grid](
        A,
        B,
        C,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_mx_scale.stride(0),
        A_mx_scale.stride(1),
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT="e2m1",
        B_DTYPE_FORMAT="e2m1",
        MUL_ROUTED_WEIGHT=True,
        top_k=top_k,
        compute_type=tl.bfloat16,
        SWIZZLE_MX_A=False,
        SWIZZLE_MX_B=False,
        NUM_XCDS=NUM_XCDS,
        **config,
    )


def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out) -> None:
    num_tokens, hidden_size = hidden_states.shape
    num_experts, gate_up_size, _ = w1.shape
    intermediate_size = gate_up_size // 2
    top_k = topk_ids.shape[1]
    num_pairs = num_tokens * top_k
    device = hidden_states.device
    config = get_optimal_moe_config(num_tokens)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts)

    # The GEMMs read the scale row of the padding sentinel (masked rows); one spare row keeps that read in bounds.
    x_fp4 = torch.empty((num_tokens, hidden_size // 2), dtype=torch.uint8, device=device)
    x_scale = torch.empty((num_tokens + 1, hidden_size // MXFP4_QUANT_BLOCK_SIZE), dtype=torch.uint8, device=device)
    mxfp4_quant(hidden_states, x_fp4, x_scale)

    intermediate = torch.empty((num_pairs, intermediate_size), dtype=torch.bfloat16, device=device)
    fused_moe_mxfp4_silu(x_fp4, w1, intermediate, x_scale, w1_scale, topk_weights, sorted_token_ids, expert_ids,
                         num_tokens_post_padded, num_pairs, top_k, config)

    inter_fp4 = torch.empty((num_pairs, intermediate_size // 2), dtype=torch.uint8, device=device)
    inter_scale = torch.empty((num_pairs + 1, intermediate_size // MXFP4_QUANT_BLOCK_SIZE), dtype=torch.uint8,
                              device=device)
    mxfp4_quant(intermediate, inter_fp4, inter_scale)

    expert_out = torch.empty((num_tokens, top_k, hidden_size), dtype=torch.bfloat16, device=device)
    fused_moe_mxfp4(inter_fp4, w2, expert_out, inter_scale, w2_scale, topk_weights, sorted_token_ids, expert_ids,
                    num_tokens_post_padded, num_pairs, 1, config)

    moe_sum(expert_out, out)
