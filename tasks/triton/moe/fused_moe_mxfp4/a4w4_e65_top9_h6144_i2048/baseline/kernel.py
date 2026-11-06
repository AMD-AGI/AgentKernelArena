# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 (a4w4) routed MoE built from AITER's Triton MXFP4 fused-MoE kernels.

The sequence is the two-stage vLLM-style Triton MoE that AITER's MXFP4 kernels serve
(op_tests/triton_tests/moe/test_moe_mx.py): align the (token, slot) pairs to the expert tiles,
quantize the bfloat16 activations to MXFP4, stage 1 (gate/up GEMM with the fused SiLU(gate) * up
epilogue on the float32 accumulator, bfloat16 output), quantize that intermediate to MXFP4, stage 2
(down GEMM scaled by the routing weight, bfloat16 per (token, slot)), then add the top_k slots of each
token in float32 (reduce_grouped).

Launchers: fused_moe_mxfp4_silu (aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py) and fused_moe_mxfp4
(aiter/ops/triton/moe/moe_op_mxfp4.py), reduced to this fixed format (MXFP4 activations and weights,
unit per-tensor scales, no swizzled scales) and without the logger. Tile configuration: AITER's gfx950
MX_FP4 MoE table (aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json), selected by token count as
get_optimal_moe_config does (aiter/ops/triton/utils/moe_config_utils.py).
"""
import torch
import triton
import triton.language as tl

from .moe_op_mxfp4 import _fused_moe_kernel_mxfp4
from .moe_op_mxfp4_silu_fused import _fused_moe_kernel_mxfp4_silu
from .quant import dynamic_mxfp4_quant
from .reduce import reduce_grouped
from .routing import moe_align_block_size

# aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json
_MOE_MX_FP4_CONFIGS = {
    "small_M": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "num_warps": 4,
                "num_stages": 4, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    "medium_M": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 8,
                 "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    "large_M": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8,
                "num_stages": 4, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
}
# aiter/ops/triton/utils/moe_config_utils.py
M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024
# aiter/ops/triton/utils/device_info.py get_num_xcds() for gfx942/gfx950
NUM_XCDS = 8
MXFP4_QUANT_BLOCK_SIZE = 32


def get_optimal_moe_config(M):
    if M < M_THRESHOLD_SMALL:
        return dict(_MOE_MX_FP4_CONFIGS["small_M"])
    if M < M_THRESHOLD_MEDIUM:
        return dict(_MOE_MX_FP4_CONFIGS["medium_M"])
    return dict(_MOE_MX_FP4_CONFIGS["large_M"])


def _effective_rows(A, sorted_token_ids, top_k, config):
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    return EM


def fused_moe_mxfp4_silu(A, B, C, A_scale, B_scale, A_mx_scale, B_mx_scale, topk_weights, sorted_token_ids,
                         expert_ids, num_tokens_post_padded, num_valid_tokens, mul_routed_weight, top_k, config):
    """Stage 1: C [tokens * top_k, N // 2] = SiLU(gate) * up for A [tokens, K // 2] and B [E, N, K // 2]."""
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    EM = _effective_rows(A, sorted_token_ids, top_k, config)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_silu[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
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
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        SWIZZLE_MX_A=False,  # TODO add swizzle support
        SWIZZLE_MX_B=False,  # TODO add swizzle support
        **config,
    )


def fused_moe_mxfp4(A, B, C, A_scale, B_scale, A_mx_scale, B_mx_scale, topk_weights, sorted_token_ids, expert_ids,
                    num_tokens_post_padded, num_valid_tokens, mul_routed_weight, top_k, config):
    """Stage 2: C [tokens, top_k, N] = (A [tokens * top_k, K // 2] x B [E, N, K // 2]) * routing weight."""
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    EM = _effective_rows(A, sorted_token_ids, top_k, config)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
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
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        SWIZZLE_MX_A=False,  # TODO add swizzle support
        SWIZZLE_MX_B=False,  # TODO add swizzle support
        NUM_XCDS=NUM_XCDS,
        **config,
    )


def _mxfp4_buffers(rows, cols, device):
    """Codes [rows, cols // 2] and column-major scales [rows, cols // 32], as dynamic_mxfp4_quant allocates.

    The scale buffer has one spare row: the GEMM kernels read the scale row of the padding sentinel
    (rows == numel, masked at the store), which must stay inside the allocation.
    """
    codes = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((cols // MXFP4_QUANT_BLOCK_SIZE, rows + 1), dtype=torch.uint8, device=device).T[:rows]
    return codes, scales


def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out) -> None:
    num_tokens, model_dim = hidden_states.shape
    num_experts, gate_up_dim, _ = w1.shape
    inter_dim = gate_up_dim // 2
    top_k = topk_ids.shape[1]
    num_pairs = num_tokens * top_k
    device = hidden_states.device
    config = get_optimal_moe_config(num_tokens)
    a_scale = torch.ones((1,), dtype=torch.float32, device=device)
    b_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts)

    x_fp4, x_scale = _mxfp4_buffers(num_tokens, model_dim, device)
    dynamic_mxfp4_quant(hidden_states, x_fp4, x_scale)

    intermediate = torch.empty((num_pairs, inter_dim), dtype=torch.bfloat16, device=device)
    fused_moe_mxfp4_silu(x_fp4, w1, intermediate, a_scale, b_scale, x_scale, w1_scale, topk_weights,
                         sorted_token_ids, expert_ids, num_tokens_post_padded, num_pairs, False, top_k, config)

    inter_fp4, inter_scale = _mxfp4_buffers(num_pairs, inter_dim, device)
    dynamic_mxfp4_quant(intermediate, inter_fp4, inter_scale)

    expert_out = torch.empty((num_tokens, top_k, model_dim), dtype=torch.bfloat16, device=device)
    fused_moe_mxfp4(inter_fp4, w2, expert_out, a_scale, b_scale, inter_scale, w2_scale, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_padded, num_pairs, True, 1, config)

    reduce_grouped(expert_out.transpose(0, 1), None, out)
