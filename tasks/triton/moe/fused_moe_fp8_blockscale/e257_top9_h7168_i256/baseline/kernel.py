# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 block-scaled routed MoE layer from AITER's Triton kernels, in the arena run() interface.

The layer chains AITER's Triton kernels in the order of a two-stage fused-MoE runner:

1. fused_flatten_fp8_group_quant: per-token 1 x 128 fp8 (e4m3fn) quantization of the hidden states;
2. moe_align_block_size: sort the (token, slot) pairs by expert, padded to BLOCK_SIZE_M;
3. fused_moe (use_fp8_w8a8, block_shape [128, 128]): grouped gate/up GEMM, bfloat16 result;
4. fused_reduce_act_mul_fp8_group_quant: silu(gate) * up, requantized per 1 x 128 to fp8;
5. fused_moe: grouped down GEMM, multiplied by the routing weight of each (token, slot);
6. reduce_grouped: sum of the top-k expert rows of every token in float32.

The tile configuration is AITER's tuned table for gfx950 FP8_W8A8 MoE
(aiter/ops/triton/configs/moe/gfx950-MOE-FP8_W8A8.json), selected by the number of
tokens as in get_optimal_moe_config (aiter/ops/triton/utils/moe_config_utils.py).
"""
import torch
import triton.language as tl

from .moe_align_block_size import moe_align_block_size
from .moe_op import fused_moe
from .quant import fused_act_mul_fp8_group_quant, fused_flatten_fp8_group_quant
from .reduce import reduce_grouped

# aiter/ops/triton/utils/moe_config_utils.py
M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

# aiter/ops/triton/configs/moe/gfx950-MOE-FP8_W8A8.json
GFX950_MOE_FP8_W8A8 = {
    "small_M": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 2,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "medium_M": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 2,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "large_M": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
}

# Weight block shape [block_n, block_k] and activation group size of the block-scaled checkpoint.
BLOCK_SHAPE = [128, 128]


def get_optimal_moe_config(M: int):
    """get_optimal_moe_config for use_fp8_w8a8=True on gfx950."""
    configs = GFX950_MOE_FP8_W8A8
    if M < M_THRESHOLD_SMALL:
        config = configs["small_M"]
    elif M < M_THRESHOLD_MEDIUM:
        config = configs["medium_M"]
    else:
        config = configs["large_M"]
    return config


def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, gate_up_size, _ = w1.shape
    top_k = topk_ids.shape[1]
    group_k = BLOCK_SHAPE[1]
    config = get_optimal_moe_config(num_tokens)

    # Per-token 1 x 128 fp8 quantization of the activations (fp32 scales).
    a1, a1_scale = fused_flatten_fp8_group_quant(
        hidden_states.view(num_tokens, 1, hidden_size), group_k
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    # Stage 1: [num_tokens, top_k, 2 * intermediate] = dequant(a1) @ dequant(w1[e]).T per routed slot.
    gate_up = torch.empty(
        (num_tokens, top_k, gate_up_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    fused_moe(
        a1, w1, gate_up, a1_scale, w1_scale, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded, False, top_k, tl.bfloat16,
        BLOCK_SHAPE, config,
    )

    # silu(gate) * up, requantized per 1 x 128 to fp8 for the down GEMM.
    a2, a2_scale = fused_act_mul_fp8_group_quant(
        gate_up.view(num_tokens * top_k, gate_up_size), group_k
    )

    # Stage 2: [num_tokens, top_k, hidden] = (dequant(a2) @ dequant(w2[e]).T) * topk_weights per slot.
    expert_out = torch.empty(
        (num_tokens, top_k, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    fused_moe(
        a2, w2, expert_out, a2_scale, w2_scale, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, tl.bfloat16,
        BLOCK_SHAPE, config,
    )

    # Combine: out[t] = sum over the top_k slots, accumulated in float32. The slot axis is the
    # reduction's split axis: x[b, t, :] = expert_out[t, b, :].
    reduce_grouped(expert_out.permute(1, 0, 2), None, out)
