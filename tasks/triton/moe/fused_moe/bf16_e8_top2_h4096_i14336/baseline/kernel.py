# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Routed-expert MoE layer from AITER's Triton MoE kernels, in the arena run() interface.

The layer runs the standard two-stage sequence of a Triton fused-MoE runner:

1. moe_align_block_size: sort the (token, slot) pairs by expert, padded to BLOCK_SIZE_M;
2. fused_moe_silu: grouped gate/up GEMM with silu(gate) * up fused into the epilogue;
3. fused_moe: grouped down GEMM, multiplied by the routing weight of each (token, slot);
4. reduce_grouped: sum of the top-k expert rows of every token in float32.

The tile configuration is AITER's tuned table for gfx950 bfloat16 MoE
(aiter/ops/triton/configs/moe/gfx950-MOE-DEFAULT.json), selected by the number of
tokens as in get_optimal_moe_config (aiter/ops/triton/utils/moe_config_utils.py).
"""
import torch
import triton.language as tl

from .moe_align_block_size import moe_align_block_size
from .moe_op import fused_moe
from .moe_op_silu_fused import fused_moe_silu
from .reduce import reduce_grouped

# aiter/ops/triton/utils/moe_config_utils.py
M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

# aiter/ops/triton/configs/moe/gfx950-MOE-DEFAULT.json
GFX950_MOE_DEFAULT = {
    "small_M": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "medium_M": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "large_M": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
}


def get_optimal_moe_config(M: int):
    """get_optimal_moe_config for unquantized bfloat16 weights on gfx950."""
    configs = GFX950_MOE_DEFAULT
    if M < M_THRESHOLD_SMALL:
        config = configs["small_M"]
    elif M < M_THRESHOLD_MEDIUM:
        config = configs["medium_M"]
    else:
        config = configs["large_M"]
    return config


def run(*, hidden_states, w13, w2, topk_weights, topk_ids, out):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, gate_up_size, _ = w13.shape
    top_k = topk_ids.shape[1]
    config = get_optimal_moe_config(num_tokens)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    # Stage 1: [num_tokens * top_k, intermediate] = silu(x @ w1.T) * (x @ w3.T) per routed slot.
    intermediate = torch.empty(
        (num_tokens * top_k, gate_up_size // 2), dtype=hidden_states.dtype, device=hidden_states.device
    )
    fused_moe_silu(
        hidden_states, w13, intermediate, None, None, None, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded, False, top_k, tl.bfloat16,
        False, False, False, None, config,
    )

    # Stage 2: [num_tokens, top_k, hidden] = (intermediate @ w2.T) * topk_weights per routed slot.
    expert_out = torch.empty(
        (num_tokens, top_k, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
    )
    fused_moe(
        intermediate, w2, expert_out, None, None, None, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, tl.bfloat16,
        False, False, False, None, config,
    )

    # Combine: out[t] = sum over the top_k slots, accumulated in float32. The slot axis is the
    # reduction's split axis: x[b, t, :] = expert_out[t, b, :].
    reduce_grouped(expert_out.permute(1, 0, 2), None, out)
