# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Routed experts with the router weight applied in stage 1 (tkw1), from ROCm/aiter's Triton MoE.

The production function, aiter's fused_moe with doweight_stage1=True, is a CK/assembly kernel; this
baseline is the same two-stage computation built from AITER's own Triton MoE kernels:

1. moe_align_block_size_triton (aiter/ops/triton/moe/moe_align_block_size.py) groups the token
   slots by expert, padded to BLOCK_SIZE_M.
2. fused_moe_silu (aiter/ops/triton/moe/moe_op_silu_fused.py) with mul_routed_weight=True: the
   gate/up GEMM, both projections scaled by the router weight in float32, rounded to bfloat16,
   then silu(gate) * up, rounded to bfloat16.
3. fused_moe (aiter/ops/triton/moe/moe_op.py) with mul_routed_weight=False: the down GEMM. With
   top_k = 1 each token has one slot, so its result is the layer output (no top-k sum).

The tile configuration is aiter's get_optimal_moe_config for unquantized weights on gfx950
(configs/moe/gfx950-MOE-DEFAULT.json), selected by the token count, used for all three steps.
"""
import torch
import triton.language as tl

from .moe_align_block_size import moe_align_block_size_triton
from .moe_config_utils import get_optimal_moe_config
from .moe_op import fused_moe
from .moe_op_silu_fused import fused_moe_silu


def run(*, hidden_states, w13, w2, topk_weights, topk_ids, out):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = w13.shape[0]
    intermediate_size = w2.shape[2]
    top_k = topk_ids.shape[1]
    assert top_k == 1, "the down GEMM writes the layer output directly only for top_k == 1"
    device = hidden_states.device

    config = get_optimal_moe_config(M=num_tokens)
    block_size = config["BLOCK_SIZE_M"]

    # Allocation as in aiter's Triton MoE tests (op_tests/triton_tests/moe/test_moe.py).
    numel = topk_ids.numel()
    sorted_token_ids = torch.full((numel + num_experts * (block_size - 1),), numel,
                                  dtype=torch.int32, device=device)
    expert_ids = torch.empty((numel + num_experts,), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
    moe_align_block_size_triton(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
                                num_tokens_post_padded)

    intermediate = torch.empty((numel, intermediate_size), dtype=hidden_states.dtype, device=device)
    fused_moe_silu(hidden_states, w13, intermediate, None, None, None, topk_weights, topk_ids,
                   sorted_token_ids, expert_ids, num_tokens_post_padded, True, top_k, tl.bfloat16,
                   False, False, False, config=config)
    fused_moe(intermediate, w2, out.view(num_tokens, top_k, hidden_size), None, None, None, topk_weights,
              topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, False, 1, tl.bfloat16,
              False, False, False, config=config)
