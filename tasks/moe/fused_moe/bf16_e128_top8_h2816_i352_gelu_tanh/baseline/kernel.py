# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert MoE layer in Triton following vLLM's fused_experts_impl (bfloat16, gelu_tanh).

align (Triton) -> gate/up GEMM -> gelu_tanh * up -> down GEMM with the routing weight -> sum over the top-k slots.
The tile configuration is vLLM's default for unquantized bfloat16 weights on ROCm (get_default_config); no tuned
configuration file exists for this expert geometry.
"""
import torch
import triton
import triton.language as tl

from .align_kernels import moe_align_block_size
from .fused_moe_kernel import fused_moe_kernel
from .moe_ops import gelu_tanh_and_mul, moe_sum


def get_default_config(M, E):
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128
    block_n = 64 if M <= 64 else 128
    block_k = 128 if M <= 64 else 64
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1
    num_warps = 4 if M <= 128 else 8
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": 2,
    }


def invoke_fused_moe_kernel(A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            mul_routed_weight, top_k, config, compute_type):
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )


def run(*, hidden_states, w13, w2, topk_weights, topk_ids, out):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13.shape
    top_k = topk_ids.size(1)
    config = get_default_config(num_tokens, num_experts)
    dtype = hidden_states.dtype
    device = hidden_states.device
    compute_type = tl.bfloat16

    intermediate_cache1 = torch.empty((num_tokens, top_k, gate_up), device=device, dtype=dtype)
    intermediate_cache2 = torch.empty((num_tokens * top_k, gate_up // 2), device=device, dtype=dtype)
    intermediate_cache3 = torch.empty((num_tokens, top_k, hidden), device=device, dtype=dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts)

    invoke_fused_moe_kernel(hidden_states, w13, intermediate_cache1, topk_weights, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, False, top_k, config, compute_type)
    gelu_tanh_and_mul(intermediate_cache1.view(-1, gate_up), intermediate_cache2)
    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3, topk_weights, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1, config, compute_type)
    moe_sum(intermediate_cache3, out)
