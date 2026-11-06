# SPDX-License-Identifier: Apache-2.0
"""Routed-expert MoE layer in Triton following SGLang's Triton fused-MoE runner (bfloat16, SiLU-gated).

align (Triton) -> gate/up GEMM -> silu(gate) * up -> down GEMM with the routing weight -> sum over the top-k slots.
The tile configuration is SGLang's default for unquantized weights (get_default_config); the tuned configuration files
shipped with SGLang cover other devices and expert geometries only.
"""
import torch
import triton
import triton.language as tl

from .align_kernels import moe_align_block_size
from .fused_moe_kernel import fused_moe_kernel
from .moe_ops import act_and_mul_triton, moe_sum_reduce_triton


def get_default_config(M, E):
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def invoke_fused_moe_kernel(A, B, C, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            mul_routed_weight, top_k, config, compute_type):
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


def run(*, hidden_states, w13, w2, topk_weights, topk_ids, out):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13.shape
    top_k = topk_ids.shape[1]
    config = get_default_config(num_tokens, num_experts)
    dtype = hidden_states.dtype
    device = hidden_states.device
    compute_type = tl.bfloat16

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts)

    intermediate_cache1 = torch.empty((num_tokens * top_k, gate_up), device=device, dtype=dtype)
    invoke_fused_moe_kernel(hidden_states, w13, intermediate_cache1, topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False, top_k, config, compute_type)

    intermediate_cache2 = torch.empty((num_tokens * top_k, gate_up // 2), device=device, dtype=dtype)
    act_and_mul_triton(intermediate_cache1, intermediate_cache2)

    intermediate_cache3 = torch.empty((num_tokens, top_k, hidden), device=device, dtype=dtype)
    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3, topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1, config, compute_type)

    moe_sum_reduce_triton(intermediate_cache3, out, 1.0)
