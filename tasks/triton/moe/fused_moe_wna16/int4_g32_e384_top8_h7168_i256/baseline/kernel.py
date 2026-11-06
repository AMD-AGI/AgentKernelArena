# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts with int4 weight-only (WNA16) GPTQ/AWQ-style weights, from vLLM's fused_experts_impl.

The sequence of vLLM's fused_experts_impl (vllm/model_executor/layers/fused_moe/fused_moe.py) for
use_int4_w4a16=True, block_shape=[0, 32] on ROCm, where should_moe_wna16_use_cuda() is false and
dispatch_fused_moe_kernel always launches the Triton fused_moe_kernel_gptq_awq:

1. moe_align_block_size groups the token slots by expert, padded to BLOCK_SIZE_M;
2. the gate/up GEMM (w13, 512 outputs) into intermediate_cache1, without the router weight;
3. silu_and_mul into intermediate_cache2;
4. the down GEMM (w2) into intermediate_cache3 with the router weight applied;
5. moe_sum over the top-k slots into the output.

The tile configuration is try_get_optimal_moe_config: vLLM's tuned table for this shape on MI355X
(configs/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json), entry closest to the token
count, used for both GEMMs. moe_align_block_size, silu_and_mul and moe_sum are CUDA ops in vLLM and are
Triton kernels here (moe_align_block_size.py, moe_ops.py).
"""
import torch
import triton.language as tl

from .fused_moe import invoke_fused_moe_wna16_triton_kernel
from .moe_align_block_size import moe_align_block_size
from .moe_ops import moe_sum, silu_and_mul

# configs/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json (triton_version key dropped,
# as get_moe_configs does).
TUNED_CONFIGS = {
    1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
        "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
        "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
        "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
        "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    24: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16},
    48: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16},
    96: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
         "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16},
    128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
          "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
          "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
          "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 32},
    1024: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
           "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 32},
    1536: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
           "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 32},
    2048: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
           "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16},
    3072: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
           "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16},
    4096: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
           "num_warps": 4, "num_stages": 2, "matrix_instr_nonkdim": 16},
    8192: {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 2, "SPLIT_K": 1,
           "num_warps": 8, "num_stages": 2, "matrix_instr_nonkdim": 32},
}


def try_get_optimal_moe_config(M: int) -> dict[str, int]:
    configs = TUNED_CONFIGS
    # If an optimal configuration map has been found, look up the optimal config
    return configs[min(configs.keys(), key=lambda x: abs(x - M))]


def run(*, hidden_states, w13_qweight, w13_scale, w2_qweight, w2_scale, topk_weights, topk_ids, out):
    # Check constraints.
    assert hidden_states.size(1) // 2 == w13_qweight.size(2), "Hidden size mismatch"
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w13_qweight.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2_qweight.stride(-1) == 1, "Stride of last dimension must be 1"

    num_tokens = hidden_states.size(0)
    E, N, _ = w13_qweight.size()
    K = w2_qweight.size(1)
    global_num_experts = E
    top_k_num = topk_ids.size(1)
    group_size = hidden_states.size(1) // w13_scale.size(2)
    block_shape = [0, group_size]

    M = num_tokens
    config = try_get_optimal_moe_config(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    compute_type = tl.bfloat16

    # The int4 path never takes vLLM's naive (unaligned) block assignment.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], global_num_experts
    )

    invoke_fused_moe_wna16_triton_kernel(
        hidden_states,
        w13_qweight,
        intermediate_cache1,
        w13_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k_num,
        config,
        compute_type=compute_type,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=block_shape,
    )

    silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_wna16_triton_kernel(
        intermediate_cache2,
        w2_qweight,
        intermediate_cache3,
        w2_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=compute_type,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=block_shape,
    )

    moe_sum(intermediate_cache3, out)
