"""Fused MoE kernel using Triton, adapted from vLLM fused_moe.py.

This is the main MoE GEMM kernel that multiplies tokens by their assigned
expert weight matrices using sorted token IDs and expert IDs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Fused MoE GEMM kernel. Each program computes a [BLOCK_SIZE_M, BLOCK_SIZE_N]
    tile of output C = A @ B^T (per expert), with optional routing weight
    multiplication.

    A: input activations [M, K]
    B: expert weights [E, N, K] (transposed layout: K is inner dim)
    C: output [EM, N]
    sorted_token_ids: token ordering sorted by expert [EM]
    expert_ids: expert index for each block [num_blocks]
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Write zeros for invalid expert
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        tl.store(c_ptrs, accumulator, mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _prepare_moe_routing(topk_ids, num_experts, block_size_m):
    """Prepare sorted_token_ids, expert_ids, num_tokens_post_padded for MoE kernel."""
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Count tokens per expert
    flat_ids = topk_ids.flatten()
    tokens_per_expert = []
    expert_token_lists = []
    for e in range(num_experts):
        mask = (flat_ids == e)
        indices = mask.nonzero(as_tuple=False).flatten()
        tokens_per_expert.append(len(indices))
        expert_token_lists.append(indices)

    # Build sorted_token_ids and expert_ids
    sorted_token_ids_list = []
    expert_ids_list = []
    for e in range(num_experts):
        indices = expert_token_lists[e]
        n = len(indices)
        if n == 0:
            continue
        # Pad to multiple of block_size_m
        padded_n = ((n + block_size_m - 1) // block_size_m) * block_size_m
        padded_indices = torch.full((padded_n,), num_tokens * topk, device=device, dtype=torch.int64)
        padded_indices[:n] = indices
        sorted_token_ids_list.append(padded_indices)
        num_blocks = padded_n // block_size_m
        expert_ids_list.extend([e] * num_blocks)

    if len(sorted_token_ids_list) == 0:
        sorted_token_ids = torch.full((block_size_m,), num_tokens * topk, device=device, dtype=torch.int64)
        expert_ids = torch.zeros(1, device=device, dtype=torch.int32)
        num_tokens_post_padded = torch.tensor([block_size_m], device=device, dtype=torch.int32)
    else:
        sorted_token_ids = torch.cat(sorted_token_ids_list)
        expert_ids = torch.tensor(expert_ids_list, device=device, dtype=torch.int32)
        num_tokens_post_padded = torch.tensor([len(sorted_token_ids)], device=device, dtype=torch.int32)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


def fused_moe(
    input: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    mul_routed_weight: bool = False,
) -> torch.Tensor:
    """
    Fused MoE GEMM.

    Args:
        input: [M, K] input activations
        expert_weights: [E, N, K] expert weight matrices
        topk_ids: [M, topk] expert assignments per token
        topk_weights: [M*topk] routing weights (optional)
        mul_routed_weight: whether to multiply by routing weights
    Returns:
        output: [M*topk, N]
    """
    assert input.dim() == 2
    assert expert_weights.dim() == 3
    M, K = input.shape
    E, N, K2 = expert_weights.shape
    assert K == K2
    topk = topk_ids.shape[1]

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_moe_routing(
        topk_ids, E, BLOCK_SIZE_M
    )

    num_valid_tokens = M * topk
    output = torch.zeros(num_valid_tokens, N, device=input.device, dtype=input.dtype)

    if topk_weights is None:
        topk_weights = torch.ones(num_valid_tokens, device=input.device, dtype=torch.float32)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    fused_moe_kernel[grid](
        input,
        expert_weights,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        input.stride(0),
        input.stride(1),
        expert_weights.stride(0),
        expert_weights.stride(2),
        expert_weights.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=topk,
        compute_type=tl.float16,
    )
    return output
