"""Fused MoE kernel with GPTQ/AWQ 4-bit/8-bit quantized weights, adapted from vLLM."""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
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
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    Fused MoE GEMM kernel for GPTQ/AWQ quantized weights.
    Supports 4-bit (INT4) and 8-bit (INT8) weight-only quantization with
    scales and optional zero points.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Consecutive programs cover N tiles of the same routed-token block.  Unlike
    # a dense GEMM, neighboring M blocks usually belong to different experts,
    # so the usual grouped-M ordering cannot reuse their weights.
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        if N % BLOCK_SIZE_N == 0:
            c_mask = token_mask[:, None]
        else:
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if N % BLOCK_SIZE_N == 0:
        offs_bn = offs_cn
    else:
        offs_bn = offs_cn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if block_k_diviable:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & k_mask.T,
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=k_mask, other=0)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        # All production groups in this kernel are aligned to a K tile.  Load
        # scale/ZP once per output column and broadcast along K, instead of
        # issuing BLOCK_SIZE_K duplicate loads for the same quantization group.
        if group_size >= BLOCK_SIZE_K and group_size % BLOCK_SIZE_K == 0:
            scale_group = (BLOCK_SIZE_K * k) // group_size
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + scale_group * stride_bsk
                + offs_bn * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs).to(tl.float32)

            if has_zp and use_int4_w4a16:
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + scale_group * stride_bzk
                    + (offs_bn // 2) * stride_bzn
                )
                b_zp = tl.load(b_zp_ptrs)
                b_zp = ((b_zp >> b_zp_shifter) & 0xF).to(tl.float32)
            elif has_zp and use_int8_w8a16:
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + scale_group * stride_bzk
                    + offs_bn * stride_bzn
                )
                b_zp = tl.load(b_zp_ptrs).to(tl.float32)
        else:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_k_true * stride_bsk
                + offs_bn[None, :] * stride_bsn
            )
            if block_k_diviable:
                b_scale = tl.load(b_scale_ptrs).to(tl.float32)
            else:
                b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            if has_zp and use_int4_w4a16:
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + offs_k_true * stride_bzk
                    + (offs_bn[None, :] // 2) * stride_bzn
                )
                if block_k_diviable:
                    b_zp = tl.load(b_zp_ptrs)
                else:
                    b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=0)
                b_zp = ((b_zp >> b_zp_shifter) & 0xF).to(tl.float32)
            elif has_zp and use_int8_w8a16:
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + offs_k_true * stride_bzk
                    + offs_bn[None, :] * stride_bzn
                )
                if block_k_diviable:
                    b_zp = tl.load(b_zp_ptrs).to(tl.float32)
                else:
                    b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    if N % BLOCK_SIZE_N == 0:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _prepare_moe_routing_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    block_size_m: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    """Stable expert grouping and padding in one asynchronous device pass."""
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    token_mask = token_offsets < num_tokens
    expert_for_token = tl.load(
        topk_ids_ptr + token_offsets, mask=token_mask, other=-1
    )
    padding_offsets = tl.arange(0, block_size_m)
    expert_block_offsets = tl.arange(0, MAX_BLOCKS_PER_EXPERT)
    output_offset = 0

    for expert in tl.static_range(0, num_experts):
        is_expert = token_mask & (expert_for_token == expert)
        token_rank = tl.cumsum(is_expert.to(tl.int32), axis=0) - 1
        token_count = tl.sum(is_expert.to(tl.int32), axis=0)
        padded_count = ((token_count + block_size_m - 1) // block_size_m) * block_size_m
        num_blocks = padded_count // block_size_m

        # The prefix rank makes this the same stable ordering produced by
        # nonzero() in the original routing path.
        tl.store(
            sorted_token_ids_ptr + output_offset + token_rank,
            token_offsets,
            mask=is_expert,
        )
        tl.store(
            sorted_token_ids_ptr + output_offset + token_count + padding_offsets,
            num_tokens,
            mask=padding_offsets < padded_count - token_count,
        )
        tl.store(
            expert_ids_ptr + output_offset // block_size_m + expert_block_offsets,
            expert,
            mask=expert_block_offsets < num_blocks,
        )
        output_offset += padded_count

    tl.store(num_tokens_post_padded_ptr, output_offset)


def _prepare_moe_routing(topk_ids, num_experts, block_size_m):
    """Prepare routing metadata without per-expert PyTorch launches or syncs."""
    flat_ids = topk_ids.flatten()
    num_tokens = flat_ids.numel()
    max_num_tokens_padded = num_tokens + num_experts * (block_size_m - 1)
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_size_m)

    sorted_token_ids = torch.empty(
        max_num_tokens_padded, device=topk_ids.device, dtype=torch.int32
    )
    expert_ids = torch.empty(
        max_num_blocks, device=topk_ids.device, dtype=torch.int32
    )
    num_tokens_post_padded = torch.empty(
        1, device=topk_ids.device, dtype=torch.int32
    )

    block_tokens = triton.next_power_of_2(num_tokens)
    max_blocks_per_expert = triton.next_power_of_2(
        triton.cdiv(num_tokens, block_size_m)
    )
    route_num_warps = min(4, max(1, block_tokens // 64))
    _prepare_moe_routing_kernel[(1,)](
        flat_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_tokens=num_tokens,
        num_experts=num_experts,
        block_size_m=block_size_m,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS_PER_EXPERT=max_blocks_per_expert,
        num_warps=route_num_warps,
        num_stages=1,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def fused_moe_gptq_awq(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    mul_routed_weight: bool = False,
    group_size: int = 128,
    use_int4: bool = True,
) -> torch.Tensor:
    """
    Fused MoE with GPTQ/AWQ quantized weights.

    Args:
        input: [M, K] fp16 activations
        qweight: [E, K//pack_factor, N] packed quantized weights
        scales: [E, K//group_size, N] dequant scales
        zeros: [E, K//group_size, N//pack_factor] or None for zero points
        topk_ids: [M, topk] expert assignments
        topk_weights: [M*topk] routing weights
        mul_routed_weight: apply routing weight to output
        group_size: quantization group size
        use_int4: True for 4-bit, False for 8-bit
    Returns:
        output: [M*topk, N]
    """
    M, K = input.shape
    E = qweight.shape[0]
    N = scales.shape[2]
    topk = topk_ids.shape[1]

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64 if N == 64 else 128
    BLOCK_SIZE_K = 32

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_moe_routing(
        topk_ids, E, BLOCK_SIZE_M
    )

    num_valid_tokens = M * topk
    output = torch.zeros(num_valid_tokens, N, device=input.device, dtype=input.dtype)

    if topk_weights is None:
        topk_weights = torch.ones(num_valid_tokens, device=input.device, dtype=torch.float32)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    has_zp = zeros is not None
    block_k_diviable = (K % BLOCK_SIZE_K == 0)

    fused_moe_kernel_gptq_awq[grid](
        input, qweight, output, scales,
        zeros if has_zp else scales,  # dummy ptr when no zp
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        input.stride(0), input.stride(1),
        qweight.stride(0), qweight.stride(1), qweight.stride(2),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1), scales.stride(2),
        zeros.stride(0) if has_zp else 0,
        zeros.stride(1) if has_zp else 0,
        zeros.stride(2) if has_zp else 0,
        block_k_diviable=block_k_diviable,
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=topk,
        compute_type=tl.float16,
        has_zp=has_zp,
        use_int4_w4a16=use_int4,
        use_int8_w8a16=not use_int4,
        num_warps=4,
        num_stages=2,
    )
    return output
