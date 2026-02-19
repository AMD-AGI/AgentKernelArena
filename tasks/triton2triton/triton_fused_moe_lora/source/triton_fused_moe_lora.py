"""Fused MoE LoRA kernel using Triton, adapted from vLLM fused_moe_lora_op.py.

This kernel fuses the MoE expert routing with LoRA adapter application.
It performs shrink (A) and expand (B) LoRA operations within the MoE
expert computation pipeline, supporting per-expert LoRA weights.

The distributed communication paths (all_gather, all_reduce for
fully_sharded mode) are excluded. Only the local compute path is included.
"""
import torch
import triton
import triton.language as tl


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    n = max(n, 1)
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _get_lora_id(
    lora_ids,
    token_lora_mapping_ptr,
    lora_idx,
    pid_m,
    top_k_num,
    naive_block_assignment: tl.constexpr,
):
    """Returns lora_id"""
    if naive_block_assignment:
        token_idx = pid_m // top_k_num
        return tl.load(token_lora_mapping_ptr + token_idx)
    else:
        return tl.load(lora_ids + lora_idx)


@triton.jit
def _get_expert_id(
    expert_ids_ptr,
    lora_id,
    pid_m,
    stride_el,
    max_loras,
    naive_block_assignment: tl.constexpr,
):
    """Returns expert_id"""
    if naive_block_assignment:
        return tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        return tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)


@triton.jit
def _get_token_offs(
    sorted_token_ids_ptr,
    lora_id,
    pid_m,
    offs,
    stride_tl,
    max_loras,
    num_valid_tokens,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Returns token offsets"""
    if naive_block_assignment:
        return tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        token_ind = stride_tl * lora_id + offs_token_id
        return tl.load(
            sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0
        )


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
    ]
)
def fused_moe_lora_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,
    # Strides
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    lora_idx = tl.program_id(axis=2)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # Get lora_id
    lora_id = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Get token offsets
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // token_mapping_factor * stride_am
        + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        if USE_B_L2_CACHE:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
        else:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


def _get_ptr(lora_weights, device):
    """Build a tensor of data pointers for grouped GEMM."""
    tensor_ptrs = [w.data_ptr() for w in lora_weights]
    return torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)


def _adjust_kernel_inputs(num_active_loras, sorted_token_ids, expert_ids):
    """Helper to adjust kernel inputs when sorted_token_ids is None."""
    if sorted_token_ids is None:
        stride_tl = 0
        stride_el = 0
        grid_lora_dim = 1
    else:
        stride_tl = sorted_token_ids.stride(0)
        stride_el = expert_ids.stride(0)
        grid_lora_dim = num_active_loras
    return grid_lora_dim, stride_tl, stride_el


@torch.inference_mode()
def fused_moe_lora(
    output: torch.Tensor,             # (num_tokens, top_k_num, out_dim)
    qcurr_hidden_states: torch.Tensor, # (num_tokens, K)
    lora_a_stacked: list,              # [(max_loras, num_experts, max_lora_rank, K)]
    lora_b_stacked: list,              # [(max_loras, num_experts, N, max_lora_rank)]
    topk_weights: torch.Tensor,        # (num_tokens, top_k_num)
    sorted_token_ids,                  # (max_loras, _) or None
    expert_ids: torch.Tensor,          # (max_loras, _) or (num_tokens * top_k,)
    num_tokens_post_padded,            # (max_loras,) or None
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: int,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool = False,
    offset: int = 0,
) -> None:
    """Fused MoE + LoRA: shrink then expand, no distributed communication."""
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]

    # Shrink config
    shrink_block_size_m = 64
    shrink_block_size_n = min(64, _next_power_of_2(N))
    shrink_block_size_k = 32
    shrink_group_size_m = 8
    shrink_num_warps = 4
    shrink_num_stages = 3
    shrink_split_k = 1

    # Expand config
    expand_block_size_m = 64
    expand_block_size_n = 64
    expand_block_size_k = max(16, min(32, _next_power_of_2(N)))
    expand_group_size_m = 8
    expand_num_warps = 4
    expand_num_stages = 3

    EM = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens * shrink_block_size_m
    )

    a_intermediate_cache1 = torch.zeros(
        (num_slices, M, top_k_num, max_lora_rank),
        dtype=output.dtype,
        device=device,
    )

    # --- SHRINK ---
    b_ptr_shrink = _get_ptr(lora_a_stacked, device)
    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    w1_lora_a_stacked = lora_a_stacked[0]

    grid_shrink = lambda META: (
        shrink_split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        grid_lora_dim,
    )
    fused_moe_lora_kernel[grid_shrink](
        qcurr_hidden_states,
        b_ptr_shrink,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_a_stacked[0].shape[0],
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(2),
        a_intermediate_cache1.stride(3),
        stride_tl,
        stride_el,
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=False,
        ADD_INPUTS=False,
        USE_B_L2_CACHE=True,
        IS_PRIMARY=True,
        BLOCK_SIZE_M=shrink_block_size_m,
        BLOCK_SIZE_N=shrink_block_size_n,
        BLOCK_SIZE_K=shrink_block_size_k,
        GROUP_SIZE_M=shrink_group_size_m,
        SPLIT_K=shrink_split_k,
        USE_GDC=False,
        launch_pdl=False,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
    )

    # --- EXPAND ---
    b_ptr_expand = _get_ptr(lora_b_stacked, device)
    K_expand = max_lora_rank
    N_expand = w1_output_dim_size

    a_intermediate_cache1_flat = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    grid_lora_dim2, stride_tl2, stride_el2 = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid_expand = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N_expand, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim2,
    )

    out_view = output[:, :, offset: offset + num_slices * N_expand]
    slice_c_size = N_expand * out_view.stride(2)

    fused_moe_lora_kernel[grid_expand](
        a_intermediate_cache1_flat,
        b_ptr_expand,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N_expand,
        K_expand,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_b_stacked[0].shape[0],
        a_intermediate_cache1_flat.stride(0),
        a_intermediate_cache1_flat.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        stride_tl2,
        stride_el2,
        slice_a_size=a_intermediate_cache1_flat.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        USE_B_L2_CACHE=True,
        IS_PRIMARY=False,
        BLOCK_SIZE_M=expand_block_size_m,
        BLOCK_SIZE_N=expand_block_size_n,
        BLOCK_SIZE_K=expand_block_size_k,
        GROUP_SIZE_M=expand_group_size_m,
        SPLIT_K=1,
        USE_GDC=False,
        launch_pdl=False,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
    )
