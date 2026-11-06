# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's fused MoE LoRA one-shot kernel (shrink + expand in one launch).

Vendored from vllm/lora/ops/triton_ops/fused_moe_lora_op.py: the kernel
``_fused_moe_lora_one_shot_kernel`` and its host wrapper
``_run_fused_moe_lora_one_shot``, the path ``_fused_moe_lora`` dispatches to for
block-sorted routing when the LoRA is not fully sharded. The LoRA slices arrive
stacked in one tensor, so the kernel offsets by a slice stride instead of
loading per-slice data pointers from a device pointer table.
"""
import torch
import triton
import triton.language as tl

# moe_lora_align_block_size block size of the routing tables (BLOCK_SIZE_M).
BLOCK_SIZE_M = 64


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _fused_moe_lora_one_shot_kernel(
    # ---- pointers ----
    x_ptr,
    A_ptr,
    B_ptr,
    out_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    lora_ids_ptr,
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    num_valid_tokens,
    top_k_num,
    max_loras,
    # ---- strides ----
    stride_xm,
    stride_xk,
    stride_A_slice,
    stride_A_lora,
    stride_A_expert,
    stride_A_r,
    stride_A_k,
    stride_B_slice,
    stride_B_lora,
    stride_B_expert,
    stride_B_n,
    stride_B_r,
    stride_om,
    stride_on,
    stride_tl_,
    stride_el,
    # ---- scalar ----
    slice_n_offset,
    # ---- constexpr (set per call) ----
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    actual_rank: tl.constexpr,
    NPID_FACTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    pid_full = tl.program_id(axis=0)
    pid_m = pid_full // NPID_FACTOR
    pid_n_outer = pid_full % NPID_FACTOR
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    # Resolve lora_id.
    if naive_block_assignment:
        token_idx_for_lora = pid_m // top_k_num
        lora_id = tl.load(token_lora_mapping_ptr + token_idx_for_lora)
    else:
        lora_id = tl.load(lora_ids_ptr + lora_idx)
    if lora_id < 0:
        return
    if lora_id >= max_loras:
        return
    enabled = tl.load(adapter_enabled_ptr + lora_id)
    if enabled == 0:
        return

    if not naive_block_assignment:
        ntpp = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_M >= ntpp:
            return

    # Resolve expert_id.
    if naive_block_assignment:
        expert_id = tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        expert_id = tl.load(
            expert_ids_ptr + ind, mask=ind < max_loras * stride_el, other=-1
        )
    if expert_id < 0:
        return

    # Compute offs_token (flat token ids).
    offs = tl.arange(0, BLOCK_M).to(tl.int64)
    if naive_block_assignment:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_M + offs
        token_ind = stride_tl_ * lora_id + offs_token_id
        offs_token = tl.load(
            sorted_token_ids_ptr + token_ind,
            mask=token_ind < max_loras * stride_tl_,
            other=num_valid_tokens,
        )
    token_mask = offs_token < num_valid_tokens

    # N range owned by this program. Splitting [0, N) into NPID_FACTOR
    # contiguous outer blocks lets us scale parallelism for small batches.
    n_per_outer = tl.cdiv(N, NPID_FACTOR)
    n_lo = pid_n_outer * n_per_outer
    n_hi = tl.minimum((pid_n_outer + 1) * n_per_outer, N)
    if n_lo >= N:
        return

    # Slice pointers.
    cur_A_ptr = A_ptr + slice_id * stride_A_slice
    cur_B_ptr = B_ptr + slice_id * stride_B_slice

    A_base = cur_A_ptr + lora_id * stride_A_lora + expert_id * stride_A_expert
    B_base = cur_B_ptr + lora_id * stride_B_lora + expert_id * stride_B_expert

    # SHRINK: tmp[BLOCK_M, BLOCK_R] = x @ A^T, accumulated in fp32 registers.
    offs_r = tl.arange(0, BLOCK_R)
    rank_mask = offs_r < actual_rank
    # Clamp rank offsets so OOB rows of A / B map to address 0; the mask
    # zeros the loaded values. Required when BLOCK_R > actual_rank
    # (e.g. rank=4 padded to 16) -- without clamping, tl.load would address
    # the next expert's memory.
    safe_offs_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)

    offs_x_row = offs_token // token_mapping_factor
    x_ptrs = x_ptr + offs_x_row[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = A_base + offs_k[:, None] * stride_A_k + safe_offs_r[None, :] * stride_A_r

    tmp = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    if EVEN_K:
        for _ in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=token_mask[:, None], other=0.0)
            a = tl.load(a_ptrs, mask=rank_mask[None, :], other=0.0)
            tmp += tl.dot(x, a)
            x_ptrs += BLOCK_K * stride_xk
            a_ptrs += BLOCK_K * stride_A_k
    else:
        for kb in range(0, K, BLOCK_K):
            k_remain = K - kb
            k_mask = offs_k < k_remain
            x = tl.load(x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=k_mask[:, None] & rank_mask[None, :], other=0.0)
            tmp += tl.dot(x, a)
            x_ptrs += BLOCK_K * stride_xk
            a_ptrs += BLOCK_K * stride_A_k

    tmp_typed = tmp.to(out_ptr.dtype.element_ty)

    # EXPAND: out[tokens, n] += tmp @ B^T, looped over BLOCK_N tiles within
    # this program's [n_lo, n_hi). The (offs_n < n_hi) mask is required
    # whenever BLOCK_N > n_per_outer to keep adjacent outer blocks from
    # writing into each other's columns.
    if MUL_ROUTED_WEIGHT:
        moe_w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0).to(
            tl.float32
        )

    out_slice_base = out_ptr + slice_id * slice_n_offset

    for n_start in range(n_lo, n_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = (offs_n < N) & (offs_n < n_hi)

        b_ptrs = (
            B_base + safe_offs_r[:, None] * stride_B_r + offs_n[None, :] * stride_B_n
        )
        b = tl.load(b_ptrs, mask=rank_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(tmp_typed, b)  # (BLOCK_M, BLOCK_N) fp32
        if MUL_ROUTED_WEIGHT:
            acc = acc * moe_w[:, None]

        out_ptrs = (
            out_slice_base
            + offs_token[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
        if ADD_INPUTS:
            prev = tl.load(out_ptrs, mask=out_mask, other=0.0)
            tl.store(out_ptrs, prev + acc.to(out_ptr.dtype.element_ty), mask=out_mask)
        else:
            tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _run_fused_moe_lora_one_shot(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool,
    block_size_m: int,
    add_inputs: bool = True,
) -> None:
    """Fast-path wrapper: launches one fused shrink+expand kernel.

    `output` has shape `(num_tokens, top_k_num, num_slices * N_per_slice)`.
    When `add_inputs=True` the kernel reads-modifies-writes `output` in place;
    when `add_inputs=False` the kernel overwrites `output` with the LoRA delta
    only (the dual-stream path, which hands it a zeroed buffer).
    """
    num_slices = lora_a_stacked.shape[0]
    device = qcurr_hidden_states.device

    A0 = lora_a_stacked[0]
    B0 = lora_b_stacked[0]
    max_loras_w = A0.shape[0]
    rank = A0.shape[2]
    K = A0.shape[3]
    N_per_slice = B0.shape[2]
    top_k_num = output.shape[1]
    # Upstream reads the route count from topk_weights.numel().
    num_valid_tokens = output.shape[0] * output.shape[1]

    # rank padding is to next pow2 with a floor of 16 (tensor-core minimum
    # K-dim). Beyond 128 the (BLOCK_M, BLOCK_R) accumulator outgrows the
    # register file.
    assert rank <= 128, (
        f"fused_moe_lora_one_shot supports max_lora_rank<=128; got rank={rank}"
    )
    BLOCK_R = max(triton.next_power_of_2(rank), 16)

    num_experts = A0.shape[1]
    EM_grid = sorted_token_ids.shape[1]
    # BLOCK_M must equal moe_lora_align_block_size's block_size.
    assert block_size_m >= 16 and (block_size_m & (block_size_m - 1)) == 0, (
        f"shrink_block_size_m must be a power of 2 and >=16; got {block_size_m}"
    )
    BLOCK_M = block_size_m
    stride_tl_ = sorted_token_ids.stride(0)
    stride_el = expert_ids.stride(0)
    # Upstream: num_active_loras, which is max_loras + 1 (the length of
    # lora_ids) unless active-LoRA specialization is enabled.
    grid_lora_dim = lora_ids.shape[0]

    if EM_grid == 0 or grid_lora_dim == 0 or num_slices == 0:
        return

    token_mapping_factor = 1 if mul_routed_weight else top_k_num

    assert output.dim() == 3, f"output must be 3-D, got {output.shape}"
    assert output.stride(0) == output.shape[1] * output.stride(1), (
        "fused_moe_lora_one_shot requires output.stride(0) == top_k*stride(1); "
        f"got shape={output.shape} strides={output.stride()}"
    )
    out_view = output.view(-1, output.shape[-1])
    M_blocks = triton.cdiv(EM_grid, BLOCK_M)

    # NPID_FACTOR heuristic: scale N-axis parallelism when base CTA count is
    # short of saturating the SM array. Cap by the cost of redundant shrink.
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    base_programs = max(M_blocks * num_slices * grid_lora_dim, 1)
    shrink_ratio = K / max(K + N_per_slice, 1)
    max_npid_by_budget = max(1, int(1.5 / max(shrink_ratio, 1e-3)) + 1)
    target = 2 * sm_count
    if base_programs >= int(1.5 * sm_count):
        npid = 1
    else:
        npid_occ = max(1, min(16, (target + base_programs - 1) // base_programs))
        npid = min(npid_occ, max_npid_by_budget)
    npid = max(1, min(npid, max(1, N_per_slice // 128)))

    if npid > 1:
        block_n, nw, ns = 128, 8, 3
    else:
        block_n, nw, ns = 128, 4, 3

    # BLOCK_K choice: hidden-sized K (>= 256) uses BLOCK_K=128; small K keeps
    # the work_per_expert heuristic.
    if K >= 256:
        block_k = 128
    else:
        work_per_expert = num_valid_tokens / max(num_experts, 1)
        block_k = 128 if work_per_expert >= 16 else 64

    grid = (M_blocks * npid, num_slices, grid_lora_dim)

    _fused_moe_lora_one_shot_kernel[grid](
        qcurr_hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        out_view,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        None,
        lora_ids,
        adapter_enabled,
        N_per_slice,
        K,
        num_valid_tokens,
        top_k_num,
        max_loras_w,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        lora_a_stacked.stride(0),
        A0.stride(0),
        A0.stride(1),
        A0.stride(2),
        A0.stride(3),
        lora_b_stacked.stride(0),
        B0.stride(0),
        B0.stride(1),
        B0.stride(2),
        B0.stride(3),
        out_view.stride(0),
        out_view.stride(1),
        stride_tl_,
        stride_el,
        N_per_slice,
        token_mapping_factor=token_mapping_factor,
        naive_block_assignment=False,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        actual_rank=rank,
        NPID_FACTOR=npid,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ADD_INPUTS=add_inputs,
        num_warps=nw,
        num_stages=ns,
    )


def run(*, hidden_states, lora_a, lora_b, sorted_token_ids, expert_ids, num_tokens_post_padded,
        lora_ids, adapter_enabled, lora_delta):
    # vLLM's dual-stream MoE LoRA path: the delta buffer starts as zeros
    # (torch.zeros_like) and the kernel writes the routes it owns.
    lora_delta.zero_()
    _run_fused_moe_lora_one_shot(
        lora_delta,
        hidden_states,
        lora_a,
        lora_b,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        lora_ids,
        adapter_enabled,
        mul_routed_weight=False,
        block_size_m=BLOCK_SIZE_M,
        add_inputs=False,
    )
