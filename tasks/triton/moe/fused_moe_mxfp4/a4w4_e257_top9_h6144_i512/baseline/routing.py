# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Token-to-expert alignment and the top-k combine, written in Triton for this task.

``moe_align_block_size`` has the contract of vLLM's ``moe_align_block_size`` op, which the AITER
MXFP4 MoE kernels consume: every (token, slot) index ``t * top_k + k`` grouped by expert, each
expert's group padded to a multiple of the GEMM tile height ``block_size`` with the sentinel
``numel`` (= num_tokens * top_k), one expert id per tile, and the padded total. AITER's own Triton
version runs one sequential loop per expert over all tokens; this one uses atomic counters. The
order of the indices inside an expert is unspecified; it changes no result, since every row of the
expert GEMMs is computed independently.

``moe_sum`` adds the top_k weighted expert outputs of each token in float32 (vLLM's ``moe_sum``).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _count_kernel(topk_ids_ptr, counts_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + ids, tl.full([BLOCK], 1, dtype=tl.int32), mask=mask)


@triton.jit
def _layout_kernel(counts_ptr, cursor_ptr, expert_ids_ptr, total_ptr, num_experts, block_size: tl.constexpr,
                   EXPERTS_P2: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    offs_e = tl.arange(0, EXPERTS_P2)
    counts = tl.load(counts_ptr + offs_e, mask=offs_e < num_experts, other=0)
    padded = (counts + block_size - 1) // block_size * block_size
    start = tl.sum(tl.where(offs_e < expert, padded, 0), axis=0)
    own = tl.sum(tl.where(offs_e == expert, padded, 0), axis=0)
    tl.store(cursor_ptr + expert, start)
    num_blocks = own // block_size
    first_block = start // block_size
    for b in range(0, num_blocks, BLOCK):
        offs = b + tl.arange(0, BLOCK)
        tl.store(expert_ids_ptr + first_block + offs, tl.full([BLOCK], 0, dtype=tl.int32) + expert,
                 mask=offs < num_blocks)
    tl.store(total_ptr, tl.sum(padded, axis=0), mask=expert == 0)


@triton.jit
def _scatter_kernel(topk_ids_ptr, cursor_ptr, sorted_ids_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    position = tl.atomic_add(cursor_ptr + ids, tl.full([BLOCK], 1, dtype=tl.int32), mask=mask)
    tl.store(sorted_ids_ptr + position, offs, mask=mask)


@triton.jit
def _moe_sum_kernel(y_ptr, out_ptr, hidden, stride_yt, stride_yk, stride_ot, TOP_K: tl.constexpr,
                    BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        acc += tl.load(y_ptr + token * stride_yt + k * stride_yk + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * stride_ot + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def moe_align_block_size(topk_ids, block_size, num_experts):
    numel = topk_ids.numel()
    max_padded = numel + num_experts * (block_size - 1)
    device = topk_ids.device
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.empty((triton.cdiv(max_padded, block_size),), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    cursor = torch.empty((num_experts,), dtype=torch.int32, device=device)
    grid = (triton.cdiv(numel, 1024),)
    _count_kernel[grid](topk_ids, counts, numel, BLOCK=1024)
    _layout_kernel[(num_experts,)](counts, cursor, expert_ids, num_tokens_post_padded, num_experts,
                                   block_size=block_size, EXPERTS_P2=triton.next_power_of_2(num_experts), BLOCK=64)
    _scatter_kernel[grid](topk_ids, cursor, sorted_ids, numel, BLOCK=1024)
    return sorted_ids, expert_ids, num_tokens_post_padded


def moe_sum(y, out):
    """out[t] = sum_k y[t, k] for y [tokens, top_k, hidden]; float32 accumulation."""
    tokens, top_k, hidden = y.shape
    block = 1024
    _moe_sum_kernel[(tokens, triton.cdiv(hidden, block))](y, out, hidden, y.stride(0), y.stride(1), out.stride(0),
                                                         TOP_K=top_k, BLOCK=block)
