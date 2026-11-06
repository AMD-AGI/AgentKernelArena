# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Token-to-expert alignment for the AITER MXFP4 MoE kernels, written in Triton for this task.

``moe_align_block_size`` produces what vLLM's ``moe_align_block_size`` (and AITER's Triton
``moe_align_block_size_triton``) produce and the fused-MoE kernels consume: every (token, slot)
index ``t * top_k + k`` grouped by expert, each expert's group padded to a multiple of the GEMM tile
height ``block_size`` with the sentinel ``numel = num_tokens * top_k``, one expert id per tile, and
the padded total. AITER's Triton version walks all tokens in one sequential loop per expert; this one
counts and scatters with atomics. The order of the indices inside an expert's group is unspecified,
which changes no result: every row of the expert GEMMs is computed and stored independently.
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
def _layout_kernel(counts_ptr, cursor_ptr, expert_ids_ptr, total_ptr, num_experts,
                   block_size: tl.constexpr, EXPERTS_P2: tl.constexpr, BLOCK: tl.constexpr):
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
    if expert == 0:
        tl.store(total_ptr, tl.sum(padded, axis=0))


@triton.jit
def _scatter_kernel(topk_ids_ptr, cursor_ptr, sorted_ids_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    position = tl.atomic_add(cursor_ptr + ids, tl.full([BLOCK], 1, dtype=tl.int32), mask=mask)
    tl.store(sorted_ids_ptr + position, offs, mask=mask)


def moe_align_block_size(topk_ids, block_size, num_experts):
    """(sorted_token_ids, expert_ids, num_tokens_post_padded) for int32 topk_ids [tokens, top_k]."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    cursor = torch.empty((num_experts,), dtype=torch.int32, device=device)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _count_kernel[grid](topk_ids, counts, numel, BLOCK=BLOCK)
    _layout_kernel[(num_experts,)](counts, cursor, expert_ids, num_tokens_post_pad, num_experts,
                                   block_size=block_size, EXPERTS_P2=triton.next_power_of_2(num_experts),
                                   BLOCK=128)
    _scatter_kernel[grid](topk_ids, cursor, sorted_ids, numel, BLOCK=BLOCK)
    return sorted_ids, expert_ids, num_tokens_post_pad
