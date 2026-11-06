# SPDX-License-Identifier: Apache-2.0
"""Triton moe_align_block_size for the fused-MoE baseline.

vLLM implements this step as a CUDA op (``ops.moe_align_block_size``); this is a Triton equivalent with the same
output contract: the (token, slot) assignments grouped by expert, every expert padded to a multiple of the GEMM tile
height with the sentinel ``numel``, one expert id per tile, and the padded total. The order of the assignments inside
an expert is unspecified (it does not change any result). Written for this task, using only atomics on counters.
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
    pid = tl.program_id(0)
    offs_e = tl.arange(0, EXPERTS_P2)
    counts = tl.load(counts_ptr + offs_e, mask=offs_e < num_experts, other=0)
    padded = (counts + block_size - 1) // block_size * block_size
    start = tl.sum(tl.where(offs_e < pid, padded, 0), axis=0)
    own = tl.sum(tl.where(offs_e == pid, padded, 0), axis=0)
    tl.store(cursor_ptr + pid, start)
    num_blocks = own // block_size
    first_block = start // block_size
    for b in range(0, num_blocks, BLOCK):
        offs = b + tl.arange(0, BLOCK)
        tl.store(expert_ids_ptr + first_block + offs, pid, mask=offs < num_blocks)
    tl.store(total_ptr, tl.sum(padded, axis=0), mask=pid == 0)


@triton.jit
def _scatter_kernel(topk_ids_ptr, cursor_ptr, sorted_ids_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    position = tl.atomic_add(cursor_ptr + ids, tl.full([BLOCK], 1, dtype=tl.int32), mask=mask)
    tl.store(sorted_ids_ptr + position, offs, mask=mask)


def moe_align_block_size(topk_ids, block_size, num_experts):
    numel = topk_ids.numel()
    max_padded = numel + num_experts * (block_size - 1)
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((triton.cdiv(max_padded, block_size),), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=topk_ids.device)
    cursor = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    grid = (triton.cdiv(numel, 1024),)
    _count_kernel[grid](topk_ids, counts, numel, BLOCK=1024)
    _layout_kernel[(num_experts,)](counts, cursor, expert_ids, num_tokens_post_padded, num_experts,
                                   block_size=block_size, EXPERTS_P2=triton.next_power_of_2(num_experts), BLOCK=32)
    _scatter_kernel[grid](topk_ids, cursor, sorted_ids, numel, BLOCK=1024)
    return sorted_ids, expert_ids, num_tokens_post_padded
