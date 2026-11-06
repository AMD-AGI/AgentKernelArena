# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_align_block_size with the contract of vLLM's moe_align_block_size (a CUDA op in vLLM).

The Python wrapper (buffer sizes) follows vllm/model_executor/layers/fused_moe/moe_align_block_size.py;
the device work, a CUDA op in vLLM, is done by Triton kernels written for this task:

1. _count_kernel: per-expert slot counts with atomic adds;
2. _scan_kernel: one program pads each count to a multiple of block_size and takes the exclusive
   prefix sum (each expert's first sorted position) and the padded total;
3. _expert_ids_kernel: one program per expert writes its expert id for each of its blocks;
4. _scatter_kernel: each slot takes a position inside its expert's segment with an atomic counter.

The order of the slots inside one expert's segment depends on the atomics; every slot is computed
independently by the GEMM, so the layer output does not depend on it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _count_kernel(topk_ids_ptr, counts_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    expert = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + expert, 1, mask=mask)


@triton.jit
def _scan_kernel(counts_ptr, offsets_ptr, num_tokens_post_pad_ptr, num_experts, block_size,
                 EXPERT_BLOCK: tl.constexpr):
    expert = tl.arange(0, EXPERT_BLOCK)
    valid = expert < num_experts
    count = tl.load(counts_ptr + expert, mask=valid, other=0)
    padded = (count + block_size - 1) // block_size * block_size
    inclusive = tl.cumsum(padded, axis=0)
    tl.store(offsets_ptr + expert, inclusive - padded, mask=valid)
    tl.store(num_tokens_post_pad_ptr, tl.sum(padded, axis=0))


@triton.jit
def _expert_ids_kernel(counts_ptr, offsets_ptr, expert_ids_ptr, block_size, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    count = tl.load(counts_ptr + expert)
    first = tl.load(offsets_ptr + expert) // block_size
    num_blocks = (count + block_size - 1) // block_size
    for start in range(0, num_blocks, BLOCK):
        index = start + tl.arange(0, BLOCK)
        tl.store(expert_ids_ptr + first + index, tl.zeros((BLOCK,), dtype=tl.int32) + expert,
                 mask=index < num_blocks)


@triton.jit
def _scatter_kernel(topk_ids_ptr, offsets_ptr, cursor_ptr, sorted_token_ids_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    expert = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    rank = tl.atomic_add(cursor_ptr + expert, 1, mask=mask)
    first = tl.load(offsets_ptr + expert, mask=mask, other=0)
    tl.store(sorted_token_ids_ptr + first + rank, offs.to(tl.int32), mask=mask)


_SLOT_BLOCK = 1024


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorted slot ids padded per expert to block_size, the expert of each block and the padded total."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    # Padding slots hold numel, which the GEMM masks out.
    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device=device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    cursor = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    offsets = torch.empty((num_experts,), dtype=torch.int32, device=device)
    grid = (triton.cdiv(numel, _SLOT_BLOCK),)
    _count_kernel[grid](topk_ids, counts, numel, BLOCK=_SLOT_BLOCK)
    _scan_kernel[(1,)](counts, offsets, num_tokens_post_pad, num_experts, block_size,
                       EXPERT_BLOCK=triton.next_power_of_2(num_experts))
    _expert_ids_kernel[(num_experts,)](counts, offsets, expert_ids, block_size, BLOCK=64)
    _scatter_kernel[grid](topk_ids, offsets, cursor, sorted_ids, numel, BLOCK=_SLOT_BLOCK)
    return sorted_ids, expert_ids, num_tokens_post_pad
