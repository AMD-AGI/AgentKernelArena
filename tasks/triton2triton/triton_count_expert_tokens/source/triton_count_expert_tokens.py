"""Count expert num tokens kernel using Triton, adapted from vLLM utils.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count number of tokens assigned to each expert from topk_ids.
    Each program handles one expert, iterating over all topk_ids entries.
    """
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Count tokens per expert from topk_ids.

    Args:
        topk_ids: [num_tokens, topk] tensor of expert assignments (signed int)
        num_experts: total number of experts
    Returns:
        expert_num_tokens: [num_experts] tensor of token counts
    """
    assert topk_ids.dtype.is_signed, "topk_ids must be signed (uses -1 for invalid)"
    expert_num_tokens = torch.empty(
        (num_experts,), device=topk_ids.device, dtype=torch.int32
    )

    grid = (num_experts,)
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[grid](
        topk_ids,
        expert_num_tokens,
        num_experts,
        topk_ids.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expert_num_tokens
