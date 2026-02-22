"""EP Scatter Phase 1 kernel using Triton, adapted from vLLM deep_gemm_utils.py.

Computes expert start locations from token counts (prefix sum with 128-alignment)
and fills m_indices with expert IDs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return ((x + y - 1) // y) * y


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    """
    Phase 1 of expert-parallel scatter:
    1. Compute prefix sum of aligned token counts -> expert_start_loc
    2. Fill m_indices with expert ID for each expert's aligned region
    """
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    tokens_per_expert = round_up_128(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num
        tl.store(m_indices_start_ptr + offs, cur_expert, mask=mask)


def ep_scatter_1(
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    """
    Compute expert start locations and fill m_indices.

    Args:
        num_recv_tokens_per_expert: [num_experts] token counts per expert
        expert_start_loc: [num_experts] output start locations (modified in-place)
        m_indices: [total_aligned] output expert IDs (modified in-place, pre-filled with -1)
    """
    num_experts = num_recv_tokens_per_expert.shape[0]
    BLOCK_E = 128
    num_warps = 8

    grid = (num_experts,)
    _fwd_kernel_ep_scatter_1[grid](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
