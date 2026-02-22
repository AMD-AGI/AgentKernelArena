"""EP Scatter Phase 2 kernel using Triton, adapted from vLLM deep_gemm_utils.py.

Scatters tokens to expert-ordered layout using atomic adds on expert_start_loc.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    """
    Scatter tokens to expert-ordered layout.
    For each token and each topk expert, atomically get a destination slot
    from expert_start_loc, then copy the token's hidden states there.
    """
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_ptr = output_tensor + dest_token_index * output_tensor_stride0
                tl.store(output_ptr + offset_in, to_copy, mask=mask)


def ep_scatter_2(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    """
    Scatter tokens to expert-ordered layout.

    Args:
        recv_x: [num_tokens, hidden_size] input tokens
        recv_topk: [num_tokens, topk] expert assignments (signed int)
        expert_start_loc: [num_experts] start locations (modified by atomic adds)
        output_tensor: [total_slots, hidden_size] output in expert order
        output_index: [num_tokens, topk] maps (token, k) -> dest slot
    """
    num_tokens = recv_x.shape[0]
    hidden_size = recv_x.shape[1]
    num_warps = 8

    grid = (min(num_tokens, 1024 * 8),)

    _fwd_kernel_ep_scatter_2[grid](
        num_tokens,
        expert_start_loc,
        recv_x, recv_x.stride(0), recv_x.stride(1),
        recv_topk, recv_topk.stride(0), recv_topk.stride(1),
        output_tensor, output_tensor.stride(0), output_tensor.stride(1),
        output_index, output_index.stride(0), output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
    )
