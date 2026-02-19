"""EP Gather kernel using Triton, adapted from vLLM deep_gemm_utils.py.

Gathers expert results back to original token order with weighted accumulation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Gather expert outputs back to original token order.
    For each token, accumulate weighted results from all assigned experts.
    Grid: (hidden_blocks, tokens).
    """
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )

            if expert_id >= 0:
                source_token_index = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    """
    Gather expert results back to original token order.

    Args:
        input_tensor: [total_expert_slots, hidden_size] expert computation results
        recv_topk_ids: [num_tokens, topk] expert assignments
        recv_topk_weight: [num_tokens, topk] routing weights
        input_index: [num_tokens, topk] source indices into input_tensor
        output_tensor: [num_tokens, hidden_size] output (modified in-place)
    """
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = min(hidden_size, 1024)
    assert hidden_size % BLOCK_D == 0
    num_warps = 2
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor, input_tensor.stride(0), input_tensor.stride(1),
        recv_topk_ids, recv_topk_ids.stride(0), recv_topk_ids.stride(1),
        recv_topk_weight, recv_topk_weight.stride(0), recv_topk_weight.stride(1),
        input_index, input_index.stride(0), input_index.stride(1),
        output_tensor, output_tensor.stride(0), output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
