"""Prepare EAGLE decode kernel from vLLM eagle worker.

Prepares decode-step inputs for EAGLE speculative decoding: copies draft tokens
to input IDs, copies hidden states, computes positions and seq_lens, and
initializes query_start_loc for CUDA graphs.
Note: "docode" preserves the original vLLM spelling.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_eagle_docode_kernel(
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    last_token_indices_ptr,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    hidden_size,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc and pad for CUDA graphs
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # output hidden states -> input hidden states
    src_idx = tl.load(last_token_indices_ptr + req_idx)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + src_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Compute position and seq_lens
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
) -> None:
    """
    Args:
        draft_tokens: [num_reqs]
        output_hidden_states: [total_tokens, hidden_size]
        last_token_indices: [num_reqs]
        target_seq_lens: [num_reqs]
        num_rejected: [num_reqs]
        positions: [max_num_tokens] (in/out)
        seq_lens: [max_num_reqs] (output)
        query_start_loc: [max_num_reqs + 1] (output)
        input_ids: [max_num_tokens] (output)
        input_hidden_states: [max_num_tokens, hidden_size] (output)
        max_model_len: int
        max_num_reqs: int
    """
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hidden_states.shape[-1]
    _prepare_eagle_docode_kernel[(num_reqs + 1,)](
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_ids,
        positions,
        input_hidden_states,
        input_hidden_states.stride(0),
        query_start_loc,
        seq_lens,
        hidden_size,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
