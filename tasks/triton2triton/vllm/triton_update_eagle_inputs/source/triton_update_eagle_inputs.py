"""Update EAGLE inputs kernel from vLLM eagle worker.

Updates input IDs, hidden states, positions, and seq_lens between
speculative decoding steps in the EAGLE draft generation loop.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _update_eagle_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    seq_lens_ptr,
    max_model_len,
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Draft token -> Input ID
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Output hidden states -> Input hidden states
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Increment position and seq_lens, clamped to max_model_len
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def update_eagle_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    input_hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
) -> None:
    """
    Args:
        draft_tokens: [num_reqs]
        output_hidden_states: [num_reqs, hidden_size]
        input_ids: [num_tokens] (in/out)
        positions: [num_tokens] (in/out)
        input_hidden_states: [num_tokens, hidden_size] (in/out)
        seq_lens: [num_reqs] (in/out)
        max_model_len: int
    """
    num_reqs, hidden_size = output_hidden_states.shape
    _update_eagle_inputs_kernel[(num_reqs,)](
        input_ids,
        positions,
        input_hidden_states,
        input_hidden_states.stride(0),
        seq_lens,
        max_model_len,
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        hidden_size,
        BLOCK_SIZE=1024,
    )
