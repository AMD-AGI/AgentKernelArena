"""Prepare M-RoPE positions kernel from vLLM v1 worker/gpu/mm/mrope_utils.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_mrope_positions_kernel(
    mrope_positions_ptr,
    mrope_positions_stride,
    prefill_mrope_positions_ptr,
    prefill_mrope_positions_stride0,
    prefill_mrope_positions_stride1,
    prefill_mrope_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    mrope_delta = tl.load(prefill_mrope_delta_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(3):
            if is_prefill:
                # Read from pre-computed M-RoPE positions.
                pos = tl.load(
                    prefill_mrope_positions_ptr
                    + req_state_idx * prefill_mrope_positions_stride0
                    + j * prefill_mrope_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                # Apply M-RoPE delta.
                pos = orig_pos + mrope_delta
            tl.store(
                mrope_positions_ptr + j * mrope_positions_stride + query_start + block,
                pos,
                mask=mask,
            )


def prepare_mrope_positions(
    mrope_positions: torch.Tensor,
    prefill_mrope_positions: torch.Tensor,
    max_model_len: int,
    prefill_mrope_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Prepare M-RoPE (Multi-dimensional Rotary Position Embedding) positions.

    For prefill requests, reads pre-computed 3D positions from prefill_mrope_positions.
    For decode requests, computes positions as orig_pos + mrope_delta.

    Args:
        mrope_positions: [3, max_num_tokens + 1] output position tensor (int64)
        prefill_mrope_positions: [max_num_reqs * 3, max_model_len] pre-computed positions (int32)
        max_model_len: maximum model sequence length
        prefill_mrope_delta: [max_num_reqs] per-request delta values (int32)
        idx_mapping: [num_reqs] batch_idx -> req_state_idx
        query_start_loc: [num_reqs + 1] cumulative token counts
        prefill_lens: [max_num_reqs] prefill lengths per request
        num_computed_tokens: [max_num_reqs] already computed tokens per request
    """
    num_reqs = idx_mapping.shape[0]
    _prepare_mrope_positions_kernel[(num_reqs,)](
        mrope_positions,
        mrope_positions.stride(0),
        prefill_mrope_positions,
        3 * max_model_len,
        max_model_len,
        prefill_mrope_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )
