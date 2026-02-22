"""Expand kernel, adapted from vLLM rejection_sampler.py."""
import torch
import triton
import triton.language as tl

MAX_SPEC_LEN = 128


@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0
    if req_idx > 0:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx).to(tl.int32)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


def expand_batch_to_tokens(x, cu_num_tokens, num_tokens, replace_from=0, replace_to=0):
    """Expand [batch_size] tensor to [num_tokens] based on cu_num_tokens."""
    batch_size = x.shape[0]
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x, x, cu_num_tokens, replace_from, replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
    )
    return expanded_x
