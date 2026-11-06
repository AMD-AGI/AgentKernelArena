# SPDX-License-Identifier: Apache-2.0
"""Request-to-token pool writer, adapted from SGLang write_req_to_token_pool_triton."""
import triton
import triton.language as tl


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_indices,  # [batch, max_prefix_len]
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
    prefix_indices_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    # SGLang receives one pointer per request; here the prefixes are the rows of one tensor.
    prefix_tensor = prefix_indices + pid * prefix_indices_stride

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def run(*, req_to_token, req_pool_indices, prefix_indices, pre_lens, seq_lens, extend_lens, out_cache_loc):
    write_req_to_token_pool_triton[(req_pool_indices.shape[0],)](
        req_to_token,
        req_pool_indices,
        prefix_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        req_to_token.stride(0),
        prefix_indices.stride(0),
    )
