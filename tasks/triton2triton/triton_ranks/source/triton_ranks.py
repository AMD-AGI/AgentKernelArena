"""Ranks kernel, adapted from vLLM logprob.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_ranks(logits, token_ids):
    """Compute ranks for the given tokens among all vocab logits."""
    batch_size, vocab_size = logits.shape
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks, logits, logits.stride(0), token_ids,
        vocab_size, BLOCK_SIZE=8192,
    )
    return token_ranks
