# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

# Smallest positive value produced by Triton's fp32 `tl.rand`; used to clamp zero draws before
# the flipped Gumbel transform below.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10)


@triton.jit
def tl_rand32(seed, offset):
    u = tl.rand(seed, offset)
    return tl.maximum(u, _TL_RAND_MIN)


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        u = tl_rand32(gumbel_seed, block)
        # Draw the large-noise tail (which decides the argmax winner) from u -> 0, where fp32 has
        # fine resolution: `log1p(-u)` == `log(1 - u)` without losing precision for small u.
        gumbel_noise = -tl.log(-tldevice.log1p(-u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


@triton.jit
def _combine_blocks_kernel(
    out_ptr,
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    num_blocks,
    BLOCK: tl.constexpr,
):
    # The upstream wrapper does this with torch.argmax + gather; it is a kernel here because
    # PyTorch compute is not allowed inside run().
    token_idx = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < num_blocks
    values = tl.load(
        local_max_ptr + token_idx * local_max_stride + offs, mask=mask, other=float("-inf")
    )
    best = tl.argmax(values, axis=0)
    token_id = tl.load(local_argmax_ptr + token_idx * local_argmax_stride + best)
    tl.store(out_ptr + token_idx, token_id)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    out: torch.Tensor,  # [num_tokens]
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    _combine_blocks_kernel[(num_tokens,)](
        out,
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        num_blocks,
        BLOCK=triton.next_power_of_2(num_blocks),
    )


def run(*, logits, expanded_idx_mapping, temperature, seed, pos, out) -> None:
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, out)
