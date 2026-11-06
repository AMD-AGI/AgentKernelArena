# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity (zero) expert combine in Triton, from vLLM fused_moe.py compute_identity_kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def run(*, hidden_states, expert_scales, out):
    num_tokens, hidden_dim = hidden_states.shape
    BLOCK_SIZE = 256
    grid = (num_tokens * (hidden_dim // BLOCK_SIZE),)
    compute_identity_kernel[grid](
        expert_scales.shape[1],
        hidden_states,
        expert_scales,
        num_tokens,
        out,
        hidden_dim,
        expert_scales.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
