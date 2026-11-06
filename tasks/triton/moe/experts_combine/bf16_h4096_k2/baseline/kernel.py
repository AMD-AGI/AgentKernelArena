# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the SGLang project
"""Experts combine in Triton, extracted from SGLang (python/sglang/srt/layers/elementwise.py).

out = (sum_k moe_hidden_states[:, k] + mlp_hidden_states) / sqrt(2): the routed-expert outputs of each token are summed
over the top-k axis, added to the shared-expert (MLP) output and scaled by 1/sqrt(2) to balance the variance of the two
branches.
"""
import triton
import triton.language as tl


@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)

    moe_x = tl.load(
        moe_hidden_states
        + start_index_rmoe
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)


def run(*, moe_hidden_states, mlp_hidden_states, out):
    num_tokens, combine_k, hidden_dim = moe_hidden_states.shape
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4),
    }
    experts_combine_kernel[(num_tokens,)](
        out,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )
