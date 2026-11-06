# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored from vllm-project/vllm (commit edbc4969a76b25c531f7b9bb16b79984e65ef023),
# vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py: the kernel vLLM uses to turn
# FusedMoE top-k ids into the routing bitmatrix of the triton_kernels MoE GEMMs (make_routing_data).
# Changes: imports made local.

import triton
import triton.language as tl


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        # Guard with `valid` to prevent negative indices from producing
        # spurious bits (on HIP, -1 // 32 == 0 and 1 << (-1 % 32) sets
        # bit 31).
        x = tl.where(
            valid[:, :, None] & (div[:, :, None] == offs[None, None, :]),
            (one << rem)[:, :, None],
            0,
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)
