# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# run() reproduces the host launch of flydsl_pa_mqa_logits_fp4 in ROCm/aiter
# aiter/ops/flydsl/kernels/pa_mqa_logits_fp4.py for a caller-supplied persistent-grid schedule (cta_info), as
# AITER's op test and SGLang's DeepSeek-V4 decode workspace do, with the settings SGLang uses
# (block_k = 256, kv_block_size = 64, 4 warps).

import torch

from .pa_mqa_logits_fp4 import compile_pa_mqa_logits_fp4

_BLOCK_K = 256
_KV_BLOCK_SIZE = 64
_NUM_WARPS = 4


def run(
    *,
    q_fp4,
    q_scale,
    kv_cache,
    kv_scale,
    weights,
    context_lens,
    block_tables,
    cta_info,
    weight_scale,
    logits,
):
    """Paged MXFP4 MQA logits (gfx950), one persistent-grid CTA per row of cta_info.

    q_fp4 [batch, next_n, heads, head_size / 2] uint8 (packed e2m1); q_scale [batch, next_n, k_tiles, 4, 16,
    padded m tiles] uint8 (E8M0, preshuffled); kv_cache [num_blocks, k_tiles, 4, 64, 16] uint8 and kv_scale
    [num_blocks, k_tiles, 4, 64] uint8 (the kernel's preshuffled block layout); weights [batch * next_n, heads]
    bfloat16; block_tables [batch, pages + 4] int32; cta_info [ctas, 4] int32 rows [batch_packed, chunk_start,
    chunk_count, ctx_len] built from context_lens (which the kernel itself does not read); logits
    [batch * next_n, max_seq_len] float32.
    """
    _, next_n, heads, head_dim_packed = q_fp4.shape
    launcher, _ = compile_pa_mqa_logits_fp4(
        block_k=_BLOCK_K,
        kv_block_size=_KV_BLOCK_SIZE,
        max_blocks_per_seq=block_tables.shape[1],
        num_warps=_NUM_WARPS,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim_packed * 2,
    )

    # The kernel stores only in-context keys. AITER's wrapper pre-fills the logits with -inf when it builds the
    # schedule itself (the evaluator hands over a NaN-poisoned buffer), so the fill is part of the call.
    logits.fill_(float("-inf"))

    launcher(
        logits,
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        logits.stride(0),
        float(weight_scale),
        cta_info.shape[0],
        torch.cuda.current_stream(),
    )
