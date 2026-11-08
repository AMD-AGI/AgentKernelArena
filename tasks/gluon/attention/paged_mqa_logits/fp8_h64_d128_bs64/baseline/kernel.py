# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# run() reproduces the launch code of the Preshuffle=True, non-varctx path of deepgemm_fp8_paged_mqa_logits in
# ROCm/aiter aiter/ops/triton/attention/pa_mqa_logits.py, with the settings vLLM's ROCm indexer and the AITER FlyDSL
# op test use for 64-token blocks (ChunkK = 256, WavePerEU = 2, no varctx schedule). The upstream wrapper compiles
# the kernel through triton.compile; here it is launched through the Gluon JIT, with the same options.

import functools

import torch

from .pa_mqa_logits_gluon import _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle

_CHUNK_K = 256
_WAVE_PER_EU = 2
_NUM_WARPS = 4
_CDNA_VERSION = 4  # gfx950
_ARCH = "gfx950"


@functools.lru_cache(maxsize=None)
def _total_cu_count(device_index):
    # aiter.ops.triton.utils.device_info.get_num_sms(): the compute-unit count of the device (256 on MI355X).
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def run(*, q, kv_cache, weights, context_lens, block_tables, logits):
    """FP8 paged MQA logits over 64-token KV blocks whose fp8 data is stored in the AITER (16, 16) preshuffle.

    q [batch, next_n, heads, head_size] float8_e4m3fn; kv_cache [num_blocks, block_size, 1, head_size + 4]
    uint8, one block being block_size * head_size preshuffled fp8 bytes followed by block_size float32
    dequantization scales; weights [batch * next_n, heads] float32; context_lens [batch] int32;
    block_tables [batch, max_model_len // block_size] int32; logits [batch * next_n, max_model_len] float32.
    """
    batch_size, next_n, heads, hidden_dim = q.shape
    block_size = kv_cache.shape[1]
    index_dim = kv_cache.shape[3]
    max_model_len = logits.shape[1]
    max_block_len = block_tables.shape[1]
    assert block_size % 16 == 0 and _CHUNK_K % block_size == 0, "preshuffle needs 16-aligned blocks that divide ChunkK"

    total_cu_count = _total_cu_count(torch.cuda.current_device())
    tile_q_count = batch_size * next_n
    split_kv = (max(1, total_cu_count // tile_q_count) + 4) // 5 * 5 * _WAVE_PER_EU

    # deepgemm_fp8_paged_mqa_logits views the cache as one row per block and slices the data and the scales out.
    kv_cache = kv_cache.view(-1, block_size * index_dim)
    num_block = kv_cache.shape[0]
    kv_cache_fp8 = kv_cache[..., : block_size * hidden_dim].view(torch.float8_e4m3fn)
    kv_cache_scale = kv_cache[..., block_size * hidden_dim :].view(torch.float32)

    # The kernel writes only the chunks that cover each context. Production hands it a logits buffer that was
    # filled with -inf beforehand (the evaluator poisons it with NaN), so the fill is part of the call.
    logits.fill_(float("-inf"))

    grid = (batch_size * next_n * split_kv, 1, 1)
    _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle[grid](
        batch_size,
        next_n,
        heads,
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        block_tables,
        weights,
        weights.stride(0),
        logits,
        logits.stride(0),
        max_model_len,
        max_block_len,
        num_block,
        SplitKV=split_kv,
        ChunkQ=heads,
        ChunkK=_CHUNK_K,
        HiddenDim=hidden_dim,
        KVBlockSize=block_size,
        CDNA_VERSION=_CDNA_VERSION,
        ARCH=_ARCH,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVE_PER_EU,
        num_stages=2,
    )
