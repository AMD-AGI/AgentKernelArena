# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# run() reproduces the launch code of the pure-Triton branch (KVBlockSize == 1) of
# deepgemm_fp8_paged_mqa_logits in ROCm/aiter aiter/ops/triton/attention/pa_mqa_logits.py, with the
# ChunkK, WavePerEU and SplitKV choices vLLM's ROCm indexer uses (ChunkK = 256, WavePerEU = 2).

import functools

import torch

from .pa_mqa_logits_kernel import _deepgemm_fp8_paged_mqa_logits

_CHUNK_K = 256
_WAVE_PER_EU = 2


@functools.lru_cache(maxsize=None)
def _total_cu_count(device_index):
    # aiter.ops.triton.utils.device_info.get_num_sms(): the compute-unit count of the device (256 on MI355X).
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def run(*, q, kv_cache, weights, context_lens, block_tables, logits):
    """FP8 paged MQA logits, one token per KV block.

    q [batch, next_n, heads, head_size] float8_e4m3fn; kv_cache [num_blocks, 1, 1, head_size + 4] uint8
    (head_size fp8 codes then the float32 dequantization scale of each token); weights [batch * next_n, heads]
    float32; context_lens [batch] int32; block_tables [batch, max_model_len] int32 (token -> KV block);
    logits [batch * next_n, max_model_len] float32.
    """
    batch_size, next_n, heads, hidden_dim = q.shape
    max_model_len = logits.shape[1]
    max_block_len = block_tables.shape[1]
    assert kv_cache.shape[1] == 1, "this scenario uses one token per KV block"

    # deepgemm_fp8_paged_mqa_logits slices the scale out of the cache row and views it as float32.
    kv_cache = kv_cache.view(-1, hidden_dim + 4)
    kv_cache_fp8 = kv_cache[:, :hidden_dim].view(torch.float8_e4m3fn)
    kv_cache_scale = kv_cache[:, hidden_dim:].view(torch.float32)

    total_cu_count = _total_cu_count(torch.cuda.current_device())
    tile_q_count = batch_size * next_n
    split_kv = (max(1, total_cu_count // tile_q_count) + 4) // 5 * 5 * _WAVE_PER_EU

    # The kernel writes only the chunks that cover each context. Production hands it a logits buffer that was
    # filled with -inf beforehand (the evaluator poisons it with NaN), so the fill is part of the call.
    logits.fill_(float("-inf"))

    grid = (batch_size * next_n * split_kv, 1, 1)
    _deepgemm_fp8_paged_mqa_logits[grid](
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
        waves_per_eu=_WAVE_PER_EU,
        ChunkQ=heads,
        ChunkK=_CHUNK_K,
        SplitKV=split_kv,
        HiddenDim=hidden_dim,
    )
