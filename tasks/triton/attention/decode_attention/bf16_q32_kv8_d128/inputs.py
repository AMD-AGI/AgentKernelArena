"""Generator for the ragged KV index structure of a decode batch."""
import math

import torch

# Split-count heuristic of SGLang's Triton attention backend (get_num_kv_splits_triton) for
# the production configuration: at most 8 splits, a 256-CU device, one token per sequence.
MAX_KV_SPLITS = 8
DEVICE_CORE_COUNT = 256


def _cdiv(a, b):
    return -(-a // b)


def _num_kv_splits(lens, n_heads, n_kv_heads):
    max_len = int(lens.max())
    min_len = int(lens.min())
    if max_len * 8 < min_len * 10:
        min_len = max_len
    splits_1 = min(_cdiv(max_len, min_len), MAX_KV_SPLITS)
    chunk_1 = _cdiv(max_len, splits_1)
    cores = int(DEVICE_CORE_COUNT * max(math.log2(max_len / 64.0), 1.0)) if max_len > 0 else 0
    group = n_heads // n_kv_heads
    block_h = min(16, group) if group > 1 else 16
    grid = lens.numel() * (n_heads if group == 1 else _cdiv(n_heads, block_h))
    splits_2 = max(1, min(_cdiv(cores, grid), MAX_KV_SPLITS))
    chunk_2 = _cdiv(max_len, splits_2)
    chunk = torch.maximum(-(-lens // chunk_1), -(-lens // chunk_2))
    return chunk.clamp(1, MAX_KV_SPLITS).to(torch.int32)


def make_inputs(case, inputs, generator, device):
    a = case["axes"]
    batch, max_ctx, pool = a["batch"], a["max_ctx"], a["pool"]
    low = max(1, max_ctx // 2)
    lens = torch.randint(low, max_ctx + 1, (batch,), generator=generator, device=device, dtype=torch.int64)
    lens[0] = max_ctx  # the longest sequence defines max_ctx
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(lens, 0).to(torch.int32)
    # KV slots are scattered over the cache pool; slots past indptr[-1] are unused.
    indices = torch.randperm(pool, generator=generator, device=device).to(torch.int64)
    return {"kv_indptr": indptr, "kv_indices": indices,
            "num_kv_splits": _num_kv_splits(lens, a["n_heads"], a["n_kv_heads"])}
