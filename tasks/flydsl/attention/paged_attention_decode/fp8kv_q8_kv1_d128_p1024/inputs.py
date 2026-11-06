# The worklist scheduler ports ROCm/aiter csrc/kernels/mla/metadata/v1_2_pa_device.cuh
# (MIT, Copyright (C) 2025-2026, Advanced Micro Devices, Inc.); the block-split expansion ports
# ROCm/FlyDSL kernels/pa_decode_fp8.py (Apache-2.0, Copyright (c) 2025 FlyDSL Project Contributors).
"""Paged FP8 KV cache, page lists and the persistent-scheduling worklist of one decode batch.

The cache holds every token of the batch quantized per token and head, as AITER's pertoken_quant
does when the cache is written: scale = max|x| / 448 over the head dimension (1 when that is 0),
code = float8_e4m3fn(x / scale). Each sequence owns max_pages physical 1024-token pages dealt in
random order; the pages it does not use and the tail of its last page hold stale, finite codes
and scales that must not affect the result.

The worklist is what AITER's get_pa_metadata_v1 (csrc/kernels/mla/metadata/v1_2_pa_device.cuh,
the single-query-tile, uniform-query-length branch) builds for 256 compute units, followed by
the block-split expansion of FlyDSL's pa_decode_ps_launch: every work item gets a partial slot
whose four 256-token sub-tile partials are merged per sequence. Production computes it once per
decode step and shares it across all layers, so it is an input here.
"""
import torch

_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, AITER's pertoken_quant bound on gfx950
_PAGE_CHUNK = 256  # pages quantized per step


def _quantize_per_token(x):
    """AITER pertoken_quant over the last axis: fp32 scale and float8_e4m3fn codes."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x / scale).to(torch.float8_e4m3fn), scale


def _schedule(pages, num_cu, group):
    """get_pa_metadata_v1 for query length 1 and one KV head (no query-tile splits).

    Returns the work rows [batch, partial, q_start, q_end, kv_start, kv_end, kv_offset,
    q_head_range] and work_indptr[num_cu + 1]. partial is -1 for a sequence handled by one work
    item and a running split index otherwise.
    """
    num_batches = len(pages)
    sum_blocks = sum(pages)
    average, reminder = divmod(sum_blocks, num_cu)
    q_head_range = (group << 16) | 0
    works, work_indptr = [], [0] * (num_cu + 1)
    cid = 0
    curr_batch, curr_kv_block, curr_n_split = 0, 0, 0
    curr_kv_begin, curr_kv_end = 0, pages[0]
    partial_idx = 0
    while cid < num_cu:
        remain = average + 1 if cid < reminder else average
        while curr_batch < num_batches:
            remain_kv = (curr_kv_end - curr_kv_begin) - curr_kv_block
            kv_start = curr_kv_begin + curr_kv_block
            if remain >= remain_kv:
                partial = partial_idx if curr_n_split > 0 else -1
                works.append([curr_batch, partial, curr_batch, curr_batch + 1, kv_start,
                              min(kv_start + remain_kv, curr_kv_end), 0, q_head_range])
                if curr_n_split > 0:
                    partial_idx += 1
                remain -= remain_kv
                curr_batch += 1
                if curr_batch < num_batches:
                    curr_kv_begin = curr_kv_end
                    curr_kv_end = curr_kv_begin + pages[curr_batch]
                    curr_kv_block, curr_n_split = 0, 0
            else:
                if remain > 0:
                    works.append([curr_batch, partial_idx, curr_batch, curr_batch + 1, kv_start,
                                  min(kv_start + remain, curr_kv_end), 0, q_head_range])
                    partial_idx += 1
                    curr_kv_block += remain
                    curr_n_split += 1
                break
        work_indptr[cid + 1] = len(works)
        cid += 1
        if curr_batch == num_batches:
            break
    for index in range(cid, num_cu + 1):
        work_indptr[index] = len(works)
    if curr_batch != num_batches:
        raise RuntimeError("the schedule left sequences unassigned")
    return works, work_indptr


def _expand_block_splits(works, tiles_per_block):
    """FlyDSL's _expand_pa_metadata_for_block_splits for query length 1.

    Every work item becomes a partial slot; each slot holds tiles_per_block sub-tile partials
    and every sequence (q range) is merged by the reduce stage.
    """
    group_order, group_slots, row_slots = [], {}, []
    for row in works:
        key = (row[2], row[3])
        if key not in group_slots:
            group_order.append(key)
            group_slots[key] = []
        slot = ("split", row[1]) if row[1] >= 0 else ("direct", row[2], row[3])
        if slot not in group_slots[key]:
            group_slots[key].append(slot)
        row_slots.append(slot)
    slot_ids = {}
    for key in group_order:
        for slot in group_slots[key]:
            slot_ids.setdefault(slot, len(slot_ids))
    for row, slot in zip(works, row_slots):
        row[1] = slot_ids[slot]
    reduce_indptr, reduce_final_map, reduce_partial_map = [0], [], []
    for key in group_order:
        reduce_final_map.append(list(key))
        for slot in group_slots[key]:
            reduce_partial_map += [slot_ids[slot] * tiles_per_block + split for split in range(tiles_per_block)]
        reduce_indptr.append(len(reduce_partial_map))
    return reduce_indptr, reduce_final_map, reduce_partial_map


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_seqs = axes["num_seqs"]
    max_seq_len = axes["max_seq_len"]
    block_size = axes["block_size"]
    max_pages = axes["max_pages"]
    num_blocks = axes["num_blocks"]
    num_kv_heads = axes["num_kv_heads"]
    head_size = axes["head_size"]
    num_cu = axes["num_cu"]
    max_work = axes["max_work"]
    group = axes["num_query_heads"] // num_kv_heads
    tiles_per_block = block_size // 256

    # Ragged contexts: the longest sequence has max_seq_len tokens, the others between half and all of it.
    low = (max_seq_len + 1) // 2
    context_lengths = torch.randint(low, max_seq_len + 1, (num_seqs,), generator=generator, device=device)
    longest = torch.randint(0, num_seqs, (1,), generator=generator, device=device)
    context_lengths[longest] = max_seq_len

    # Each sequence owns max_pages scattered physical pages and uses the first ceil(len / 1024).
    owned = torch.randperm(num_blocks, generator=generator, device=device).view(num_seqs, max_pages)
    lengths = context_lengths.tolist()
    pages = [-(-length // block_size) for length in lengths]
    kv_indptr = [0]
    for count in pages:
        kv_indptr.append(kv_indptr[-1] + count)
    used = torch.arange(max_pages, device=device)[None, :] < torch.tensor(pages, device=device)[:, None]
    kv_page_indices = torch.zeros(num_blocks, dtype=torch.int32, device=device)
    kv_page_indices[:kv_indptr[-1]] = owned[used].to(torch.int32)

    key_cache = torch.empty((num_blocks, num_kv_heads, head_size // 16, block_size, 16),
                            dtype=torch.float8_e4m3fn, device=device)
    value_cache = torch.empty((num_blocks, num_kv_heads, block_size // 16, head_size, 16),
                              dtype=torch.float8_e4m3fn, device=device)
    key_scale = torch.empty((num_blocks, num_kv_heads, block_size, 1), dtype=torch.float32, device=device)
    value_scale = torch.empty_like(key_scale)
    for start in range(0, num_blocks, _PAGE_CHUNK):
        stop = min(num_blocks, start + _PAGE_CHUNK)
        count = stop - start
        shape = (count, num_kv_heads, block_size, head_size)
        k_codes, k_scale = _quantize_per_token(
            torch.randn(shape, generator=generator, device=device, dtype=torch.float32))
        v_codes, v_scale = _quantize_per_token(
            torch.randn(shape, generator=generator, device=device, dtype=torch.float32))
        # key: [page, head, d // 16, slot, d % 16]; value: [page, head, slot // 16, d, slot % 16]
        key_cache[start:stop] = k_codes.view(count, num_kv_heads, block_size, head_size // 16, 16).permute(0, 1, 3, 2, 4)
        value_cache[start:stop] = v_codes.view(count, num_kv_heads, block_size // 16, 16, head_size).permute(0, 1, 2, 4, 3)
        key_scale[start:stop] = k_scale
        value_scale[start:stop] = v_scale

    works, work_indptr = _schedule(pages, num_cu, group)
    reduce_indptr, reduce_final_map, reduce_partial_map = _expand_block_splits(works, tiles_per_block)
    if len(works) > max_work:
        raise RuntimeError(f"{len(works)} work items exceed the declared {max_work}")
    work_info = torch.zeros((max_work, 8), dtype=torch.int32)
    if works:
        work_info[:len(works)] = torch.tensor(works, dtype=torch.int32)
    partial_map = torch.zeros(tiles_per_block * max_work, dtype=torch.int32)
    partial_map[:len(reduce_partial_map)] = torch.tensor(reduce_partial_map, dtype=torch.int32)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32).to(device)

    return {
        "key_cache": key_cache, "value_cache": value_cache, "key_scale": key_scale, "value_scale": value_scale,
        "context_lengths": context_lengths.to(torch.int32), "kv_indptr": ints(kv_indptr),
        "kv_page_indices": kv_page_indices, "work_indptr": ints(work_indptr), "work_info": work_info.to(device),
        "reduce_indptr": ints(reduce_indptr), "reduce_final_map": ints(reduce_final_map).view(num_seqs, 2),
        "reduce_partial_map": partial_map.to(device),
    }
