# The worklist planner ports kn_get_mla_metadata_v1_2_parallel of ROCm/aiter
# csrc/kernels/mla/metadata/v1_2_device.cuh (MIT, Copyright (C) 2025-2026, Advanced Micro Devices, Inc.).
"""Paged FP8 latent KV cache, token page lists and the persistent-scheduling worklist of one MLA decode batch.

The cache is a pool of one-token pages (page_size 1) of 576 float8_e4m3fn values ([512 latent | 64 rope]); each
sequence owns ctx scattered rows and lists the first seq_len of them in kv_page_indices, so rows past a sequence's
length and the rows it does not list hold stale, finite codes that must not affect the result. Sequence lengths are
ragged: the longest has ctx tokens, the others between half of it and all of it.

The worklist is what AITER's get_mla_metadata_v1 builds in fast mode for this kernel's configuration (128 heads,
one query token per sequence, FP8 queries and cache, page size 1, 256 compute units, kv granularity 16 and a fixed
cost of 16 blocks per work item): the parallel planner of v1_2_device.cuh, ported step by step. Production builds it
once per decode step and shares it across all layers, so it is an input here. One deliberate difference: blocks are 64
tokens instead of 16 and the last work item of a sequence takes the partial tail, so that no work item is shorter than the
kernel's 64-token tile (see README.md).
"""
import torch

# AITER schedules in blocks of kv_granularity = max(page_size, 16) = 16 tokens with a fixed cost of 16 blocks per work
# item. The FlyDSL kernel mishandles work items shorter than its 64-token tile (README.md), so this port schedules in
# blocks of 64 tokens (the kernel's BLOCK_N), keeps the same fixed cost in tokens (4 blocks = 256 tokens) and lets the last
# work item of a sequence take the partial tail block, so no work item is shorter than 64 tokens.
_KV_GRANULARITY = 64
_OVERHEAD = 4
_CHUNK_ROWS = 1 << 16


def _plan(seq_lens, num_cu):
    """kn_get_mla_metadata_v1_2_parallel for one query token per sequence and one query tile.

    Returns work rows [batch, partial_qo_loc, qo_start, qo_end, kv_start, kv_end, kv_offset, 0], work_indptr
    [num_cu + 1], reduce_indptr [num_batches + 1], reduce_final_map [num_batches, 2] and the partial map.
    partial_qo_loc is -1 for a sequence handled by one work item. The FlyDSL kernel writes every work item to a
    partial slot, so the caller must pass shapes in which every sequence is split (see make_inputs).
    """
    num_batches = len(seq_lens)
    # SGLang's aiter backend (srt/layers/attention/aiter_backend.py): cap 64 without draft tokens, and under DP attention
    # min(ceil(cu / batch), cap), which gives the same num_splits as the fixed cap.
    max_split_per_batch = min(-(-num_cu // num_batches), 64)
    num_splits = min(num_cu, max_split_per_batch * num_batches)
    blocks = [max(1, length // _KV_GRANULARITY) for length in seq_lens]  # the partial tail joins the last block
    sum_blocks = sum(count + _OVERHEAD for count in blocks)
    payload = -(-sum_blocks // num_splits) + _OVERHEAD
    blocks_per_cu = payload - _OVERHEAD

    # Phase 1: one thread scans the batches and records where each one starts.
    starts = []
    curr_cu, remain, num_works, last_reduce, partial_idx = 0, payload, 0, 0, 0
    for count in blocks:
        starts.append((curr_cu, remain, num_works, last_reduce, partial_idx))
        if count + _OVERHEAD <= remain:
            num_works += 1
            remain -= count + _OVERHEAD
        else:
            if remain > _OVERHEAD:
                remain_blocks = count - (remain - _OVERHEAD)
                fresh = -(-remain_blocks // blocks_per_cu)
                frags = fresh + 1
                curr_cu += fresh
                remain = payload - ((remain_blocks - (fresh - 1) * blocks_per_cu) + _OVERHEAD)
            else:
                fresh = -(-count // blocks_per_cu)
                frags = fresh
                curr_cu += fresh
                remain = payload - ((count - (fresh - 1) * blocks_per_cu) + _OVERHEAD)
            num_works += frags
            if frags > 1:
                last_reduce += frags
                partial_idx += frags
    total_works, total_reduce = num_works, last_reduce

    # Phase 2: every batch writes its work items, reduce entries and the closing work_indptr entries.
    work_info = [[0] * 8 for _ in range(num_batches + num_cu - 1)]
    work_indptr = [0] + [total_works] * num_cu
    reduce_indptr = [0] * (num_batches + 1)
    reduce_final_map = [[0, 0] for _ in range(num_batches)]
    reduce_partial_map = [0] * min(num_batches + num_cu - 1, (num_cu - 1) * 2)
    kv_begin = 0
    for bid, (length, count) in enumerate(zip(seq_lens, blocks)):
        start_cu, remain, works_before, reduce_before, partial_before = starts[bid]
        kv_end = kv_begin + length
        if count + _OVERHEAD <= remain:
            fresh, frags, waste = 1, 1, False
        elif remain > _OVERHEAD:
            fresh = -(-(count - (remain - _OVERHEAD)) // blocks_per_cu)
            frags, waste = fresh + 1, False
        else:
            fresh = -(-count // blocks_per_cu)
            frags, waste = fresh, True
        is_split = frags > 1
        first_frag_blocks = remain - _OVERHEAD
        if is_split:
            reduce_indptr[bid + 1] = reduce_before + frags
            reduce_final_map[bid] = [bid, bid + 1]
        else:
            reduce_indptr[bid + 1] = reduce_before
        if waste and start_cu + 1 <= num_cu:
            work_indptr[start_cu + 1] = works_before
        for frag in range(frags):
            if not is_split:
                block_begin, block_end, frag_cu = 0, count, start_cu
            elif not waste:
                block_begin = 0 if frag == 0 else first_frag_blocks + (frag - 1) * blocks_per_cu
                block_end = first_frag_blocks + frag * blocks_per_cu if frag < fresh else count
                frag_cu = start_cu + frag
            else:
                block_begin = frag * blocks_per_cu
                block_end = (frag + 1) * blocks_per_cu if frag < fresh - 1 else count
                frag_cu = start_cu + 1 + frag
            frag_start = kv_begin + block_begin * _KV_GRANULARITY
            frag_end = kv_end if block_end == count else kv_begin + block_end * _KV_GRANULARITY
            work_idx = works_before + frag
            partial = partial_before + frag if is_split else -1
            work_info[work_idx] = [bid, partial, bid, bid + 1, frag_start, frag_end, kv_end - frag_end, 0]
            assert frag_end - frag_start >= _KV_GRANULARITY, "a work item is shorter than the kernel tile"
            if is_split:
                reduce_partial_map[reduce_before + frag] = partial_before + frag
            if frag != frags - 1 and frag_cu + 1 <= num_cu:
                work_indptr[frag_cu + 1] = work_idx + 1
        kv_begin = kv_end
    for index in range(num_batches, len(reduce_indptr)):
        reduce_indptr[index] = total_reduce
    return work_info, work_indptr, reduce_indptr, reduce_final_map, reduce_partial_map


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, ctx = axes["batch"], axes["ctx"]
    head_size = axes["head_size"]
    num_cu = axes["num_cu"]
    num_pages = axes["num_pages"]

    # Ragged contexts: the longest sequence has ctx tokens, the others between half and all of it.
    seq_lens = torch.randint((ctx + 1) // 2, ctx + 1, (batch,), generator=generator, device=device)
    seq_lens[torch.randint(0, batch, (1,), generator=generator, device=device)] = ctx
    lengths = seq_lens.tolist()

    # Each sequence owns ctx scattered rows of the pool and lists the first seq_len of them.
    owned = torch.randperm(num_pages, generator=generator, device=device).view(batch, ctx)
    used = torch.arange(ctx, device=device)[None, :] < seq_lens[:, None]
    kv_page_indices = torch.zeros(num_pages, dtype=torch.int32, device=device)
    kv_indptr = [0]
    for length in lengths:
        kv_indptr.append(kv_indptr[-1] + length)
    kv_page_indices[:kv_indptr[-1]] = owned[used].to(torch.int32)

    # FP8 codes of N(0, 1) latent / rope values, cast directly as in FlyDSL's test (no scales).
    kv_cache = torch.empty((num_pages, 1, 1, head_size), dtype=torch.float8_e4m3fn, device=device)
    flat = kv_cache.view(num_pages, head_size)
    for start in range(0, num_pages, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, num_pages)
        flat[start:stop] = torch.randn((stop - start, head_size), generator=generator, device=device,
                                       dtype=torch.float32).to(torch.float8_e4m3fn)

    work_info, work_indptr, reduce_indptr, reduce_final_map, reduce_partial_map = _plan(lengths, num_cu)
    if any(row[1] < 0 for row in work_info[:work_indptr[-1]]) or len(reduce_partial_map) != axes["max_split_tiles"]:
        raise RuntimeError("the worklist has an unsplit sequence or does not match its declared shapes")

    def ints(values):
        return torch.tensor(values, dtype=torch.int32).to(device)

    return {
        "kv_cache": kv_cache,
        "kv_indptr": ints(kv_indptr),
        "kv_page_indices": kv_page_indices,
        "work_indptr": ints(work_indptr),
        "work_info": ints(work_info),
        "reduce_indptr": ints(reduce_indptr),
        "reduce_final_map": ints(reduce_final_map),
        "reduce_partial_map": ints(reduce_partial_map),
    }
