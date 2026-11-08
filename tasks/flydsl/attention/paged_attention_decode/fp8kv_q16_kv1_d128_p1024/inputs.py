# The worklist scheduler ports compile_pa_metadata_v1 of ROCm/FlyDSL kernels/attention/pa_metadata.py
# (tag v0.3.2; Apache-2.0, Copyright (c) 2025 FlyDSL Project Contributors).
"""Paged FP8 KV cache, page lists and the persistent-scheduling worklist of one decode batch.

The cache holds every token of the batch quantized per token and head, as AITER's pertoken_quant
does when the cache is written: scale = max|x| / 448 over the head dimension (1 when that is 0),
code = float8_e4m3fn(x / scale). Each sequence owns max_pages physical 1024-token pages dealt in
random order; the pages it does not use and the tail of its last page hold stale, finite codes
and scales that must not affect the result.

The worklist is what FlyDSL's get_pa_metadata_v1 (the scheduler kernel of pa_metadata.py, for the
uniform query length 1) builds for 256 compute units: the 256-token partitions of all sequences are
dealt out evenly, work_info kv_start / kv_end are cumulative partition indices, and the reduce maps
come out of the scheduler directly. Production computes it once per decode step and shares it across
all layers, so it is an input here.
"""
import torch

_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, AITER's pertoken_quant bound on gfx950
_PAGE_CHUNK = 256  # pages quantized per step
_PARTITION = 256  # tokens per scheduled partition (KV_COMPUTE_BLOCK of pa_metadata.py)
_QUERY_LENGTH = 1


def _quantize_per_token(x):
    """AITER pertoken_quant over the last axis: fp32 scale and float8_e4m3fn codes."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x / scale).to(torch.float8_e4m3fn), scale


def _schedule(parts, num_cu, group, num_heads_k=1):
    """compile_pa_metadata_v1 for query length 1: one thread of the scheduler kernel, step by step.

    parts[b] is the number of 256-token partitions of sequence b. Returns work_info rows
    [batch, partial, q_start, q_end, kv_start, kv_end, 0, q_head_range], work_indptr[num_cu + 1],
    reduce_indptr[num_batches + 1], reduce_final_map rows and reduce_partial_map. partial is -1 for a
    work item that finishes a sequence without a split and the partial slot otherwise.
    """
    num_batches = len(parts)
    ql = _QUERY_LENGTH
    splits_per_head = num_cu // num_heads_k
    average, reminder = divmod(sum(parts), splits_per_head)

    def remain_for(cid):
        return average + (1 if cid % splits_per_head < reminder else 0)

    max_work = (num_batches + num_cu - 1) * num_heads_k
    work_info = [[0] * 8 for _ in range(max_work + 1)]
    work_indptr = [0] * (num_cu + 1)
    reduce_indptr = [0] * (num_batches + 1)
    reduce_final_map = [[0, 0] for _ in range(num_batches)]
    reduce_partial_map = [0] * min(num_batches + num_cu - 1, (num_cu - 1) * 2)
    cid = 0
    num_works = 0
    for khead in range(num_heads_k):
        q_head_range = ((khead + 1) * group << 16) | ((khead * group) & 0xFFFF)
        batch, kv_blk, n_split, pidx, kv_beg = 0, 0, 0, 0, 0
        kv_end = parts[0]
        remain = remain_for(cid)
        lri = grt = 0
        while cid < num_cu and batch < num_batches:
            remain_kv = (kv_end - kv_beg) - kv_blk
            do_finish = remain >= remain_kv
            kv_start = kv_beg + kv_blk
            has_split = n_split > 0
            if do_finish:
                ploc = pidx if has_split else -1
                work_end = kv_end
            else:
                ploc = pidx
                work_end = min(kv_start + remain, kv_end)
            work_info[num_works] = [batch, ploc, batch * ql, (batch + 1) * ql, kv_start, work_end, 0, q_head_range]
            do_reduce = do_finish and has_split
            num_splits = n_split + 1
            # Written on every step; a step that is not a reduce group is overwritten by the next one.
            reduce_indptr[grt + 1] = lri + num_splits
            reduce_final_map[grt] = [batch * ql, (batch + 1) * ql]
            if do_reduce:
                for sidx in range(num_splits):
                    reduce_partial_map[lri + sidx] = pidx - (n_split - sidx) * ql
            if do_finish:
                num_works += 1
                work_indptr[cid + 1] = num_works
                new_parts = parts[batch + 1] if batch + 1 < num_batches else 0
                if has_split:
                    pidx += ql
                batch += 1
                kv_blk, n_split = 0, 0
                kv_beg, kv_end = kv_end, kv_end + new_parts
                remain -= remain_kv
            else:
                if remain > 0:
                    num_works += 1
                    pidx += ql
                    kv_blk += remain
                    n_split += 1
                work_indptr[cid + 1] = num_works
                cid += 1
                remain = remain_for(cid)
            if do_reduce:
                lri += num_splits
                grt += 1
        if cid < num_cu:
            cid += 1
    for index in range(cid, num_cu + 1):
        work_indptr[index] = num_works
    for index in range(grt, num_batches + 1):
        reduce_indptr[index] = lri
    return work_info[:max_work], work_indptr, reduce_indptr, reduce_final_map, reduce_partial_map


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

    parts = [-(-length // _PARTITION) for length in lengths]
    partition_indptr = [0]
    for count in parts:
        partition_indptr.append(partition_indptr[-1] + count)
    works, work_indptr, reduce_indptr, reduce_final_map, reduce_partial_map = _schedule(parts, num_cu, group)
    work_info = torch.tensor(works, dtype=torch.int32)
    if work_info.shape != (max_work, 8) or len(reduce_partial_map) != axes["max_split_tiles"]:
        raise RuntimeError("the worklist does not match its declared shapes")

    def ints(values):
        return torch.tensor(values, dtype=torch.int32).to(device)

    return {
        "key_cache": key_cache, "value_cache": value_cache, "key_scale": key_scale, "value_scale": value_scale,
        "context_lengths": context_lengths.to(torch.int32), "kv_indptr": ints(kv_indptr),
        "kv_page_indices": kv_page_indices, "partition_indptr": ints(partition_indptr),
        "work_indptr": ints(work_indptr), "work_info": work_info.to(device),
        "reduce_indptr": ints(reduce_indptr), "reduce_final_map": ints(reduce_final_map).view(num_seqs, 2),
        "reduce_partial_map": ints(reduce_partial_map),
    }
