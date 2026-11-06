"""Packed FP8 KV pool, sparse index tables and valid lengths for DSv4 sparse decode.

Pool layout (one page = page_size tokens, 584 bytes per token): the page holds
page_size payloads of 576 bytes (448 float8_e4m3fn codes, then 64 bfloat16 rope
values), followed by page_size scale records of 8 bytes (7 UE8M0 exponents, one
per 64 codes, then a zero pad byte). Every slot is written the way the DSv4 KV
quantizer writes it: a bfloat16 key vector whose 448 leading dimensions are
quantized in tiles of 64 with scale 2**ceil(log2(amax / 448)).

Index tables. A case of at most DECODE_MAX_ROWS rows is a decode batch, one row
per request, drawn as in the source task: uniform legal slots over the whole
pool. Most rows use the full 128-entry window; some rows are shorter (sequence
starts) and a few are empty. A larger case is a prefill chunk, one row per
prompt token as the ROCm DSv4 backend expands it: the chunk is split into
prompts (some continuing an already cached prefix), each prompt's tokens occupy
its own shuffled pages of the pool, and the row of the token at position p lists
the slots of positions p, p - 1, ..., p - 127 (-1 for negative positions) with
length min(p + 1, 128), so neighbouring rows share their windows.

In every case, positions past a row's length are -1 except in rows whose tail
keeps legal slots that must not contribute, and some rows have -1 holes inside
their prefix. Rows 0-7 are fixed boundary rows of uniform legal slots: lengths
128, 0 (legal tail), 1 (whose only entry is -1), 64, 63, 65 and 127 (legal
tails), and 128 with every third entry -1.
"""
import torch

NOPE = 448
ROPE = 64
TILE = 64
PAYLOAD = NOPE + 2 * ROPE       # 576 bytes
SCALE_RECORD = 8
RECORD = PAYLOAD + SCALE_RECORD  # 584 bytes
FP8_MAX = 448.0
PAGE_CHUNK = 64
DECODE_MAX_ROWS = 4096           # the source workload's largest decode batch
PROMPT_MIN, PROMPT_MAX = 256, 8192
CACHED_PREFIX_MAX = 8192


def _write_pool(cache, generator, device):
    pages, page_size = cache.shape[0], cache.shape[1]
    raw = cache.view(torch.uint8).view(pages, page_size * RECORD)
    for first in range(0, pages, PAGE_CHUNK):
        last = min(first + PAGE_CHUNK, pages)
        n = last - first
        # Token and tile magnitudes vary so the stored exponents vary.
        token_scale = torch.exp2(torch.rand((n, page_size, 1), generator=generator, device=device) * 2 - 1)
        tile_scale = torch.exp2(torch.rand((n, page_size, NOPE // TILE, 1), generator=generator,
                                           device=device) - 0.5)
        key = (torch.randn((n, page_size, NOPE + ROPE), generator=generator, device=device)
               * token_scale).to(torch.bfloat16)
        nope = key[..., :NOPE].float().view(n, page_size, NOPE // TILE, TILE) * tile_scale
        nope = nope.to(torch.bfloat16).float()
        amax = nope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        exponent = torch.ceil(torch.log2(amax / FP8_MAX))
        codes = (nope * torch.exp2(-exponent)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        payload = torch.cat((codes.reshape(n, page_size, NOPE).view(torch.uint8),
                             key[..., NOPE:].contiguous().view(torch.uint8)), dim=-1)
        scales = torch.zeros((n, page_size, SCALE_RECORD), dtype=torch.uint8, device=device)
        scales[..., :NOPE // TILE] = (exponent.squeeze(-1) + 127).to(torch.uint8)
        raw[first:last, :page_size * PAYLOAD] = payload.reshape(n, page_size * PAYLOAD)
        raw[first:last, page_size * PAYLOAD:] = scales.reshape(n, page_size * SCALE_RECORD)


def _decode_tables(rows, topk, capacity, generator, device):
    """One request per row: uniform legal slots, mostly full windows."""
    indices = torch.randint(0, capacity, (rows, topk), generator=generator, device=device)
    draw = torch.rand((rows,), generator=generator, device=device)
    short_len = torch.randint(1, topk, (rows,), generator=generator, device=device)
    lengths = torch.where(draw < 0.125, short_len, torch.full_like(short_len, topk))
    lengths = torch.where(draw > 0.99, torch.zeros_like(lengths), lengths)
    return indices, lengths


def _prefill_tables(rows, topk, pages, page_size, generator, device):
    """One row per prompt token: each row's window of the last min(p + 1, topk) positions."""
    sizes = []
    while sum(sizes) < rows:
        sizes.append(int(torch.randint(PROMPT_MIN, PROMPT_MAX + 1, (1,), generator=generator,
                                       device=device).item()))
    sizes[-1] -= sum(sizes) - rows
    count = len(sizes)
    cached = torch.randint(1, CACHED_PREFIX_MAX + 1, (count,), generator=generator, device=device)
    fresh = torch.rand((count,), generator=generator, device=device) < 0.5
    starts = torch.where(fresh, torch.zeros_like(cached), cached)
    size = torch.tensor(sizes, dtype=torch.int64, device=device)
    # Pages that hold each prompt's window positions [start - topk + 1, start + size).
    first_page = (starts - (topk - 1)).clamp_min(0) // page_size
    last_page = (starts + size - 1) // page_size
    page_count = last_page - first_page + 1
    if int(page_count.sum()) > pages:
        raise ValueError(f"a {rows}-row prefill chunk needs more than {pages} pages")
    page_base = torch.cumsum(page_count, 0) - page_count
    shuffled = torch.randperm(pages, generator=generator, device=device)
    prompt = torch.repeat_interleave(torch.arange(count, device=device), size)
    position = starts[prompt] + torch.arange(rows, device=device) - (torch.cumsum(size, 0) - size)[prompt]
    token = position[:, None] - torch.arange(topk, device=device)[None, :]
    page = shuffled[page_base[prompt][:, None] + token.clamp_min(0) // page_size - first_page[prompt][:, None]]
    indices = torch.where(token >= 0, page * page_size + token.clamp_min(0) % page_size,
                          torch.full_like(token, -1))
    return indices, (position + 1).clamp(max=topk)


def _index_tables(rows, topk, pages, page_size, generator, device):
    capacity = pages * page_size
    if rows <= DECODE_MAX_ROWS:
        indices, lengths = _decode_tables(rows, topk, capacity, generator, device)
    else:
        indices, lengths = _prefill_tables(rows, topk, pages, page_size, generator, device)
    draw = torch.rand((rows, 2), generator=generator, device=device)
    legal_tail = draw[:, 0] < 0.5
    holes = draw[:, 1] < 1.0 / 16
    fixed = min(rows, 8)
    indices[:fixed] = torch.randint(0, capacity, (fixed, topk), generator=generator, device=device)
    lengths[:fixed] = torch.tensor([topk, 0, 1, topk // 2, topk // 2 - 1, topk // 2 + 1, topk - 1, topk],
                                   dtype=lengths.dtype, device=device)[:fixed]
    legal_tail[:fixed] = torch.tensor([True, True, True, True, True, False, True, True],
                                      device=device)[:fixed]
    holes[:fixed] = torch.tensor([False] * 7 + [True], device=device)[:fixed]
    position = torch.arange(topk, device=device)[None, :]
    past = position >= lengths[:, None]
    tail = torch.randint(0, capacity, (rows, topk), generator=generator, device=device)
    indices = torch.where(past, torch.where(legal_tail[:, None], tail, torch.full_like(tail, -1)), indices)
    indices = indices.masked_fill(holes[:, None] & ~past & (position % 3 == 2), -1)
    if rows > 2:
        indices[2, 0] = -1
    return indices.to(torch.int32), lengths.to(torch.int32)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    rows, topk = axes["batch"], axes["topk"]
    if axes["packed_width"] != RECORD:
        raise ValueError("the DSv4 packed record is 584 bytes")
    cache = torch.empty((axes["pages"], axes["page_size"], axes["one"], axes["packed_width"]),
                        dtype=torch.float8_e4m3fn, device=device)
    _write_pool(cache, generator, device)
    indices, lengths = _index_tables(rows, topk, axes["pages"], axes["page_size"], generator, device)
    return {"kv_cache": cache,
            "sparse_indices": indices.view(rows, axes["query_length"], topk),
            "sparse_lens": lengths}
