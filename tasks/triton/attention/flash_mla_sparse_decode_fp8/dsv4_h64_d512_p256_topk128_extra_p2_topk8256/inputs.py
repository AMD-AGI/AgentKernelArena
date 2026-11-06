"""Packed FP8 KV pools and sparse index tables of a DeepSeek-V4 C128A decode batch.

Every request (row) has a context of c tokens, drawn between ctx/2 and ctx (row 0 has
exactly ctx). Its sliding-window (main) entries are the last min(c, 128) tokens, kept
in a 256-slot ring page of the main pool; its compressed (extra) entries are all
c // 128 C128 tokens of its history, kept in its own pages of the extra pool (two
compressed tokens per page, one page per 256 original tokens). Pages are handed out
in shuffled order, as a block allocator does after requests have come and gone.

Both pools hold production-quantized latent rows: 448 NoPE values as OCP E4M3 codes
with one UE8M0 scale per 64 values (scale = 2^ceil(log2(amax / 448)), as the DSv4
cache writers compute it) and 64 RoPE values in bfloat16.

Rows 1 to 10 of a batch, when present, exercise the rest of the index contract: an
empty row, -1 holes inside both prefixes, legal slots past both prefixes, arbitrary
slots anywhere in the pools, an empty main prefix, an extra prefix whose only
entry is -1, and main prefixes of 1, 63, 65 and 127 entries (the rest of the
window left in the table) next to an unchanged extra prefix.
"""
import torch

NOPE_DIM, ROPE_DIM, GROUP = 448, 64, 64
NUM_GROUPS = NOPE_DIM // GROUP
PAYLOAD_BYTES, SCALE_BYTES = 576, 8
RECORD_BYTES = PAYLOAD_BYTES + SCALE_BYTES
FP8_MAX = 448.0
TOKENS_PER_PAGE = 256  # original tokens covered by one page of either pool
CHUNK_SLOTS = 1 << 18  # slots quantized per step, to bound temporary memory


def _fill_pool(pages, page_size, generator, device):
    """A [pages, page_size, 1, 584] pool of quantized latent rows (bytes in float8_e4m3fn storage)."""
    pool = torch.empty((pages, page_size * RECORD_BYTES), dtype=torch.uint8, device=device)
    step = max(1, CHUNK_SLOTS // page_size)
    for start in range(0, pages, step):
        stop = min(pages, start + step)
        count = (stop - start) * page_size
        # Latent channels differ in magnitude: each 64-value group gets its own spread.
        spread = torch.exp2(torch.rand((count, NUM_GROUPS, 1), generator=generator, device=device) * 4.0 - 2.0)
        nope = torch.randn((count, NUM_GROUPS, GROUP), generator=generator, device=device) * spread
        nope = nope.to(torch.bfloat16).float()
        amax = nope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(amax / FP8_MAX))
        codes = (nope / torch.exp2(exponent)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        rope = torch.randn((count, ROPE_DIM), generator=generator, device=device).to(torch.bfloat16)
        block = pool[start:stop]
        payload = block[:, : page_size * PAYLOAD_BYTES].view(stop - start, page_size, PAYLOAD_BYTES)
        payload[..., :NOPE_DIM] = codes.view(torch.uint8).view(stop - start, page_size, NOPE_DIM)
        payload[..., NOPE_DIM:] = rope.view(torch.uint8).view(stop - start, page_size, 2 * ROPE_DIM)
        scales = block[:, page_size * PAYLOAD_BYTES :].view(stop - start, page_size, SCALE_BYTES)
        scales[..., :NUM_GROUPS] = (exponent + 127.0).to(torch.uint8).view(stop - start, page_size, NUM_GROUPS)
        scales[..., NUM_GROUPS] = 0  # the record's padding byte
    return pool.view(torch.float8_e4m3fn).view(pages, page_size, 1, RECORD_BYTES)


def _contexts(batch, ctx, generator, device):
    low = (ctx + 1) // 2
    contexts = torch.randint(low, ctx + 1, (batch,), generator=generator, device=device, dtype=torch.int64)
    contexts[0] = ctx
    return contexts


def _main_table(contexts, pages, page_size, width, generator, device):
    """Each request's window of the last min(c, width) tokens in its ring page."""
    batch = contexts.shape[0]
    page = torch.randperm(pages, generator=generator, device=device)[:batch]
    j = torch.arange(width, device=device)
    window = contexts.clamp(max=width)
    position = (contexts - window)[:, None] + j[None, :]
    slots = page[:, None] * page_size + position % page_size
    table = torch.where(j[None, :] < window[:, None], slots, torch.full_like(slots, -1))
    return table, window


def _extra_table(contexts, ctx, pages, page_size, width, generator, device):
    """All c // 128 compressed tokens of each request, in order, through its pages."""
    batch = contexts.shape[0]
    per_request = pages // batch
    owned = torch.randperm(pages, generator=generator, device=device).view(batch, per_request)
    count = contexts // (TOKENS_PER_PAGE // page_size)
    j = torch.arange(width, device=device)
    page_of = (j // page_size).clamp(max=per_request - 1)
    slots = owned[:, page_of] * page_size + (j % page_size)[None, :]
    table = torch.where(j[None, :] < count[:, None], slots, torch.full_like(slots, -1))
    return table, count


def _edge_rows(tables, lengths, capacities, generator, device):
    """Rows 1-10: the parts of the index contract a production batch rarely shows."""
    batch = tables[0].shape[0]
    if batch > 1:  # no KV entry at all: zero output, lse = +inf
        for table, length in zip(tables, lengths):
            table[1] = -1
            length[1] = 0
    if batch > 2:  # -1 holes inside both prefixes select nothing
        for table, length in zip(tables, lengths):
            j = torch.arange(table.shape[1], device=device)
            table[2] = torch.where((j % 3 == 2) & (j < length[2]), torch.full_like(table[2], -1), table[2])
    if batch > 3:  # legal slots past both prefixes must not contribute
        for table, length, capacity in zip(tables, lengths, capacities):
            j = torch.arange(table.shape[1], device=device)
            noise = torch.randint(0, capacity, (table.shape[1],), generator=generator, device=device)
            table[3] = torch.where(j >= length[3], noise, table[3])
    if batch > 4:  # arbitrary (possibly repeated) slots anywhere in the pools
        for table, length, capacity in zip(tables, lengths, capacities):
            j = torch.arange(table.shape[1], device=device)
            noise = torch.randint(0, capacity, (table.shape[1],), generator=generator, device=device)
            table[4] = torch.where(j < length[4], noise, table[4])
    if batch > 5:  # empty main prefix, extra prefix unchanged
        tables[0][5] = -1
        lengths[0][5] = 0
    if batch > 6:  # history too short for any extra entry: one -1 inside the prefix
        tables[1][6] = -1
        lengths[1][6] = 1
    # Rows 7-10: a partial window of 1, 63, 65 or 127 entries next to the unchanged extra
    # prefix; the rest of the window stays in the table, past the prefix.
    for row, keep in zip(range(7, 11), (1, 63, 65, 127)):
        if batch > row:
            lengths[0][row] = lengths[0][row].clamp(max=keep)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, ctx = int(axes["batch"]), int(axes["ctx"])
    pages, page_size, width = int(axes["pages"]), int(axes["page_size"]), int(axes["topk"])
    extra_pages, extra_page_size = int(axes["extra_pages"]), int(axes["extra_page_size"])
    extra_width = int(axes["extra_topk"])
    kv_cache = _fill_pool(pages, page_size, generator, device)
    extra_kv_cache = _fill_pool(extra_pages, extra_page_size, generator, device)
    contexts = _contexts(batch, ctx, generator, device)
    main, main_len = _main_table(contexts, pages, page_size, width, generator, device)
    extra, extra_len = _extra_table(contexts, ctx, extra_pages, extra_page_size, extra_width, generator, device)
    _edge_rows((main, extra), (main_len, extra_len), (pages * page_size, extra_pages * extra_page_size),
               generator, device)
    return {
        "kv_cache": kv_cache,
        "sparse_indices": main.to(torch.int32).view(batch, 1, width),
        "sparse_lens": main_len.to(torch.int32),
        "extra_kv_cache": extra_kv_cache,
        "extra_sparse_indices": extra.to(torch.int32).view(batch, 1, extra_width),
        "extra_sparse_lens": extra_len.to(torch.int32),
    }
