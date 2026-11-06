"""Float64 reference: sparse decode attention with sinks over two packed FP8 KV pools."""
import math

import torch

NOPE_DIM, ROPE_DIM, GROUP = 448, 64, 64
NUM_GROUPS = NOPE_DIM // GROUP
PAYLOAD_BYTES, SCALE_BYTES = 576, 8
TOKEN_BUDGET = 1 << 17  # rows * selected entries decoded at once


def _selected(table, lengths):
    """Each row's selected slots (nonnegative entries of its prefix), compacted in order."""
    width = table.shape[-1]
    position = torch.arange(width, device=table.device)
    valid = (position[None, :] < lengths.to(torch.int64)[:, None]) & (table >= 0)
    count = int(valid.sum(dim=-1).max()) if table.shape[0] else 0
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)[:, :count]
    slots = torch.gather(table.to(torch.int64), 1, order)
    keep = torch.gather(valid, 1, order)
    return torch.where(keep, slots, torch.zeros_like(slots)), keep


def _decode(pool, slots):
    """The 512-wide float64 latent rows stored at `slots` of a [pages, page_size, 1, 584] byte pool.

    A page holds page_size payloads of 576 bytes (448 E4M3 codes, then 64 bfloat16 values),
    then page_size scale records of 8 bytes (7 UE8M0 exponents, one per 64 codes, and a
    padding byte). NoPE value d is e4m3(code[d]) * 2^(scale[d // 64] - 127).
    """
    pages, page_size = pool.shape[0], pool.shape[1]
    raw = pool.view(torch.uint8).reshape(pages, -1)
    payload = raw[:, : page_size * PAYLOAD_BYTES].view(pages, page_size, PAYLOAD_BYTES)
    scales = raw[:, page_size * PAYLOAD_BYTES :].view(pages, page_size, SCALE_BYTES)
    page, within = slots // page_size, slots % page_size
    record = payload[page, within]
    codes = record[..., :NOPE_DIM].contiguous().view(torch.float8_e4m3fn).to(torch.float32).to(torch.float64)
    exponent = scales[page, within][..., :NUM_GROUPS].to(torch.float64) - 127.0
    nope = codes * torch.exp2(exponent).repeat_interleave(GROUP, dim=-1)
    rope = record[..., NOPE_DIM:].contiguous().view(torch.bfloat16).to(torch.float64)
    return torch.cat((nope, rope), dim=-1)


def reference(*, q, kv_cache, sparse_indices, sparse_lens, extra_kv_cache, extra_sparse_indices,
              extra_sparse_lens, sm_scale, sinks):
    batch, _, heads, dim = q.shape
    output = torch.zeros((batch, 1, heads, dim), dtype=torch.float64, device=q.device)
    lse = torch.full((batch, 1, heads), math.inf, dtype=torch.float64, device=q.device)
    main_slots, main_keep = _selected(sparse_indices[:, 0], sparse_lens)
    extra_slots, extra_keep = _selected(extra_sparse_indices[:, 0], extra_sparse_lens)
    entries = main_slots.shape[1] + extra_slots.shape[1]
    if entries == 0:
        return {"output": output, "lse": lse}
    sink = sinks.to(torch.float64)
    step = max(1, TOKEN_BUDGET // entries)
    for start in range(0, batch, step):
        rows = slice(start, min(batch, start + step))
        kv = torch.cat((_decode(kv_cache, main_slots[rows]), _decode(extra_kv_cache, extra_slots[rows])), dim=1)
        keep = torch.cat((main_keep[rows], extra_keep[rows]), dim=1)
        logits = torch.einsum("bhd,btd->bht", q[rows, 0].to(torch.float64), kv) * sm_scale
        logits = logits.masked_fill(~keep[:, None, :], -math.inf)
        peak = logits.amax(dim=-1)
        present = keep.any(dim=-1)[:, None]
        anchor = torch.where(present, peak, torch.zeros_like(peak))
        total = torch.exp(logits - anchor[..., None]).sum(dim=-1)
        # The log-sum-exp covers the selected logits only; the sink is not a logit.
        lse[rows, 0] = torch.where(present, anchor + torch.log(total), torch.full_like(total, math.inf))
        # The sink joins the softmax denominator without contributing a value row.
        shift = torch.maximum(peak, sink[None, :])
        weights = torch.exp(logits - shift[..., None])
        denominator = weights.sum(dim=-1) + torch.exp(sink[None, :] - shift)
        output[rows, 0] = torch.einsum("bht,btd->bhd", weights, kv) / denominator[..., None]
    return {"output": output, "lse": lse}
