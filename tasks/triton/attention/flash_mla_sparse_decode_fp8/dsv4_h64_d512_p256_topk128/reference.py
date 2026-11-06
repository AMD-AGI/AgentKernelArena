"""Float64 DSv4 sparse attention over a packed FP8 KV pool, used only for correctness.

A slot s = page * page_size + position decodes to a 512-dimensional vector that
is both key and value: dimensions 0..447 are the float8_e4m3fn codes of its
payload times 2**(scale - 127), with one UE8M0 scale byte per 64 codes, and
dimensions 448..511 are its 64 bfloat16 rope values. Row b selects the
nonnegative entries among the first sparse_lens[b] of sparse_indices[b, 0].
"""
import torch

NOPE = 448
TILE = 64
PAYLOAD = 576
RECORD = 584
ROW_CHUNK = 256


def _decode(kv_cache, slots):
    pages, page_size = kv_cache.shape[0], kv_cache.shape[1]
    raw = kv_cache.view(torch.uint8).reshape(pages, page_size * RECORD)
    page = slots // page_size
    position = slots % page_size
    device = kv_cache.device
    payload = raw[page[:, None], position[:, None] * PAYLOAD + torch.arange(PAYLOAD, device=device)]
    scale_bytes = raw[page[:, None], page_size * PAYLOAD + position[:, None] * 8
                      + torch.arange(NOPE // TILE, device=device)]
    codes = payload[:, :NOPE].contiguous().view(torch.float8_e4m3fn).double()
    scales = torch.exp2(scale_bytes.double() - 127.0).repeat_interleave(TILE, dim=1)
    rope = payload[:, NOPE:].contiguous().view(torch.bfloat16).double()
    return torch.cat((codes * scales, rope), dim=1)


def reference(*, q, kv_cache, sparse_indices, sparse_lens, sm_scale, sinks):
    rows, query_length, heads, head_dim = q.shape
    topk = sparse_indices.shape[-1]
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    lse = torch.empty(q.shape[:-1], dtype=torch.float64, device=q.device)
    sink = sinks.double()[None, :]
    position = torch.arange(topk, device=q.device)[None, :]
    for first in range(0, rows, ROW_CHUNK):
        last = min(first + ROW_CHUNK, rows)
        count = last - first
        index = sparse_indices[first:last, 0].long()
        valid = (position < sparse_lens[first:last, None].long()) & (index >= 0)
        kv = _decode(kv_cache, index.clamp_min(0).reshape(-1)).view(count, topk, head_dim)
        logits = torch.einsum("chd,ctd->cht", q[first:last, 0].double(), kv) * sm_scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        row_lse = torch.logsumexp(logits, dim=-1)                     # -inf for an empty row
        top = torch.maximum(row_lse, sink)
        weights = torch.exp(logits - top[..., None]) / (torch.exp(row_lse - top) + torch.exp(sink - top))[..., None]
        out[first:last, 0] = torch.einsum("cht,ctd->chd", weights, kv)
        lse[first:last, 0] = torch.where(torch.isneginf(row_lse), torch.full_like(row_lse, float("inf")), row_lse)
    return {"output": out, "lse": lse}
