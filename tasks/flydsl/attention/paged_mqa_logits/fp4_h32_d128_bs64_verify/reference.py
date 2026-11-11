"""float64 reference for the FP4 lightning-indexer logits of a DeepSeek-V4.1-Flash DSpark verify step."""
import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_PAGE = 64
_KEY_CHUNK = 32768  # keys scored per step, bounding the temporaries


def _unpack(packed):
    """float64 e2m1 values of packed codes [..., B] -> [..., 2 B] (element 2i in the low nibble of byte i)."""
    levels = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    return torch.stack([levels[(packed & 15).long()], levels[(packed >> 4).long()]], dim=-1).flatten(-2)


def reference(*, q_fp4, q_scale, weights, kv_cache, kv_scale, page_table, bounds, owners):
    rows, heads = q_fp4.shape[0], q_fp4.shape[1]
    chunks = q_scale.shape[2]
    width = int(bounds.max())  # the logits' columns: the longest row's visible columns (ctx)
    logits = torch.full((rows, width), float("-inf"), dtype=torch.float64, device=q_fp4.device)

    # queries [rows, heads, 128]: chunk c of head h (dims 32 c .. 32 c + 31) times 2 ** (q_scale[t, 0, c, h % 16,
    # h // 16] - 127)
    h = torch.arange(heads, device=q_fp4.device)
    q_exp = q_scale[:, 0][:, :, h % 16, h // 16].double().transpose(1, 2)  # [rows, heads, chunks]
    query = (_unpack(q_fp4).reshape(rows, heads, chunks, -1) * torch.exp2(q_exp - 127.0)[..., None]).reshape(
        rows, heads, -1)
    head_weight = weights.double()

    t = 0
    while t < rows:
        # the rows that share this row's page table (a request's rows) score the same keys
        u = t + 1
        while u < rows and torch.equal(page_table[u], page_table[t]):
            u += 1
        group = torch.arange(t, u, device=q_fp4.device)
        seen = bounds[t:u].long()
        limit = int(seen.max())
        for start in range(0, limit, _KEY_CHUNK):
            stop = min(limit, start + _KEY_CHUNK)
            cols = torch.arange(start, stop, device=q_fp4.device)
            page = page_table[t][cols // _PAGE].long()
            slot = cols % _PAGE
            # key row `slot` of plane page `page`: payload [chunks, 16 B]; its chunk-c scale at (slot % 16) * 4 +
            # slot // 16
            payload = _unpack(kv_cache[page, 0, :, slot, :])  # [n, chunks, 32]
            k_exp = kv_scale[page[:, None], 0, torch.arange(chunks, device=q_fp4.device)[None, :],
                             ((slot % 16) * 4 + slot // 16)[:, None]].double()  # [n, chunks]
            key = (payload * torch.exp2(k_exp - 127.0)[..., None]).reshape(stop - start, -1)
            score = torch.einsum("ghd,nd->ghn", query[group], key)
            value = (head_weight[group][:, :, None] * torch.relu(score)).sum(dim=1)  # [rows of group, n]
            visible = cols[None, :] < seen[:, None]
            logits[t:u, start:stop] = torch.where(visible, value, torch.full_like(value, float("-inf")))
        t = u
    return logits
