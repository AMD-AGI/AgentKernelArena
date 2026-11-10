"""Float64 reference of the Muse Spark verify attention (one full-attention layer, one TP8 rank).

Row t of the step is draft position j = t % 8 of request r = t // 8, whose committed context is
seq_lens[r] keys. Query head h (16 a rank) reads KV head h // 8. The request's keys are positions
0 .. seq_lens[r] + 7 (the step's 8 rows are already in the cache), key i at slot
page_table[r, i // 16] * 16 + i % 16 of the float8_e4m3fn planes, used as exact values. Row t sees
key i when i <= seq_lens[r] + j (causal; the layer is a full-attention layer, no window):

    o[t, h] = softmax_i(scale * q[t, h] . K[i, h // 8]) @ V[:, h // 8],   scale = 64 ** -0.5
"""

import torch

Q_LEN = 8
PAGE = 16
HEAD_DIM = 64
KV_HEADS = 2
GQA = 8
SOFTMAX_SCALE = HEAD_DIM ** -0.5


def reference(*, q, k_cache, v_cache, page_table, seq_lens):
    rows = q.shape[0]
    R = seq_lens.shape[0]
    assert rows == R * Q_LEN
    out = torch.zeros(rows, KV_HEADS, GQA, HEAD_DIM, dtype=torch.float64, device=q.device)
    qpos = torch.arange(Q_LEN, device=q.device)
    for r in range(R):
        prefix = int(seq_lens[r])
        keys = prefix + Q_LEN
        pages = page_table[r, :-(-keys // PAGE)].long()
        slots = (pages[:, None] * PAGE + torch.arange(PAGE, device=q.device)[None, :]).reshape(-1)[:keys]
        visible = torch.arange(keys, device=q.device)[None, :] <= (prefix + qpos)[:, None]  # [8, keys]
        qr = q[r * Q_LEN:(r + 1) * Q_LEN].double().reshape(Q_LEN, KV_HEADS, GQA, HEAD_DIM)
        for h in range(KV_HEADS):
            K = k_cache[slots, h].double()  # [keys, 64]
            V = v_cache[slots, h].double()
            s = torch.einsum("jgd,kd->jgk", qr[:, h], K) * SOFTMAX_SCALE
            s = s.masked_fill(~visible[:, None, :], float("-inf"))
            out[r * Q_LEN:(r + 1) * Q_LEN, h] = torch.einsum("jgk,kd->jgd", torch.softmax(s, dim=-1), V)
    return out.reshape(rows, KV_HEADS * GQA * HEAD_DIM)
