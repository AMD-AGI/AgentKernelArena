"""High-precision reference of MiniMax lightning-attention prefill, used only for correctness.

Tokens are split into blocks of 256 (the last may be partial). Per head with decay slope s:

* inside a block, token t attends to tokens j <= t of the same block with weight
  exp(-s (t - j)) <q_t, k_j>;
* a block's state is sum_c exp(-s (m - 1 - c)) k_c v_c^T over its m tokens (the decay is counted
  from the block's last token); the state entering block b is the history decayed by exp(-s m)
  per block plus the earlier blocks' states, starting from kv_history;
* token p (0-based) of a block adds exp(-s p) q_p (state entering the block).

Cross-block weights are therefore exp(-s (t - j - 1)), one step less decay than inside a block;
this is the behavior of the production kernels, not an error of the reference. The final state is
the state after the last block, which replaces kv_history.
"""
import torch

BLOCK = 256


def reference(*, q, k, v, slope_rate, kv_history):
    heads, tokens, dim = q.shape
    value_dim = v.shape[-1]
    blocks = -(-tokens // BLOCK)
    pad = blocks * BLOCK - tokens
    last = tokens - (blocks - 1) * BLOCK
    sizes = torch.full((blocks,), BLOCK, dtype=torch.float64, device=q.device)
    sizes[-1] = last
    pos = torch.arange(BLOCK, dtype=torch.float64, device=q.device)
    diff = pos[:, None] - pos[None, :]
    out = torch.empty(heads, tokens, value_dim, dtype=torch.float64, device=q.device)
    final = torch.empty(heads, dim, value_dim, dtype=torch.float64, device=q.device)

    def blocked(x):
        x = x.double()
        if pad:
            x = torch.cat([x, x.new_zeros(pad, x.shape[-1])])
        return x.view(blocks, BLOCK, x.shape[-1])

    for h in range(heads):
        s = slope_rate[h].double()
        qh, kh, vh = blocked(q[h]), blocked(k[h]), blocked(v[h])
        causal = torch.where(diff >= 0, torch.exp(-s * diff), torch.zeros_like(diff))
        scores = torch.einsum("nid,njd->nij", qh, kh) * causal
        result = torch.einsum("nij,nje->nie", scores, vh)
        # Decay of every key to the end of its block; padding keys are zero.
        key_weight = torch.where(pos[None, :] < sizes[:, None],
                                 torch.exp(-s * (sizes[:, None] - 1 - pos[None, :])),
                                 torch.zeros_like(pos[None, :]))
        block_state = torch.einsum("nc,ncd,nce->nde", key_weight, kh, vh)
        state = kv_history[h].double()
        entering = []
        for b in range(blocks):
            entering.append(state)
            state = torch.exp(-s * sizes[b]) * state + block_state[b]
        final[h] = state
        cross = torch.einsum("nid,nde->nie", qh, torch.stack(entering))
        result = result + cross * torch.exp(-s * pos)[None, :, None]
        out[h] = result.reshape(-1, value_dim)[:tokens]
    return out, final
