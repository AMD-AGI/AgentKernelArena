"""Chunk-local gate cumsum and strictly lower triangular chunk blocks of the Gated DeltaNet prefill.

g_cumsum and A_raw are what the fused cumsum / K K^T kernel produces from keys k, write strengths beta
and the raw per-token log-decay `-exp(A_log) * softplus(a + dt_bias)` (non-positive): the decay is
drawn with `a + dt_bias ~ N(0, 1)` and a per-head rate `exp(A_log)` log-uniform in [0.02, 0.4] (a design
choice of this task), cumulated inside each 64-token chunk in float32 and scaled by log2(e) for
g_cumsum. A_raw[i, h, j] = beta[i, h] * <k[i, h // 2], k[j, h // 2]> * exp(gc[i] - gc[j]) for j < i
inside a chunk and exactly 0 elsewhere (the producer writes zeros on and above the diagonal, which the
solve kernels rely on), computed here in float32.

The fused kernel and the split path's solve kernel read only the 16 x 16 diagonal blocks of a chunk block
(on and above the diagonal included, unmasked inside the forward substitution, hence the zeros) and the
six 16 x 16 blocks strictly below them. The six blocks strictly above the diagonal blocks (columns
j // 16 > i // 16) are never read by the baseline and are not part of the operator (the reference applies
tril(-1) itself), so here they hold random values that an implementation must ignore. This keeps the input
free of an all-zero or bfloat16-exact masked region.
"""
import math

import torch

CHUNK_SIZE = 64
LOG2_E = math.log2(math.e)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads, key_heads = axes["n_tokens"], axes["num_heads"], axes["num_key_heads"]
    group = heads // key_heads
    low, high = math.log(0.02), math.log(0.4)
    rate = torch.exp(torch.rand(heads, generator=generator, device=device, dtype=torch.float32) * (high - low) + low)
    x = torch.randn(tokens, heads, generator=generator, device=device, dtype=torch.float32)
    step = -rate * torch.nn.functional.softplus(x)
    del x
    pad = (-tokens) % CHUNK_SIZE
    if pad:
        step = torch.cat([step, step.new_zeros(pad, heads)])
    gc = step.view(-1, CHUNK_SIZE, heads).cumsum(dim=1)  # [chunks, 64, heads], natural-log cumulative decay
    del step
    keys = inputs["k"]
    beta = inputs["beta"]
    if pad:
        keys = torch.cat([keys, keys.new_zeros(pad, *keys.shape[1:])])
        beta = torch.cat([beta, beta.new_zeros(pad, heads)])
    keys = keys.view(-1, CHUNK_SIZE, key_heads, keys.shape[-1])
    beta = beta.view(-1, CHUNK_SIZE, heads)
    lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=device), diagonal=-1)
    index = torch.arange(CHUNK_SIZE, device=device)
    above_blocks = (index[None, :] // 16) > (index[:, None] // 16)  # [row, col]: block strictly above the diagonal
    a_raw = torch.empty(gc.shape[0], CHUNK_SIZE, heads, CHUNK_SIZE, dtype=torch.float32, device=device)
    for h in range(heads):
        kh = keys[:, :, h // group].float()                     # [chunks, 64, K]
        gram = kh @ kh.transpose(1, 2)                          # [chunks, 64, 64]
        decay = torch.exp(gc[:, :, h, None] - gc[:, None, :, h])
        block = beta[:, :, h, None] * gram * decay
        block = torch.where(lower, block, torch.zeros_like(block))
        garbage = torch.rand(block.shape, generator=generator, device=device, dtype=torch.float32) * 0.6 - 0.3
        a_raw[:, :, h] = torch.where(above_blocks, garbage, block)
    g_cumsum = (gc * LOG2_E).permute(2, 0, 1).reshape(heads, -1)[:, :tokens].contiguous()
    return {"A_raw": a_raw.view(-1, heads, CHUNK_SIZE)[:tokens].contiguous(), "g_cumsum": g_cumsum}
