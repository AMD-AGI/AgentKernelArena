"""Chunk-local gates and the solved intra-chunk inverse of the Gated DeltaNet prefill.

In production `g` is the chunk-local cumulative sum of the per-token log-decay, which is
non-positive, and `A` is the inverse of (I + tril(beta * K K^T * exp(g_i - g_j))) of every
64-token chunk and value head (the output of the triangular solve that precedes `recompute_w_u`):
unit lower triangular, stored as bfloat16 [T, H, 64] with zeros outside the chunk's valid rows
and columns. The generator draws per-token log-decays in [-0.2, 0], cumulates them inside
each chunk in float32, and computes the inverse from the generated keys and beta in float32.
"""
import torch

CHUNK_SIZE = 64
CHUNKS_PER_STEP = 128  # whole chunks per step, to bound temporaries


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    group = heads // axes["num_key_heads"]
    k, beta = inputs["k"], inputs["beta"]
    pad = (-tokens) % CHUNK_SIZE
    chunks = (tokens + pad) // CHUNK_SIZE

    step = -torch.rand(tokens, heads, generator=generator, device=device, dtype=torch.float32) * 0.2
    step = torch.cat([step, step.new_zeros(pad, heads)])
    g_all = step.view(chunks, CHUNK_SIZE, heads).cumsum(dim=1)

    kp = torch.cat([k, k.new_zeros(pad, *k.shape[1:])]).view(chunks, CHUNK_SIZE, *k.shape[1:])
    bp = torch.cat([beta, beta.new_zeros(pad, heads)]).view(chunks, CHUNK_SIZE, heads)
    a_all = torch.empty(chunks, CHUNK_SIZE, heads, CHUNK_SIZE, dtype=torch.bfloat16, device=device)
    lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=device), diagonal=-1)
    eye = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=device)
    for start in range(0, chunks, CHUNKS_PER_STEP):
        stop = min(start + CHUNKS_PER_STEP, chunks)
        kc = kp[start:stop].float().repeat_interleave(group, dim=2)       # [c, 64, H, K]
        gram = torch.einsum("cihk,cjhk->chij", kc, kc)
        gc = g_all[start:stop].permute(0, 2, 1)                           # [c, H, 64]
        decay = torch.exp(gc[:, :, :, None] - gc[:, :, None, :])
        bc = bp[start:stop].float().permute(0, 2, 1)                      # [c, H, 64]
        block = torch.where(lower, bc[:, :, :, None] * gram * decay, torch.zeros_like(gram))
        inverse = torch.linalg.solve_triangular(block + eye, eye.expand_as(block), upper=False,
                                                unitriangular=True)
        a_all[start:stop] = inverse.permute(0, 2, 1, 3).to(torch.bfloat16)
    g = g_all.reshape(-1, heads)[:tokens].contiguous()
    a = a_all.reshape(-1, heads, CHUNK_SIZE)[:tokens].contiguous()
    return {"g": g, "A": a}
