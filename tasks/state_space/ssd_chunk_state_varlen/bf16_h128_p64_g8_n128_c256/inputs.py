"""Sequence boundaries and decay inputs of the Mamba2 varlen chunk state.

Tokens of n_seqs sequences are packed back to back; the boundaries are n_seqs - 1 distinct uniformly random
cut points, so every sequence has at least one token and sequences start and end anywhere inside a chunk.
dt is the discretization step of every token, U(0, 0.1), and dA_cumsum its decayed chunk-local running sum
with a negative per-head rate over the valid tokens, so it is non-increasing and constant in the padding of
a partial last chunk, exactly as the chunk cumsum pass produces it. The padding slots of dt (token >= n_tokens)
are never read: every sequence ends at or before n_tokens and the operator only sums its own tokens. They
keep their U(0, 0.1) draw instead of the cumsum pass's zeros so that a candidate cannot depend on them and a
mostly padded chunk (1 token) is not a block of exact zeros.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, seqs, heads = axes["n_tokens"], axes["n_seqs"], axes["num_heads"]
    size, chunks = axes["chunk_size"], axes["n_chunks"]
    if seqs > tokens:
        raise ValueError("n_seqs must not exceed n_tokens")
    dt = torch.rand(heads, chunks, size, generator=generator, device=device, dtype=torch.float32) * 0.1
    token = torch.arange(chunks, device=device)[:, None] * size + torch.arange(size, device=device)[None, :]
    valid = token < tokens
    rate = 0.02 + 0.18 * torch.rand(heads, 1, 1, generator=generator, device=device, dtype=torch.float32)
    dA_cumsum = torch.cumsum(dt * valid[None] * -rate, dim=-1)
    cuts = torch.randperm(tokens - 1, generator=generator, device=device)[: seqs - 1] + 1
    cuts, _ = torch.sort(cuts)
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    cu_seqlens = torch.cat([zero, cuts, zero + tokens]).to(torch.int32)
    return {"dt": dt, "dA_cumsum": dA_cumsum, "cu_seqlens": cu_seqlens}
