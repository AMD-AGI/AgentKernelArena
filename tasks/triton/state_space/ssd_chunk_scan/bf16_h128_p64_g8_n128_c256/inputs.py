"""Chunk metadata and decay inputs of the Mamba2 chunk scan.

Tokens form consecutive chunks of 256 (the last clipped), and every 8 chunks start a new sequence. dt is
the discretization step of every token, U(0, 0.1), and dA_cumsum its decayed chunk-local running sum with a
negative per-head rate over the chunk's valid tokens, so it is non-increasing and constant in the padding of a
partial chunk, exactly as the chunk cumsum pass produces it. The padding slots of dt are never read for a
written output row (the operator sums over k <= m < chunk length); they keep their U(0, 0.1) draw instead of
the cumsum pass's zeros so that a candidate cannot depend on them and a mostly padded chunk (1 token) is not a
block of exact zeros.
"""
import torch

CHUNKS_PER_SEQUENCE = 8


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    size, chunks = axes["chunk_size"], axes["n_chunks"]
    dt = torch.rand(heads, chunks, size, generator=generator, device=device, dtype=torch.float32) * 0.1
    boundaries = (torch.arange(chunks + 1, device=device, dtype=torch.int64) * size).clamp(max=tokens)
    lengths = boundaries[1:] - boundaries[:-1]
    valid = torch.arange(size, device=device)[None, :] < lengths[:, None]
    rate = 0.02 + 0.18 * torch.rand(heads, 1, 1, generator=generator, device=device, dtype=torch.float32)
    dA_cumsum = torch.cumsum(dt * valid[None] * -rate, dim=-1)
    seq_idx = torch.div(torch.arange(chunks, device=device), CHUNKS_PER_SEQUENCE, rounding_mode="floor")
    return {"dt": dt, "dA_cumsum": dA_cumsum, "cu_chunk_seqlens": boundaries.to(torch.int32),
            "seq_idx": seq_idx.to(torch.int32)}
