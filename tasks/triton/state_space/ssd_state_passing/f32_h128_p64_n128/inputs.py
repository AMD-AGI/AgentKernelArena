"""Chunk grouping and decay inputs of the Mamba2 state passing.

The n_chunks chunks are split into n_seqs consecutive non-empty groups at uniformly random cut points.
dA_cumsum is the decayed chunk-local running sum of a random step dt in [0, 0.1) with a negative per-head
rate, so it is non-increasing within a chunk, exactly as the chunk cumsum pass produces it.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    chunks, seqs, heads, size = axes["n_chunks"], axes["n_seqs"], axes["num_heads"], axes["chunk_size"]
    if seqs > chunks:
        raise ValueError("n_seqs must not exceed n_chunks")
    dt = torch.rand(heads, chunks, size, generator=generator, device=device, dtype=torch.float32) * 0.1
    rate = 0.02 + 0.18 * torch.rand(heads, 1, 1, generator=generator, device=device, dtype=torch.float32)
    dA_cumsum = torch.cumsum(dt * -rate, dim=-1)
    cuts = torch.randperm(chunks - 1, generator=generator, device=device)[: seqs - 1] + 1
    cuts, _ = torch.sort(cuts)
    last = torch.cat([cuts - 1, torch.full((1,), chunks - 1, dtype=torch.int64, device=device)])
    return {"dA_cumsum": dA_cumsum, "last_chunk_indices": last.to(torch.int32)}
