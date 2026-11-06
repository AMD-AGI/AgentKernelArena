"""Chunk boundaries of one packed sequence for the SSD chunk Gram matrix."""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, chunk, chunks = axes["n_tokens"], axes["chunk_size"], axes["n_chunks"]
    bounds = torch.arange(chunks + 1, device=device, dtype=torch.int64) * chunk
    return {"cu_chunk_seqlens": bounds.clamp(max=tokens).to(torch.int32)}
