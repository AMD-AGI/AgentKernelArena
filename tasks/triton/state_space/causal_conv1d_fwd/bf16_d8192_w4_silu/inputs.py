"""Packed-sequence metadata for the causal conv1d forward task."""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, n_seqs, cache_lines = axes["n_tokens"], axes["n_seqs"], axes["cache_lines"]
    # Random cut points split n_tokens into n_seqs non-empty sequences of mixed length.
    if n_seqs > 1:
        cuts = torch.randperm(n_tokens - 1, generator=generator, device=device)[: n_seqs - 1] + 1
        cuts = cuts.sort().values.to(torch.int32)
    else:
        cuts = torch.empty(0, dtype=torch.int32, device=device)
    bounds = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device), cuts,
        torch.full((1,), n_tokens, dtype=torch.int32, device=device),
    ])
    # Each sequence owns a distinct cache slot; the pool also holds slots no sequence touches.
    slots = torch.randperm(cache_lines, generator=generator, device=device)[:n_seqs].to(torch.int32)
    has_state = (torch.rand(n_seqs, generator=generator, device=device) < 0.5).to(torch.int32)
    return {"query_start_loc": bounds, "cache_indices": slots, "has_initial_state": has_state}
