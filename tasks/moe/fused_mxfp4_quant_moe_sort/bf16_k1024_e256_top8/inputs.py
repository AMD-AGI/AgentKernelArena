"""Block-padded MoE sorting table for a balanced router (the layout of AITER's moe_sorting).

The n_tokens * 8 (token, slot) pairs are spread over 256 experts as evenly as possible with a random assignment, sorted
by expert (token order within an expert), and each expert's group is padded with the sentinel entry to a multiple of 32
rows. The padded length equals the declared sorted_len exactly, so every sorted row is a valid output row.
"""
import torch

BLOCK = 32


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, top_k, experts = axes["n_tokens"], axes["top_k"], axes["num_experts"]
    pairs = n_tokens * top_k
    expert_of_pair = (torch.arange(pairs, device=device) % experts)[torch.randperm(pairs, generator=generator, device=device)]
    order = torch.sort(expert_of_pair, stable=True).indices          # pair index, grouped by expert, ascending
    sorted_expert = expert_of_pair[order]
    counts = torch.bincount(expert_of_pair, minlength=experts)
    padded = (counts + BLOCK - 1) // BLOCK * BLOCK
    group_start = torch.cumsum(counts, 0) - counts
    padded_start = torch.cumsum(padded, 0) - padded
    rank = torch.arange(pairs, device=device) - group_start[sorted_expert]
    position = padded_start[sorted_expert] + rank
    sorted_len = int(padded.sum())
    assert sorted_len == axes["sorted_len"], (sorted_len, axes["sorted_len"])
    sentinel = (top_k << 24) | n_tokens
    sorted_ids = torch.full((sorted_len,), sentinel, dtype=torch.int64, device=device)
    token, slot = order // top_k, order % top_k
    sorted_ids[position] = (slot << 24) | token
    num_valid_ids = torch.tensor([sorted_len, n_tokens], dtype=torch.int32, device=device)
    return {"sorted_ids": sorted_ids.to(torch.int32), "num_valid_ids": num_valid_ids}
