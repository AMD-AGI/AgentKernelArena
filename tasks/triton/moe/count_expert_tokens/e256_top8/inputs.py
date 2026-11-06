"""Routing table of a mixture-of-experts layer: each token picks 8 distinct experts of 256.

A fraction of the entries is replaced by -1, the marker vLLM uses for an assignment that is not served (an expert
owned by another rank or a dropped token); those entries must not be counted.
"""
import torch

INVALID_FRACTION = 0.06


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    num_experts, top_k = case["axes"]["num_experts"], case["axes"]["top_k"]
    # Gate scores -> top-k distinct experts per token, like a router; skewed popularity so counts differ per expert.
    popularity = torch.rand(num_experts, generator=generator, device=device) * 2.0
    scores = torch.rand(n_tokens, num_experts, generator=generator, device=device) + popularity
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    drop = torch.rand(n_tokens, top_k, generator=generator, device=device) < INVALID_FRACTION
    ids = torch.where(drop, torch.full_like(ids, -1), ids)
    return {"topk_ids": ids}
