"""Routing metadata of an expert-parallel MoE layer after the grouped GEMM.

Every token selected 8 distinct experts of 256 (a fraction of the assignments is -1, not served here). The rows of
`input_tensor` are the expert-ordered outputs, one row per (token, expert) assignment; `input_index[t, k]` is the row of
`input_tensor` holding the output of assignment (t, k), so the valid indices are distinct. Weights are softmax
routing weights.
"""
import torch

INVALID_FRACTION = 0.06


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, top_k, num_experts = axes["n_tokens"], axes["top_k"], axes["num_experts"]
    scores = torch.rand(n_tokens, num_experts, generator=generator, device=device)
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    drop = torch.rand(n_tokens, top_k, generator=generator, device=device) < INVALID_FRACTION
    ids = torch.where(drop, torch.full_like(ids, -1), ids)
    weights = torch.softmax(torch.randn(n_tokens, top_k, generator=generator, device=device), dim=-1)
    slots = torch.randperm(n_tokens * top_k, generator=generator, device=device).to(torch.int32)
    return {"recv_topk_ids": ids, "recv_topk_weight": weights, "input_index": slots.reshape(n_tokens, top_k)}
