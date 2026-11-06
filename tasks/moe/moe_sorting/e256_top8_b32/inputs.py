"""Balanced top-8 routing of 256 experts with distinct experts per token.

Tokens are taken in chunks of 32; within a chunk the 256 (token, slot) positions hold every expert exactly once in a
random order, so each token's 8 experts are distinct and every expert receives n_tokens / 32 assignments (a partial
last chunk gives one more assignment to a random subset of experts). Balanced counts make the padded length of the
sorted output a closed-form function of n_tokens (the declared sorted_len), so every output element is written and can
be compared.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, top_k, experts = axes["n_tokens"], axes["top_k"], axes["num_experts"]
    assert experts == top_k * 32
    chunks = -(-n_tokens // 32)
    scores = torch.rand(chunks, experts, generator=generator, device=device)
    expert_order = scores.argsort(dim=1)                     # a random permutation of the experts per chunk
    ids = expert_order.reshape(chunks * 32, top_k)[:n_tokens]
    return {"topk_ids": ids.to(torch.int32).contiguous()}
