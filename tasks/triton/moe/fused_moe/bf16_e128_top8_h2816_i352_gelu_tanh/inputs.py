"""Router decisions of a top-8 MoE layer over 128 experts.

Each token selects 8 distinct experts from random gate logits (the top-8 of a normal draw per expert) and its routing
weights are the softmax over the 8 selected logits, so they sum to one per token (renormalized top-k routing).
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_tokens, top_k, num_experts = axes["num_tokens"], axes["top_k"], axes["num_experts"]
    logits = torch.randn(num_tokens, num_experts, generator=generator, device=device, dtype=torch.float32)
    values, ids = logits.topk(top_k, dim=-1)
    return {"topk_ids": ids.to(torch.int32), "topk_weights": torch.softmax(values, dim=-1)}
