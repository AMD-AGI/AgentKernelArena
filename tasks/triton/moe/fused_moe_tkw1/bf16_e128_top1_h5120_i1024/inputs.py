"""Expert weights and router decisions of a Llama-4-style top-1 MoE layer over 128 experts.

Weights. Each expert matrix is normal with standard deviation 1/sqrt(K) of its reduction (K = 5120 for the
gate/up projection w13, K = 1024 for the down projection w2), rounded to bfloat16, so the gate, the activation
and the output stay of order one. The weights are drawn a few experts at a time: the bank is 4 GB per draw, and a
single float32 draw of it would need four times that.

Routing. Each token selects the expert with the largest router logit, and its routing weight is the sigmoid of that
logit (Llama 4 scores the selected expert with a sigmoid and does not renormalize). The logits are a per-token
offset, a per-expert popularity bias (standard deviation 0.25, drawn anew with every input draw) and independent
normal values per token and expert. The offset spreads the selected sigmoid weight over (0, 1) across tokens instead
of clustering near one. The bias makes the expert loads uneven, as a trained router's are: at 16384 tokens the most
loaded expert receives about 350 to 800 tokens (3 to 6 times the mean, more than one 256-row tile) and the least
loaded only a few, so an expert's token segment spanning several M tiles is exercised.
"""
import math

import torch

_CHUNK = 8
_POPULARITY_STD = 0.25


def _normal_bank(shape, std, generator, device):
    bank = torch.empty(shape, dtype=torch.bfloat16, device=device)
    for start in range(0, shape[0], _CHUNK):
        stop = min(start + _CHUNK, shape[0])
        chunk = torch.randn((stop - start, *shape[1:]), generator=generator, device=device, dtype=torch.float32)
        bank[start:stop] = chunk.mul_(std).to(torch.bfloat16)
    return bank


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_tokens, top_k, num_experts = axes["num_tokens"], axes["top_k"], axes["num_experts"]
    hidden, inter = axes["hidden_size"], axes["intermediate_size"]
    w13 = _normal_bank((num_experts, 2 * inter, hidden), 1.0 / math.sqrt(hidden), generator, device)
    w2 = _normal_bank((num_experts, hidden, inter), 1.0 / math.sqrt(inter), generator, device)
    popularity = torch.randn((1, num_experts), generator=generator, device=device, dtype=torch.float32)
    offset = torch.randn((num_tokens, 1), generator=generator, device=device, dtype=torch.float32) * 1.5 - 2.5
    logits = torch.randn((num_tokens, num_experts), generator=generator, device=device, dtype=torch.float32)
    logits = (logits + popularity * _POPULARITY_STD + offset).to(torch.bfloat16)
    values, ids = logits.topk(top_k, dim=-1)
    return {"w13": w13, "w2": w2, "topk_weights": torch.sigmoid(values.float()), "topk_ids": ids.to(torch.int32)}
