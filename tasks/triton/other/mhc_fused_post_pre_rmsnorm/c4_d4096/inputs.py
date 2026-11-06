"""Task-owned input generator: approximately doubly stochastic incoming stream-combination matrices."""
import torch


def make_inputs(case, inputs, generator, device):
    """comb_mix as the previous layer's pre block leaves it: positive, rows and columns near one.

    A softmax of normal logits over the last axis followed by 20 row/column normalizations.
    """
    tokens = case["axes"]["tokens"]
    hc_mult = case["axes"]["hc_mult"]
    logits = torch.randn((tokens, hc_mult, hc_mult), generator=generator, device=device, dtype=torch.float32)
    comb = torch.softmax(logits, dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-1, keepdim=True)
        comb = comb / comb.sum(dim=-2, keepdim=True)
    return {"comb_mix": comb.contiguous()}
