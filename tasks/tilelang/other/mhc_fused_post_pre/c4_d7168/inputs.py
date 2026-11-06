"""Task-owned input generator: row-stochastic residual combination matrices."""
import torch


def make_inputs(case, inputs, generator, device):
    tokens = case["axes"]["tokens"]
    hc_mult = case["axes"]["hc_mult"]
    logits = torch.randn((tokens, hc_mult, hc_mult), generator=generator, device=device,
                         dtype=torch.float32)
    return {"comb_res_mix": torch.softmax(logits, dim=-1)}
