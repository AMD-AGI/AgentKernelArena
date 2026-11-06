"""Seeded inputs with special values planted at the start of the tensor."""
import torch

SPECIAL = [-3.5, -0.0, 0.5, 100.5, float("inf"), -float("inf"), 65504.0]


def make_inputs(case, inputs, generator, device):
    """Normal float16 data whose first elements are signed zero, infinities and the largest finite value.

    NaN is left out on purpose: the evaluator poisons outputs with NaN, so a NaN input could not
    distinguish a copied element from an unwritten one.
    """
    x = torch.randn(case["axes"]["numel"], generator=generator, device=device, dtype=torch.float32).to(torch.float16)
    count = min(x.numel(), len(SPECIAL))
    x[:count] = torch.tensor(SPECIAL[:count], dtype=torch.float16, device=device)
    return {"x": x}
