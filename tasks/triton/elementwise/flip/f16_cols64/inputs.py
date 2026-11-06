"""Seeded inputs with special values planted at the start of the tensor."""
import torch

SPECIAL = [-3.5, -0.0, 0.5, 100.5, float("inf"), -float("inf"), 65504.0]


def make_inputs(case, inputs, generator, device):
    """Normal data whose first elements are signed zero, infinities and the dtype's largest finite value.

    NaN is left out on purpose: the evaluator poisons outputs with NaN, so a NaN input could not
    distinguish a copied element from an unwritten one.
    """
    shape = [case["axes"][name] for name in ["n_rows", "n_cols"]]
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(torch.float16)
    flat = x.view(-1)
    count = min(flat.numel(), len(SPECIAL))
    flat[:count] = torch.tensor(SPECIAL[:count], dtype=torch.float16, device=device)
    return {"x": x}
