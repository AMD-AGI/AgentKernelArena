"""Seeded inputs with extreme values planted at the start of the tensor."""
import torch

SPECIAL = [-2147483648, 2147483647, -16777217, 16777217, 0, -3, 5]


def make_inputs(case, inputs, generator, device):
    """Random int32 data in [-100, 100) whose first elements are extreme and float-unrepresentable values."""
    shape = [case["axes"][name] for name in ["n_rows", "n_cols"]]
    x = torch.randint(-100, 100, shape, generator=generator, device=device, dtype=torch.int32)
    flat = x.view(-1)
    count = min(flat.numel(), len(SPECIAL))
    flat[:count] = torch.tensor(SPECIAL[:count], dtype=torch.int32, device=device)
    return {"x": x}
