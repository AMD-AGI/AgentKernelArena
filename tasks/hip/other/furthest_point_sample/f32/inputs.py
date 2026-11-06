"""Task-owned input generator: points on a dyadic grid so that every distance is exact.

Coordinates are odd integers / 128, so about three quarters of them need more than the 8 significant bits
of bfloat16 (a plain integer grid would be half exact in bfloat16), while squared distances stay exact."""
import torch

GRID = 128.0  # coordinates are odd integers / 128 in (-8, 8)


def make_inputs(case, inputs, generator, device):
    batch = case["axes"]["batch"]
    n_points = case["axes"]["n_points"]
    half = torch.randint(-512, 512, (batch, n_points, 3), generator=generator, device=device,
                         dtype=torch.int64)
    ints = 2 * half + 1  # odd, |value| <= 1023
    return {"points_xyz": ints.to(torch.float32) / GRID}
