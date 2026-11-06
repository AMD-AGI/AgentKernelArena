"""Task-owned input generator: points on a dyadic grid so that every distance is exact.

Coordinates are odd integers / 128, so most are not representable in bfloat16 (9 or more significant bits)."""
import torch

GRID = 128.0  # coordinates are odd integers / 128 in (-8, 8)


def make_inputs(case, inputs, generator, device):
    batch = case["axes"]["batch"]
    n_points = case["axes"]["n_points"]
    n_query = case["axes"]["n_query"]
    ints = 2 * torch.randint(-512, 512, (batch, n_points, 3), generator=generator, device=device,
                             dtype=torch.int64) + 1
    xyz = ints.to(torch.float32) / GRID
    # Query centres are points of the cloud (what a furthest-point-sampling stage hands to knn).
    pick = torch.randint(0, n_points, (batch, n_query), generator=generator, device=device)
    centers = torch.gather(xyz, 1, pick.unsqueeze(-1).expand(batch, n_query, 3))
    return {"xyz": xyz, "center_xyz": centers}
