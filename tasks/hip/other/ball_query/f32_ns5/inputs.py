"""Point cloud and sampled centers for ball query.

Coordinates are drawn from a clipped standard normal cloud and snapped to odd multiples of 1/4096 (fine enough that
the values are not exactly representable in bfloat16), so every squared distance that can fall inside
a ball of radius <= 1 is exactly representable in float32 (a component difference is m/4096 with
|m| <= 4096, its square has at most 24 significant bits, and partial sums stay below 1): the kernel's
float32 comparison against the squared radii then agrees with the float64 reference for any evaluation
order or fused multiply-add contraction. Larger distances may round in float32 but stay far above 1. The centers are a random subset of the points (as farthest-point
sampling produces), so each center has a zero-distance match and duplicate points occur.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, n_points, n_centers = axes["batch"], axes["n_points"], axes["n_centers"]
    raw = torch.randn((batch, n_points, 3), generator=generator, device=device, dtype=torch.float32)
    # odd multiples of 1/4096: a value with |x| >= 1/16 needs more than 8 significant bits, so it is never
    # exact in bfloat16 even for a one-point cloud
    xyz = (2.0 * torch.floor(raw.clamp(-4.0, 4.0) * 2048.0) + 1.0) / 4096.0
    order = torch.rand((batch, n_points), generator=generator, device=device).argsort(dim=1)[:, :n_centers]
    centers = torch.gather(xyz, 1, order.unsqueeze(-1).expand(batch, n_centers, 3))
    return {"xyz": xyz, "center_xyz": centers}
