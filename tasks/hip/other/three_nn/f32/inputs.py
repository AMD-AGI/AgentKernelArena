"""Target and source point clouds for three_nn.

Coordinates are uniform on a 1/1024 grid in [1, 2) restricted to odd multiples of 1/1024, so every
coordinate has 11 significant bits (not exactly representable in bfloat16 or float16) and every
coordinate difference is an even multiple of 1/1024, at most 1023/1024. Squared differences are then
integer multiples of 2**-20 below 2**20 and their sums stay below 2**22, so every squared distance is
exactly representable in float32: the kernel's float32 distances equal the float64 reference's for any
evaluation order or fused multiply-add contraction, and the nearest-neighbor ranking (including its
ties, which the lowest index wins) is unambiguous.
"""
import torch

_LEVELS = 512  # odd multiples 1025/1024, 1027/1024, ..., 2047/1024


def _cloud(generator, device, batch, count):
    level = torch.randint(0, _LEVELS, (batch, count, 3), generator=generator, device=device)
    return (1024 + 2 * level + 1).to(torch.float32) / 1024.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    return {"unknown": _cloud(generator, device, axes["batch"], axes["n_target"]),
            "known": _cloud(generator, device, axes["batch"], axes["n_source"])}
