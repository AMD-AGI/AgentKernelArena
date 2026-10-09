"""Evaluator-owned CPU inputs and mathematical truth for RMSNorm validation."""
from __future__ import annotations


INPUTS = ("x", "residual", "weight")
DOMAINS = {"x": (-0.75, 0.875), "residual": (-0.625, 0.5), "weight": (-0.125, 0.125)}
CHALLENGES = ("zero_residual", "zero_sum", "near_cancellation", "small_amplitude")


def cpu_reference(torch, inputs, eps):
    """Use the unrounded FP32 sum; keep every oracle allocation on the CPU."""
    if any(inputs[name].device.type != "cpu" for name in INPUTS):
        raise RuntimeError("RMSNorm oracle inputs must be CPU-owned")
    x, residual, weight = (inputs[name] for name in INPUTS)
    normed = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="cpu")
    summed = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="cpu")
    # Bound temporary FP32 memory while retaining the complete native row width.
    for first in range(0, x.shape[0], 256):
        rows = slice(first, first + 256)
        s = x[rows].float() + residual[rows].float()
        scale = torch.rsqrt(s.square().mean(dim=-1, keepdim=True) + float(eps))
        normed[rows].copy_((s * scale) * (1.0 + weight.float()))
        summed[rows].copy_(s)
    return normed, summed


class CPUInputs:
    """A continuous CPU RNG supplies new BF16 values before each invocation.

    Ordinary timing uses the original three uniform input domains. Additional
    correctness challenges are applied only after the reported sample set.
    No expected output, expected-output copy, or reference computation is sent
    to the GPU. Only the three input tensors cross that boundary.
    """

    def __init__(self, torch, args, seed):
        self.torch = torch
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.geometry = {name: (tuple(args[name].shape), tuple(args[name].stride())) for name in INPUTS}
        self.dtype = args["x"].dtype
        self.eps = float(args["eps"])
        self.generated = 0

    def next(self, kind="ordinary"):
        if kind not in ("ordinary", *CHALLENGES):
            raise ValueError(f"unknown RMSNorm input challenge: {kind}")
        values = {}
        for name in INPUTS:
            shape, stride = self.geometry[name]
            value = self.torch.empty_strided(shape, stride, dtype=self.dtype, device="cpu")
            value.uniform_(*DOMAINS[name], generator=self.rng)
            values[name] = value
        x, residual = values["x"], values["residual"]
        if kind == "zero_residual":
            residual.zero_()
        elif kind == "zero_sum":
            x.uniform_(-0.5, 0.5, generator=self.rng)
            residual.copy_(-x)
        elif kind == "near_cancellation":
            x.uniform_(-0.25, 0.25, generator=self.rng)
            # BF16-rounded residual leaves small, finite, nonuniform sums.
            delta = self.torch.empty_like(x, dtype=self.torch.float32, device="cpu")
            delta.uniform_(-2.0**-10, 2.0**-10, generator=self.rng)
            residual.copy_(-x.float() + delta)
        elif kind == "small_amplitude":
            # E[s^2] is below eps=1e-6, so dropping eps changes the answer.
            x.uniform_(-2.0**-12, 2.0**-12, generator=self.rng)
            residual.uniform_(-2.0**-13, 2.0**-13, generator=self.rng)
        expected = cpu_reference(self.torch, values, self.eps)
        self.generated += 1
        return values, expected
