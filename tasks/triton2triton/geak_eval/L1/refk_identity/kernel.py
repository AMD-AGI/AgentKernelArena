#!/usr/bin/env python3
"""
Identity Kernel — Triton implementation extracted via torch.compile(backend='inductor').

Copies an input tensor to an output tensor element-wise.
Triton kernel generated from PyTorch's `output.copy_(input)` on float16 1-D tensors.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# TRITON KERNEL — extracted from torch.compile inductor output
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 128}, num_warps=2),
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _identity_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask).to(tl.float32)
    tl.store(out_ptr0 + xindex, tmp0, xmask)


# ============================================================================
# PYTHON WRAPPER
# ============================================================================


def identity_triton(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    xnumel = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]),)
    _identity_kernel[grid](input_tensor, output_tensor, xnumel)
    return output_tensor


# ============================================================================
# ENTRY POINT (callable form of the kernel for downstream profilers)
# ============================================================================


def triton_op(size, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.empty(size, device="cuda", dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    output = torch.empty_like(data)
    return identity_triton(data, output)


# ============================================================================
# SYNTHETIC INPUT BUILDER
# ============================================================================


def get_inputs(size, seed=42, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    data = torch.empty(size, device=device, dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    output = torch.empty_like(data)
    return data, output


# ============================================================================
# CONFIG SPACE — matches test_submission_harness.py ALL_CONFIGS
# ============================================================================


EVAL_CONFIGS = [
    # tests from task.yml
    {"size": 127, "seed": 4242},
    {"size": 128, "seed": 5236},
    {"size": 129, "seed": 1001},
    {"size": 256, "seed": 5531},
    {"size": 512, "seed": 9173},
    # benchmarks from task.yml
    {"size": 1024, "seed": 54352},
    {"size": 2048, "seed": 93246},
    {"size": 4096, "seed": 6256},
    {"size": 8192, "seed": 8841},
    {"size": 16384, "seed": 6252},
    {"size": 32768, "seed": 52624},
    {"size": 65536, "seed": 125432},
]

PROFILE_CONFIGS = [
    {"size": 1024, "seed": 54352},
    {"size": 8192, "seed": 8841},
    {"size": 65536, "seed": 125432},
]

RTOL, ATOL = 1e-5, 1e-5
