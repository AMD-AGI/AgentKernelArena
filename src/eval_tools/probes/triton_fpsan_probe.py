#!/usr/bin/env python3
"""Synthetic Triton FPSan equivalence/mismatch probe.

Run with ``TRITON_INSTRUMENTATION_MODE=fpsan`` and optional ``wrong``.  The
structured output is intentionally the same record consumed by the plugin.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def reference_kernel(x, y, z, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out + offsets, (tl.load(x + offsets) + tl.load(y + offsets)) + tl.load(z + offsets))


@triton.jit
def equivalent_kernel(x, y, z, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out + offsets, tl.load(x + offsets) + (tl.load(y + offsets) + tl.load(z + offsets)))


@triton.jit
def wrong_kernel(x, y, z, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out + offsets, (tl.load(x + offsets) + tl.load(y + offsets)) + (tl.load(z + offsets) + 1.0))


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    wrong = len(sys.argv) > 1 and sys.argv[1] == "wrong"
    if os.environ.get("TRITON_INSTRUMENTATION_MODE") != "fpsan":
        print("TRITON_INSTRUMENTATION_MODE=fpsan is required", file=sys.stderr)
        return 2
    size = 256
    x = torch.full((size,), 1.0e20, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -1.0e20)
    z = torch.full_like(x, 3.1415927)
    reference = torch.empty_like(x)
    candidate = torch.empty_like(x)
    reference_kernel[(1,)](x, y, z, reference, BLOCK=size)
    selected = wrong_kernel if wrong else equivalent_kernel
    selected[(1,)](x, y, z, candidate, BLOCK=size)
    torch.cuda.synchronize()
    record = {
        "instrumented": True,
        "reference_digest": digest(reference),
        "candidate_digest": digest(candidate),
        "expected": "mismatch" if wrong else "equivalent",
        "triton_version": triton.__version__,
    }
    print("AKA_FPSAN_RESULT " + json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
