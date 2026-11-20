"""Page-fault guard control: a Triton copy, optionally storing far past its output.

Usage: page_fault_probe.py [safe|far]

Run with PyTorch's caching allocator and ROCr's fragment allocator off; the
far store then reaches an unmapped page and the GPU faults.
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(source, destination, n, shift, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(source + offsets, mask=mask)
    tl.store(destination + offsets, values, mask=mask)
    if shift > 0:
        tl.store(destination + offsets + shift, values, mask=mask)


def main() -> int:
    far = len(sys.argv) > 1 and sys.argv[1] == "far"
    n = 4096
    source = torch.arange(n, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    copy_kernel[(triton.cdiv(n, 1024),)](source, destination, n, (1 << 26) if far else 0, BLOCK=1024)
    torch.cuda.synchronize()
    torch.testing.assert_close(source, destination)
    print("FAR_RUN_COMPLETED" if far else "SAFE_RUN_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
