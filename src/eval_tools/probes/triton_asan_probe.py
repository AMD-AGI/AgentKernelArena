#!/usr/bin/env python3
"""Synthetic Triton GPU-ASan safe/OOB probe."""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(source, destination, shift: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE) + shift
    values = tl.load(source + offsets)
    tl.store(destination + offsets, values)


def main() -> int:
    shift = 1 if len(sys.argv) > 1 and sys.argv[1] == "oob" else 0
    size = 4096
    source = torch.arange(size, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    copy_kernel[(1,)](source, destination, shift, BLOCK_SIZE=size)
    torch.cuda.synchronize()
    if not shift:
        torch.testing.assert_close(source, destination)
    print("OOB_RUN_COMPLETED" if shift else "SAFE_RUN_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
