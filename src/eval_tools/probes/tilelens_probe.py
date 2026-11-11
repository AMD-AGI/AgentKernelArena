"""TileLens control: a Triton copy, optionally off by one element, checked symbolically.

Usage: tilelens_probe.py [safe|oob]

Uses the same programmatic API as the schema-v3 lane: a Sanitizer client with
abort_on_error off, whose records are read after the launch.
"""

from __future__ import annotations

import json
import sys

import torch
import triton
import triton.language as tl
import tilelens
from tilelens.clients import Sanitizer


@triton.jit
def copy_kernel(source, destination, shift: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK) + shift
    tl.store(destination + offsets, tl.load(source + offsets))


def main() -> int:
    shift = 1 if len(sys.argv) > 1 and sys.argv[1] == "oob" else 0
    client = Sanitizer(abort_on_error=False)
    kernel = tilelens.trace(client=client)(copy_kernel)
    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    kernel[(1,)](source, destination, shift, BLOCK=1024)
    print("TILELENS_RESULT " + json.dumps({"shift": shift, "records": len(client.records)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
