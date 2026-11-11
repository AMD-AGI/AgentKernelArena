"""PyTorch CUDA Sanitizer control: a side-stream read, with or without stream waits.

``safe`` orders the side stream after the write that produced the tensor and
the default stream after the side-stream read; ``race`` omits both waits, so
the read races the write. Usage: csan_probe.py [safe|race]
"""

from __future__ import annotations

import sys

import torch
import torch.cuda._sanitizer as cuda_sanitizer


def main() -> int:
    race = len(sys.argv) > 1 and sys.argv[1] == "race"
    cuda_sanitizer.enable_cuda_sanitizer()
    a = torch.ones(1 << 16, device="cuda")
    side = torch.cuda.Stream()
    if not race:
        side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        total = a.sum()
    if not race:
        torch.cuda.current_stream().wait_stream(side)
    a.add_(1)
    torch.cuda.synchronize()
    print("CSAN_RUN_COMPLETED", "race" if race else "safe", float(total), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
