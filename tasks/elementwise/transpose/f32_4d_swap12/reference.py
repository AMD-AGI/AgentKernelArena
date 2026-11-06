"""Reference for x.transpose(1, 2).contiguous(); a pure data movement, so the fp32 result is exact."""
import torch


def reference(*, x):
    return x.transpose(1, 2).contiguous()
