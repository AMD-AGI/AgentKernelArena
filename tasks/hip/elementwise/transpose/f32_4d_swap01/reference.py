"""Reference for x.transpose(0, 1).contiguous(); a pure data movement, so the fp32 result is exact."""
import torch


def reference(*, x):
    return x.transpose(0, 1).contiguous()
