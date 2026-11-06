"""High-precision 2D max-pooling reference used only for correctness."""
import torch
import torch.nn.functional as F


def reference(*, x):
    # Max pooling selects an input element, so the float64 result is exactly representable in float32.
    return F.max_pool2d(x.double(), kernel_size=4, stride=1, padding=1, dilation=1)
