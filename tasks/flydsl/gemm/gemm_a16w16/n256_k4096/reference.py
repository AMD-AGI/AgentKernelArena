import torch


def reference(*, a, b):
    """out = a @ b.T accumulated in float64 (rounded to bfloat16 by the comparison)."""
    return a.double() @ b.double().T
