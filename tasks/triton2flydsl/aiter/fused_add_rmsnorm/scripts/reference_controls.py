"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['_torch_rmsnorm'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[1.,3.]]);g=t([2.,4.])
    check(r._torch_rmsnorm(x,g,torch.float32),t([[2./math.sqrt(5+r.EPS),12./math.sqrt(5+r.EPS)]]),"RMS uses mean-square 5 with affine weights")
    return rows
