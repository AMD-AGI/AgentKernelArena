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
    r = references(['_torch_reference'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[3.,-3.,4.,-4.]])
    check(r._torch_reference(x,2.,t([[2.,3.]])),t([[8/(1+math.exp(-2)),18/(1+math.exp(3))]]),"one-sided gate clamp, two-sided up clamp, weights")
    return rows
