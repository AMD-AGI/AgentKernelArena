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
    r = references(['reference_l2norm'], {"F": F, "SQRT2": math.sqrt(2)})
    check(r.reference_l2norm(t([[3.,4.]]),eps=0.),t([[.6,.8]]),"3-4-5 Euclidean normalization")
    return rows
