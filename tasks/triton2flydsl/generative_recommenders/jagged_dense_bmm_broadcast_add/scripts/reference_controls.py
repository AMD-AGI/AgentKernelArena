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
    r = references(['_torch_ref'], {"F": F, "SQRT2": math.sqrt(2)})
    offsets=torch.tensor([0,1,1,3]);x=t([[1.,2.],[3.,4.],[5.,6.]])
    dense=t([[[1.],[2.]],[[99.],[99.]],[[3.],[4.]]])
    check(r._torch_ref(offsets,x,dense,t([[10.],[99.],[20.]]),False),t([[15.],[45.],[59.]]),"empty segment and per-batch bias")
    check(r._torch_ref(offsets,x,dense,t([[10.],[20.],[30.]]),True),t([[15.],[45.],[69.]]),"per-row bias")
    return rows
