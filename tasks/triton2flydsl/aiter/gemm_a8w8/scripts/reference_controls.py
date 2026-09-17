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
    x = t([[1., 2.], [3., 4.]])
    w = t([[5., 6.], [7., 8.]])
    expected = t([[17., 23.], [39., 53.]])
    check(r._torch_ref(x,w,t([[2.],[3.]]),t([[4.,5.]]),t([1.,2.]),torch.float32),expected*t([[8.,10.],[12.,15.]])+t([1.,2.]),"scaled dot products and bias")
    return rows
