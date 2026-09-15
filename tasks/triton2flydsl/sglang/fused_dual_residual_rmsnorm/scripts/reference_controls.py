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
    r = references(['reference'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[1.,1.]]);res=t([[2.,2.]])
    out,mid=r.reference(x,res,t([1.,1.]),t([2.,2.]),eps=0.)
    check(mid,t([[3.,3.]]),"first RMS result becomes residual")
    check(out,t([[2.,2.]]),"second RMS normalization with weight two")
    return rows
