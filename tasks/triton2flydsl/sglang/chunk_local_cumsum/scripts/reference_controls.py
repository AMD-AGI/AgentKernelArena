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
    r = references(['reference_cumsum'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([1.,2.,3.,4.,5.]).reshape(1,5,1)
    for reverse,expected in [(False,[2.,6.,6.,14.,10.]),(True,[6.,4.,14.,8.,10.])]:
        check(r.reference_cumsum(x,{"reverse":reverse,"scale":2.},BT=2),t(expected).reshape_as(x),"chunk reset and direction "+str(reverse))
    return rows
