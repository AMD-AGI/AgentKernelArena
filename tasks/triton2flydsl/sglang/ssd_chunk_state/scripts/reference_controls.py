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
    x=t([2.,3.]).reshape(1,2,1,1);b=t([4.,5.]).reshape(1,2,1,1);dt=t([.5,2.]).reshape(1,1,1,2);g=torch.zeros(1,1,1,2)
    check(r.reference(x,b,dt,g,dict(b=1,H=1,P=1,G=1,N=1,cs=2,C=1)),t([34.]).reshape(1,1,1,1,1),"chunk state is 2*4*.5 + 3*5*2")
    return rows
