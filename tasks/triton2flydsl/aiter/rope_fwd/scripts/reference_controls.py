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
    r = references(['_ref_rope_sbhd_fwd'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([1.,2.,3.,4.,9.,10.]).reshape(1,1,1,6)
    for style,expected in [(0,[-3.,-4.,1.,2.,9.,10.]),(1,[-2.,1.,-4.,3.,9.,10.])]:
        check(r._ref_rope_sbhd_fwd(x,torch.full((1,1,1,2),math.pi/2),style,True,False,0),t(expected).reshape_as(x),"90 degree rotation and unchanged suffix "+str(style))
    return rows
