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
    r = references(['reference_wu'], {"F": F, "SQRT2": math.sqrt(2)})
    a=torch.zeros(1,2,1,64);a[0,0,0,0]=1;a[0,1,0,:2]=t([2.,1.])
    inp=dict(B=1,T=2,Hg=1,H=1,K=1,V=1,NT=1,k=t([2.,3.]).reshape(1,2,1,1),v=t([4.,5.]).reshape(1,2,1,1),beta=t([.5,2.]).reshape(1,2,1),g=torch.zeros(1,2,1),A=a)
    w,u=r.reference_wu(inp)
    check(w,t([1.,8.]).reshape_as(w),"lower triangular transform of beta-scaled keys")
    check(u,t([2.,14.]).reshape_as(u),"lower triangular transform of beta-scaled values")
    return rows
