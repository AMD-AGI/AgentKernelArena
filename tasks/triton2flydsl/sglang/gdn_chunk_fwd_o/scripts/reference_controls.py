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
    r = references(['reference_o'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(B=1,T=2,Hg=1,H=1,K=1,V=1,NT=1,scale=.5,q=t([1.,2.]).reshape(1,2,1,1),k=t([3.,4.]).reshape(1,2,1,1),v=t([5.,6.]).reshape(1,2,1,1),h=t([7.]).reshape(1,1,1,1,1),g=torch.zeros(1,2,1))
    check(r.reference_o(inp),t([11.,46.]).reshape(1,2,1,1),"causal intra contribution plus initial-state contribution")
    return rows
