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
    q=t([1.,2.]).reshape(1,1,1,2);k=t([3.,4.]).reshape_as(q);v=t([5.,6.]).reshape_as(q)
    cache=torch.ones(1,1,2,2)
    o,s=r.reference(q,k,v,cache,t([0.]),torch.tensor([0]),dict(B=1,H=1,D=2))
    check(s,t([[[[16.,19.],[21.,25.]]]]),"outer product plus undecayed cache")
    check(o,t([[58.,69.]]),"query contracts updated cache")
    return rows
