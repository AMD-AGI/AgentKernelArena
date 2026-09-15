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
    cfg=dict(nt=1,hd=8,rd=6,n_qh=1,n_kh=1,section=(1,1,1),interleaved=False,neox=True)
    x=t([[1.,2.,3.,4.,5.,6.,9.,10.]])
    # time:90 degrees, height:0, width:180. Last two dimensions untouched.
    cache=t([[0.,0.,0.,1.,1.,1.],[1.,1.,1.,0.,0.,0.],[-1.,-1.,-1.,0.,0.,0.]])
    q,k=r.reference(x,x,cache,torch.tensor([[0],[1],[2]]),cfg)
    e=t([[-4.,2.,-3.,1.,5.,-6.,9.,10.]])
    check(q,e,"different temporal/spatial positions and rotary suffix")
    check(k,e,"key rotation follows same sections")
    return rows
