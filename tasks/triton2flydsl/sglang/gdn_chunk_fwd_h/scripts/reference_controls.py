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
    r = references(['reference_h', '_close'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(B=1,T=1,Hg=1,H=1,K=1,V=1,NT=1,k=t([2.]).reshape(1,1,1,1),w=t([3.]).reshape(1,1,1,1),u=t([20.]).reshape(1,1,1,1),g=torch.zeros(1,1,1),init=t([4.]).reshape(1,1,1,1),idx=torch.tensor([0]))
    h,v,state=r.reference_h(inp)
    check(h,t([4.]).reshape_as(h),"chunk starts with original state 4")
    check(v,t([8.]).reshape_as(v),"delta is 20 - 3*4")
    check(state,t([20.]).reshape_as(state),"state updated to 4 + 8*2")
    rows.append(control(state,t([20.]).reshape_as(state),lambda a,b:r._close(a,b,.01,.01,.001)[0],"state comparator rejects wrong update"))
    return rows
