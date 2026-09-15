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
    r = references(['_compare_gdn_output', 'reference_wu'], {"F": F, "SQRT2": math.sqrt(2)})
    a=torch.zeros(1,2,1,64);a[0,0,0,0]=1;a[0,1,0,:2]=t([2.,1.])
    inp=dict(B=1,T=2,Hg=1,H=1,K=1,V=1,NT=1,k=t([2.,3.]).reshape(1,2,1,1),v=t([4.,5.]).reshape(1,2,1,1),beta=t([.5,2.]).reshape(1,2,1),g=torch.zeros(1,2,1),A=a)
    w,u=r.reference_wu(inp)
    check(w,t([1.,8.]).reshape_as(w),"lower triangular transform of beta-scaled keys")
    check(u,t([2.,14.]).reshape_as(u),"lower triangular transform of beta-scaled values")
    ew=t([1.,8.]).reshape_as(w); eu=t([2.,14.]).reshape_as(u)
    def accept_w(a,b):
        try: r._compare_gdn_output((a,eu),(b,eu),inp); return True
        except AssertionError: return False
    def accept_u(a,b):
        try: r._compare_gdn_output((ew,a),(ew,b),inp); return True
        except AssertionError: return False
    rows.append(control(w,ew,accept_w,"actual WY key comparator independent lower-triangular product"))
    rows.append(control(u,eu,accept_u,"actual WY value comparator independent lower-triangular product"))
    inp["g"]=torch.full((1,2,1),math.log(.5))
    w,u=r.reference_wu(inp);ew=ew*.5
    rows.append(control(w,ew,accept_w,"decay halves keys but not values"))
    rows.append(control(u,eu,accept_u,"value output independent of key decay"))
    return rows
