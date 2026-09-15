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
    r = references(['reference_decode'], {"F": F, "SQRT2": math.sqrt(2)})
    # Normalized q=k=1/sqrt(1+1e-6); g=-log(2), beta=1/2.
    u=1/math.sqrt(1.000001)
    state=2.+.5*(6.-2.*u)*u
    inp=dict(B=1,H=1,HV=1,K=1,V=1,scale=1.,mixed_qkv=t([[1.,1.,6.]]),a=t([[0.]]),b=t([[0.]]),A_log=t([0.]),dt_bias=t([0.]),cache_indices=torch.tensor([0]),ssm_states=t([4.]).reshape(1,1,1,1))
    o,s=r.reference_decode(inp)
    check(s,t([state]).reshape_as(s),"scalar decay, delta rule, state update")
    check(o,t([state*u]).reshape_as(o),"normalized query reads updated state")
    return rows
