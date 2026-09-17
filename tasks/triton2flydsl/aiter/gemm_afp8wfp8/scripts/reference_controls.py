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
    r = references(['_torch_ref'], {"F": F, "SQRT2": math.sqrt(2)})
    K=r.W_SCALE_K_GROUP;N=r.W_SCALE_N_GROUP
    x=torch.ones(1,K);w=torch.ones(N,K)
    xs=torch.full((1,K//r.SCALE_GROUP_SIZE),128,dtype=torch.uint8)
    ws=torch.full((1,1),129,dtype=torch.uint8)
    check(r._torch_ref(x,w,xs,ws,torch.float32),torch.full((1,N),float(K*8)),"E8M0 128 means 2; 129 means 4")
    return rows
