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
    q=torch.zeros(2,1,2);ke=torch.zeros(2,1,2);ve=t([6.,8.,10.,12.]).reshape(2,1,2)
    kb=torch.zeros(1,1,2);vb=t([2.,4.]).reshape(1,1,2)
    cfg=dict(prefix=[1],extend=[2],head=1,kv_head=1,Lq=2,Lv=2,causal=True)
    check(r.reference(q,ke,ve,kb,vb,torch.tensor([0]),cfg),t([4.,6.,6.,8.]).reshape_as(q),"prefix plus causal extension uniform means")
    return rows
