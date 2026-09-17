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
    q=torch.zeros(1,2,2);k=torch.zeros(3,1,2);v=t([100.,100.,2.,4.,6.,8.]).reshape(3,1,2)
    cfg=dict(seqs=[2],head=2,kv_head=1,Lk=2,Lv=2)
    check(r.reference(q,k,v,torch.tensor([2,1]),cfg),t([[[4.,6.],[4.,6.]]]),"decode indexed KV mean with GQA")
    return rows
