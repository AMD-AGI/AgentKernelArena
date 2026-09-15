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
    r = references(['ref_sparse_mla'], {"F": F, "SQRT2": math.sqrt(2)})
    q=torch.zeros(2,1,3);kv=t([2.,4.,10.,6.,8.,20.]).reshape(1,2,1,3)
    idx=torch.tensor([[1,0,-1],[-1,-1,-1]])
    check(r.ref_sparse_mla(q,kv,idx,2,2,1.),t([[[4.,6.]],[[0.,0.]]]),"sparse selection, ignored sentinel and empty row")
    return rows
