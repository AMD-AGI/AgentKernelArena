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
    r = references(['_ref_masked_attention', '_compare'], {"F": F, "SQRT2": math.sqrt(2)})
    q=torch.zeros(1,2,2);k=torch.zeros(2,1,2);v=t([2.,4.,6.,8.]).reshape(2,1,2)
    actual=r._ref_masked_attention(q,k,v,1.)
    check(actual,t([4.,6.,4.,6.]).reshape(1,2,2),"single decode query averages both KV tokens with GQA")
    return rows
