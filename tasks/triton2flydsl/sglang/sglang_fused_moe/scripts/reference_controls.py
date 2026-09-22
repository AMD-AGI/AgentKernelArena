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
    r = references(['reference_moe'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(M=1,K=1,I=1,E=2,topk=2,hidden=t([[1.]]),w1=t([[[1.],[2.]],[[2.],[3.]]]),w2=t([[[4.]],[[5.]]]),topk_ids=torch.tensor([[1,0]]),topk_weights=t([[.25,.75]]))
    e=.25*(2/(1+math.exp(-2))*3*5)+.75*(1/(1+math.exp(-1))*2*4)
    check(r.reference_moe(inp),t([[e]]),"two explicit experts, SiLU gate, second GEMM and route sum")
    return rows
