"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close
from scripts.replay_checks import require_tensor_contract


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['_compare_flat_output', '_ref_moe_gemm'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    a=t([[1.,2.]])
    b=t([[[3.,4.]],[[5.,6.]]]);ids=torch.tensor([[1,0]]);weights=t([[.25,.75]])
    for mul in (False,True):
        expected=t([[[17.],[11.]]])
        if mul:expected=expected*t([[[.25],[.75]]])
        check(r._ref_moe_gemm(a,b,ids,weights,2,mul),expected,"explicit expert permutation and routed weights "+str(mul))
    def accept(a,b):
        try: r._compare_flat_output(a,b); return True
        except AssertionError: return False
    actual = r._ref_moe_gemm(a,b,ids,weights,2,True).to(torch.bfloat16)
    rows.append(control(actual,expected,accept,"actual measured MoE comparator explicit weighted projections"))
    return rows
