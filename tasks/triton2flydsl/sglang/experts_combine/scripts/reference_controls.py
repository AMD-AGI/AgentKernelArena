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
    moe=t([[[1.,2.],[3.,4.]]]);mlp=t([[5.,6.]])
    check(r.reference(moe,mlp),t([[9./math.sqrt(2),12./math.sqrt(2)]]),"expert sum plus MLP divided by sqrt(2)")
    return rows
