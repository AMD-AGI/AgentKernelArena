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
    r = references(['_compare_combine_output', 'reference'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    moe=t([[[1.,2.],[3.,4.]]]);mlp=t([[5.,6.]])
    check(r.reference(moe,mlp),t([[9./math.sqrt(2),12./math.sqrt(2)]]),"expert sum plus MLP divided by sqrt(2)")
    expected=t([[9./math.sqrt(2),12./math.sqrt(2)]])
    def accept(a,b):
        try: r._compare_combine_output(a,b,{"dtype":"fp32"}); return True
        except AssertionError: return False
    rows.append(control(r.reference(moe,mlp),expected,accept,"actual combine comparator independent scaled sum"))
    return rows
