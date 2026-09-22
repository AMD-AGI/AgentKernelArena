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
    r = references(['reference_norm_gate'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(x=t([[1.,3.]]),g=t([[0.,math.log(3.)]]),weight=t([2.,4.]),bias=t([1.,2.]))
    check(r.reference_norm_gate(inp,False,"sigmoid",eps=0.),t([[-.5,4.5]]),"layer norm, affine, then sigmoid gate")
    check(r.reference_norm_gate(inp,True,"sigmoid",eps=0.),t([[(2/math.sqrt(5)+1)*.5,(12/math.sqrt(5)+2)*.75]]),"RMS norm keeps nonzero mean")
    return rows
