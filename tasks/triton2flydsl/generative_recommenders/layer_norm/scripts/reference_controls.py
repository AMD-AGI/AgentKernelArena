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
    r = references(['_torch_ref', '_close'], {"F": F, "SQRT2": math.sqrt(2)})
    a=r._torch_ref(t([[1.,3.]]),t([2.,4.]),t([1.,2.]),1e-5)
    e=t([[1-2/math.sqrt(1.00001),2+4/math.sqrt(1.00001)]])
    check(a,e,"population variance and affine layer norm")
    rows.append(control(a,e,lambda a,b:r._close(b,a)[0],"task comparator rejects incorrect normalization"))
    return rows
