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
    x=t([[1.,2.],[3.,4.]])
    a=r._torch_ref(torch.tensor([0,1,1,2]),x,t([[10.,20.],[99.,99.],[30.,40.]]))
    e=t([[11.,22.],[33.,44.]])
    check(a,e,"jagged batch map skips empty segment")
    rows.append(control(a,e,lambda a,b:r._close(b,a)[0],"task comparator rejects incorrect broadcast"))
    return rows
