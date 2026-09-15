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
    r = references(['_compare_gdn_output', 'reference_kkt'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(B=1,T=2,Hg=1,H=1,K=1,k=t([2.,3.]).reshape(1,2,1,1),beta=t([.5,.25]).reshape(1,2,1),g=torch.zeros(1,2,1))
    e=torch.zeros(1,2,1,r.BT);e[0,1,0,0]=1.5
    check(r.reference_kkt(inp),e,"strict lower triangle: 3*2*.25")
    def accept(a,b):
        try: r._compare_gdn_output(a,b,inp); return True
        except AssertionError: return False
    rows.append(control(r.reference_kkt(inp),e,accept,"actual KKT comparator strict-lower entry1.5"))
    inp["g"]=t([0.,math.log(.5)]).reshape(1,2,1)
    e=e.clone();e[0,1,0,0]=.75
    rows.append(control(r.reference_kkt(inp),e,accept,"independent decay halves KKT entry to0.75"))
    return rows
