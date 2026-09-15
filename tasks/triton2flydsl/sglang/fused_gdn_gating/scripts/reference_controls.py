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
    r = references(['_compare_router_output', 'reference_gating'], {"F": F, "SQRT2": math.sqrt(2)})
    inp=dict(a=t([[0.,30.]]),b=t([[0.,math.log(3.)]]),A_log=t([0.,0.]),dt_bias=t([0.,0.]))
    g,beta=r.reference_gating(inp,beta=1.,threshold=20.)
    check(g,t([[[-math.log(2.),-30.]]]),"softplus origin and linear threshold branch")
    check(beta,t([[[.5,.75]]]),"sigmoid odds 1 and 3")
    expected = (t([[[-math.log(2.),-30.]]]), t([[[.5,.75]]]))
    def accept_g(a,b):
        try: r._compare_router_output((a,expected[1]),(b,expected[1]),torch.float32); return True
        except AssertionError: return False
    def accept_beta(a,b):
        try: r._compare_router_output((expected[0],a),(expected[0],b),torch.float32); return True
        except AssertionError: return False
    rows.append(control(g,expected[0],accept_g,"actual GDN comparator origin and linear branch"))
    rows.append(control(beta,expected[1],accept_beta,"actual sigmoid comparator odds1and3"))
    return rows
