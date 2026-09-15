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
    r = references(['_compare_combine_output', 'reference_o'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    inp=dict(B=1,T=2,Hg=1,H=1,K=1,V=1,NT=1,scale=.5,q=t([1.,2.]).reshape(1,2,1,1),k=t([3.,4.]).reshape(1,2,1,1),v=t([5.,6.]).reshape(1,2,1,1),h=t([7.]).reshape(1,1,1,1,1),g=torch.zeros(1,2,1))
    check(r.reference_o(inp),t([11.,46.]).reshape(1,2,1,1),"causal intra contribution plus initial-state contribution")
    expected=t([11.,46.]).reshape(1,2,1,1)
    def accept(a,b):
        try: r._compare_combine_output(a,b,torch.float32); return True
        except AssertionError: return False
    rows.append(control(r.reference_o(inp),expected,accept,"actual GDN comparator independent causal outputs11and46"))
    # One-token decay: (q*h*exp(log(1/2)) + q*k*v)*scale
    # = (2*7/2 + 2*3*5)/2 = 18.5, independent scalar arithmetic.
    inp.update(T=1,q=t([2.]).reshape(1,1,1,1),k=t([3.]).reshape(1,1,1,1),
               v=t([5.]).reshape(1,1,1,1),g=t([math.log(.5)]).reshape(1,1,1))
    rows.append(control(r.reference_o(inp),t([18.5]).reshape(1,1,1,1),accept,
                        "GDN independent decay and initial-state contribution18.5"))
    return rows
