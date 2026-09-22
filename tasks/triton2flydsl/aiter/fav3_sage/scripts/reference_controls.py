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
    r = references(['_attention_reference', '_compare'], {"F": F, "SQRT2": math.sqrt(2)})
    q=torch.zeros(1,2,2,2);k=torch.zeros(1,2,1,2);v=t([2.,4.,6.,8.]).reshape(1,2,1,2)
    for causal in (False,True):
        actual=r._attention_reference(q,k,v,1.,causal,-1)
        expected=t([2.,4.,4.,6.] if causal else [4.,6.,4.,6.]).reshape(1,2,1,2).repeat(1,1,2,1)
        check(actual,expected,"zero logits give uniform visible GQA values "+str(causal))
        accept=lambda a,b:r._compare(a,b)["norm_max_err"]<=r.NORM_MAX_ERR_TOL and r._compare(a,b)["frac_exceeding_pct"]<=r.MAX_DIFF_PCT
        rows.append(control(actual,expected,accept,"production comparator positive/negative control"))
    return rows
