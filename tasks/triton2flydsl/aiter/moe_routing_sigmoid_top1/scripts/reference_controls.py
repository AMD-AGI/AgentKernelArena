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
    r = references(['_torch_routing_ref'], {"F": F, "SQRT2": math.sqrt(2)})
    ids,weights,scores=r._torch_routing_ref(t([[1.,0.]]),t([[0.,math.log(3.)],[0.,0.]]),2,True)
    check(ids.float(),t([[1.,2.]]),"winning expert plus shared expert index")
    check(weights,t([[.75,1.]]),"sigmoid log(3) and shared unit weight")
    check(scores,t([[.5,.75]]),"independent sigmoid probabilities")
    return rows
