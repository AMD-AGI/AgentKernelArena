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
    x=t([[1.,0.]]);w=t([[0.,0.],[math.log(3.),0.]])
    weights,ids=r.reference(x,w,dict(cap=0,topk=1),None)
    check(weights,t([[.75]]),"softmax odds 1:3")
    check(ids.float(),t([[1.]]),"unique winning expert")
    return rows
