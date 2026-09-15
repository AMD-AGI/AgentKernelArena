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
    r = references(['_reference_softmax'], {"F": F, "SQRT2": math.sqrt(2)})
    check(r._reference_softmax(t([[0.,math.log(3.)]])),t([[.25,.75]]),"odds 1:3 normalize to probabilities")
    return rows
