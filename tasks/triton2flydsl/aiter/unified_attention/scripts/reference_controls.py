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
    r = references(['ref_paged_attn'], {"F": F, "SQRT2": math.sqrt(2)})
    q=torch.zeros(1,2,2);k=torch.zeros(2,2,1,2);v=t([100.,100.,100.,100.,2.,4.,6.,8.]).reshape(2,2,1,2)
    check(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[1]]),1.,torch.float32),t([[[4.,6.],[4.,6.]]]),"physical block selection and GQA mean")
    check(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[1]]),1.,torch.float32,sliding_window=1),t([[[6.,8.],[6.,8.]]]),"one-token sliding window")
    return rows
