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
    r = references(['_torch_mxfp8_quant_from_fp32'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[1.,-1.]*16]);q,s=r._torch_mxfp8_quant_from_fp32(x)
    check(q.float(),x*256,"unit magnitude uses 2^-8 scale")
    check(s.float(),t([[119.]]),"E8M0 exponent byte 127-8")
    return rows
