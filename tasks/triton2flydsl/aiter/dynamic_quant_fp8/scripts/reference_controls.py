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
    r = references(['_reference_quant'], {"F": F, "SQRT2": math.sqrt(2)})
    for dtype,maximum in [(torch.int8,127.),(torch.float8_e4m3fn,448.)]:
        x=t([[maximum,-maximum]])
        for mode in ("static","dyn_tensor","dyn_token"):
            q,s=r._reference_quant(x,dtype,mode,t([1.]))
            check(q.float(),x,"unit-scale quantization "+str(dtype)+" "+mode)
            check(s.reshape(-1),t([1.]),"known quantization scale "+mode)
    return rows
