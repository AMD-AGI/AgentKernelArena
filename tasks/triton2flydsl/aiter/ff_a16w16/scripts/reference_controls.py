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
    r = references(['_torch_ref'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[1.,-1.]]);w=torch.eye(2)
    for activation in (None,"relu","silu","gelu"):
        key="gelu_tanh" if activation=="gelu" else activation
        expected=[]
        for z in (1.,-1.):
            if activation is None:y=z
            elif activation=="relu":y=max(z,0.)
            elif activation=="silu":y=z/(1+math.exp(-z))
            else:y=.5*z*(1+math.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3)))
            expected.append(y)
        # Harness uses the upstream silu_exp2 selector.
        if activation=="silu":key="silu_exp2"
        check(r._torch_ref(x,w,w,key),t([expected]),"two identity GEMMs and scalar "+str(key))
    return rows
