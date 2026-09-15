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
    r = references(['_compare_flat_output', '_ref_rope_sbhd_fwd'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    x=t([1.,2.,3.,4.,9.,10.]).reshape(1,1,1,6)
    for style,expected in [(0,[-3.,-4.,1.,2.,9.,10.]),(1,[-2.,1.,-4.,3.,9.,10.])]:
        check(r._ref_rope_sbhd_fwd(x,torch.full((1,1,1,2),math.pi/2),style,True,False,0),t(expected).reshape_as(x),"90 degree rotation and unchanged suffix "+str(style))
    def accept(a,b):
        try: r._compare_flat_output(a,b); return True
        except AssertionError: return False
    actual = r._ref_rope_sbhd_fwd(x,torch.full((1,1,1,2),math.pi/2),1,True,False,0).to(torch.bfloat16)
    rows.append(control(actual,t([-2.,1.,-4.,3.,9.,10.]).reshape_as(x).to(torch.bfloat16),accept,
                        "actual measured RoPE comparator explicit90degree GPTJ rotation"))
    return rows
