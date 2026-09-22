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
    r = references(['_compare_state_output', 'reference'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    x=t([[1.,1.]]);res=t([[2.,2.]])
    out,mid=r.reference(x,res,t([1.,1.]),t([2.,2.]),eps=0.)
    check(mid,t([[3.,3.]]),"first RMS result becomes residual")
    check(out,t([[2.,2.]]),"second RMS normalization with weight two")
    # Unitless +/-sqrt(EPS) makes a missing first epsilon visibly wrong.
    x=t([[math.sqrt(r.EPS),-math.sqrt(r.EPS)]])
    out,mid=r.reference(x,torch.zeros_like(x),t([1.,1.]),t([2.,2.]))
    expected_mid=t([[1/math.sqrt(2),-1/math.sqrt(2)]])
    expected_out=expected_mid*(2/math.sqrt(.5+r.EPS))
    check(mid,expected_mid,"EPS-scale first RMS: +/-1/sqrt2")
    check(out,expected_out,"second RMS denominator sqrt(0.5+EPS)")
    tiny=t([[math.sqrt(r.EPS),-math.sqrt(r.EPS)]])
    second,kept=r.reference(torch.zeros_like(tiny),tiny,t([1.,1.]),t([2.,2.]))
    check(kept,tiny,"zero first input preserves small residual")
    check(second,t([[math.sqrt(2),-math.sqrt(2)]]),"EPS-scale second RMS: +/-sqrt2")
    def accept_output(a,b):
        try: r._compare_state_output((a,expected_mid),(b,expected_mid),{"dtype":"fp32"}); return True
        except AssertionError: return False
    def accept_mid(a,b):
        try: r._compare_state_output((expected_out,a),(expected_out,b),{"dtype":"fp32"}); return True
        except AssertionError: return False
    rows.append(control(out,expected_out,accept_output,"actual final-output comparator EPS-scale answer"))
    rows.append(control(mid,expected_mid,accept_mid,"actual residual comparator EPS-scale answer"))
    return rows
