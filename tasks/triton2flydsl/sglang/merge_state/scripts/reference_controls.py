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
    r = references(['_compare_state_output', 'reference_merge'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    o,lse=r.reference_merge(t([[[2.,4.]]]),t([[0.]]),t([[[6.,8.]]]),t([[math.log(3.)]]))
    check(o,t([[[5.,7.]]]),"partial states weighted 1:3")
    check(lse,t([[math.log(4.)]]),"log-sum-exp of weights 1 and 3")
    expected_o = t([[[5.,7.]]]); expected_lse = t([[math.log(4.)]])
    def accept_value(a,b):
        try: r._compare_state_output((a,expected_lse),(b,expected_lse),{"dtype":"fp32"}); return True
        except AssertionError: return False
    def accept_lse(a,b):
        try: r._compare_state_output((expected_o,a),(expected_o,b),{"dtype":"fp32"}); return True
        except AssertionError: return False
    rows.append(control(o,expected_o,accept_value,"actual merge value comparator independent1:3mean"))
    rows.append(control(lse,expected_lse,accept_lse,"actual merge LSE comparator independentlog4"))
    return rows
