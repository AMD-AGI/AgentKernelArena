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
    r = references(['_compare_gr_output', '_torch_ref', '_close'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    a=r._torch_ref(t([[1.,3.]]),t([2.,4.]),t([1.,2.]),1e-5)
    e=t([[1-2/math.sqrt(1.00001),2+4/math.sqrt(1.00001)]])
    check(a,e,"population variance and affine layer norm")
    rows.append(control(a,e,lambda a,b:r._close(b,a)[0],"task comparator rejects incorrect normalization"))
    actual_zero = r._torch_ref(torch.ones(2,2), None, None, r.EPS)
    expected_zero = torch.zeros(2,2)
    check(actual_zero, expected_zero, "independent exactly zero operator output")
    def accept(a,b):
        try:
            r._compare_gr_output(a,b)
            return True
        except AssertionError:
            return False
    rows.append(control(actual_zero.to(torch.bfloat16), expected_zero, accept,
                        "zero-reference numerical comparator rejects wrong output"))
    return rows
