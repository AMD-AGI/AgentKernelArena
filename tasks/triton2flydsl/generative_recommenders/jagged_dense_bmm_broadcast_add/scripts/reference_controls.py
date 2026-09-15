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
    r = references(['_compare_gr_output', '_torch_ref'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    offsets=torch.tensor([0,1,1,3]);x=t([[1.,2.],[3.,4.],[5.,6.]])
    dense=t([[[1.],[2.]],[[99.],[99.]],[[3.],[4.]]])
    check(r._torch_ref(offsets,x,dense,t([[10.],[99.],[20.]]),False),t([[15.],[45.],[59.]]),"empty segment and per-batch bias")
    check(r._torch_ref(offsets,x,dense,t([[10.],[20.],[30.]]),True),t([[15.],[45.],[69.]]),"per-row bias")
    actual_zero = r._torch_ref(torch.tensor([0,2]), torch.zeros(2,2), torch.ones(1,2,2), torch.zeros(1,2), False)
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
