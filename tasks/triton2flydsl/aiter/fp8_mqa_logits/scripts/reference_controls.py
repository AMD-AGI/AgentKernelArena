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
    r = references(['_ref_fp8_mqa_logits'], {"F": F, "SQRT2": math.sqrt(2)})
    q=t([[[1.,0.],[0.,1.]]]);kv=t([[2.,4.],[-1.,3.]])
    a=r._ref_fp8_mqa_logits(q,kv,t([2.,3.]),t([[.5,2.]]),torch.tensor([0]),torch.tensor([2]))
    check(a,t([[18.,18.]]),"head weighting, KV scales and ReLU")
    a=r._ref_fp8_mqa_logits(q,kv,t([2.,3.]),t([[.5,2.]]),torch.tensor([1]),torch.tensor([2]))
    accept_mask = lambda z: bool(torch.isneginf(z[0,0]) and z[0,1]==18)
    assert accept_mask(a) and not accept_mask(torch.zeros_like(a))
    rows.append({"control":"excluded logits retain negative infinity mask","known_answer":"PASS","negative_output":"rejected"})
    return rows
