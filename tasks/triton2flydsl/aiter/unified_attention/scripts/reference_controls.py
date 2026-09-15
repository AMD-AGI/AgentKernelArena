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
    r = references(['_compare_unified_output', 'ref_paged_attn'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    q=torch.zeros(1,2,2);k=torch.zeros(2,2,1,2);v=t([100.,100.,100.,100.,2.,4.,6.,8.]).reshape(2,2,1,2)
    check(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[1]]),1.,torch.float32),t([[[4.,6.],[4.,6.]]]),"physical block selection and GQA mean")
    check(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[1]]),1.,torch.float32,sliding_window=1),t([[[6.,8.],[6.,8.]]]),"one-token sliding window")
    def accept(a,b):
        try: r._compare_unified_output(a.to(torch.bfloat16),b); return True
        except AssertionError: return False
    rows.append(control(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[1]]),1.,torch.float32),
                        t([[[4.,6.],[4.,6.]]]),accept,"actual paged comparator physical-block GQA mean"))
    q=t([[[1.,0.]]]);k=t([0.,0.,2.,0.]).reshape(1,2,1,2);v=t([2.,4.,6.,8.]).reshape(1,2,1,2)
    weight=math.exp(math.tanh(2.))/(1.+math.exp(math.tanh(2.)))
    expected=t([[[2.+4.*weight,4.+4.*weight]]])
    rows.append(control(r.ref_paged_attn(q,k,v,[1],[2],torch.tensor([[0]]),1.,torch.float32,soft_cap=1.),
                        expected,accept,"nonuniform softcap weights independently computed in FP64"))
    return rows
