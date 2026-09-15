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
    r = references(['reference', '_compare_attention_output'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    q=torch.zeros(2,2,2);k=torch.zeros(2,1,2);v=t([2.,4.,6.,8.]).reshape(2,1,2)
    cfg=dict(seqs=[2],head=2,kv_head=1,d=2,causal=True)
    check(r.reference(q,k,v,cfg),t([2.,4.,2.,4.,4.,6.,4.,6.]).reshape_as(q),"causal prefill prefix means with GQA")
    def accept(a, b):
        try:
            r._compare_attention_output(a, b)
            return True
        except AssertionError:
            return False
    expected = t([2.,4.,2.,4.,4.,6.,4.,6.]).reshape_as(q)
    rows.append(control(r.reference(q,k,v,cfg).to(torch.bfloat16), expected, accept,
                        "actual measured-path comparator checks independent causal prefix means"))
    return rows
