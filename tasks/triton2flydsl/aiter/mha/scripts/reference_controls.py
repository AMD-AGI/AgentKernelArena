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
    r = references(['torch_mha_ref', '_compare', '_compare_attention_output'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    q=torch.zeros(1,2,2,2);k=torch.zeros(1,2,1,2);v=t([2.,4.,6.,8.]).reshape(1,2,1,2)
    for causal in (False,True):
        actual=r.torch_mha_ref(q,k,v,1.,causal)
        expected=t([2.,4.,4.,6.] if causal else [4.,6.,4.,6.]).reshape(1,2,1,2).repeat(1,1,2,1)
        check(actual,expected,"zero logits give uniform visible GQA values "+str(causal))
        accept=lambda a,b:r._compare(b,a)[0]<=r.NORM_ERR_TOL
        rows.append(control(actual,expected,accept,"production comparator positive/negative control"))
    def accept(a, b):
        try:
            r._compare_attention_output(a, b)
            return True
        except AssertionError:
            return False
    rows.append(control(actual.to(torch.float16), expected.to(torch.float16), accept,
                        "actual measured-path comparator checks independent uniform GQA mean"))
    return rows
