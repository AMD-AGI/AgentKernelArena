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
    r = references(['_ref_masked_attention', 'torch_mla_extend', '_compare', '_compare_mla_output'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract})
    q=torch.zeros(1,2,2);k=torch.zeros(2,1,2);v=t([2.,4.,6.,8.]).reshape(2,1,2)
    actual=r._ref_masked_attention(q,k,v,1.)
    check(actual,t([4.,6.,4.,6.]).reshape(1,2,2),"single decode query averages both KV tokens with GQA")
    # Reverse page order and a two-query bottom-right causal mask make this
    # independent control exercise the page gather and visible-prefix logic.
    query = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    pages = torch.tensor([[[[2.,4.,0.]], [[6.,8.,0.]]],
                          [[[10.,12.,0.]], [[14.,16.,0.]]]], dtype=torch.bfloat16)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    expected = torch.tensor([[[12.,14.],[12.,14.]],
                             [[8.6875,10.6875],[8.6875,10.6875]]], dtype=torch.bfloat16)
    # Last-query mean uses BF16 softmax(1/3)=0.333984375, as specified by the
    # retained reference; its sum26*0.333984375 rounds to8.6875.
    actual = r.torch_mla_extend(query,pages,cu,lengths,table,2,1.,torch.bfloat16)
    check(actual, expected, "reverse paged gather and bottom-right causal BF16 means")
    def accept(a,b):
        try:
            r._compare_mla_output(a,b)
            return True
        except AssertionError:
            return False
    rows.append(control(actual,expected,accept,"actual MLA comparator independent paged means"))
    return rows
