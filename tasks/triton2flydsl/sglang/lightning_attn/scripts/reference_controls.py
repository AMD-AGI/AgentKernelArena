"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close
from scripts.replay_checks import require_tensor_contract, require_unchanged


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['reference', '_compare_lightning_output'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract, "require_unchanged": require_unchanged})
    q=t([1.,2.]).reshape(1,1,1,2);k=t([3.,4.]).reshape_as(q);v=t([5.,6.]).reshape_as(q)
    cache=torch.ones(1,1,2,2)
    o,s=r.reference(q,k,v,cache,t([0.]),torch.tensor([0]),dict(B=1,H=1,D=2))
    check(s,t([[[[16.,19.],[21.,25.]]]]),"outer product plus undecayed cache")
    check(o,t([[58.,69.]]),"query contracts updated cache")
    cfg=dict(B=1,H=1,D=2);slots=torch.tensor([0])
    expected=(t([[58.,69.]]),t([[[[16.,19.],[21.,25.]]]]))
    for index in range(2):
        def accept(a,b,index=index):
            actual=list(expected);want=list(expected);actual[index]=a;want[index]=b
            try: r._compare_lightning_output(actual,want,q,slots,cfg); return True
            except AssertionError: return False
        rows.append(control((o,s)[index],expected[index],accept,f"actual lightning comparator output{index}"))
    # Slope ln2 halves the previous state. This changes every controlled entry.
    od,sd=r.reference(q,k,v,cache,t([math.log(2.)]),slots,cfg)
    check(sd,t([[[[15.5,18.5],[20.5,24.5]]]]),"independent one-half decay state")
    check(od,t([[56.5,67.5]]),"independent one-half decay output")
    return rows
