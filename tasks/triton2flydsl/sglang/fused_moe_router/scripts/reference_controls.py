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
    r = references(['_compare_router_output', 'reference'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[1.,0.]]);w=t([[0.,0.],[math.log(3.),0.]])
    weights,ids=r.reference(x,w,dict(cap=0,topk=1),None)
    check(weights,t([[.75]]),"softmax odds 1:3")
    check(ids.float(),t([[1.]]),"unique winning expert")
    expected_weights=t([[.75]]); expected_ids=torch.tensor([[1]],dtype=torch.int32)
    def accept_weights(a,b):
        try: r._compare_router_output((a,expected_ids),(b,expected_ids),{}); return True
        except AssertionError: return False
    def accept_ids(a,b):
        try: r._compare_router_output((expected_weights,a),(expected_weights,b),{}); return True
        except AssertionError: return False
    rows.append(control(weights,expected_weights,accept_weights,"actual router comparator odds1:3"))
    rows.append(control(ids,expected_ids,accept_ids,"actual router comparator exact expert IDs"))
    # Two-coordinate dot products, softcap and bias independently computed in
    # Python FP64, including top-k2 where selected weights must not sum to one.
    x=t([[1.,-2.]]); w=t([[2.,1.],[-1.,2.],[3.,-1.]])
    bias=t([.5,-.25,1.]); cfg=dict(cap=3.,topk=2)
    logits=[3.*math.tanh(v/3.)+b for v,b in zip([0.,-5.,5.],[.5,-.25,1.])]
    denom=sum(math.exp(v) for v in logits)
    order=sorted(range(3),key=lambda i:logits[i],reverse=True)[:2]
    expected_weights=t([[math.exp(logits[i])/denom for i in order]])
    expected_ids=torch.tensor([order],dtype=torch.int32)
    weights,ids=r.reference(x,w,cfg,bias)
    rows.append(control(weights,expected_weights,accept_weights,"softcap then bias, unnormalized topk2 FP64 oracle"))
    rows.append(control(ids,expected_ids,accept_ids,"softcap/bias exact descending expert IDs"))
    return rows
