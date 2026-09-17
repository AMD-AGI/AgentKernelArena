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
    r = references(['reference', '_compare_mrope_output', '_reference_glm'], {"F": F, "SQRT2": math.sqrt(2), "require_tensor_contract": require_tensor_contract, "require_unchanged": require_unchanged})
    cfg=dict(nt=1,hd=8,rd=6,n_qh=1,n_kh=1,section=(1,1,1),interleaved=False,neox=True)
    x=t([[1.,2.,3.,4.,5.,6.,9.,10.]])
    # time:90 degrees, height:0, width:180. Last two dimensions untouched.
    cache=t([[0.,0.,0.,1.,1.,1.],[1.,1.,1.,0.,0.,0.],[-1.,-1.,-1.,0.,0.,0.]])
    q,k=r.reference(x,x,cache,torch.tensor([[0],[1],[2]]),cfg)
    e=t([[-4.,2.,-3.,1.,5.,-6.,9.,10.]])
    check(q,e,"different temporal/spatial positions and rotary suffix")
    check(k,e,"key rotation follows same sections")
    axes=torch.tensor([2,0,1,3],dtype=torch.int32)
    gq,gk=r._reference_glm(x,x,cache,torch.tensor([[0],[1],[2]]),axes,cfg)
    ge=t([[-1.,-5.,3.,-4.,2.,6.,9.,10.]])
    check(gq,ge,"GLM independent axis permutation:width,time,height with suffix")
    check(gk,ge,"GLM key independent axis permutation")
    def accept(a,b):
        try: r._compare_mrope_output((a,a),(b,b)); return True
        except AssertionError: return False
    rows.append(control(q,e,accept,"actual M-RoPE comparator accepts rotation and rejects wrong output"))
    rows.append(control(gq,ge,accept,"actual GLM comparator accepts axis permutation and rejects wrong output"))
    return rows
