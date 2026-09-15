"""Small independent known answers for protected numerical references.

These controls are not workload replacements and produce no benchmark score.
"""
import math
import torch


def equal(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.004, rtol=0.004)


def rejects(check, bad):
    try:
        check(bad)
    except (AssertionError, ValueError):
        return
    raise AssertionError("Task comparison accepted the deliberately wrong control")


def attention_data():
    query = torch.tensor([[[math.log(3), 0.], [0., math.log(3)]]])
    key = torch.tensor([[[[0., 0.]], [[1., 0.]]]])
    value = torch.tensor([[[[2., 4.]], [[6., 8.]]]])
    expected = torch.tensor([[[5., 7.], [4., 6.]]])
    return {"query":query, "key":key, "value":value,
            "output":torch.empty_like(query), "scale":1., "ctx_len":2,
            "sliding_window":0}, expected


def moe_data(activation):
    x = torch.tensor([[1., 2.], [-1., 1.]], dtype=torch.bfloat16)
    w1 = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]], dtype=torch.bfloat16)
    w2 = torch.tensor([[[1.], [2.]], [[3.], [-1.]]], dtype=torch.bfloat16)
    ids = torch.tensor([[0,1],[1,0]])
    weights = torch.tensor([[0.25,0.75],[0.6,0.4]])
    expected = torch.zeros(2,2)
    for t in range(2):
        for slot in range(2):
            e=int(ids[t,slot]);a,b=map(float,x[t]);gate,up=(a,b) if e==0 else (b,a)
            if activation=='gelu_tanh':g=0.5*gate*(1+math.tanh(math.sqrt(2/math.pi)*(gate+0.044715*gate**3)))*up
            elif activation=='situv2':g=4*math.tanh(gate/4)/(1+math.exp(-gate))*25*math.tanh(up/25)
            else:g=gate/(1+math.exp(-gate))*up
            expected[t,0]+=g*float(w2[e,0,0])*float(weights[t,slot])
            expected[t,1]+=g*float(w2[e,1,0])*float(weights[t,slot])
    inputs={"x":x,"hidden":x,"w1":w1,"w2":w2,"w1_deq":w1,"w2_deq":w2,
            "topk_ids":ids,"topk_weights":weights,"inter":1,"activation":activation}
    return inputs,expected.to(torch.bfloat16)


def assert_moe_cosine(got,expected):
    error=1-torch.nn.functional.cosine_similarity(got.float().flatten(),expected.float().flatten(),dim=0)
    assert torch.isfinite(got).all() and float(error)<0.03


def assert_relative_error(h,got,expected,tol):
    assert h._relerr(got,expected)<tol


def check_reference(h):
    inputs={"H":1,"D":1,"scale":1.,"mode":"chunk","q":torch.ones(1,1,1,1),"k":torch.ones(1,1,1,1),
            "v":torch.tensor([[[[2.]]]]),"raw_g":torch.zeros(1,1,1,1),"A_log":torch.zeros(1),"dt_bias":torch.zeros(1),
            "raw_beta":torch.zeros(1,1,1),"total_t":1,"segments":[(0,1)],"seg_state0":[torch.tensor([[[3.]]],dtype=torch.float64)]}
    norm=math.sqrt(1.+1e-6);decayed=3.*math.exp(-2.5)
    updated=decayed+(2.-decayed/norm)*0.5/norm
    expected=torch.tensor([[[[updated/norm]]]],dtype=torch.float64)
    equal(h._golden(inputs),expected)
    inputs["mode"]="packed_decode";inputs["mixed_qkv"]=torch.tensor([[1.,1.,2.]])
    equal(h._golden(inputs),expected)
    rejects(lambda bad:torch.testing.assert_close(bad,expected),torch.zeros_like(expected))
    return {"known_answer": "PASS", "negative_control": "PASS", "scored": False}
