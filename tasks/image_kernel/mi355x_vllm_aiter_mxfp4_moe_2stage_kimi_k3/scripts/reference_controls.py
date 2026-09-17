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
            # This reference stores stage1 as BF16 before the down projection.
            # Round the independent scalar answer at the same public boundary.
            g = float(torch.tensor(g, dtype=torch.bfloat16))
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
    import aiter
    from _reference_fused_moe import torch_moe_stage1,torch_moe_stage2
    inputs,expected=moe_data('situv2')
    activation=aiter.ActivationType.Situv2
    kwargs={"dtype":torch.bfloat16,"activation":activation,"quant_type":aiter.QuantType.No}
    kwargs.update(situ_beta=4.,situ_linear_beta=25.)
    a=torch_moe_stage1(inputs["x"],inputs["w1"],inputs["w2"],inputs["topk_weights"],inputs["topk_ids"],**kwargs)
    actual=torch_moe_stage2(a,inputs["w1"],inputs["w2"],inputs["topk_weights"],inputs["topk_ids"],dtype=torch.bfloat16,quant_type=aiter.QuantType.No)
    equal(actual,expected)
    case={"id":"reference-control","params":{"min_cosine":0.999,"max_rel_norm_err":0.05}}
    good=h._moe_deviation(expected,expected);h._assert_moe_within_tolerance(case,*good,"known answer")
    rejects(lambda bad: h._moe_deviation(bad, expected), expected.float())
    rejects(lambda bad: h._moe_deviation(bad, expected), expected.flatten())
    rejects(lambda bad:h._assert_moe_within_tolerance(case,*h._moe_deviation(bad,expected),"negative control"),torch.zeros_like(expected))
    return {"known_answer": "PASS", "negative_control": "PASS", "scored": False}
