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
    from reference_mxfp8 import dequant_mxfp8_to_bf16
    encoded=torch.ones(1,64).to(torch.float8_e4m3fn)
    scales=torch.tensor([[127,128]],dtype=torch.uint8)
    expected=torch.tensor([[1.]*32+[2.]*32],dtype=torch.bfloat16)
    equal(dequant_mxfp8_to_bf16(encoded,scales),expected)
    inputs={"a_q":encoded,"a_s":scales,"w13_fp8":encoded.repeat(2,2,1),"w13_scale":scales.repeat(2,2,1),
            "w2_fp8":torch.ones(2,64,32).to(torch.float8_e4m3fn),"w2_scale":torch.full((2,64,1),127,dtype=torch.uint8),
            "topk_ids":torch.tensor([[1,0]]),"topk_weights":torch.tensor([[0.25,0.75]]),"top_k":2,"T":1,"I":1,"H":64,
            "gemm2_args":{"a_q":torch.ones(2,32).to(torch.float8_e4m3fn),"a_scale":torch.full((2,1),127,dtype=torch.uint8)}}
    gemm1,gemm2=h._timed_references(inputs)
    equal(gemm1,torch.full((2,2),160.,dtype=torch.bfloat16))
    equal(gemm2,torch.tensor([[8.]*64,[24.]*64]))
    rejects(lambda bad:assert_relative_error(h,bad,gemm2,0.08),torch.zeros_like(gemm2))
    return {"known_answer": "PASS", "negative_control": "PASS", "scored": False}
