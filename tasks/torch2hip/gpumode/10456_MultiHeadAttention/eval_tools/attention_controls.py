# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Additional declared correctness cases; no added performance samples."""
import math
import torch


def generate(row, device="cpu"):
    p = row["params"]
    generator = torch.Generator(device="cpu").manual_seed(p["input_seed"])
    inputs = [torch.randn(1, p["sequence_length"], 4, generator=generator).to(device) for _ in range(3)]
    if p["mask"] == "causal":
        inputs.append(torch.ones(1, p["sequence_length"], p["sequence_length"], dtype=torch.uint8, device=device).tril())
    return inputs


def validate_rows(rows):
    expected = [("heads4_lower_boundary",139,"none",4242), ("heads4_upper_mask",230,"causal",4243)]
    if len(rows) != len(expected):
        raise ValueError("Attention routing controls must contain both boundary cases")
    for row,(name,length,mask,seed) in zip(rows,expected):
        p=row["params"]
        if (row["test_case_id"],p["sequence_length"],p["mask"],p["input_seed"]) != (name,length,mask,seed):
            raise ValueError("Attention routing control identity changed")
        if row.get("checks") != ["correctness"] or p.get("heads") != 4 or p.get("model_init_seed") != 0:
            raise ValueError("Attention controls require a fresh four-head model and correctness-only scope")


def reference(model, inputs, heads=4, ignore_mask=False):
    """Independent FP64 per-head attention; no call to the module/functional path."""
    q,k,v=inputs[:3]
    dim=q.shape[-1]//heads
    q=q.double() @ model.q_linear1.detach().double()
    k=k.double() @ model.k_linear1.detach().double()
    v=v.double() @ model.v_linear1.detach().double()
    outputs=[]
    for head in range(heads):
        sl=slice(head*dim,(head+1)*dim)
        scores=(q[...,sl] @ k[...,sl].transpose(-1,-2))/math.sqrt(dim)
        if len(inputs)>3 and not ignore_mask:
            scores=scores.masked_fill(inputs[3]==0,-1e9)
        weights=torch.exp(scores-scores.amax(dim=-1,keepdim=True))
        weights=weights/weights.sum(dim=-1,keepdim=True)
        outputs.append(weights @ v[...,sl])
    result=torch.cat(outputs,dim=-1) @ model.out.weight.detach().double().T
    result=result+model.out.bias.detach().double()
    return result.to(inputs[0].dtype)


def self_test(module_class,functional_class,rows):
    validate_rows(rows)
    for row in rows:
        torch.manual_seed(0)
        module=module_class(heads=4,d_model=4).eval()
        functional=functional_class(heads=4,d_model=4).eval()
        functional.load_state_dict(module.state_dict())
        inputs=generate(row)
        with torch.no_grad():
            expected=reference(module,inputs)
            for model in (module,functional):
                torch.testing.assert_close(model(*inputs),expected,rtol=1e-4,atol=1e-5)
            if torch.allclose(reference(module,inputs,heads=2),expected,rtol=1e-4,atol=1e-5):
                raise ValueError("Attention control does not distinguish head routing")
            if len(inputs)>3 and torch.allclose(reference(module,inputs,ignore_mask=True),expected,rtol=1e-4,atol=1e-5):
                raise ValueError("Attention control does not distinguish ignored mask")
