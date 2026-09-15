"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[[[1.,2.,3.,4.,5.,6.,7.,8.]]]],dtype=torch.bfloat16)
    c=torch.zeros(1,1,1,4);s=torch.ones_like(c)
    expected=torch.tensor([[[[-3.,-4.,1.,2.,-7.,-8.,5.,6.]]]],dtype=torch.bfloat16)
    results = [verify(m.Model(1,1)(x,c,s,c,s),expected,"independent height/width 90-degree rotations",atol=0,rtol=0)]
    from reference_support import references
    compare = references(["_compare"])._compare
    reference = torch.ones(2000, dtype=torch.bfloat16)
    if not compare(reference, reference)[0]:
        raise AssertionError("2D RoPE comparator rejected an exact known answer")
    sparse_error = reference.clone()
    sparse_error[0] = 1e20
    accepted, worst, percentage = compare(reference, sparse_error)
    if accepted or worst <= 0.01 or percentage < 99.9:
        raise AssertionError("2D RoPE gate failed its sparse-outlier negative control")
    results.append({"control": "sparse catastrophic error cannot bypass worst-element bound",
                    "known_answer": "PASS", "negative_output": "rejected", "comparator": "task _compare"})
    return results
