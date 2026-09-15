"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    a=torch.ones(2,128,dtype=torch.bfloat16);a[1]*=-1
    w=torch.ones(128,128,dtype=torch.bfloat16);w[1::2]*=-1
    expected=torch.full((2,128),128.,dtype=torch.bfloat16);expected[1]*=-1;expected[:,1::2]*=-1
    a=a.unsqueeze(0).repeat(2,1,1);w=w.unsqueeze(0).repeat(2,1,1);expected=expected.unsqueeze(0).repeat(2,1,1)
    a[1]*=-1;expected[1]*=-1
    return [verify(m.Model()(a,w),expected,"signed constant dot products with independent integer sums",atol=0,rtol=0)]
