"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[3.,4.]],dtype=torch.bfloat16);w=torch.tensor([2.,3.],dtype=torch.bfloat16)
    expected=(torch.tensor([[6.,12.]])/math.sqrt(12.5+1e-5)).to(torch.bfloat16)
    return [verify(m.Model()(x,w),expected,"hand-computed weighted RMS normalization",atol=0,rtol=0)]
