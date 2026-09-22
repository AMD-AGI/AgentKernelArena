"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.ones(1,2,dtype=torch.bfloat16);w=torch.ones(2,dtype=torch.bfloat16);dt=m._FP8_DTYPE;mx=float(torch.finfo(dt).max)
    expected=(torch.full((1,2),mx).to(dt),torch.tensor([[1/math.sqrt(1+1e-5)/mx]]))
    return [verify(m.Model()(x,w),expected,"unit RMS row and dynamic FP8 scale")]
