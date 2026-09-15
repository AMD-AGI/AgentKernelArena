"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[1.,3.]],dtype=torch.bfloat16);w=torch.tensor([2.,3.],dtype=torch.bfloat16);b=torch.tensor([.5,-.5],dtype=torch.bfloat16)
    expected=torch.tensor([[-2/math.sqrt(1+1e-5)+.5,3/math.sqrt(1+1e-5)-.5]],dtype=torch.bfloat16)
    r=torch.ones_like(x);actual=m.Model()(x,r,w,b);expected=(expected,x+r)
    return [verify(actual,expected,"population mean/variance and affine output plus residual",atol=0,rtol=0)]
