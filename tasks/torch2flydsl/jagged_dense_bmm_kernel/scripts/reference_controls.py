"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    w=torch.tensor([[[5.,6.]],[[7.,8.]]],dtype=torch.bfloat16);b=torch.tensor([[1.],[2.]],dtype=torch.bfloat16)
    expected=torch.tensor([[18.],[55.]],dtype=torch.bfloat16)
    return [verify(m.Model()(x,w,b,torch.tensor([0,1,2])),expected,"separate jagged-group dot products and distinct biases",atol=0,rtol=0)]
