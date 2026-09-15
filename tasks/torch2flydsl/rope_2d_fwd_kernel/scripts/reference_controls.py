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
    return [verify(m.Model(1,1)(x,c,s,c,s),expected,"independent height/width 90-degree rotations",atol=0,rtol=0)]
