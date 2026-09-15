"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[[1.,2.,3.,4.]],[[5.,6.,7.,8.]]],dtype=torch.bfloat16)
    freqs=torch.full((1,1,1,2),math.pi/2)
    expected=torch.tensor([[[-3.,-4.,1.,2.]],[[-7.,-8.,5.,6.]]],dtype=torch.bfloat16)
    return [verify(m.Model()(x,torch.tensor([0,1,2]),freqs),expected,"packed sequence-local position reset and 90-degree rotations",atol=0,rtol=0)]
