"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[[[1.,2.,3.,4.]]]],dtype=torch.bfloat16)
    c=torch.zeros(1,1,1,2);s=torch.ones_like(c)
    expected=torch.tensor([[[[-3.,-4.,1.,2.]]]],dtype=torch.bfloat16)
    first=verify(m.Model(0,True,False)(x,c,s),expected,"90-degree NeoX rotation",atol=0,rtol=0)
    expected=torch.tensor([[[[-2.,1.,-4.,3.]]]],dtype=torch.bfloat16)
    return [first,verify(m.Model(1,True,False)(x,c,s),expected,"90-degree GPT-J rotation",atol=0,rtol=0)]
