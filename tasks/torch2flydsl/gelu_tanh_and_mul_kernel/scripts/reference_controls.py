"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    xs=[-1.,0.,1.,2.]
    gate=torch.tensor([xs],dtype=torch.bfloat16)
    x=torch.cat([gate,torch.full_like(gate,2.)],dim=1)
    expected=torch.tensor([[(z*.5*(1+math.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3))))*2 for z in xs]],dtype=torch.bfloat16)
    return [verify(m.Model()(x),expected,"scalar activation formula and independent multiplication",atol=0,rtol=0)]
