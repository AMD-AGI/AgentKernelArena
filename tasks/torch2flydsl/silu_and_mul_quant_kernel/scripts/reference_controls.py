"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.cat([torch.ones(1,128),torch.full((1,128),2.)],dim=1).to(torch.bfloat16)
    dt=m._FP8_DTYPE;mx=float(torch.finfo(dt).max);value=2/(1+math.exp(-1))
    expected=(torch.full((1,128),mx).to(dt),torch.tensor([[value/mx]]))
    return [verify(m.Model()(x),expected,"scalar SiLU product followed by one FP8 group")]
