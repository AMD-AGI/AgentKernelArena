"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    model=m.Model();dt=m._FP8_DTYPE;mx=float(torch.finfo(dt).max)
    x=torch.ones(2,256,dtype=torch.bfloat16);x[1]*=-1
    expected_q=torch.full((2,256),mx).to(dt);expected_q=torch.cat([expected_q[:1],(-expected_q[1:].float()).to(dt)])
    expected_s=torch.full((2,1),1/mx)
    return [verify(model(x),(expected_q,expected_s),"constant signed extrema quantization codes and inverse range scale")]
