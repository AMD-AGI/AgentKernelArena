"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[1.,2.]],dtype=torch.bfloat16);s=torch.tensor([2.,3.])
    expected=(torch.tensor([[42,127]],dtype=torch.int8),torch.tensor([[6/127]]))
    return [verify(m.Model()(x,s),expected,"independent smoothing, truncation and int8 scale")]
