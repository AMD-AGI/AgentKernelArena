"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.ones(1,2,dtype=torch.bfloat16);s=torch.tensor([1.,-1.]);w=torch.ones(2,dtype=torch.bfloat16)
    expected=(torch.tensor([[127,-127]],dtype=torch.int8),torch.tensor([[1/127]]))
    return [verify(m.Model(0.)(x,s,w),expected,"zero-epsilon unit RMS control, signed smoothing and int8 quantization")]
