"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    model=m.Model(2,4,2,2)
    q=torch.ones(1,8,dtype=torch.bfloat16);kv=torch.ones(1,4,dtype=torch.bfloat16);w=torch.full((4,),2.,dtype=torch.bfloat16)
    c=torch.ones(1,1,1,1);s=torch.zeros_like(c)
    actual=model(q,kv,w,c,s,torch.tensor([0]))
    expected=(torch.ones(1,2,4,dtype=torch.bfloat16),torch.full((1,4),2.,dtype=torch.bfloat16))
    return [verify(actual,expected,"headwise RMS, KV-only weight and identity rope tail",atol=0,rtol=0)]
