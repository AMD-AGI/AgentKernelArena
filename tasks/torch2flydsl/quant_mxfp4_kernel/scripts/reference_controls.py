"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    # [1,-1] at scale 1/4 has signed E2M1 codes [6,14], packed as 0xe6.
    x=torch.tensor([[1.,-1.]*16],dtype=torch.bfloat16)
    packed,scale=m.Model()(x)
    # Byte views are the original comparator's contract for these shell dtypes.
    expected_p=torch.full((1,16),0xe6,dtype=torch.uint8)
    expected_s=torch.tensor([[125]],dtype=torch.uint8)
    record=verify((packed.view(torch.uint8),scale.view(torch.uint8)),(expected_p,expected_s),"hand-derived MXFP4 nibbles, packing and E8M0 scale",atol=0,rtol=0)
    return [record]
