"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.cat([torch.ones(1,32),torch.full((1,32),-2.)],dim=1).to(torch.bfloat16)
    expected_q=torch.cat([torch.full((1,32),256.),torch.full((1,32),-256.)],dim=1).to(torch.float8_e4m3fn)
    return [verify(m.Model()(x),(expected_q,torch.tensor([[119,120]],dtype=torch.uint8)),"independent power-of-two exponents for two MXFP8 blocks")]
