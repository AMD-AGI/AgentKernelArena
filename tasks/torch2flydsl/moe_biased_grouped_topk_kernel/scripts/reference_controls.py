"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[0.,1.,2.,3.]])
    model=m.Model(4,1,2,1,True,2.5);model.correction_bias.data.zero_();expected_w=torch.tensor([[2.5]])
    return [verify(model(x),(expected_w,torch.tensor([[3]],dtype=torch.int32)),"unique top expert with independently computed routing weight")]
