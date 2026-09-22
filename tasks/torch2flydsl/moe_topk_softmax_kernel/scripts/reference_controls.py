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
    model=m.Model(4,1,1.,False);expected_w=torch.tensor([[math.exp(3)/sum(math.exp(i) for i in range(4))]])
    return [verify(model(x),(expected_w,torch.tensor([[3]],dtype=torch.int32)),"unique top expert with independently computed routing weight")]
