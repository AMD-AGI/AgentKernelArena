"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    model=m.Model(2,1,2);ids=torch.tensor([[1],[0]],dtype=torch.int32);w=torch.tensor([[.25],[.75]])
    sid=(1<<24)|2
    expected=(torch.tensor([1,sid,0,sid,sid],dtype=torch.int32),torch.tensor([.75,0.,.25,0.,0.]),torch.tensor([0,1,-1],dtype=torch.int32),torch.tensor([4,2],dtype=torch.int32))
    return [verify(model(ids,w),expected,"explicit packed token ordering, weights, padding and valid counts",atol=0,rtol=0)]
