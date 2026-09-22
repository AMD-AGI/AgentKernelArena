"""Independent CPU known answers and deliberately wrong-output controls.
These controls supplement, never replace, the full GPU case suite.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_support import references, control, close

def run():
    import torch
    import math
    h=references(["_torch_reference"])
    a=torch.tensor([[1.,2.],[3.,4.]])
    b=torch.tensor([[5.,6.],[7.,8.]])
    expected=torch.tensor([[17.,46.],[78.,212.]]).to(torch.bfloat16).float()
    return [control(h._torch_reference(a,b,torch.tensor([1.,2.]),torch.tensor([1.,2.])),expected,close(h.ATOL,h.RTOL),"hand-computed GEMM with distinct row/column scales")]
