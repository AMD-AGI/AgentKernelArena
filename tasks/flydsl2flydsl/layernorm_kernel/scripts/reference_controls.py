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
    h=references(["reference_layernorm"])
    x=torch.tensor([[1.,3.]])
    expected=torch.tensor([[-2./math.sqrt(1.+1e-5)+.5,3./math.sqrt(1.+1e-5)-.5]])
    actual=h.reference_layernorm(x,torch.tensor([2.,3.]),torch.tensor([.5,-.5]))
    return [control(actual,expected,close(h.ATOL,h.RTOL),"two-element population variance and affine transform")]
