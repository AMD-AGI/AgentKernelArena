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
    h=references(["reference_rope_neox"])
    x=torch.tensor([[[1.,2.,3.,4.]],[[5.,6.,7.,8.]]])
    cos=torch.tensor([[1.,1.],[0.,0.]]); sin=torch.tensor([[0.,0.],[1.,1.]])
    expected=torch.tensor([[[1.,2.,3.,4.]],[[-7.,-8.,5.,6.]]])
    return [control(h.reference_rope_neox(x,cos,sin,torch.tensor([0,1])),expected,close(h.ATOL,h.RTOL),"identity and 90-degree NeoX rotations")]
