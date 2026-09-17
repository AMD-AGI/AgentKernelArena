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
    h=references(["reference_gemm"])
    a=torch.tensor([[1.,2.],[3.,4.]])
    b=torch.tensor([[5.,6.],[7.,8.]])
    expected=torch.tensor([[17.,23.],[39.,53.]])
    accept=lambda a,b: bool(torch.isfinite(a).all() and (a-b).abs().max()/(b.abs().max()+1e-6)<=h.RTOL)
    return [control(h.reference_gemm(a,b),expected,accept,"hand-computed 2x2 A @ B.T")]
