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
    h=references(["reference_rms_norm"])
    x=torch.tensor([[3.,4.]])
    expected=torch.tensor([[6.,12.]])/math.sqrt(12.5+1e-5)
    return [control(h.reference_rms_norm(x,torch.tensor([2.,3.])), expected,close(h.ATOL,h.RTOL),"RMS= sqrt((9+16)/2+eps)")]
