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
    h=references(["reference_flash_attn"])
    q=torch.zeros(1,2,1,2); k=torch.zeros_like(q)
    v=torch.tensor([[[[2.,4.]],[[6.,8.]]]])
    accept=lambda a,b: bool(torch.isfinite(a).all() and (a-b).abs().max()<h.ATOL_BY_DTYPE["f16"])
    return [control(h.reference_flash_attn(q,k,v,causal=True),torch.tensor([[[[2.,4.]],[[4.,6.]]]]),accept,"causal uniform prefix means"),control(h.reference_flash_attn(q,k,v,causal=False),torch.tensor([[[[4.,6.]],[[4.,6.]]]]),accept,"noncausal uniform mean")]
