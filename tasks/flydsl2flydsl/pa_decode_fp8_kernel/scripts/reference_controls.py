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
    h=references(["reference_masked_attention"])
    q=torch.zeros(1,2,2);k=torch.zeros(3,1,2)
    v=torch.tensor([[[2.,4.]],[[6.,8.]],[[10.,12.]]])
    accept=lambda a,b: bool(torch.isfinite(a).all() and (a-b).abs().max()<=5e-3)
    return [control(h.reference_masked_attention(q,k,v,1.,torch.float32),torch.tensor([[[6.,8.],[6.,8.]]]),accept,"GQA repeated heads and zero-logit average"),control(h.reference_masked_attention(q,k,v,1.,torch.float32,sliding_window=1),torch.tensor([[[8.,10.],[8.,10.]]]),accept,"last two keys in decode window")]
