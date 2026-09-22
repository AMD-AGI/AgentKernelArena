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
    h=references(["reference_softmax"])
    x=torch.tensor([[0.,0.],[0.,math.log(3.)]])
    expected=torch.tensor([[.5,.5],[.25,.75]])
    return [control(h.reference_softmax(x), expected, close(h.ATOL,h.RTOL), "uniform and 1:3 logits")]
