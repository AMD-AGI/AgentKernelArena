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
    h=references(["reference_topk"],dict(DTYPE_FP32=torch.float32))
    x=torch.tensor([[0.,math.log(2.),math.log(3.)]])
    probs,w,ids,tei=h.reference_topk(x,2)
    out=[control(probs,torch.tensor([[1/6,2/6,3/6]]),close(1e-5,0),"1:2:3 softmax probabilities"),control(w,torch.tensor([[.6,.4]]),close(1e-5,0),"renormalized top two weights")]
    if ids.tolist()!=[[2,1]] or tei.tolist()!=[[0,1]]: raise AssertionError("TopK indices/slot convention known answer failed")
    return out
