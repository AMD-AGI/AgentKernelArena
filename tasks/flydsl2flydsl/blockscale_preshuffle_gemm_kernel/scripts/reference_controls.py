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
    h=references(["_torch_blockscale_reference"])
    # First 128 products contribute 128*2*3; next block 128*5*7.
    a=torch.ones(1,256); b=torch.ones(128,256)
    inp=dict(M=1,N=128,K=256,scale_k=2,a_fp8=a,b_fp8=b,scale_a=torch.tensor([[2.],[5.]]),scale_b=torch.tensor([[3.,7.]]))
    expected=torch.full((1,128),5248.)
    # The deliberate error must exceed this output's relative tolerance.
    actual=h._torch_blockscale_reference(inp)
    check=close(h.ATOL,h.RTOL)
    if not check(actual,expected) or check(expected+10000,expected): raise AssertionError("blockscale known answer/control failed")
    return [{"control":"two independent 128-wide scale blocks","known_answer":"PASS","negative_output":"rejected"}]
