"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    # Independent sub-step evidence for the same grouped-GEMM function used
    # by the complete protected MoE reference; full quantization/gating is
    # additionally checked against AITER on every original GPU case.
    acts=torch.tensor([[1.,2.],[3.,4.]])
    weights=torch.tensor([[[5.,6.],[7.,8.]],[[1.,2.],[3.,4.]]])
    ids=torch.tensor([[0],[1]],dtype=torch.int64)
    expected=torch.tensor([[[17.,23.]],[[11.,25.]]])
    actual=m._grouped_gemm_stage1(acts,weights,ids)
    return [verify(actual,expected,"independent per-expert routed matrix products",atol=0,rtol=0)]
