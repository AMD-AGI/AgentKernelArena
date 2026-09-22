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
    h=references(["reference_swa_decode"])
    H=h.HEAD_SIZE; B=h.KV_BLOCK_SIZE; G=h.QUERY_GROUP_SIZE; X=h.X
    # Physical block one is selected: first stored value=2, second=6,
    # third=10; a window of one keeps the last two, hence mean eight.
    kc=torch.zeros(2,1,H//X,B,X);vc=torch.zeros(2,1,H,B)
    vc[1,:,:,0]=2;vc[1,:,:,1]=6;vc[1,:,:,2]=10
    data=dict(num_seqs=1,num_kv_heads=1,sliding_window=1,query=torch.zeros(1,G,H),block_tables=torch.tensor([[1]]),context_lengths=torch.tensor([3]),key_scale=torch.tensor([1.]),value_scale=torch.tensor([.5]),key_cache=kc,value_cache=vc)
    expected=torch.full((1,1,1,G,H),4.,dtype=torch.bfloat16)
    accept=lambda a,b: bool(torch.isfinite(a).all() and (a-b).abs().max()<=h.ATOL)
    return [control(h.reference_swa_decode(data),expected,accept,"physical page mapping, sliding window, dequant scale and GQA")]
