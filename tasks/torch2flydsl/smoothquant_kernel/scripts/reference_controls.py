"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[1.,2.]],dtype=torch.bfloat16);s=torch.tensor([2.,3.])
    expected=(torch.tensor([[42,127]],dtype=torch.int8),torch.tensor([[6/127]]))
    rows = [verify(m.Model()(x,s),expected,"independent smoothing, truncation and int8 scale")]
    from semantic_controls import run_controls
    from reference_support import references
    from scripts.replay_checks import require_unchanged
    h = references(["_compare_quant_outputs", "_checked_quant_pair", "_compare"])
    def compare(actual, expected, inputs):
        h._compare_quant_outputs(actual, expected, inputs, m)
    rows.extend(run_controls(m.Model(), compare, require_unchanged, device="cpu"))
    return rows
