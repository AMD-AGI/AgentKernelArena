"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify
from limit_controls import LIMIT_PROBES, GATE_VALUES, UP_VALUES

def run():
    import torch
    import math
    m=load_model()
    xs=[-1.,0.,1.,2.]
    gate=torch.tensor([xs],dtype=torch.bfloat16)
    x=torch.cat([gate,torch.full_like(gate,2.)],dim=1)
    expected=torch.tensor([[(z/(1+math.exp(-z)))*2 for z in xs]],dtype=torch.bfloat16)
    controls = [verify(m.Model()(x),expected,"scalar activation formula and independent multiplication",atol=0,rtol=0)]
    x = torch.tensor([GATE_VALUES + UP_VALUES], dtype=torch.bfloat16)
    for limit in LIMIT_PROBES:
        expected_values = []
        for gate, up in zip(GATE_VALUES, UP_VALUES):
            # Independent scalar clamp with the documented BF16 gate recast.
            g = float(torch.tensor(min(gate, limit), dtype=torch.bfloat16))
            u = max(-limit, min(up, limit))
            expected_values.append(g / (1 + math.exp(-g)) * u)
        expected = torch.tensor([expected_values], dtype=torch.bfloat16)
        controls.append(verify(m.Model(limit)(x), expected,
            f"positive limit={limit}: scalar clamps and BF16 gate recast", atol=0, rtol=0))
    return controls
