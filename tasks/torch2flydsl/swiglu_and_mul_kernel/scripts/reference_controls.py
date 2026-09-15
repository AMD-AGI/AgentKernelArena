"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify
from swiglu_controls import SATURATION_GATE, SATURATION_LINEAR

def run():
    import torch
    import math
    m=load_model()
    xs=[-1.,0.,1.,2.]
    gate=torch.tensor([xs],dtype=torch.bfloat16)
    x=torch.cat([gate,torch.full_like(gate,2.)],dim=1)
    expected=torch.tensor([[(z/(1+math.exp(-m.ALPHA*z)))*3 for z in xs]],dtype=torch.bfloat16)
    controls = [verify(m.Model()(x),expected,"scalar activation formula and independent multiplication",atol=0,rtol=0)]
    x = torch.tensor([SATURATION_GATE + SATURATION_LINEAR], dtype=torch.bfloat16)
    # Independent Python scalar oracle: do not copy Model's torch.clamp code.
    expected = torch.tensor([[
        min(g, 7.) / (1 + math.exp(-1.702 * min(g, 7.))) * (max(-7., min(y, 7.)) + 1.)
        for g, y in zip(SATURATION_GATE, SATURATION_LINEAR)
    ]], dtype=torch.bfloat16)
    controls.append(verify(m.Model()(x), expected,
        "independent scalar gate-upper and linear-two-sided saturation answers", atol=0, rtol=0))
    return controls
