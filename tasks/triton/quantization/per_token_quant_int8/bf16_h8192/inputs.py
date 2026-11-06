"""Activation generator of the per-token INT8 quantization scenario.

Standard-normal rows, with deterministic special rows so the exact-code comparison covers the edge
semantics a random draw almost never reaches:
- every 97th row (starting at row 5) is all zero (the 1e-10 absolute-maximum floor);
- every 97th row (starting at row 11) holds integers in [-254, 254] with a forced maximum of 254, so
  the code multiplier 127/254 is exactly 0.5 and every odd integer lands exactly on a rounding tie
  (distinguishes round-half-away-from-zero from half-to-even and from truncation).
Integers up to 256 are exact in bfloat16.
"""
import torch


def make_inputs(case, inputs, generator, device):
    rows, width = case["axes"]["n_tokens"], case["axes"]["hidden_size"]
    x = torch.randn((rows, width), generator=generator, device=device, dtype=torch.float32)
    zero_rows = torch.arange(5, max(rows, 5), 97, device=device)
    tie_rows = torch.arange(11, max(rows, 11), 97, device=device)
    x[zero_rows] = 0.0
    if tie_rows.numel():
        ints = torch.randint(-254, 255, (tie_rows.numel(), width), generator=generator,
                             device=device).float()
        ints[:, 0] = 254.0
        x[tie_rows] = ints
    return {"x": x.to(torch.bfloat16)}



