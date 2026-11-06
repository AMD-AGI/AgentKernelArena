"""Activation generator with per-group dynamic range and all-zero groups.

Every 128-element group is a standard-normal draw times 2^e with e uniform in [-12, 6], and about
one group in 32 is all zero, so the scale spans five decades across a row and the eps floor of
the group scale is exercised on every case. Rows are generated in chunks to bound temporaries.
"""
import torch

GROUP_SIZE = 128
_CHUNK_ROWS = 8192


def make_inputs(case, inputs, generator, device):
    rows = case["axes"]["n_tokens"]
    width = case["axes"]["hidden_size"]
    groups = width // GROUP_SIZE
    x = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    for start in range(0, rows, _CHUNK_ROWS):
        n = min(_CHUNK_ROWS, rows - start)
        values = torch.randn((n, groups, GROUP_SIZE), generator=generator, device=device)
        exponent = torch.randint(-12, 7, (n, groups, 1), generator=generator, device=device)
        zero = torch.rand((n, groups, 1), generator=generator, device=device) < (1.0 / 32.0)
        gain = torch.exp2(exponent.float()).masked_fill(zero, 0.0)
        x[start:start + n] = (values * gain).reshape(n, width).to(torch.bfloat16)
    return {"x": x}
