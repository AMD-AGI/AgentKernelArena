"""High-precision float64 GRU-style gated selection reference used only for correctness."""
import torch

_CHUNK_ROWS = 8192


def reference(*, x1, x2, reset_weight, reset_bias, update_weight, update_bias, proposal_weight, proposal_bias):
    wr, br = reset_weight.double(), reset_bias.double()
    wu, bu = update_weight.double(), update_bias.double()
    wp, bp = proposal_weight.double(), proposal_bias.double()
    out = torch.empty(x1.shape, dtype=torch.float64, device=x1.device)
    for start in range(0, x1.shape[0], _CHUNK_ROWS):
        a = x1[start:start + _CHUNK_ROWS].double()
        b = x2[start:start + _CHUNK_ROWS].double()
        joined = torch.cat([a, b], dim=-1)
        reset = torch.sigmoid(joined @ wr.t() + br)
        update = torch.sigmoid(joined @ wu.t() + bu)
        proposal = torch.tanh(torch.cat([reset * a, b], dim=-1) @ wp.t() + bp)
        out[start:start + _CHUNK_ROWS] = (1.0 - update) * a + update * proposal
    return out
