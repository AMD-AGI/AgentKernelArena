"""High-precision gated residual + LayerNorm + select-modulate reference used only for correctness."""
import torch

_CHUNK_ROWS = 4096


def reference(*, x, residual, residual_gate, scale0, shift0, gate0, scale1, shift1, gate1, index, epsilon):
    batch, seq_len, hidden = x.shape
    out = torch.empty((batch, seq_len, hidden), dtype=torch.float64, device=x.device)
    residual_out = torch.empty_like(out)
    gate_out = torch.empty_like(out)
    for b in range(batch):
        mods = [m[b].double() for m in (scale0, shift0, gate0, scale1, shift1, gate1)]
        for start in range(0, seq_len, _CHUNK_ROWS):
            rows = slice(start, start + _CHUNK_ROWS)
            # The residual stream is kept unrounded here; the kernel normalizes its fp32 value.
            res = residual[b, rows].double() + residual_gate[b, rows].double() * x[b, rows].double()
            mean = res.mean(dim=-1, keepdim=True)
            centered = res - mean
            var = centered.square().mean(dim=-1, keepdim=True)
            normed = centered * torch.rsqrt(var + epsilon)
            second = (index[b, rows] != 0).unsqueeze(-1)
            scale = torch.where(second, mods[3], mods[0])
            shift = torch.where(second, mods[4], mods[1])
            gate = torch.where(second, mods[5], mods[2])
            out[b, rows] = normed * (1.0 + scale) + shift
            residual_out[b, rows] = res
            gate_out[b, rows] = gate
    return out, residual_out, gate_out
