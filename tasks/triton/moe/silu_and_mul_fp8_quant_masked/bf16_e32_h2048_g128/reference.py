"""Float64 reference: masked SiLU * mul and per-group FP8 quantization (unrounded codes)."""
import torch

GROUP_SIZE = 128
FP8_MAX = 448.0  # float8_e4m3fn
EPS = 1e-10


def reference(*, y, tokens_per_expert):
    E, T, H2 = y.shape
    H = H2 // 2
    G = H // GROUP_SIZE
    y_q = torch.zeros(E, T, H, dtype=torch.float64, device=y.device)
    y_s = torch.zeros(E, G, T, dtype=torch.float64, device=y.device)
    for e in range(E):
        n = int(tokens_per_expert[e])
        if n == 0:
            continue
        x = y[e, :n].double()
        gate, up = x[:, :H], x[:, H:]
        value = gate * torch.sigmoid(gate) * up
        groups = value.reshape(n, G, GROUP_SIZE)
        scale = groups.abs().amax(dim=-1).clamp_min(EPS) / FP8_MAX
        q = (groups / scale[..., None]).clamp(-FP8_MAX, FP8_MAX)
        y_q[e, :n] = q.reshape(n, H)
        y_s[e, :, :n] = scale.transpose(0, 1)
    return y_q, y_s
