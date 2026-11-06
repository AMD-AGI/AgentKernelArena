"""High-precision float64 position-wise FFN + residual + LayerNorm reference used only for correctness."""
import torch

_CHUNK_TOKENS = 8192


def reference(*, x, w1_weight, w1_bias, w2_weight, w2_bias, ln_gamma, ln_beta, eps):
    w1, b1 = w1_weight.double(), w1_bias.double()
    w2, b2 = w2_weight.double(), w2_bias.double()
    gamma, beta = ln_gamma.double(), ln_beta.double()
    epsilon = float(eps)
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_TOKENS):
        xc = x[start:start + _CHUNK_TOKENS].double()
        hidden = torch.relu(xc @ w1.t() + b1)
        y = hidden @ w2.t() + b2 + xc
        mean = y.mean(dim=-1, keepdim=True)
        var = (y - mean).square().mean(dim=-1, keepdim=True)
        out[start:start + _CHUNK_TOKENS] = (y - mean) * torch.rsqrt(var + epsilon) * gamma + beta
    return out
