"""Float64 non-local block (dot-product, ELU energy), used only for correctness."""
import torch

MAX_ENERGY_ELEMENTS = 1 << 26  # float64 energy block of at most 512 MB

def reference(*, x, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias,
              gamma_weight, gamma_bias):
    x64 = x.double()
    n = x64.shape[2]
    q = torch.einsum("oc,bcn->bon", query_weight.double(), x64) + query_bias.double()[None, :, None]
    k = torch.einsum("oc,bcn->bon", key_weight.double(), x64) + key_bias.double()[None, :, None]
    v = torch.einsum("oc,bcn->bon", value_weight.double(), x64) + value_bias.double()[None, :, None]
    y = torch.zeros_like(v)
    row_chunk = max(16, MAX_ENERGY_ELEMENTS // (x64.shape[0] * n))
    for i0 in range(0, n, row_chunk):  # chunk the summed position i to bound the n x n energy
        energy = torch.einsum("bdi,bdj->bij", q[:, :, i0:i0 + row_chunk], k)
        energy = torch.where(energy > 0, energy, torch.expm1(energy)) / n
        y += torch.einsum("bci,bij->bcj", v[:, :, i0:i0 + row_chunk], energy)
    return torch.einsum("oc,bcn->bon", gamma_weight.double(), y) + gamma_bias.double()[None, :, None]
