"""Float64 reference for swish(x @ weight.T + bias) * scaling_factor, used only for correctness."""
import torch

# Output features per float64 weight slice: 4096 x 32768 doubles (1 GiB) at a time instead of the
# whole 8 GiB float64 weight.
CHUNK = 4096


def reference(*, x, weight, bias, scaling_factor):
    x64 = x.double()
    out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, weight.shape[0], CHUNK):
        stop = min(start + CHUNK, weight.shape[0])
        # nn.Linear: z = x @ weight.T + bias.
        z = torch.addmm(bias[start:stop].double(), x64, weight[start:stop].double().t())
        # Swish z * sigmoid(z), then the scaling factor.
        out[:, start:stop] = torch.sigmoid(z).mul_(z).mul_(scaling_factor)
    return out
