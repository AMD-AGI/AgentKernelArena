"""Float64 reference for GroupNorm(swish(x @ weight.T + bias) + extra_bias), used only for correctness."""
import torch

NUM_GROUPS = 64


def reference(*, x, weight, bias, extra_bias, gn_weight, gn_bias, epsilon):
    # nn.Linear: z = x @ weight.T + bias.
    z = torch.addmm(bias.double(), x.double(), weight.double().t())
    # Swish z * sigmoid(z), then the per-feature extra bias.
    h = torch.sigmoid(z).mul_(z).add_(extra_bias.double())
    del z
    rows, features = h.shape
    # GroupNorm over 64 groups of 64 contiguous features per row: mean and biased variance.
    grouped = h.view(rows, NUM_GROUPS, features // NUM_GROUPS)
    variance, mean = torch.var_mean(grouped, dim=-1, correction=0, keepdim=True)
    normalized = grouped.sub_(mean).mul_(torch.rsqrt(variance + epsilon)).view(rows, features)
    return normalized.mul_(gn_weight.double()).add_(gn_bias.double())
