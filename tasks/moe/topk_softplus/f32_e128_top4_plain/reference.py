"""float64 sqrt-softplus top-k routing reference, used only for correctness."""
import torch


def reference(*, gating_output, correction_bias, route_scale):
    topk = 4
    zero = torch.zeros((), dtype=torch.float64, device=gating_output.device)
    scores = torch.sqrt(torch.logaddexp(gating_output.double(), zero))
    choice = scores + correction_bias.double().unsqueeze(0)
    ids = torch.topk(choice, k=topk, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights * float(route_scale)
    return weights, ids.to(torch.int32)
