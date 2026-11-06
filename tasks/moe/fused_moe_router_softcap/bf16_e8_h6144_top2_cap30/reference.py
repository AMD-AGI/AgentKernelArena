"""float64 soft-capped top-2 router reference, used only for correctness."""
import torch

_CHUNK = 16384


def reference(*, x, router_weight, moe_softcapping):
    topk = 2
    w = router_weight.double()
    weights, ids = [], []
    for start in range(0, x.shape[0], _CHUNK):
        logits = x[start:start + _CHUNK].double() @ w.T
        cap = float(moe_softcapping)
        if cap != 0.0:  # upstream treats 0 as "soft-capping disabled"
            logits = torch.tanh(logits / cap) * cap
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(logits, topk, dim=-1).indices
        weights.append(probs.gather(1, top))
        ids.append(top.to(torch.int32))
    return torch.cat(weights), torch.cat(ids)
