"""Float64 reference of the fused router: sigmoid(x @ w), top-1 expert, shared expert column."""
import torch

CHUNK_ROWS = 8192


def reference(*, x, w):
    n_tokens = x.shape[0]
    n_experts = w.shape[1]
    w64 = w.double()
    ids = torch.empty((n_tokens, 2), dtype=torch.int32, device=x.device)
    weights = torch.empty((n_tokens, 2), dtype=torch.float64, device=x.device)
    for start in range(0, n_tokens, CHUNK_ROWS):
        logits = x[start:start + CHUNK_ROWS].double() @ w64
        top_id = logits.argmax(dim=1)  # first maximum wins
        top_logit = logits.gather(1, top_id[:, None])[:, 0]
        # sigmoid is monotonic, so the first maximal logit is the first maximal score.
        ids[start:start + CHUNK_ROWS, 0] = top_id.to(torch.int32)
        weights[start:start + CHUNK_ROWS, 0] = torch.sigmoid(top_logit)
    ids[:, 1] = n_experts
    weights[:, 1] = 1.0
    return ids, weights
