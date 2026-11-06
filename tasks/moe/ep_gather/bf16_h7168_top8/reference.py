"""High-precision expert-parallel gather, used only for correctness."""
import torch

CHUNK = 1024


def reference(*, input_tensor, recv_topk_ids, recv_topk_weight, input_index):
    n_tokens, top_k = recv_topk_ids.shape
    out = torch.empty(n_tokens, input_tensor.shape[1], dtype=torch.float64, device=input_tensor.device)
    for start in range(0, n_tokens, CHUNK):
        stop = min(start + CHUNK, n_tokens)
        valid = recv_topk_ids[start:stop] >= 0
        slots = input_index[start:stop].to(torch.int64).clamp(min=0)
        weight = recv_topk_weight[start:stop].double() * valid
        rows = input_tensor[slots].double()  # [chunk, top_k, hidden]
        out[start:stop] = (rows * weight.unsqueeze(-1)).sum(dim=1)
    return out
