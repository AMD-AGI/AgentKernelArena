"""Exact MoE sorting with block padding, used only for correctness."""
import torch


def reference(*, topk_ids, topk_weights, num_experts, block_size):
    n_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    pairs = n_tokens * top_k
    expert = topk_ids.reshape(-1).to(torch.int64)                  # pair p = token * top_k + slot
    order = torch.sort(expert, stable=True).indices                 # by expert, (token, slot) ascending inside
    sorted_expert = expert[order]
    counts = torch.bincount(expert, minlength=num_experts)
    blocks = (counts + block_size - 1) // block_size
    padded = blocks * block_size
    group_start = torch.cumsum(counts, 0) - counts
    padded_start = torch.cumsum(padded, 0) - padded
    position = padded_start[sorted_expert] + torch.arange(pairs, device=device) - group_start[sorted_expert]
    length = int(padded.sum())
    sentinel = (top_k << 24) | n_tokens
    sorted_ids = torch.full((length,), sentinel, dtype=torch.int64, device=device)
    sorted_ids[position] = ((order % top_k) << 24) | (order // top_k)
    sorted_weights = torch.zeros(length, dtype=torch.float32, device=device)
    sorted_weights[position] = topk_weights.reshape(-1)[order].float()
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=device), blocks)
    num_valid_ids = torch.tensor([length, n_tokens], dtype=torch.int64, device=device)
    moe_buf = torch.zeros(n_tokens, 7168, dtype=torch.float32, device=device)
    return {"sorted_ids": sorted_ids.to(torch.int32), "sorted_weights": sorted_weights,
            "sorted_expert_ids": expert_ids.to(torch.int32), "num_valid_ids": num_valid_ids.to(torch.int32),
            "moe_buf": moe_buf}
