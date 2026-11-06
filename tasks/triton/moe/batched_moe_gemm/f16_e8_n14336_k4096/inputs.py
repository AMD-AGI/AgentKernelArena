"""Per-expert token counts for the batched MoE GEMM task."""
import torch


def make_inputs(case, inputs, generator, device):
    n_experts = case["axes"]["n_experts"]
    max_tokens = case["axes"]["max_tokens"]
    counts = torch.randint(0, max_tokens + 1, (n_experts,), generator=generator, device=device,
                           dtype=torch.int64)
    # One full, one empty and one single-token expert exercise the tile, skip and ragged-tail paths.
    counts[0] = max_tokens
    counts[1] = 0
    counts[2] = 1
    if case["id"].endswith("_empty"):
        # Every expert empty: the whole output must be zero and no tile may read A or B.
        counts.zero_()
    return {"expert_num_tokens": counts.to(torch.int32)}
