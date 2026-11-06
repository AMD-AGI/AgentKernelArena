"""vLLM LoRA token-routing metadata for a batch with four active adapters.

Each token is assigned to one adapter group at random (group sizes are skewed). The metadata is what vLLM's
LoRAKernelMeta produces: token indices sorted by group (stable), the token count and the start offset of each group,
and the adapter slot served by each group (a permutation of the slots, so the slot is not the group index).
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, n_adapters = axes["n_tokens"], axes["n_adapters"]
    lora_ids = torch.randperm(n_adapters, generator=generator, device=device).to(torch.int32)
    share = torch.rand(n_adapters, generator=generator, device=device) + 0.25
    boundaries = (share / share.sum()).cumsum(0)[:-1]
    group = torch.bucketize(torch.rand(n_tokens, generator=generator, device=device), boundaries)
    token_indices_sorted = torch.sort(group, stable=True).indices.to(torch.int64)
    counts = torch.bincount(group, minlength=n_adapters)
    start = torch.cat([torch.zeros(1, dtype=counts.dtype, device=device), counts.cumsum(0)])
    return {
        "token_indices_sorted": token_indices_sorted,
        "num_tokens_per_lora": counts.to(torch.int32),
        "lora_token_start_loc": start.to(torch.int32),
        "lora_ids": lora_ids,
    }
