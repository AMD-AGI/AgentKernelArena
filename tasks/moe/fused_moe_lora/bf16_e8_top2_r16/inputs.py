"""Routing and LoRA-slot metadata for the fused MoE LoRA task.

The tables follow vLLM's moe_lora_align_block_size layout: for every LoRA slot,
the routes (token * top_k + k) of that slot's tokens grouped by expert, each
group padded to a multiple of block_size with the value num_routes, and one
expert id per block (-1 past the padded total).
"""
import torch


def _alignment(route_experts, member, num_experts, block, padded, num_routes, generator, device):
    """sorted ids, per-block experts and padded total of the routes flagged in ``member``."""
    routes = torch.nonzero(member).flatten()
    experts = route_experts[routes]
    # The CUDA alignment kernel places routes with atomics: no order inside a group.
    key = experts.double() + torch.rand(routes.shape, generator=generator, device=device, dtype=torch.float64)
    order = torch.argsort(key)
    routes, experts = routes[order], experts[order]
    counts = torch.bincount(experts, minlength=num_experts)
    padded_counts = (counts + block - 1) // block * block
    ends = torch.cumsum(padded_counts, 0)
    group_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(routes.numel(), device=device) - group_start[experts]
    sorted_ids = torch.full((padded,), num_routes, dtype=torch.int32, device=device)
    sorted_ids[(ends - padded_counts)[experts] + rank] = routes.to(torch.int32)
    total = int(ends[-1])
    block_experts = torch.full((-(-padded // block),), -1, dtype=torch.int32, device=device)
    starts = torch.arange(total // block, device=device) * block
    block_experts[: total // block] = torch.searchsorted(ends, starts, right=True).to(torch.int32)
    return sorted_ids, block_experts, total


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, top_k = axes["n_tokens"], axes["top_k"]
    num_experts, max_loras, block = axes["num_experts"], axes["max_loras"], axes["block_size"]
    padded = axes["max_num_tokens_padded"]
    num_routes = n_tokens * top_k

    # Router: top_k distinct experts per token.
    scores = torch.rand((n_tokens, num_experts), generator=generator, device=device)
    route_experts = torch.argsort(scores, dim=1)[:, :top_k].reshape(-1)

    # One adapter slot holds a LoRA without MoE weights (adapter_enabled = 0,
    # as vLLM's reset_lora leaves it); the extra entry max_loras is never set.
    adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    adapter_enabled[:max_loras] = 1
    disabled = int(torch.randint(0, max_loras, (1,), generator=generator, device=device))
    adapter_enabled[disabled] = 0
    enabled_slots = [slot for slot in range(max_loras) if slot != disabled]

    # Requests of contiguous tokens, each with one adapter or none (-1); the
    # first request always uses an enabled MoE adapter.
    n_requests = max(1, min(n_tokens, 8))
    choice = torch.randint(-1, max_loras, (n_requests,), generator=generator, device=device)
    choice[0] = enabled_slots[int(torch.randint(0, len(enabled_slots), (1,), generator=generator, device=device))]
    sizes = [n_tokens // n_requests + (1 if i < n_tokens % n_requests else 0) for i in range(n_requests)]
    token_lora = torch.repeat_interleave(choice, torch.tensor(sizes, device=device))
    route_lora = torch.repeat_interleave(token_lora, top_k)

    # active_lora_ids: the sorted distinct adapters of the batch (-1 first when
    # base-model tokens are present), padded with -1 to max_loras + 1 entries.
    present = torch.unique(token_lora, sorted=True).to(torch.int32)
    lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=device)
    lora_ids[: present.numel()] = present
    listed = set(present.tolist())

    sorted_token_ids = torch.empty((max_loras, padded), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_loras, axes["max_num_m_blocks"]), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32, device=device)
    for slot in range(max_loras):
        if slot in listed and int(adapter_enabled[slot]):
            member = route_lora == slot
        else:
            # The alignment kernel skips this slot (vLLM leaves its token
            # rows uninitialized and its experts at -1); fill all its rows
            # with a stale, in-range alignment so it must be skipped by id.
            member = torch.rand((num_routes,), generator=generator, device=device) < 0.5
        rows = _alignment(route_experts, member, num_experts, block, padded, num_routes, generator, device)
        sorted_token_ids[slot], expert_ids[slot], num_tokens_post_padded[slot] = rows
    return {"sorted_token_ids": sorted_token_ids, "expert_ids": expert_ids,
            "num_tokens_post_padded": num_tokens_post_padded, "lora_ids": lora_ids,
            "adapter_enabled": adapter_enabled}
