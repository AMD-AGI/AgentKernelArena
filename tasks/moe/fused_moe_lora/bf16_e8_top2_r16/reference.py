"""Float64 reference for the fused MoE LoRA (w13 gate/up) delta, used only for correctness."""
import torch

TOP_K = 2
BLOCK_SIZE = 64


def reference(*, hidden_states, lora_a, lora_b, sorted_token_ids, expert_ids, num_tokens_post_padded,
              lora_ids, adapter_enabled):
    n_tokens = hidden_states.shape[0]
    num_slices, max_loras, _, _, _ = lora_a.shape
    n_per_slice = lora_b.shape[3]
    num_routes = n_tokens * TOP_K
    x = hidden_states.double()
    delta = torch.zeros(num_routes, num_slices, n_per_slice, dtype=torch.float64, device=x.device)
    enabled = adapter_enabled.tolist()
    for slot in sorted(set(lora_ids.tolist())):
        # Base-model tokens (-1) and adapters without MoE weights get no delta.
        if slot < 0 or slot >= max_loras or enabled[slot] == 0:
            continue
        n_blocks = min(-(-int(num_tokens_post_padded[slot]) // BLOCK_SIZE), expert_ids.shape[1])
        routes = sorted_token_ids[slot, : n_blocks * BLOCK_SIZE].long().reshape(n_blocks, BLOCK_SIZE)
        experts = expert_ids[slot, :n_blocks].long()[:, None].expand(n_blocks, BLOCK_SIZE)
        valid = (routes < num_routes) & (experts >= 0)
        routes, experts = routes[valid], experts[valid]
        for expert in experts.unique().tolist():
            chosen = routes[experts == expert]
            # shrink: [slices, routes, rank]; expand: [routes, slices, n_per_slice]
            shrink = torch.einsum("tk,srk->str", x[chosen // TOP_K], lora_a[:, slot, expert].double())
            delta[chosen] = torch.einsum("str,snr->tsn", shrink, lora_b[:, slot, expert].double())
    return delta.reshape(n_tokens, TOP_K, num_slices * n_per_slice)
