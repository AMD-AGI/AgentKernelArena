"""High-precision LoRA shrink reference used only for correctness."""
import torch

_CHUNK = 8192


def reference(*, x, lora_a, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling):
    n_tokens = x.shape[0]
    out = torch.zeros(n_tokens, lora_a.shape[1], dtype=torch.float64, device=x.device)
    for group in range(lora_ids.shape[0]):
        slot = int(lora_ids[group])
        if slot < 0:
            continue
        start = int(lora_token_start_loc[group])
        count = int(num_tokens_per_lora[group])
        weight = lora_a[slot].double()
        for offset in range(0, count, _CHUNK):
            rows = token_indices_sorted[start + offset:start + min(offset + _CHUNK, count)].long()
            out[rows] = (x[rows].double() @ weight.T) * float(scaling)
    return out
