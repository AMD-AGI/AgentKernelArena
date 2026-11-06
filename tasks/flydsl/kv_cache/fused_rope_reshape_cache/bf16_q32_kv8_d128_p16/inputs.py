import torch


def make_inputs(case, inputs, generator, device):
    """NeoX cos/sin tables (base 10000, first-half angles) and a slot mapping with unique slots.

    About one token in sixteen is padding (slot -1), as in a padded batch.
    """
    axes = case["axes"]
    n_tokens, max_position = axes["n_tokens"], axes["max_position"]
    head_dim, num_slots = axes["head_dim"], axes["num_slots"]
    inv_freq = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    slots = torch.randperm(num_slots, generator=generator, device=device)[:n_tokens]
    padding = torch.rand(n_tokens, generator=generator, device=device) < 1.0 / 16.0
    slot_mapping = torch.where(padding, torch.full_like(slots, -1), slots).to(torch.int32)
    return {"cos_cache": torch.cos(angles).to(torch.bfloat16),
            "sin_cache": torch.sin(angles).to(torch.bfloat16),
            "slot_mapping": slot_mapping}
