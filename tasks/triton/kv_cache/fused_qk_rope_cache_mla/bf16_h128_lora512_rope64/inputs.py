import torch


def make_inputs(case, inputs, generator, device):
    """Interleaved-RoPE cos/sin tables (base 10000) and a slot mapping with unique, valid slots.

    No slot is negative: the baseline kernel leaves k_pe_out unwritten for a padded (negative-slot)
    token, so padding would make that output undefined.
    """
    axes = case["axes"]
    n_tokens, max_position = axes["n_tokens"], axes["max_position"]
    rope_dim, num_slots = axes["rope_dim"], axes["num_slots"]
    inv_freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    slots = torch.randperm(num_slots, generator=generator, device=device)[:n_tokens]
    slot_mapping = slots.to(torch.int64)
    return {"cos": torch.cos(angles).to(torch.bfloat16), "sin": torch.sin(angles).to(torch.bfloat16),
            "slot_mapping": slot_mapping}
