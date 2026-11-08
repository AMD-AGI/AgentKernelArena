"""RoPE tables, positions, ring-buffer routing for the sliding-window write."""
import torch

_TOKENS_PER_SEQUENCE = 4  # one decode step with three speculative tokens


def make_inputs(case, inputs, generator, device):
    """Tokens come in sequences of up to four consecutive positions, so no two tokens of a sequence share a ring cell.

    The kernel writes ring cell (slot, position % window) from several programs without ordering, so two written
    tokens that address the same cell would make swa_kv depend on the schedule. Sequences own distinct slots,
    consecutive positions (at most 4 < window) of one sequence have distinct ring indices, and write_indices is a
    permutation of the token rows, so every cell is written exactly once.
    """
    axes = case["axes"]
    n, max_position, rope_dim = axes["n_tokens"], axes["max_position"], axes["rope_dim"]
    num_slots = axes["num_slots"]
    inv_freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    token = torch.arange(n, device=device)
    sequence = token // _TOKENS_PER_SEQUENCE
    num_sequences = (n + _TOKENS_PER_SEQUENCE - 1) // _TOKENS_PER_SEQUENCE
    base = torch.randint(0, max_position - _TOKENS_PER_SEQUENCE, (num_sequences,), generator=generator, device=device)
    positions = base[sequence] + token % _TOKENS_PER_SEQUENCE
    return {
        "positions": positions.to(torch.int64),
        "cos": torch.cos(angles).to(torch.bfloat16),
        "sin": torch.sin(angles).to(torch.bfloat16),
        "write_indices": torch.randperm(n, generator=generator, device=device).to(torch.int32),
        "batch_id_per_token": sequence.to(torch.int32),
        "state_slot_mapping": torch.randperm(num_slots, generator=generator, device=device).to(torch.int32),
    }
