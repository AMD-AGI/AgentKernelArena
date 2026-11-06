"""Log-sum-exp generator: normal values with empty partitions and extreme finite values."""
import torch

EMPTY_FRACTION = 1.0 / 32.0
EXTREME = 1000.0


def make_inputs(case, inputs, generator, device):
    """Prefix and suffix log-sum-exp, [num_heads, num_tokens].

    About 1/32 of the entries have an empty prefix (+inf, the FlashAttention-2 encoding) and 1/32 an
    empty suffix (-inf, the FlashAttention-3 encoding); the other partition is always populated.
    A further 1/32 each have a prefix of +1000 (prefix dominates), -1000 (suffix dominates) and both
    partitions at +1000 (equal weights at a large offset), the extreme finite values the previous
    task's controls used.
    """
    shape = tuple(inputs["prefix_output"].shape[1:2]) + (inputs["prefix_output"].shape[0],)
    p = torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * 2.0
    s = torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * 2.0
    kind = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
    p = torch.where(kind < EMPTY_FRACTION, torch.full_like(p, float("inf")), p)
    s = torch.where((kind >= EMPTY_FRACTION) & (kind < 2 * EMPTY_FRACTION),
                    torch.full_like(s, float("-inf")), s)
    f = EMPTY_FRACTION
    high = (kind >= 2 * f) & (kind < 3 * f)
    low = (kind >= 3 * f) & (kind < 4 * f)
    both = (kind >= 4 * f) & (kind < 5 * f)
    p = torch.where(high | both, torch.full_like(p, EXTREME), torch.where(low, torch.full_like(p, -EXTREME), p))
    s = torch.where(both, torch.full_like(s, EXTREME), s)
    return {"prefix_lse": p, "suffix_lse": s}
