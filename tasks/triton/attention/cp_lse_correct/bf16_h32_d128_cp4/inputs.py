"""Generator for the stacked per-rank log-sum-exp values."""
import torch

# A rank whose KV slice is empty for a token reports -inf; a small fraction of entries is masked.
EMPTY_RANK_FRACTION = 0.02


def make_inputs(case, inputs, generator, device):
    a = case["axes"]
    shape = (a["cp_size"], a["n_tokens"], a["n_heads"])
    lses = torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * 2.0 + 4.0
    empty = torch.rand(shape, generator=generator, device=device, dtype=torch.float32) < EMPTY_RANK_FRACTION
    # Keep at least one finite rank per (token, head) so the global LSE is finite.
    empty[0] = False
    lses = torch.where(empty, torch.full_like(lses, float("-inf")), lses)
    return {"lses": lses}
