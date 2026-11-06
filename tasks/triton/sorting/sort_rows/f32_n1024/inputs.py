import torch


def make_inputs(case, inputs, generator, device):
    """Float32 rows, half all-distinct normal values and half rows with many repeated values.

    Even rows are standard normal (all values distinct). Odd rows draw each element at random from a pool
    of n/8 normal float32 values of that row, so every value repeats about eight times on average while the
    values themselves stay full-precision float32 (not exactly representable in bfloat16 or float16).
    Sorting must keep every duplicate.
    """
    n_rows = case["axes"]["n_rows"]
    n = case["axes"]["row_size"]
    x = torch.randn(n_rows, n, generator=generator, device=device, dtype=torch.float32)
    pool = torch.randn(n_rows, n // 8, generator=generator, device=device, dtype=torch.float32)
    idx = torch.randint(0, n // 8, (n_rows, n), generator=generator, device=device)
    repeated = torch.gather(pool, 1, idx)
    odd = (torch.arange(n_rows, device=device) % 2 == 1).unsqueeze(-1)
    return {"x": torch.where(odd, repeated, x)}
