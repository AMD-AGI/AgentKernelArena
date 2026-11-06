import torch


def make_inputs(case, inputs, generator, device):
    """Float32 rows that are standard-normal in distribution with all values distinct.

    Each row holds the normal quantiles at (j + 0.5) / n, j = 0..n-1, placed at random columns, so the
    top-k set and its order are unambiguous. This keeps the exact comparison independent of how an
    implementation breaks ties.
    """
    n_rows = case["axes"]["n_rows"]
    n = case["axes"]["vocab_size"]
    levels = torch.special.ndtri((torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n)
    levels = levels.to(torch.float32)
    x = torch.empty(n_rows, n, dtype=torch.float32, device=device)
    chunk = 64
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        order = torch.rand(stop - start, n, generator=generator, device=device).argsort(dim=-1)
        x[start:stop].scatter_(1, order, levels.expand(stop - start, n))
    return {"x": x}
