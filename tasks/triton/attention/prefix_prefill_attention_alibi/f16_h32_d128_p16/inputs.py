"""Block table, sequence metadata and ALiBi slopes for the paged prefix-prefill scenario."""
import torch


def make_inputs(case, inputs, generator, device):
    a = case["axes"]
    batch, q_len, ctx_len, n_heads = a["batch"], a["q_len"], a["ctx_len"], a["n_heads"]
    max_blocks, n_pages = a["max_blocks"], a["n_pages"]
    # Every logical page of every sequence is mapped to a distinct, randomly placed physical page.
    pages = torch.randperm(n_pages, generator=generator, device=device).to(torch.int32)
    b_loc = pages.reshape(batch, max_blocks)
    b_start_loc = torch.arange(batch + 1, device=device, dtype=torch.int32) * q_len
    b_seq_len = torch.full((batch,), ctx_len + q_len, device=device, dtype=torch.int32)
    # ALiBi slopes of a power-of-two head count: 2^(-8 (h + 1) / n_heads) (Press et al.; Bloom, MPT).
    assert n_heads & (n_heads - 1) == 0
    exponent = -8.0 * torch.arange(1, n_heads + 1, dtype=torch.float64) / n_heads
    alibi_slopes = torch.exp2(exponent).to(torch.float32).to(device)
    return {"b_loc": b_loc, "b_start_loc": b_start_loc, "b_seq_len": b_seq_len,
            "alibi_slopes": alibi_slopes}
