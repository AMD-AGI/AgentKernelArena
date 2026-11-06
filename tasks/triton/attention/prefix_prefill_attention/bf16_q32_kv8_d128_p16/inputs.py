"""Block table and sequence metadata for the paged prefix-prefill scenario."""
import torch


def make_inputs(case, inputs, generator, device):
    a = case["axes"]
    batch, q_len, ctx_len = a["batch"], a["q_len"], a["ctx_len"]
    max_blocks, n_pages = a["max_blocks"], a["n_pages"]
    # Every logical page of every sequence is mapped to a distinct, randomly placed physical page.
    pages = torch.randperm(n_pages, generator=generator, device=device).to(torch.int32)
    b_loc = pages.reshape(batch, max_blocks)
    b_start_loc = torch.arange(batch + 1, device=device, dtype=torch.int32) * q_len
    b_seq_len = torch.full((batch,), ctx_len + q_len, device=device, dtype=torch.int32)
    return {"b_loc": b_loc, "b_start_loc": b_start_loc, "b_seq_len": b_seq_len}
