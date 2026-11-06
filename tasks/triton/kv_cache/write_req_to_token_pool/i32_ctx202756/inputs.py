"""Scheduler state of one prefill step for the request-to-token pool writer.

Each request owns a distinct row of the pool. It has a cached prefix of 0..max_prefix_len tokens
(slot indices already known) and an extension of max_extend_len minus 0..8 new tokens (so lengths
differ per request, include zero for tiny max_extend_len and cross the 512-element tile boundary
for larger ones). ``out_cache_loc`` holds the newly allocated slots of all extensions back to back,
padded to ``n_requests * max_extend_len``.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, max_extend = axes["n_requests"], axes["max_extend_len"]
    max_prefix, max_rows, width = axes["max_prefix_len"], axes["max_batch"], axes["max_context_len"]
    rows = torch.randperm(max_rows, generator=generator, device=device)[:batch].to(torch.int64)
    pre = torch.randint(0, max_prefix + 1, (batch,), generator=generator, device=device, dtype=torch.int64)
    shrink = torch.randint(0, min(max_extend, 8) + 1, (batch,), generator=generator, device=device,
                           dtype=torch.int64)
    extend = max_extend - shrink
    prefix = torch.randint(0, width, (batch, max_prefix), generator=generator, device=device,
                           dtype=torch.int64)
    out_cache_loc = torch.randint(0, width, (batch * max_extend,), generator=generator, device=device,
                                  dtype=torch.int64)
    return {"req_pool_indices": rows, "prefix_indices": prefix, "pre_lens": pre,
            "seq_lens": pre + extend, "extend_lens": extend, "out_cache_loc": out_cache_loc}
